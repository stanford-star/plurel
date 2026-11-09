"""Measure generation: each workload at each size, repeated, every run in a fresh process so its peak
memory is its own. Phases are timed apart, in the order a user of the priors runs them: draw (the
structure from its prior), sample, database (relbench form), write (to parquet) and read (load
back).

    python benchmarks/run.py --out a.json                                  # every workload and size, 3 runs
    python benchmarks/run.py --sizes 100000 --repeats 1 --out /tmp/smoke.json   # a smoke run

Every run draws with seed 0, so repeats differ only in timing. There is no warm-up: every run is a
cold process, and imports are not timed. Results go to one JSON file: the environment, then one
record per run, failures included."""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = (100_000, 250_000, 500_000, 1_000_000)  # rows drawn, in total


def environment() -> dict[str, object]:
    def run(*command: str) -> str:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    if sys.platform == "darwin":
        cpu = run("sysctl", "-n", "machdep.cpu.brand_string")
    else:
        names = (
            [
                line
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ]
            if Path("/proc/cpuinfo").exists()
            else []
        )
        cpu = names[0].split(":", 1)[1].strip() if names else platform.processor() or "unknown"
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": run("git", "-C", str(HERE), "rev-parse", "--short", "HEAD"),
        # changes to what is measured: the library, not the harness itself
        "dirty": bool(run("git", "-C", str(HERE.parent), "status", "--porcelain", "--", "plurel")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "memory_gib": round(memory / 2**30, 1),
        "python": platform.python_version(),
        "packages": {
            p: version(p) for p in ("plurel", "numpy", "pandas", "pyarrow", "scipy", "relbench")
        },
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def peak_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1)  # bytes vs KiB


def one(workload: str, rows: int, seed: int) -> dict[str, object]:
    """One run, in this process: every phase timed, sizes and peak memory recorded."""
    sys.path.insert(0, str(HERE))
    from workloads import WORKLOADS

    from plurel import create_database, read_database, split_timestamps, write_database

    seconds: dict[str, float] = {}

    def timed(phase, step):
        start = time.perf_counter()
        out = step()
        seconds[phase] = round(time.perf_counter() - start, 4)
        return out

    schema, counts = timed("draw", lambda: WORKLOADS[workload](rows))
    frames = timed("sample", lambda: schema.sample(counts, seed=seed))
    db = timed("database", lambda: create_database(schema, frames))
    with tempfile.TemporaryDirectory() as scratch:
        timed_tables = any(table.time_col for table in db.table_dict.values())
        split = (
            dict(zip(("val_timestamp", "test_timestamp"), split_timestamps(db)))
            if timed_tables
            else {}
        )
        path = timed(
            "write", lambda: write_database(db, Path(scratch) / "db", name=workload, **split)
        )
        disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        timed("read", lambda: read_database(path))
    return {
        "seconds": seconds,
        "total_seconds": round(sum(seconds.values()), 4),
        "rows": sum(len(frame) for frame in frames.values()),
        "cells": int(sum(frame.size for frame in frames.values())),
        "tables": len(frames),
        "disk_mib": round(disk / 2**20, 2),
        "peak_mib": peak_mib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workloads", nargs="*", help="names from benchmarks/workloads.py; all by default"
    )
    parser.add_argument("--sizes", nargs="*", type=int, default=list(SIZES))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0], help="data seeds")
    parser.add_argument("--repeats", type=int, default=3, help="runs of each seed")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--one", nargs=3, metavar=("WORKLOAD", "ROWS", "SEED"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.one:
        print(json.dumps(one(args.one[0], int(args.one[1]), int(args.one[2]))))
        return
    sys.path.insert(0, str(HERE))
    from workloads import WORKLOADS

    names = args.workloads or list(WORKLOADS)
    if unknown := sorted(set(names) - set(WORKLOADS)):
        sys.exit(f"unknown workloads {unknown}; there are {sorted(WORKLOADS)}")
    started = environment()  # before the runs, so a checkout changing meanwhile is caught
    records = []
    for name in names:
        for rows in args.sizes:
            for seed, repeat in ((s, r) for s in args.seeds for r in range(args.repeats)):
                record = {"workload": name, "size": rows, "repeat": repeat, "seed": seed}
                command = [
                    sys.executable,
                    "-I",
                    __file__,
                    "--out",
                    "-",
                    "--one",
                    name,
                    str(rows),
                    str(seed),
                ]
                try:
                    done = subprocess.run(
                        command, capture_output=True, text=True, timeout=args.timeout
                    )
                    if done.returncode:
                        record["failure"] = (
                            done.stderr.strip().splitlines()[-1]
                            if done.stderr
                            else f"exit {done.returncode}"
                        )
                        record["failure_kind"] = (
                            "out of memory" if done.returncode in (-9, 137) else "error"
                        )
                    else:
                        record |= json.loads(done.stdout.strip().splitlines()[-1])
                except subprocess.TimeoutExpired:
                    record["failure"], record["failure_kind"] = (
                        f"over {args.timeout:g} s",
                        "timeout",
                    )
                records.append(record)
                status = (
                    record.get("failure")
                    or f"{record['total_seconds']:.2f} s, {record['peak_mib']:.0f} MiB"
                )
                print(f"{name:12} {rows:>9} seed {seed} #{repeat}: {status}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if environment()["commit"] != started["commit"]:
        started["commit_changed_during_runs"] = True
        print("warning: the checkout changed commit during the runs", file=sys.stderr)
    args.out.write_text(json.dumps({"environment": started, "runs": records}, indent=1) + "\n")
    print(f"wrote {len(records)} runs to {args.out}")


if __name__ == "__main__":
    main()
