"""Measure generation as a corpus is generated: draw tables and databases from the default priors,
one per seed, every draw in a fresh process so its peak memory is its own, and time its phases in
the order a user of the priors runs them: draw (the structure from its prior), sample, database
(relbench form), write (to parquet) and read (load back).

    python benchmarks/run.py --out a.json                               # 100 tables and 100 databases
    python benchmarks/run.py --count 3 --out /tmp/smoke.json             # a smoke run

Draw `seed` samples with data seed 0. There is no warm-up: every run is a cold process, and imports
are not timed. Results go to one JSON file: the environment, then one record per draw, failures
included."""

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


def one(kind: str, seed: int) -> dict[str, object]:
    """One draw, in this process: every phase timed, its shape, sizes and peak memory recorded."""
    sys.path.insert(0, str(HERE))
    from workloads import KINDS

    from plurel import create_database, read_database, split_timestamps, write_database

    seconds: dict[str, float] = {}

    def timed(phase, step):
        start = time.perf_counter()
        out = step()
        seconds[phase] = round(time.perf_counter() - start, 4)
        return out

    schema, counts = timed("draw", lambda: KINDS[kind](seed))
    frames = timed("sample", lambda: schema.sample(counts, seed=0))
    db = timed("database", lambda: create_database(schema, frames))
    with tempfile.TemporaryDirectory() as scratch:
        timed_tables = any(table.time_col for table in db.table_dict.values())
        split = (
            dict(zip(("val_timestamp", "test_timestamp"), split_timestamps(db)))
            if timed_tables
            else {}
        )
        path = timed("write", lambda: write_database(db, Path(scratch) / "db", name=kind, **split))
        disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        timed("read", lambda: read_database(path))
    return {
        "seconds": seconds,
        "total_seconds": round(sum(seconds.values()), 4),
        "tables": len(frames),
        "keys": len(schema.keys),
        "rows": sum(len(frame) for frame in frames.values()),
        "cells": int(sum(frame.size for frame in frames.values())),
        "disk_mib": round(disk / 2**20, 2),
        "peak_mib": peak_mib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinds", nargs="*", help="table, database; both by default")
    parser.add_argument(
        "--count", type=int, default=100, help="draws of each kind, seeds 0 to count - 1"
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per draw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--one", nargs=2, metavar=("KIND", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(one(args.one[0], int(args.one[1]))))
        return
    sys.path.insert(0, str(HERE))
    from workloads import KINDS

    kinds = args.kinds or list(KINDS)
    if unknown := sorted(set(kinds) - set(KINDS)):
        sys.exit(f"unknown kinds {unknown}; there are {sorted(KINDS)}")
    started = environment()  # before the runs, so a checkout changing meanwhile is caught
    records = []
    for kind in kinds:
        for seed in range(args.count):
            record = {"kind": kind, "seed": seed}
            command = [sys.executable, "-I", __file__, "--out", "-", "--one", kind, str(seed)]
            try:
                done = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
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
                record["failure"], record["failure_kind"] = f"over {args.timeout:g} s", "timeout"
            records.append(record)
            status = record.get("failure") or (
                f"{record['rows']:>8,} rows, {record['tables']:>3} tables: "
                f"{record['total_seconds']:.2f} s, {record['peak_mib']:.0f} MiB"
            )
            print(f"{kind:9} {seed:>4}: {status}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if environment()["commit"] != started["commit"]:
        started["commit_changed_during_runs"] = True
        print("warning: the checkout changed commit during the runs", file=sys.stderr)
    args.out.write_text(json.dumps({"environment": started, "runs": records}, indent=1) + "\n")
    print(f"wrote {len(records)} draws to {args.out}")


if __name__ == "__main__":
    main()
