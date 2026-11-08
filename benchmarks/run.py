"""Measure generation: each workload at each size, repeated, every run in a fresh process so its peak
memory is its own. Phases are timed apart: construct (declare and check the schema), realize (draw
ranges and fit cuts), sample, database (relbench form), write (to parquet) and read (load back).

    python benchmarks/run.py --out a.json                     # the full ladder: 3 seeds, 2 repeats each
    python benchmarks/run.py --sizes 1000 --seeds 0 --repeats 1 --out /tmp/smoke.json   # a smoke run

Seeds vary the draw, repeats rerun one seed, so their spreads tell draws from timing noise apart.
There is no warm-up: every run is a cold process, and imports are not timed. Results go to one
JSON file: the environment, then one record per run, failures and skipped sizes included."""

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
SIZES = (1_000, 10_000, 100_000, 1_000_000)


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
        # changes to what is measured: the library and the recipes, not the harness itself
        "dirty": bool(
            run("git", "-C", str(HERE.parent), "status", "--porcelain", "--", "plurel", "examples")
        ),
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


def copied_cells(generator, frames) -> int:
    """Cells holding given values, copied rather than drawn: real columns and given keys."""
    copied = 0
    for name, scm in generator.tables.items():
        given = [
            c for c, entry in scm.columns.items() if getattr(entry, "values", None) is not None
        ]
        given += [c for c, fk in scm.fkeys.items() if fk.values is not None]
        copied += len(frames[name]) * len(given)
    return copied


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

    schema, counts = timed("construct", lambda: WORKLOADS[workload](rows))
    generator = timed("realize", lambda: schema.realize(seed))
    frames = timed("sample", lambda: generator.sample(counts, seed=seed))
    db = timed("database", lambda: create_database(generator, frames))
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
    given = [t for t, scm in generator.tables.items() if scm.size is not None]
    retained = sum(len(frames[t]) for t in given)
    total = sum(len(frame) for frame in frames.values())
    cells = int(sum(frame.size for frame in frames.values()))
    return {
        "seconds": seconds,
        "total_seconds": round(sum(seconds.values()), 4),
        "rows": total,
        "generated_rows": total - retained,
        "retained_rows": retained,
        "cells": cells,
        "generated_cells": cells - copied_cells(generator, frames),
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
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=2, help="runs of each seed")
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
    from workloads import LARGEST, WORKLOADS

    names = args.workloads or list(WORKLOADS)
    if unknown := sorted(set(names) - set(WORKLOADS)):
        sys.exit(f"unknown workloads {unknown}; there are {sorted(WORKLOADS)}")
    started = environment()  # before the runs, so a checkout changing meanwhile is caught
    records = []
    for name in names:
        for rows in args.sizes:
            if rows > LARGEST.get(name, rows):
                records.append(
                    {"workload": name, "size": rows, "skipped": f"above {LARGEST[name]:,} rows"}
                )
                continue
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
