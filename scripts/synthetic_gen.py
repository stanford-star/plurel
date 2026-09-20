"""Generate PluRel databases in the relbench format, one per seed.

Example:
    $ pixi run python scripts/synthetic_gen.py --seed_offset 0 --num_dbs 1000 --num_proc 16
"""

import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import perf_counter

from plurel import SchemaPrior, create_database, split_timestamps, write_database

OUT_DIR = "~/.cache/relbench"
DB_PREFIX = "plurel"


def generate(
    seed: int, out_dir: str, db_prefix: str, val_share: float, test_share: float, overwrite: bool
) -> str:
    start = perf_counter()
    prior = SchemaPrior()
    schema = prior.realize(seed)
    db = create_database(schema, schema.sample(prior.rows(schema, seed), seed=seed))
    val_timestamp, test_timestamp = split_timestamps(db, val_share, test_share)
    name = f"{db_prefix}-{seed}"
    write_database(
        db,
        Path(out_dir).expanduser() / name,
        name=name,
        val_timestamp=val_timestamp,
        test_timestamp=test_timestamp,
        description=f"PluRel synthetic relational database (seed {seed}).",
        overwrite=overwrite,
    )
    rows = sum(len(table.df) for table in db.table_dict.values())
    return f"{name}: {len(db.table_dict)} tables, {rows} rows, {perf_counter() - start:.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        required=True,
        help="first seed; databases are named <db_prefix>-<seed>",
    )
    parser.add_argument(
        "--num_dbs", type=int, required=True, help="number of databases, one per seed"
    )
    parser.add_argument(
        "--num_proc", type=int, default=cpu_count(), help="worker processes (default: all cores)"
    )
    parser.add_argument(
        "--db_prefix", default=DB_PREFIX, help=f"database name prefix (default: {DB_PREFIX})"
    )
    parser.add_argument(
        "--out_dir",
        default=OUT_DIR,
        help=f"directory receiving one folder per database (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--val_share",
        type=float,
        default=0.1,
        help="share of timestamped rows in the validation window (default: 0.1)",
    )
    parser.add_argument(
        "--test_share",
        type=float,
        default=0.1,
        help="share of timestamped rows in the test window (default: 0.1)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace databases that already exist"
    )
    args = parser.parse_args()
    if args.num_dbs < 1:
        parser.error("num_dbs must be at least one")
    seeds = range(args.seed_offset, args.seed_offset + args.num_dbs)
    worker = partial(
        generate,
        out_dir=args.out_dir,
        db_prefix=args.db_prefix,
        val_share=args.val_share,
        test_share=args.test_share,
        overwrite=args.overwrite,
    )
    with Pool(args.num_proc) as pool:
        for line in pool.imap_unordered(worker, seeds):
            print(line, flush=True)


if __name__ == "__main__":
    main()
