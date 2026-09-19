import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from tqdm import tqdm

from plurel.config import Config
from plurel.dataset import SyntheticDataset
from plurel.utils import set_random_seed

# Default name prefix for generated synthetic DBs: <DB_PREFIX>-<seed>.
DB_PREFIX = "plurel"


def generate_plurel_db(
    seed: int,
    db_prefix: str = DB_PREFIX,
):
    torch.set_num_threads(1)
    set_random_seed(0)
    db_name = f"{db_prefix}-{seed}"
    print(f"Creating dataset: {db_name}")

    cache_dir = Path(f"~/.cache/relbench/{db_name}").expanduser()
    dataset = SyntheticDataset(
        seed=seed,
        config=Config(cache_dir=cache_dir),
    )

    # generate and cache db in relbench-3.0.0 format (manifest.yaml + db/*.parquet).
    # To preprocess for training, use the relational-transformer repo:
    # https://github.com/rishabh-ranjan/relational-transformer
    #   pixi run preprocess --dataset <cache_dir> --out-dir <pre_dir>
    dataset.get_db()


def main(
    seed_offset: int,
    num_dbs: int,
    num_proc: int,
    db_prefix: str = DB_PREFIX,
):
    seeds = [idx + seed_offset for idx in range(num_dbs)]
    worker = partial(generate_plurel_db, db_prefix=db_prefix)

    with Pool(processes=num_proc) as p:
        list(
            tqdm(
                p.imap_unordered(worker, seeds),
                total=len(seeds),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset.")

    parser.add_argument(
        "--seed_offset",
        type=int,
        required=True,
        help="Seed offset for database generation. DBs will be named <db_prefix>-<seed>.",
    )

    parser.add_argument(
        "--num_dbs",
        type=int,
        required=True,
        help="Number of databases to generate.",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use (default: number of CPU cores).",
    )

    parser.add_argument(
        "--db_prefix",
        type=str,
        default=DB_PREFIX,
        help=f"DB name prefix; DBs are named <db_prefix>-<seed> (default: {DB_PREFIX}).",
    )

    args = parser.parse_args()

    main(
        seed_offset=args.seed_offset,
        num_dbs=args.num_dbs,
        num_proc=args.num_proc,
        db_prefix=args.db_prefix,
    )
