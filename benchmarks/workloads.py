"""The benchmark's workloads: one random table, and random databases of 2, 8, 32 and 128 tables,
each drawn once from a fixed seed, 1. A workload at `rows` draws exactly that many rows; a database
splits them across its tables in the shares its prior drew, so its structure and the split are the
same in every run."""

from collections.abc import Callable

import numpy as np

from plurel import IntegersRange, Schema
from plurel.prior import SchemaPrior, TablePrior

# the structure's seed: every run draws the same table and databases. Seed 0 drew a 128-table
# database whose smallest tables are too small for their links below 500k rows; 1 is the first
# seed whose workloads all sample at every size
SEED = 1

Build = Callable[[int], tuple[Schema, dict[str, int]]]


def split(shares: dict[str, int], total: int) -> dict[str, int]:
    """`total` rows across the tables in proportion to `shares`, rounded to sum to `total`."""
    raw = np.asarray(list(shares.values()), dtype=float) / sum(shares.values()) * total
    counts = np.floor(raw).astype(int)
    counts[np.argsort(counts - raw)[: total - counts.sum()]] += 1
    return dict(zip(shares, counts.tolist()))


def table(rows: int) -> tuple[Schema, dict[str, int]]:
    """A random table of 20 columns."""
    return Schema({"table": TablePrior(column_count=IntegersRange(20, 20)).realize(SEED)}), {
        "table": rows
    }


def database(tables: int) -> Build:
    """A random database of `tables` tables."""

    def build(rows: int) -> tuple[Schema, dict[str, int]]:
        prior = SchemaPrior(table_count=IntegersRange(tables, tables))
        schema = prior.realize(SEED)
        return schema, split(prior.rows(schema, SEED), rows)

    return build


WORKLOADS: dict[str, Build] = {
    "table": table,
    "database-2": database(2),
    "database-8": database(8),
    "database-32": database(32),
    "database-128": database(128),
}
