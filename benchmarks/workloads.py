"""What the benchmark draws: tables and databases from the default priors, one per seed, as a corpus
is generated. Database `seed` is `SchemaPrior().realize(seed)` at the row counts
`prior.rows(database, seed)` draws for its tables; table `seed` is `TablePrior().realize(seed)` at a
row count drawn the way the database prior draws an event table's: warped, then drawn."""

from collections.abc import Callable

import numpy as np

from plurel import Schema
from plurel.prior import SchemaPrior, TablePrior

Draw = Callable[[int], tuple[Schema, dict[str, int]]]


def database(seed: int) -> tuple[Schema, dict[str, int]]:
    prior = SchemaPrior()
    schema = prior.realize(seed)
    return schema, prior.rows(schema, seed)


def table(seed: int) -> tuple[Schema, dict[str, int]]:
    rng = np.random.default_rng(seed)
    rows = int(SchemaPrior().activity_row_count.warp(rng).draw(rng))
    return Schema({"table": TablePrior().realize(seed)}), {"table": rows}


KINDS: dict[str, Draw] = {"table": table, "database": database}
