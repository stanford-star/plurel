"""Populations from plurel's :class:`TablePrior`: the prior realized under a seed, with a categorical
target chosen among its columns. The prior draws the DAG, the ops, the edge families, hidden nodes,
noise, marginals, binned and one-hot columns and missingness; nothing is planted on top."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from plurel.population import Population, TableMetadata, category_order, table_id
from plurel.prior import IntegersRange, TablePrior
from plurel.schema import SCM

# Every coin comes from a lane of [RANDOM_DOMAIN, seed, lane, ...], so a seed gives the same SCM and
# the same row draws anywhere.
RANDOM_DOMAIN = 0x524E444D
_LANE_WORLD = 1
_LANE_SAMPLE = 2
_TARGET_TRIES = 20

PRIOR = TablePrior()


def realize_random(seed: int, *, columns: int | tuple[int, int] | None = None) -> Population:
    """The prior realized under ``seed`` as a :class:`Population`; ``columns`` pins or bounds the observed column count,
    target included. The target is a categorical column with parents when the draw has one, else
    any categorical column, and it is never missing; a draw without one is redrawn under the next
    attempt of the same seed."""
    prior = PRIOR
    if columns is not None:
        lo, hi = (columns, columns) if isinstance(columns, int) else columns
        prior = replace(prior, column_count=IntegersRange(lo, hi))
    for attempt in range(_TARGET_TRIES):
        rng = np.random.default_rng([RANDOM_DOMAIN, int(seed), _LANE_WORLD, attempt])
        drawn = prior.realize(rng)
        observed = {name: c for name, c in drawn.columns.items() if c.kind != "key"}
        categorical = [name for name, c in observed.items() if c.kind == "categorical"]
        parented = [name for name in categorical if drawn.nodes[observed[name].node].edges]
        candidates = parented or categorical
        if not candidates:
            continue
        target = str(rng.choice(candidates))
        # a label is never blank: the target keeps every row the prior would have hidden
        observed[target] = replace(observed[target], missing=0.0)
        plan = SCM(drawn.nodes, observed)
        features = [name for name in observed if name != target]
        metadata = TableMetadata(
            feature_names=features,
            feature_types=[observed[c].kind for c in features],
            target=target,
            plan=plan,
            category_order=category_order(plan),
            realization={"table_id": table_id("random", seed), "seed": int(seed)},
        )
        return Population(metadata=metadata, lanes=(RANDOM_DOMAIN, int(seed), _LANE_SAMPLE), render=None)  # fmt: skip
    raise ValueError(f"no categorical column in {_TARGET_TRIES} draws of seed {seed}")
