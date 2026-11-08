"""Recognize expected failures and repair them without silently changing the intended schema.

Tiny samples, tied causes, fixed real-row counts and impossible relationship counts each have
different semantics. Empty retained parent tables are valid when unmatched keys may be null.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

from collections.abc import Callable

import pandas as pd

from plurel import (
    Categorical,
    Counts,
    ForeignKey,
    Numeric,
    PrimaryKey,
    Range,
    Schema,
    Table,
    Uniform,
)


def expect_error(action: Callable, message: str) -> None:
    try:
        action()
    except ValueError as error:
        assert message in str(error)
        print("Expected:", error)
    else:
        raise AssertionError(f"expected an error mentioning {message!r}")


family = Table({"x": Numeric(), "y": Numeric(causes={"x": Range(0.2, 0.8)})})
expect_error(lambda: family.sample(10, seed=0), "realize(seed)")
assert len(family.realize(0).sample(10, seed=0)) == 10

expect_error(lambda: Table({"x": Numeric(causes={"y": 1}), "y": Numeric(causes={"x": 1})}), "cycle")
# Repair a cycle by choosing an upstream cause; mutually dependent mechanisms are not a DAG.
assert Table({"x": Numeric()}).sample(0, seed=0).empty
assert len(Table({"x": Numeric()}).sample(1, seed=0)) == 1
ranked = Table({"x": Numeric(), "y": Numeric(Uniform(0, 1), causes={"x": 1})})
expect_error(lambda: ranked.sample(1, seed=0), "at least two")
assert len(ranked.sample(2, seed=0)) == 2

real = Table.from_frame(pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]}))
expect_error(lambda: real.sample(8, seed=0), "given rows")
assert len(real.sample(seed=0)) == 4

# Current limitation: categorical cuts can put tied, zero-noise inputs in one class.
# Inspect the shares without requiring this limitation to persist if tie handling improves.
tied = Table.from_frame(
    pd.DataFrame({"x": [1.0] * 20}),
    extra={"label": Categorical({"a": 0.5, "b": 0.5}, causes={"x": 1}, noise=0)},
)
print(
    "Requested 50/50 with tied inputs; observed:",
    tied.sample(seed=0).label.value_counts(normalize=True).to_dict(),
)
print("Add noise when random variation in the labels is intended")

counted = Schema(
    {
        "parents": Table({"id": PrimaryKey()}),
        "children": Table({"parent_id": ForeignKey("parents", counts=Counts(2, least=1, most=3))}),
    }
)
expect_error(lambda: counted.sample({"parents": 2, "children": 7}, seed=0), "at most 3")
assert len(counted.sample({"parents": 2, "children": 6}, seed=0)["children"]) == 6

empty = Table.from_frame(
    pd.DataFrame({"id": pd.Series([], dtype="string"), "region": pd.Series([], dtype="string")}),
    primary_key="id",
)
children = Table.from_frame(
    pd.DataFrame({"region": ["north", "south"]}),
    extra={"parent_id": ForeignKey("empty", match={"region": "region"}, unmatched="null")},
)
frames = Schema({"empty": empty, "children": children}).sample(seed=0)
assert frames["empty"].empty and frames["children"].parent_id.isna().all()
print("edge_cases: expected failures were explained and valid repairs passed")
