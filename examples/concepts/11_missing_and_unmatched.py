"""Missing values, missing matches and random missing keys have different controls.

Recorded(fill=...) chooses the numeric value seen by a mechanism; it does not impute the
output. Derived propagates blanks. unmatched='null' preserves unmatchable rows; nullable
independently drops an additional random share of keys.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

import numpy as np
import pandas as pd

from plurel import Derived, ForeignKey, Numeric, Recorded, Schema, Table

source = pd.DataFrame({"price": [10.0, np.nan, 30.0]})
table = Table.from_frame(
    source,
    extra={
        "filled_read": Numeric(causes={Recorded("price", fill=0): 2}, noise=0),
        "formula": Derived(lambda price: 2 * price),
    },
)
frame = table.sample(seed=0)
np.testing.assert_allclose(frame.filled_read, [20, 0, 60])
assert frame.price.isna().equals(source.price.isna())
assert frame.formula.isna().equals(source.price.isna())

parents = Table.from_frame(pd.DataFrame({"id": ["N"], "region": ["north"]}), primary_key="id")
real_children = pd.DataFrame({"region": ["north"] * 200 + ["south"] * 200})
for rate in (0.0, 0.25):
    children = Table.from_frame(
        real_children,
        extra={
            "parent_id": ForeignKey(
                "parents", match={"region": "region"}, unmatched="null", nullable=rate
            )
        },
    )
    child = Schema({"parents": parents, "children": children}).sample(seed=0)["children"]
    assert child.loc[child.region == "south", "parent_id"].isna().all()
    north_nulls = child.loc[child.region == "north", "parent_id"].isna().sum()
    if rate == 0:
        assert north_nulls == 0
    else:
        assert abs(north_nulls / 200 - rate) < 0.1
    print(f"Random missingness {rate}: {north_nulls} eligible north keys missing")
print("missing_and_unmatched: output blanks, read fills, unmatched rows and random nulls differ")
