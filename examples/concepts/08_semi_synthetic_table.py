"""Keep source rows and author extra columns: semi-synthetic augmentation.

from_frame retains the source's values and row count. It does not fit a generator that
produces more source-like rows. Here an embedded fixture stands in for a user DataFrame.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

import numpy as np
import pandas as pd

from plurel import Derived, Numeric, Range, Recorded, Table

real = pd.DataFrame(
    {"order_id": ["O1", "O2", "O3"], "Unit Price": [10.0, 20.0, 30.0], "Quantity": [2, 3, 4]}
)
family = Table.from_frame(
    real,
    primary_key="order_id",
    extra={
        "total": Derived(
            lambda price, quantity: price * quantity,
            columns={"price": "Unit Price", "quantity": "Quantity"},
        ),
        "estimated_cost": Numeric(causes={Recorded("total"): Range(0.5, 0.8)}, noise=1),
    },
    name="orders",
    target="estimated_cost",
    docs={
        "Unit Price": "price per item",
        "estimated_cost": "authored noisy cost in currency units",
    },
)
generator = family.realize(seed=0)
first, second = generator.sample(seed=10), generator.sample(seed=11)
for frame in (first, second):
    pd.testing.assert_frame_equal(frame[list(real)], real)
    np.testing.assert_allclose(frame.total, [20, 60, 120])
assert not first.estimated_cost.equals(second.estimated_cost)
assert generator.describe()["target"] == "estimated_cost"
print(first)
print("semi_synthetic_table: kept source rows, added exact arithmetic and an authored noisy target")
