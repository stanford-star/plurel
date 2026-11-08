"""Scored categoricals fit class biases, including over retained real rows.

Scores choose a class under noise; realize fits class biases toward authored shares.
This does not synthesize more source rows. Calibration-size details are documented in
README.md#author-many-tables-from-one-description at the repository root.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

import numpy as np
import pandas as pd

from plurel import Categorical, Table

# An embedded fixture standing in for an imported table.
real = pd.DataFrame({"id": [f"C{i}" for i in range(1000)], "age": np.linspace(18, 80, 1000)})
family = Table.from_frame(
    real,
    primary_key="id",
    extra={
        "offer": Categorical({"standard": 0.75, "premium": 0.25}, scores={"premium": {"age": 1.0}})
    },
    target="offer",
)
try:
    family.sample(seed=10)
except ValueError as error:
    assert "fits categorical scores" in str(error)
    print("Expected:", error)
else:
    raise AssertionError("unfitted scores should require realization")
generator = family.realize(seed=0)
description = generator.describe()
print("Fitted score biases:", description["columns"]["offer"]["calibrated"]["bias"])
frame = generator.sample(seed=10)  # no row count: these are all the retained rows
pd.testing.assert_frame_equal(frame[["id", "age"]], real)
share = frame.offer.eq("premium").mean()
assert abs(share - 0.25) < 0.06
assert generator.describe() == description
# The age coefficient is an effect on the real column's standardized rank, not per year.
print(f"Premium share: {share:.3f}; retained rows: {len(frame)}")
print(f"score_calibration: calibrated a label mechanism and preserved all {len(real)} source rows")
