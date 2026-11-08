"""Scored categoricals fit class biases, including over retained real rows.

Scores always need realization, even with fixed coefficients. They choose a class by noisy
scores; calibration adjusts biases toward authored shares. This does not synthesize more
source rows. Large retained tables use a bounded calibration subset in current v2.
"""

import numpy as np
import pandas as pd

from plurel import Categorical, Table

# A synthetic fixture standing in for an imported table, larger than the current
# 4096-row calibration cap. The cap is an implementation detail, not a sampling limit.
real = pd.DataFrame({"id": [f"C{i}" for i in range(6000)], "age": np.linspace(18, 80, 6000)})
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
    assert "realize" in str(error)
    print("Scored categoricals need realization even with fixed score coefficients")
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
print("score_calibration: calibrated a label mechanism and preserved all 6000 source rows")
