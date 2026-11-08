"""Calibration pins categorical cut points using a temporary draw of an authored mechanism.

Without pinned cuts, an ordinary categorical cuts each draw at its requested shares.
With pinned cuts, future draws fluctuate around the calibrated shares. Numeric marginals
still rank-map within each draw. Calibration is not learning a distribution from real data.
"""

import numpy as np

from plurel import Categorical, Numeric, Table

table = Table(
    {
        "score": Numeric(),
        "decision": Categorical({"no": 0.8, "yes": 0.2}, causes={"score": 1.0}),
    }
)
direct_shares = [table.sample(1000, seed=s).decision.eq("yes").mean() for s in range(5)]
np.testing.assert_allclose(direct_shares, 0.2)

# Even though all weights are fixed, realize fits the cut on a calibration draw.
# rows=2000 is the calibration size, not a limit on later sample sizes.
generator = table.realize(seed=0, rows=2000)
before = generator.describe()
assert "calibrated" not in table.describe()["columns"]["decision"]
print("Fitted cuts:", before["columns"]["decision"]["calibrated"]["cuts"])
calibrated_shares = [generator.sample(1000, seed=s).decision.eq("yes").mean() for s in range(5)]
assert len(set(calibrated_shares)) > 1
assert all(abs(share - 0.2) < 0.08 for share in calibrated_shares)
assert generator.describe() == before
print("Per-draw cuts:", direct_shares)
print("Pinned cuts:", calibrated_shares)
# Tied, zero-noise latents can miss requested shares. Key-dependent categoricals
# retain per-draw cuts; see 10_relational_schema.py and 15_edge_cases.py.
print("calibration: fixed weights can still acquire calibrated categorical cuts")
