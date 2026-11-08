"""Separate a latent mechanism, a recorded marginal, and arithmetic in recorded units.

A bare cause reads a latent. Recorded reads the emitted value; Derived computes a formula
over emitted numbers. A marginal places its own sampled values in the latent's rank order.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

import numpy as np

from plurel import Derived, Numeric, Recorded, Table, Uniform

table = Table(
    {
        "price": Numeric(Uniform(100, 200)),
        "latent_twice": Numeric(causes={"price": 2.0}, noise=0),
        "recorded_twice": Numeric(causes={Recorded("price"): 2.0}, noise=0),
        "formula_twice": Derived(lambda price: 2 * price),
        "ranked": Numeric(Uniform(0, 1), causes={"price": 2.0}, noise=0),
    }
)
frame, latents = table.sample_with_latents(1000, seed=0)

np.testing.assert_allclose(frame.latent_twice, 2 * latents["price"][:, 0])
np.testing.assert_allclose(frame.recorded_twice, 2 * frame.price)
np.testing.assert_allclose(frame.formula_twice, 2 * frame.price, atol=1e-6)
assert frame.price.between(100, 200).all()
assert frame.ranked.between(0, 1).all()
assert np.array_equal(np.argsort(frame.price), np.argsort(frame.ranked))
# The last column follows price's order while retaining its own 0-to-1 marginal.
# Derived rounds its result to six decimal places; Recorded mechanisms retain their units.
print(frame.head().round(3))
print("latent_and_recorded: checked latent effects, recorded arithmetic, and rank mapping")
