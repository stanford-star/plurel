"""Realize a family into a generator, then sample datasets with a separate seed.

Realization draws unresolved parameters and calibrates applicable categorical mechanisms.
It returns a Table or Schema, not a DataFrame. This numeric example needs no calibration.
"""

from plurel import Numeric, Range, Table

family = Table(
    {"x": Numeric(), "y": Numeric(causes={"x": Range(0.2, 0.8)}, noise=0.5)},
    name="linear_family",
)
mechanism_seed, data_seed = 1, 10
generator = family.realize(mechanism_seed)
description = generator.describe()
assert description == family.realize(mechanism_seed).describe()

first = generator.sample(1000, seed=data_seed)
second = generator.sample(1000, seed=data_seed + 1)
assert not first.equals(second)
assert generator.describe() == description, "sampling leaves the generator unchanged"
assert first.equals(generator.sample(1000, seed=data_seed))

# Different mechanisms with the SAME data seed: the upstream x draws stay fixed,
# while the chosen coefficient changes y. Inspect the choices without guessing them.
other = family.realize(mechanism_seed + 1)
other_frame = other.sample(1000, seed=data_seed)
assert first.x.equals(other_frame.x)
assert not first.y.equals(other_frame.y)
assert other.describe() != description
print(generator.explain())
print(other.explain())
print("realization: mechanism seeds change generators; data seeds change their draws")
