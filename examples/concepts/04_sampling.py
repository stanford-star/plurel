"""Sampling is repeatable for fixed inputs, but a larger draw is not an extension of a smaller one.

Row counts participate in rank mapping and relational allocation. Independent draws are
independent datasets, not streaming chunks of one stable population.
"""

from plurel import Numeric, PrimaryKey, Table, Uniform

generator = Table(
    {
        "id": PrimaryKey(),
        "x": Numeric(Uniform(0, 1)),
        "y": Numeric(Uniform(10, 20), causes={"x": 1.0}),
    }
).realize(0)

small = generator.sample(100, seed=10)
assert small.equals(generator.sample(100, seed=10))
assert not small.equals(generator.sample(100, seed=11))
larger = generator.sample(200, seed=10)
assert not small.equals(larger.head(100))

# Primary keys are local to a draw, so concatenating draws requires key remapping.
# For a relational database, remap the foreign keys to those parents as well.
another = generator.sample(100, seed=11)
assert small.id.equals(another.id)
print("Repeated draw identical:", small.equals(generator.sample(100, seed=10)))
print("Larger draw has the same prefix:", small.equals(larger.head(100)))
print("sampling: seeds, row counts, names, code and environment define repeatability")
