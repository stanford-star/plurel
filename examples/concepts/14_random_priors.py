"""A prior chooses structures as well as mechanisms; an authored family keeps its declared structure.

These small priors produce anonymous synthetic tables. They offer structural diversity,
while authored schemas express domain choices. A row-count seed is separate from a data seed.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

from plurel import IntegersRange, SchemaPrior, TablePrior

table_prior = TablePrior(node_count=IntegersRange(4, 6), column_count=IntegersRange(3, 4))
table = table_prior.realize(seed=1)
frame = table.sample(100, seed=10)
assert len(frame) == 100
assert frame.equals(table.sample(100, seed=10))

schema_prior = SchemaPrior(
    table_count=IntegersRange(2, 3),
    table_prior=table_prior,
    entity_row_count=IntegersRange(20, 30),
    activity_row_count=IntegersRange(50, 80),
    # Small tables need small link cluster counts as well as small row counts.
    link_level_count=IntegersRange(1, 2),
    link_cluster_count=IntegersRange(1, 2),
)
schema = schema_prior.realize(seed=1)
rows = schema_prior.rows(schema, seed=2)
frames = schema.sample(rows, seed=10)
assert {name: len(frame) for name, frame in frames.items()} == rows
print(schema.explain())
print("Drawn row counts:", rows)
print("random_priors: drew small structures, mechanisms, row counts and then data")
