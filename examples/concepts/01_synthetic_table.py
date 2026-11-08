"""Author one fully synthetic table: choose columns, distributions, and dependencies.

No source rows are supplied. Fixed declarations can sample directly; a target is metadata
identifying a prediction column, not a request to fit a predictive model.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

from plurel import Categorical, Numeric, PrimaryKey, Table, Uniform

customers = Table(
    {
        "customer_id": PrimaryKey(),
        "age": Numeric(Uniform(18, 80), round=1, doc="age in years"),
        "tenure": Numeric(Uniform(0, 15), causes={"age": 0.8}),
        "renewed": Categorical({"no": 0.3, "yes": 0.7}, causes={"tenure": 1.0}),
    },
    name="customers",
    doc="One invented customer per row; illustrative mechanisms.",
    target="renewed",
)

frame = customers.sample(1000, seed=10)
assert frame.shape == (1000, 4)
assert frame.customer_id.is_unique
assert frame.age.between(18, 80).all()
assert frame.tenure.between(0, 15).all()
assert set(frame.renewed) == {"no", "yes"}
# The coefficients control dependence in latent space. They are not years per year;
# 02_latent_and_recorded.py demonstrates the distinction with exact equalities.
print(customers.explain())
print(frame.head())
print("synthetic_table: generated 1000 new rows from fixed declarations")
