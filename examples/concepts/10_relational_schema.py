"""A schema connects tables through keys, bounded counts and recorded-value summaries.

A counting key can infer its table's size. A supplied size is instead an exact total allocated
within the bounds. A key-dependent categorical keeps per-draw cuts even after realization.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

import numpy as np

from plurel import (
    Categorical,
    Counts,
    ForeignKey,
    Numeric,
    PrimaryKey,
    Recorded,
    Schema,
    Summary,
    Table,
    Uniform,
)

schema = Schema(
    {
        "teams": Table(
            {
                "id": PrimaryKey(),
                "budget": Numeric(Uniform(100, 200)),
                "payroll": Numeric(
                    causes={Summary("staff", "team_id", Recorded("pay"), "sum"): 1}, noise=0
                ),
            }
        ),
        "staff": Table(
            {
                "team_id": ForeignKey("teams", counts=Counts(3, least=1, most=5)),
                "pay": Numeric(Uniform(10, 50)),
                "grade": Categorical({"junior": 0.5, "senior": 0.5}, causes={"team_id.budget": 1}),
            }
        ),
    }
)
generator = schema.realize(0)
assert "calibrated" not in generator.describe()["tables"]["staff"]["columns"]["grade"]
for rows in ({"teams": 4}, {"teams": 4, "staff": 12}):
    frames = generator.sample(rows, seed=10)
    teams, staff = frames["teams"], frames["staff"]
    assert staff.team_id.isin(teams.id).all()
    counts = staff.team_id.value_counts().reindex(teams.id, fill_value=0)
    assert counts.between(1, 5).all()
    totals = staff.groupby("team_id").pay.sum().reindex(teams.id, fill_value=0)
    np.testing.assert_allclose(teams.payroll, totals)
    if "staff" in rows:
        assert len(staff) == rows["staff"]
    print("Requested:", rows, "actual staff:", len(staff), "counts:", counts.tolist())
print("relational_schema: bounded allocation and exact recorded-value aggregation")
