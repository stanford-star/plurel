"""Seeded draws whose exact bytes the compatibility contract promises, and their sha256 digests.

`python test/fingerprints.py` prints whether the stored digests in fingerprints.json still match;
`--update` rewrites them, which a change that moves draws on purpose does and its PR says so. See
COMPATIBILITY.md for what is promised exactly and what is not.
"""

import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from plurel import (
    ByLevel,
    Categorical,
    Counts,
    Derived,
    Exponential,
    ForeignKey,
    Hidden,
    HSBMLink,
    LogNormal,
    Numeric,
    Pareto,
    PrimaryKey,
    Range,
    Recorded,
    Scaled,
    Schema,
    Summary,
    Table,
    Timestamp,
    TreeLink,
    Uniform,
)
from plurel.prior import SchemaPrior

STORED = Path(__file__).with_name("fingerprints.json")


def digest(frames: dict[str, pd.DataFrame]) -> str:
    """One hash of every table's name, columns, dtypes and values."""
    h = hashlib.sha256()
    for name in sorted(frames):
        frame = frames[name]
        h.update(name.encode())
        h.update(str(list(frame.columns)).encode())
        h.update(str(list(map(str, frame.dtypes))).encode())
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()


def shop() -> Schema:
    customers = Table(
        {
            "id": PrimaryKey(),
            "segment": Categorical({"gold": 0.2, "silver": 0.3, "bronze": 0.5}),
            "age": Numeric(Uniform(18, 90), round=1, missing=0.1, missing_by=("spend", "high")),
            "spend": Numeric(
                LogNormal(), causes={"segment": {"gold": 1.0}, "age": 0.5}, round=0.01
            ),
            "churn": Categorical(("no", "yes"), causes={"spend": -1.0}, missing=0.05),
            "band": Numeric(
                ByLevel("segment", {"gold": (50, 100), "silver": (20, 60), "bronze": (0, 30)}, 2.0)
            ),
            "ratio": Derived(lambda spend, band: spend / (band + 1.0)),
            "joined": Timestamp("2019-01-01", "2020-01-01", causes={"age": 0.5}, missing=0.02),
        }
    )
    orders = Table(
        {
            "id": PrimaryKey(),
            "customer_id": ForeignKey(
                "customers", HSBMLink(popularity=Pareto(2.5)), nullable=0.05, fill=0.0
            ),
            "time": Timestamp("2020-01-01", "2024-01-01", causes={"customer_id.joined": 1.0}),
            "amount": Numeric(LogNormal(), causes={"customer_id.spend": 1.0}),
            "shipped": Timestamp(causes={Recorded("time"): 1.0}, noise=Exponential(86400)),
        },
        time_column="time",
    )
    return Schema({"customers": customers, "orders": orders})


def teams() -> Schema:
    teams = Table(
        {
            "id": PrimaryKey(),
            "size": Numeric(causes={Summary("staff", "team_id"): 1.0}, noise=0),
            "payroll": Numeric(
                causes={Summary("staff", "team_id", Recorded("pay"), "sum"): 1.0}, noise=0
            ),
        }
    )
    staff = Table(
        {
            "id": PrimaryKey(),
            "team_id": ForeignKey("teams"),
            "boss_id": ForeignKey("staff", TreeLink(roots=0.05), fill=0.0),
            "reports": Hidden(causes={Summary("staff", "boss_id"): 1.0}),
            "grade": Categorical((1, 2, 3), causes={"reports": 1.0}, ordered=True),
            "pay": Numeric(LogNormal(11, 0.3), causes={"grade": 1.0, ("grade", "reports"): 0.5}),
        }
    )
    return Schema({"teams": teams, "staff": staff})


def school() -> Schema:
    courses = Table.from_frame(
        pd.DataFrame({"id": range(12), "level": ["intro", "advanced"] * 6}), primary_key="id"
    )
    students = Table(
        {
            "id": PrimaryKey(),
            "year": Numeric(Scaled("count", 1, 4)),
            "level": Categorical(("intro", "advanced")),
        }
    )
    enrollments = Table(
        {
            "student_id": ForeignKey("students", counts=Counts(Range(2.0, 3.0), least=1, most=5)),
            "level": Categorical(("intro", "advanced"), causes={"student_id.year": 1.0}),
            "course_id": ForeignKey(
                "courses", match={"level": "level"}, unique="student_id", unmatched="null", fill=0.0
            ),
            "grade": Numeric(causes={Recorded("student_id.year"): 10.0}, noise=5.0),
        }
    )
    return Schema({"students": students, "courses": courses, "enrollments": enrollments})


def workloads() -> Iterator[tuple[str, Schema, dict[str, int]]]:
    """Each named draw: a schema, already a generator, and its row counts."""
    yield "shop", shop().realize(0), {"customers": 300, "orders": 2000}
    yield "teams", teams().realize(0), {"teams": 8, "staff": 200}
    yield "school", school().realize(1), {"students": 200}
    prior = SchemaPrior()
    for seed in range(4):
        generator = prior.realize(seed)
        rows = {t: min(n, 2000) for t, n in prior.rows(generator, seed).items()}
        yield f"prior/{seed}", generator, rows


def draw() -> dict[str, str]:
    return {
        f"{name}/seed {seed}": digest(generator.sample(rows, seed=seed))
        for name, generator, rows in workloads()
        for seed in (0, 1)
    }


if __name__ == "__main__":
    drawn = draw()
    if "--update" in sys.argv:
        STORED.write_text(json.dumps(drawn, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(drawn)} fingerprints to {STORED}")
    else:
        stored = json.loads(STORED.read_text())
        moved = sorted(name for name in drawn if stored.get(name) != drawn[name])
        print("all match" if not moved else f"moved: {moved}")
        sys.exit(1 if moved else 0)
