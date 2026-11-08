"""Declarations as users and agents write them, checked by pyright: run `pixi run typecheck`. The
module also runs, so the examples stay valid at runtime."""

import pandas as pd

from plurel import (
    MEDIUM,
    WEAK,
    ByLevel,
    Categorical,
    Counts,
    Derived,
    Exponential,
    ForeignKey,
    Hidden,
    HSBMLink,
    LogNormal,
    Noise,
    Numeric,
    Pareto,
    PrimaryKey,
    Range,
    Real,
    Recorded,
    Scaled,
    Schema,
    Shares,
    Summary,
    Table,
    Timestamp,
    Uniform,
)

customers = Table(
    {
        "id": PrimaryKey(),
        "age": Numeric(Uniform(18, 80), round=1, doc="years"),
        "segment": Categorical({"bronze": 0.6, "silver": 0.3, "gold": 0.1}, causes={"age": 1.0}),
        "tier": Categorical(Shares(("a", "b")), scores={"a": {"age": WEAK}}),
        "spend": Numeric(
            LogNormal(3, 0.5),
            causes={
                "age": (0.8, "tanh"),
                "segment": {"gold": 1.5},
                ("segment", "age"): {"gold": 0.3},
                "taste": Range(0.2, 0.6),
            },
            noise=Noise("gaussian", MEDIUM),
            missing=0.05,
            clip=(0.0, None),
        ),
        "band": Numeric(
            ByLevel("segment", {"bronze": (0, 10), "silver": (5, 20), "gold": (10, 50)})
        ),
        "taste": Hidden(),
        "ratio": Derived(lambda spend, age: spend / age),
        "joined": Timestamp("2020-01-01", "2024-01-01", causes={"age": -0.5}),
        "score": Numeric(Scaled("score", 0, 100), causes={Recorded("age", fill=40.0): 0.1}),
    },
    doc="people who buy",
    target="spend",
)

orders = Table(
    {
        "id": PrimaryKey(),
        "customer_id": ForeignKey("customers", counts=Counts(3.0, {"age": 0.5}, most=10)),
        "store_id": ForeignKey("stores", HSBMLink(popularity=Pareto(2.5)), nullable=0.1, fill=0.0),
        "when": Timestamp("2020-01-01", "2025-01-01", causes={"customer_id.joined": 1.0}),
        "shipped": Timestamp(causes={Recorded("when"): 1.0}, noise=Exponential(2 * 86400)),
        "amount": Numeric(causes={Recorded("customer_id.spend", fill=0.0): 2.0}, noise=5.0),
    },
    time_column="when",
)

clerks = Table({"store_id": ForeignKey("stores"), "pay": Numeric(Uniform(10, 40))})
stores = Table(
    {
        "id": PrimaryKey(),
        "region": Categorical(("n", "s")),
        "staff": Numeric(causes={Summary("clerks", "store_id", Recorded("pay"), "mean", 0.0): 1.0}),
    }
)

people = Table.from_frame(
    pd.DataFrame({"id": ["a", "b"], "city": ["Lima", "Oslo"]}),
    primary_key="id",
    extra={"churned": Categorical(("no", "yes"), causes={"city": {"Lima": 1.0}})},
)
given = Table({"x": Real([1.0, 2.0, 3.0]), "y": Numeric(causes={"x": 1.0})})

schema = Schema({"customers": customers, "orders": orders, "stores": stores, "clerks": clerks})
realized = schema.realize(0)
frames: dict[str, pd.DataFrame] = realized.sample(
    {"customers": 100, "stores": 10, "clerks": 30}, seed=0
)
frame: pd.DataFrame = given.sample(seed=0)
description: dict[str, object] = realized.describe()
text: str = people.realize(0).explain()


def rejected() -> None:
    """What the annotations reject, never run; each ignore must stay needed, as pyright checks."""
    Table({"x": "not an entry"})  # pyright: ignore[reportArgumentType]
    Numeric(causes={"x": "strong"})  # pyright: ignore[reportArgumentType]
    Numeric(causes={3: 1.0})  # pyright: ignore[reportArgumentType]
    ForeignKey("p", match={"region": 1})  # pyright: ignore[reportArgumentType]
