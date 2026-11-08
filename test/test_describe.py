import json

import pytest

from plurel import (
    Categorical,
    Counts,
    Derived,
    ForeignKey,
    Hidden,
    LogNormal,
    Numeric,
    PrimaryKey,
    Range,
    Recorded,
    Schema,
    Summary,
    Table,
    Timestamp,
    Uniform,
)
from plurel.prior import SchemaPrior


@pytest.fixture
def schema():
    customers = Table(
        {
            "id": PrimaryKey(),
            "age": Numeric(Uniform(18, 90), round=1, doc="years since birth"),
            "segment": Categorical(
                ("bronze", "gold"), causes={"age": Range(0.5, 1.0)}, ordered=True
            ),
            "spend": Numeric(
                LogNormal(3, 0.5),
                causes={"age": 0.8, "segment": {"gold": 1.5}, ("segment", "age"): {"gold": 0.3}},
                missing=0.05,
            ),
            "taste": Hidden(),
            "orders": Numeric(
                causes={Summary("orders", "customer_id", Recorded("amount"), "sum"): 1.0}
            ),
            "ratio": Derived(lambda spend, age: spend / age),
        },
        doc="people who buy",
        target="spend",
    )
    orders = Table(
        {
            "id": PrimaryKey(),
            "customer_id": ForeignKey("customers", counts=Counts(3.0, {"taste": 1.0})),
            "amount": Numeric(causes={Recorded("customer_id.spend", fill=0.0): 2.0}, noise=1.0),
        }
    )
    return Schema({"customers": customers, "orders": orders})


def test_a_description_names_authored_columns_and_how_each_reads_its_causes(schema):
    realized = schema.realize(0)
    description = realized.describe()
    assert json.loads(json.dumps(description)) == description
    customers = description["tables"]["customers"]
    assert list(customers["columns"]) == list(schema.tables["customers"].declared)
    assert customers["doc"] == "people who buy" and customers["target"] == "spend"
    spend = customers["columns"]["spend"]
    assert spend["marginal"] == {"type": "LogNormal", "mean": 3, "sigma": 0.5}
    assert {"reads": "age", "values": "latent", "weight": 0.8, "transform": "identity"} in spend[
        "causes"
    ]
    assert {"reads": "segment", "values": "latent", "effects": {"gold": 1.5}} in spend["causes"]
    assert any(c.get("product") == ["segment", "age"] for c in spend["causes"])
    amount = description["tables"]["orders"]["columns"]["amount"]["causes"][0]
    assert amount == {
        "reads": "customer_id.spend",
        "values": "recorded",
        "fill": 0.0,
        "weight": 2.0,
        "transform": "identity",
    }
    summary = customers["columns"]["orders"]["causes"][0]
    assert (
        summary["summary"] == "orders"
        and summary["values"] == "recorded"
        and summary["how"] == "sum"
    )
    key = description["tables"]["orders"]["columns"]["customer_id"]
    assert key["kind"] == "foreign key" and key["counts"]["causes"] == {"taste": 1.0}
    assert customers["columns"]["age"]["doc"] == "years since birth"
    assert customers["columns"]["ratio"] == {
        "kind": "derived",
        "inputs": ["spend", "age"],
        "function": "<lambda>",
    }


def test_resolved_values_sit_beside_the_declaration_and_calibration(schema):
    segment = schema.realize(0).describe()["tables"]["customers"]["columns"]["segment"]
    weight = segment["causes"][0]["weight"]
    assert 0.5 <= weight <= 1.0 and segment["declared"]["causes"] == {"age": {"range": [0.5, 1.0]}}
    assert len(segment["calibrated"]["cuts"]) == 1
    fixed = Table({"x": Numeric(), "c": Categorical(("a", "b"), causes={"x": 1.0})})
    assert "calibrated" not in fixed.describe()["columns"]["c"]
    assert fixed.describe() == fixed.describe()


def test_any_schema_describes_and_explains():
    prior = SchemaPrior().realize(3)
    assert json.dumps(prior.describe())
    text = prior.explain()
    assert all(f"{t}" in text for t in prior.tables)
    shop = Table(
        {"when": Timestamp("2020-01-01", "2021-01-01"), "n": Numeric(causes={"when": 1.0})},
        time_column="when",
    )
    assert "time_column when" in shop.explain() and "<- 1 * when" in shop.explain()
