import numpy as np
import pandas as pd
import pytest

from plurel import (
    Categorical,
    Counts,
    Derived,
    ForeignKey,
    Laplace,
    Noise,
    Normal,
    Numeric,
    PrimaryKey,
    Range,
    Recorded,
    Scaled,
    Schema,
    Shares,
    Table,
    Uniform,
)


def people(real):
    if real:
        frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "region": ["n", "s", "n", "s"]})
        return Table.from_frame(frame, primary_key="id")
    return Table({"id": PrimaryKey(), "region": Categorical(("n", "s"))})


def test_realizing_respects_unique_keys_and_capped_counts_over_few_real_parents():
    few = people(real=True)
    profiles = Table(
        {
            "person_id": ForeignKey("people", unique=True),
            "value": Numeric(Scaled("score", 0, 1)),
            "tier": Categorical(Shares(("x", "y"))),
        }
    )
    realized = Schema({"people": few, "profiles": profiles}).realize(0)
    frame = realized.sample({"profiles": 4}, seed=0)["profiles"]
    assert sorted(frame.person_id) == ["a", "b", "c", "d"]
    orders = Table({"person_id": ForeignKey("people", counts=Counts(Range(1.0, 2.0), most=2))})
    counted = Schema({"people": few, "orders": orders}).realize(0).sample(seed=0)["orders"]
    assert counted.person_id.value_counts().max() <= 2


def test_an_infeasible_calibration_draw_asks_for_feasible_rows():
    schema = Schema(
        {
            "students": Table({"id": PrimaryKey()}),
            "courses": Table.from_frame(pd.DataFrame({"id": [1, 2, 3]}), primary_key="id"),
            "enrollments": Table(
                {
                    "student_id": ForeignKey("students"),
                    "course_id": ForeignKey("courses", unique="student_id"),
                    "grade": Categorical(Shares(("a", "b"))),
                }
            ),
        }
    )
    with pytest.raises(ValueError, match=r"realize\(seed, rows\)"):
        schema.realize(0, {"students": 100})
    realized = schema.realize(0, {"enrollments": 100})
    frame = realized.sample({"students": 400, "enrollments": 100}, seed=0)["enrollments"]
    assert not frame.duplicated(["student_id", "course_id"]).any()


@pytest.mark.parametrize(
    "noise", [Noise("gaussian", 1.0), Noise("laplace", 1.0), Normal(std=1.0), Laplace(), 0.5]
)
def test_counts_take_any_fixed_noise_and_sample_without_realizing(noise):
    schema = Schema(
        {
            "parents": Table({"id": PrimaryKey(), "size": Numeric()}),
            "children": Table(
                {"parent_id": ForeignKey("parents", counts=Counts(5.0, {"size": 0.5}, noise))}
            ),
        }
    )
    first = schema.sample({"parents": 200}, seed=0)["children"]
    assert len(first) > 500 and first.equals(schema.sample({"parents": 200}, seed=0)["children"])
    declared = schema.tables["children"].declared["parent_id"].counts.noise
    assert declared is noise


def test_ranged_noise_in_counts_needs_realizing():
    schema = Schema(
        {
            "parents": Table({"id": PrimaryKey()}),
            "children": Table(
                {
                    "parent_id": ForeignKey(
                        "parents", counts=Counts(5.0, noise=Noise("gaussian", Range(0.5, 1.0)))
                    )
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="realize"):
        schema.sample({"parents": 10}, seed=0)
    assert len(schema.realize(0).sample({"parents": 50}, seed=0)["children"]) > 50


@pytest.mark.parametrize(
    "key, match",
    [
        ({"prefer": {Recorded("typo"): 1.0}}, "'typo' as recorded, which is no column"),
        ({"prefer": {Recorded("region"): 1.0}}, "'region' as recorded, which is a categorical"),
        ({"counts": Counts(2.0, {Recorded("region"): 1.0})}, "counts its rows in 'people'"),
        ({"prefer": {"typo": 1.0}}, r"by \['typo'\], no columns of 'people'"),
        ({"prefer": {Recorded("id"): 1.0}}, "'id' as recorded, which is a key column"),
    ],
)
def test_keys_check_what_they_read_of_their_parent_when_built(key, match):
    with pytest.raises(ValueError, match=match):
        Schema(
            {
                "people": people(real=False),
                "orders": Table({"person_id": ForeignKey("people", **key)}),
            }
        )


def mixed(real, ranged):
    weight = Range(1.0, 2.0) if ranged else 1.5
    customers = Table(
        {
            "id": PrimaryKey(),
            "region": Categorical(("n", "s")),
            "budget": Numeric(Uniform(10, 100), round=1),
        }
        if not real
        else {
            "id": PrimaryKey(["a", "b", "c", "d", "e", "f"]),
            "region": Categorical(("n", "s")),
            "budget": Numeric(Uniform(10, 100), round=1),
        },
        doc="who buys",
    )
    stores = Table({"id": PrimaryKey(), "region": Categorical(("n", "s")), "size": Numeric()})
    visits = Table(
        {
            "customer_id": ForeignKey("customers", counts=Counts(3.0, {"budget": weight}, most=3)),
            "region": Categorical(("n", "s"), causes={"customer_id.region": {"s": 3.0}}),
            "store_id": ForeignKey(
                "stores",
                match={"region": "region"},
                prefer={"size": 1.0},
                unique="customer_id",
                nullable=0.01,
                fill=0.0,
            ),
            "spent": Numeric(causes={Recorded("customer_id.budget"): 0.5}, noise=0),
            "share": Derived(lambda spent: spent / 100.0),
            "double": Derived(lambda share: 2 * share),
        }
    )
    return Schema({"customers": customers, "stores": stores, "visits": visits})


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("ranged", [False, True])
@pytest.mark.parametrize("total", [None, 12])
def test_mixed_schemas_keep_their_invariants(real, ranged, total):
    schema = mixed(real, ranged)
    if ranged:
        schema = schema.realize(1)
    rows = {"customers": 6, "stores": 8} | ({} if total is None else {"visits": total})
    frames = schema.sample(rows, seed=0)
    customers, stores, visits = frames["customers"], frames["stores"], frames["visits"]
    if total is not None:
        assert len(visits) == total
    assert visits.customer_id.isin(customers.id).all()
    assert visits.customer_id.value_counts().max() <= 3
    linked = visits.dropna(subset=["store_id"])
    assert linked.store_id.isin(stores.id).all()
    assert not linked.duplicated(["customer_id", "store_id"]).any()
    by_store = linked.merge(stores, left_on="store_id", right_on="id", suffixes=("", "_store"))
    assert (by_store.region == by_store.region_store).all()
    budget = visits.merge(customers, left_on="customer_id", right_on="id").budget
    np.testing.assert_array_equal(visits.spent, 0.5 * budget)
    np.testing.assert_allclose(visits.double, 2 * np.round(visits.spent / 100.0, 6))
    assert schema.tables["customers"].doc == "who buys"
    again = schema.sample(rows, seed=0)
    assert all(frames[t].equals(again[t]) for t in frames)
