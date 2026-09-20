import numpy as np
import pandas as pd
import pytest

from plurel import (
    DEFAULT_CALENDAR,
    SCM,
    Column,
    Combine,
    Exponential,
    LinearEffect,
    LogNormal,
    MatrixEffect,
    Normal,
    Root,
    Softmax,
    generator,
)
from plurel.links import HSBMLink, RandomLink, TreeLink
from plurel.schema import AGGREGATES, FK, Port, Schema

ROWS = {"customers": 300, "orders": 2000, "employees": 150}
EMBEDDING = np.arange(9.0).reshape(3, 3)


def customers():
    return SCM(
        {
            "segment": Softmax(biases=(0.0, 0.5, -0.5)),
            "value": Root(),
            "n_orders": Port("orders", "amount", aggregate="count"),
            "spend": Port("orders", "amount", aggregate="sum"),
            "churn": Combine(
                (LinearEffect("spend", -0.1), LinearEffect("value")), noise=Normal(std=0.1)
            ),
        },
        {
            "segment": Column("segment", "categorical", categories=("a", "b", "c")),
            "value": Column("value"),
            "n_orders": Column("n_orders"),
            "churn": Column("churn"),
        },
    )


def orders():
    return SCM(
        {
            "segment": Port("customers", "segment", dim=3, fill=0.0),
            "value": Port("customers", "value", fill=0.0),
            "embedding": Combine((MatrixEffect("segment", EMBEDDING),), noise=None),
            "amount": Combine(
                (LinearEffect("value", 2.0), MatrixEffect("embedding", np.ones((3, 1)))),
                noise=Normal(std=0.5),
            ),
        },
        {"amount": Column("amount"), "value": Column("value")},
    )


def employees():
    return SCM(
        {
            "level": Root(),
            "manager_level": Port(None, "level", via="manager_id", fill=0.0),
            "pay": Combine(
                (LinearEffect("level"), LinearEffect("manager_level", 0.5)), noise=Normal(std=0.1)
            ),
        },
        {"level": Column("level"), "pay": Column("pay")},
    )


FKEYS = (
    FK("orders", "customer_id", "customers", HSBMLink((2,), (2,)), nullable=0.1),
    FK("employees", "manager_id", "employees", TreeLink(roots=0.2)),
)


@pytest.fixture
def schema():
    return Schema({"customers": customers(), "orders": orders(), "employees": employees()}, FKEYS)


def test_sample_propagates_latents_in_both_directions(schema):
    frames, latents = schema.sample_with_latents(ROWS, seed=0)
    assert set(frames) == set(ROWS) and all(len(frames[t]) == n for t, n in ROWS.items())
    assert schema.order.index(("orders", "amount")) < schema.order.index(("customers", "spend"))
    fk = frames["orders"]["customer_id"]
    assert fk.dtype == "Int64" and 0.05 < fk.isna().mean() < 0.15
    linked = fk.notna().to_numpy()
    index = fk.to_numpy(dtype=float, na_value=-1).astype(int)
    np.testing.assert_array_equal(
        latents["orders"]["value"][linked, 0], latents["customers"]["value"][index[linked], 0]
    )
    assert not latents["orders"]["value"][~linked].any()
    np.testing.assert_array_equal(
        latents["orders"]["embedding"],
        EMBEDDING[latents["customers"]["segment"][index].argmax(1)] * linked[:, None],
    )
    counts = np.bincount(index[linked], minlength=ROWS["customers"])
    np.testing.assert_array_equal(frames["customers"]["n_orders"], counts)
    spend = np.zeros(ROWS["customers"])
    np.add.at(spend, index[linked], latents["orders"]["amount"][linked, 0])
    np.testing.assert_allclose(latents["customers"]["spend"][:, 0], spend)
    assert np.corrcoef(spend, frames["customers"]["churn"])[0, 1] < -0.3
    manager = frames["employees"]["manager_id"]
    roots = manager.isna().to_numpy()
    assert roots[0] and (manager.dropna().to_numpy() < np.flatnonzero(~roots)).all()
    level = latents["employees"]["level"][:, 0]
    np.testing.assert_array_equal(
        latents["employees"]["manager_level"][~roots, 0],
        level[manager.dropna().to_numpy().astype(int)],
    )
    assert not latents["employees"]["manager_level"][roots].any()


def test_sampling_is_deterministic_and_interventions_keep_common_random_numbers(schema):
    frames, latents = schema.sample_with_latents(ROWS, seed=0)
    again, _ = schema.sample_with_latents(ROWS, seed=0)
    for table in ROWS:
        pd.testing.assert_frame_equal(frames[table], again[table])
    forced, latents_forced = schema.sample_with_latents(
        ROWS, seed=0, interventions={"customers": {"value": 0.0}}
    )
    pd.testing.assert_series_equal(forced["orders"]["customer_id"], frames["orders"]["customer_id"])
    pd.testing.assert_frame_equal(forced["employees"], frames["employees"])
    assert not latents_forced["orders"]["value"].any()
    residual = latents["orders"]["amount"] - 2.0 * latents["orders"]["value"]
    residual_forced = latents_forced["orders"]["amount"] - 2.0 * latents_forced["orders"]["value"]
    np.testing.assert_allclose(residual, residual_forced)
    ported = {"orders": {"value": 1.0}, "customers": {"segment": np.eye(3)[2]}}
    _, latents_ported = schema.sample_with_latents(ROWS, seed=0, interventions=ported)
    assert (latents_ported["orders"]["value"] == 1.0).all()
    linked = frames["orders"]["customer_id"].notna().to_numpy()
    expected = np.tile(EMBEDDING[2], (linked.sum(), 1))
    np.testing.assert_array_equal(latents_ported["orders"]["embedding"][linked], expected)
    assert orders().sample(50, seed=0).shape == (50, 2)


def test_aggregates_handle_empty_groups():
    values = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    index = np.array([0, 0, 2])
    nan = np.nan
    expected = {
        "count": [[2.0], [0.0], [1.0]],
        "sum": [[4.0, 40.0], [0.0, 0.0], [5.0, 50.0]],
        "mean": [[2.0, 20.0], [nan, nan], [5.0, 50.0]],
        "max": [[3.0, 30.0], [nan, nan], [5.0, 50.0]],
        "min": [[1.0, 10.0], [nan, nan], [5.0, 50.0]],
    }
    assert set(expected) == set(AGGREGATES)
    for name, aggregate in AGGREGATES.items():
        np.testing.assert_array_equal(aggregate(values, index, 3), expected[name])
    port = Port("orders", "amount", aggregate="mean")
    with pytest.raises(ValueError, match="fill"):
        port.resolve(values, index, 3)
    np.testing.assert_array_equal(
        Port("orders", "amount", aggregate="mean", fill=-1.0).resolve(values, index, 3)[1],
        [-1.0, -1.0],
    )
    np.testing.assert_array_equal(
        Port("orders", "amount", aggregate="sum").resolve(values, index, 3), expected["sum"]
    )
    with pytest.raises(ValueError, match="fill"):
        Port("customers", "value").resolve(values[:, :1], np.array([0, -1]), 2)
    np.testing.assert_array_equal(
        Port("customers", "value").resolve(values[:, :1], np.array([2, 0]), 2), [[5.0], [1.0]]
    )


def test_schema_validation():
    tables = {"customers": customers(), "orders": orders(), "employees": employees()}
    with pytest.raises(ValueError):
        Schema(tables, (FK("orders", "customer_id", "shops"),))
    with pytest.raises(ValueError):
        Schema(tables, (FK("orders", "amount", "customers"),))
    with pytest.raises(ValueError):
        Schema(tables, FKEYS + (FK("orders", "customer_id", "customers"),))
    with pytest.raises(ValueError):
        Schema(tables, FKEYS[1:])
    twice = FKEYS + (FK("orders", "referrer_id", "customers", RandomLink()),)
    with pytest.raises(ValueError):
        Schema(tables, twice)
    via = {
        "customers": SCM(
            {
                **customers().mechanisms,
                "n_orders": Port("orders", "amount", via="customer_id", aggregate="count"),
                "spend": Port("orders", "amount", via="customer_id", aggregate="sum"),
            },
            customers().columns,
        ),
        "orders": SCM(
            {
                **orders().mechanisms,
                "segment": Port("customers", "segment", via="customer_id", dim=3),
                "value": Port("customers", "value", via="referrer_id"),
            },
            orders().columns,
        ),
    }
    assert Schema({**tables, **via}, twice).ports[("orders", "value")][1].column == "referrer_id"
    for bad in (
        {"segment": Port("customers", "segment")},
        {"value": Port("customers", "nothing")},
        {"value": Port("customers", "value", aggregate="sum")},
    ):
        with pytest.raises(ValueError):
            Schema(
                {**tables, "orders": SCM({**orders().mechanisms, **bad}, orders().columns)}, FKEYS
            )
    with pytest.raises(ValueError):
        Port("orders", "amount", aggregate="count", dim=3)
    with pytest.raises(ValueError):
        Port("orders", "amount", aggregate="median")
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", nullable=1.0)
    looped = SCM(
        {**customers().mechanisms, "value": Port("orders", "value", aggregate="mean")},
        customers().columns,
    )
    with pytest.raises(ValueError):
        Schema({**tables, "customers": looped}, FKEYS)
    schema = Schema(tables, FKEYS)
    with pytest.raises(ValueError):
        schema.sample({"customers": 10, "orders": 10})
    with pytest.raises(ValueError):
        schema.sample(ROWS, interventions={"shops": {"x": 1.0}})
    with pytest.raises(ValueError):
        schema.sample(ROWS, interventions={"customers": {"nothing": 1.0}})
    with pytest.raises(ValueError, match="does not fit"):
        schema.sample(ROWS, interventions={"customers": {"segment": np.ones(2)}})
    empty = schema.sample({"customers": 5, "orders": 0, "employees": 0}, seed=0)
    assert len(empty["orders"]) == 0 and list(empty["orders"]) == ["amount", "value", "customer_id"]
    assert not empty["customers"]["n_orders"].any()


def test_evaluation_order_within_a_generation_does_not_matter(schema):
    linking, noise, _ = generator(0).spawn(3)
    links = schema.links(ROWS, linking)
    expected = schema.propagate(ROWS, links, noise, {})
    streams = dict(zip(schema.order, generator(0).spawn(3)[1].spawn(len(schema.order))))
    latents = {table: {} for table in ROWS}
    for generation in schema.generations:
        for table, name in reversed(generation):
            node = (table, name)
            latents[table][name] = schema.evaluate(node, ROWS, latents, links, streams[node], {})
    for table in ROWS:
        assert set(latents[table]) == set(expected[table])
        for name, latent in latents[table].items():
            np.testing.assert_array_equal(latent, expected[table][name])


def test_influence_flows_child_to_parent_to_other_child():
    tables = {
        "a": SCM({"total": Port("b", "x", aggregate="sum")}, {"total": Column("total")}),
        "b": SCM({"x": Root()}, {"x": Column("x")}),
        "c": SCM({"from_a": Port("a", "total")}, {"from_a": Column("from_a")}),
        "d": SCM(
            {"from_b": Port("b", "x"), "from_c": Port("c", "from_a")},
            {"from_b": Column("from_b"), "from_c": Column("from_c")},
        ),
    }
    fkeys = (FK("b", "a_id", "a"), FK("c", "a_id", "a"), FK("d", "b_id", "b"), FK("d", "c_id", "c"))
    schema = Schema(tables, fkeys)
    positions = [
        schema.order.index(node)
        for node in (("b", "x"), ("a", "total"), ("c", "from_a"), ("d", "from_c"))
    ]
    assert positions == sorted(positions)
    rows = {"a": 20, "b": 500, "c": 100, "d": 1000}
    frames, latents = schema.sample_with_latents(rows, seed=0)
    total = np.zeros(rows["a"])
    np.add.at(total, frames["b"]["a_id"].to_numpy(), latents["b"]["x"][:, 0])
    np.testing.assert_allclose(frames["a"]["total"], total)
    np.testing.assert_allclose(frames["c"]["from_a"], total[frames["c"]["a_id"].to_numpy()])
    np.testing.assert_allclose(
        frames["d"]["from_c"], frames["c"]["from_a"].to_numpy()[frames["d"]["c_id"].to_numpy()]
    )
    np.testing.assert_allclose(
        frames["d"]["from_b"], frames["b"]["x"].to_numpy()[frames["d"]["b_id"].to_numpy()]
    )


def test_ports_read_through_the_key_they_name():
    buyer_seller = SCM(
        {
            "buyer_value": Port("customers", "value", via="buyer_id"),
            "seller_value": Port("customers", "value", via="seller_id"),
        },
        {"buyer_value": Column("buyer_value"), "seller_value": Column("seller_value")},
    )
    both = SCM(
        {
            "value": Root(),
            "bought": Port("orders", "buyer_value", via="buyer_id", aggregate="count"),
            "sold": Port("orders", "seller_value", via="seller_id", aggregate="count"),
        },
        {"bought": Column("bought"), "sold": Column("sold")},
    )
    fkeys = (FK("orders", "buyer_id", "customers"), FK("orders", "seller_id", "customers"))
    schema = Schema({"customers": both, "orders": buyer_seller}, fkeys)
    rows = {"customers": 50, "orders": 2000}
    frames, latents = schema.sample_with_latents(rows, seed=0)
    value = latents["customers"]["value"][:, 0]
    buyer = frames["orders"]["buyer_id"].to_numpy()
    seller = frames["orders"]["seller_id"].to_numpy()
    assert not np.array_equal(buyer, seller)
    np.testing.assert_array_equal(frames["orders"]["buyer_value"], value[buyer])
    np.testing.assert_array_equal(frames["orders"]["seller_value"], value[seller])
    np.testing.assert_array_equal(frames["customers"]["bought"], np.bincount(buyer, minlength=50))
    np.testing.assert_array_equal(frames["customers"]["sold"], np.bincount(seller, minlength=50))
    assert Schema({"t": SCM({"x": Root()}, {"x": Column("x")})}).sample({"t": 5}, seed=0)[
        "t"
    ].shape == (5, 1)
    assert Schema({"t": SCM({}, {})}).sample({"t": 4}, seed=0)["t"].shape == (4, 0)


class BadLink:
    def __init__(self, index):
        self.index = index

    def sample(self, n_child, n_parent, rng):
        return self.index


def test_orphans_and_bad_inputs_surface_instead_of_looking_like_data():
    customers = SCM({"segment": Softmax(biases=(0.0, 0.0, 0.0)), "value": Root()}, {})
    orders = SCM(
        {
            "segment": Port("customers", "segment", dim=3, fill=0.0),
            "value": Port("customers", "value", fill=np.nan),
        },
        {
            "segment": Column("segment", "categorical", categories=("a", "b", "c")),
            "amount": Column("value", marginal=LogNormal()),
        },
    )
    fkeys = (FK("orders", "customer_id", "customers", nullable=0.3),)
    schema = Schema({"customers": customers, "orders": orders}, fkeys)
    frames = schema.sample({"customers": 20, "orders": 1000}, seed=0)
    orphan = frames["orders"]["customer_id"].isna()
    assert 0.2 < orphan.mean() < 0.4
    pd.testing.assert_series_equal(frames["orders"]["segment"].isna(), orphan, check_names=False)
    pd.testing.assert_series_equal(frames["orders"]["amount"].isna(), orphan, check_names=False)
    assert (frames["orders"]["amount"].dropna() > 0).all()
    with pytest.raises(ValueError, match="integers"):
        schema.sample({"customers": 20.0, "orders": 10}, seed=0)
    unfilled = SCM({"value": Port("customers", "value")}, {"value": Column("value")})
    with pytest.raises(ValueError, match="fill"):
        Schema({"customers": customers, "orders": unfilled}, fkeys).sample(
            {"customers": 20, "orders": 10}, seed=0
        )
    assert (
        Schema(
            {"customers": customers, "orders": unfilled},
            (FK("orders", "customer_id", "customers"),),
        )
        .sample({"customers": 20, "orders": 10}, seed=0)["orders"]
        .notna()
        .all()
        .all()
    )
    downstream = SCM(
        {
            "value": Port("customers", "value", fill=np.nan),
            "double": Combine((LinearEffect("value", 2.0),)),
        },
        {},
    )
    with pytest.raises(ValueError, match="non-finite"):
        Schema({"customers": customers, "orders": downstream}, fkeys).sample(
            {"customers": 20, "orders": 10}, seed=0
        )
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", TreeLink())
    for bad in (np.zeros(10), np.zeros(9, dtype=int), np.full(10, 20), np.full(10, -2)):
        broken = Schema(
            {"customers": customers, "orders": orders},
            (FK("orders", "customer_id", "customers", BadLink(bad)),),
        )
        with pytest.raises(ValueError, match="link"):
            broken.sample({"customers": 20, "orders": 10}, seed=0)


def test_child_events_follow_their_parent_events():
    customers = SCM(
        {"signup": Root(noise=DEFAULT_CALENDAR)},
        {"signup": Column("signup", "timestamp")},
    )
    orders = SCM(
        {
            "signup": Port("customers", "signup", fill=np.nan),
            "when": Combine((LinearEffect("signup"),), noise=Exponential(30 * 24 * 3600.0)),
        },
        {"when": Column("when", "timestamp")},
        time_column="when",
    )
    schema = Schema(
        {"customers": customers, "orders": orders}, (FK("orders", "customer_id", "customers"),)
    )
    frames = schema.sample({"customers": 50, "orders": 500}, seed=0)
    signup = frames["customers"]["signup"].to_numpy()[frames["orders"]["customer_id"].to_numpy()]
    when = frames["orders"]["when"].to_numpy()
    assert (when >= signup).all() and frames["orders"]["when"].dtype == "datetime64[ns]"
    assert frames["customers"]["signup"].is_monotonic_increasing
