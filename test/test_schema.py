import numpy as np
import pandas as pd
import pytest

from plurel import (
    DEFAULT_CALENDAR,
    SCM,
    Column,
    Exponential,
    Gumbel,
    LinearEdge,
    LogNormal,
    MatrixEdge,
    Node,
    Normal,
    generator,
)
from plurel.links import HSBMLink, RandomLink, TreeLink
from plurel.schema import AGGREGATES, FK, Childless, Foreign, Orphan, Schema, Summary

ROWS = {"customers": 300, "orders": 2000, "employees": 150}
EMBEDDING = np.arange(9.0).reshape(3, 3)


def copy(tail, dim=1):
    return Node((LinearEdge(tail, dim=dim),), noise=None)


def customers():
    return SCM(
        {
            "segment": Node(bias=(0.0, 0.5, -0.5), onehot=True, noise=Gumbel()),
            "value": Node(),
            "n_orders": copy(Summary("orders", "customer_id", "amount", "count")),
            "spend": copy(Summary("orders", "customer_id", "amount", "sum")),
            "churn": Node((LinearEdge("spend", -0.1), LinearEdge("value")), noise=Normal(std=0.1)),
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
            "segment": copy(Foreign("customer_id", "segment"), dim=3),
            "value": copy(Foreign("customer_id", "value")),
            "embedding": Node((MatrixEdge("segment", EMBEDDING),), noise=None),
            "amount": Node(
                (LinearEdge("value", 2.0), MatrixEdge("embedding", np.ones((3, 1)))),
                noise=Normal(std=0.5),
            ),
        },
        {"amount": Column("amount"), "value": Column("value")},
    )


def employees():
    return SCM(
        {
            "level": Node(),
            "manager_level": copy(Foreign("manager_id", "level")),
            "pay": Node(
                (LinearEdge("level"), LinearEdge("manager_level", 0.5)), noise=Normal(std=0.1)
            ),
        },
        {"level": Column("level"), "pay": Column("pay")},
    )


FKEYS = (
    FK("orders", "customer_id", "customers", HSBMLink((2,), (2,)), nullable=0.1, fill=0.0),
    FK("employees", "manager_id", "employees", TreeLink(roots=0.2), fill=0.0),
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
    with pytest.raises(ValueError, match="Schema"):
        orders().sample(50, seed=0)


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
    schema = Schema(
        {"a": SCM({"x": Node()}, {}), "b": SCM({"x": Node(dim=2)}, {})},
        (FK("b", "a_id", "a", nullable=0.5),),
    )
    rows, latents = {"a": 3, "b": 3}, {"a": {"x": values[:, :1]}, "b": {"x": values}}
    links = {("b", "a_id"): index}
    with pytest.raises(ValueError, match="fill"):
        schema.resolve("a", Summary("b", "a_id", "x", "mean"), rows, latents, links)
    np.testing.assert_array_equal(
        schema.resolve("a", Summary("b", "a_id", "x", "mean", fill=-1.0), rows, latents, links)[1],
        [-1.0, -1.0],
    )
    np.testing.assert_array_equal(
        schema.resolve("a", Summary("b", "a_id", "x", "sum"), rows, latents, links),
        expected["sum"],
    )
    with pytest.raises(ValueError, match="fill"):
        schema.resolve(
            "b", Foreign("a_id", "x"), rows, latents, {("b", "a_id"): np.array([0, -1, 2])}
        )
    np.testing.assert_array_equal(
        schema.resolve(
            "b", Foreign("a_id", "x"), rows, latents, {("b", "a_id"): np.array([2, 0, 1])}
        ),
        [[5.0], [1.0], [3.0]],
    )


def test_schema_validation():
    tables = {"customers": customers(), "orders": orders(), "employees": employees()}
    with pytest.raises(ValueError):
        Schema(tables, (FK("orders", "customer_id", "shops"),))
    with pytest.raises(ValueError):
        Schema(tables, (FK("orders", "amount", "customers"),))
    with pytest.raises(ValueError):
        Schema(tables, FKEYS + (FK("orders", "customer_id", "customers"),))
    with pytest.raises(ValueError, match="no key column"):
        Schema(tables, FKEYS[1:])
    twice = FKEYS + (FK("orders", "referrer_id", "customers", RandomLink()),)
    referred = SCM(
        {**orders().nodes, "value": copy(Foreign("referrer_id", "value"))}, orders().columns
    )
    schema = Schema({**tables, "orders": referred}, twice)
    assert schema.source("orders", Foreign("referrer_id", "value")) == ("customers", "value")
    for bad, message in (
        ({"value": copy(Foreign("nothing", "value"))}, "no key column"),
        ({"value": copy(Foreign("customer_id", "nothing"))}, "unknown node"),
        ({"value": copy(Summary("orders", "customer_id", "amount", "sum"))}, "does not point"),
        ({"value": copy(("customers", "value"))}, "not a node name"),
    ):
        with pytest.raises(ValueError, match=message):
            Schema({**tables, "orders": SCM({**orders().nodes, **bad}, orders().columns)}, FKEYS)
    with pytest.raises(ValueError, match="marker"):
        Schema(
            {**tables, "orders": SCM(orders().nodes, {"amount": Column("amount", missing=("x",))})},
            FKEYS,
        )
    with pytest.raises(ValueError, match="does not point"):
        Schema(
            {
                **tables,
                "orders": SCM(
                    orders().nodes,
                    {"amount": Column("amount", missing=Childless("orders", "customer_id"))},
                ),
            },
            FKEYS,
        )
    with pytest.raises(ValueError):
        Summary("orders", "customer_id", "amount", "median")
    with pytest.raises(ValueError):
        Summary("orders", "customer_id", "amount", "count", fill=0.0)
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", nullable=1.0)
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", fill=np.inf)
    looped = SCM(
        {**customers().nodes, "value": copy(Summary("orders", "customer_id", "value", "mean"))},
        customers().columns,
    )
    with pytest.raises(ValueError, match="acyclic"):
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


def test_order_puts_every_node_after_its_parents(schema):
    position = {node: k for k, node in enumerate(schema.order)}
    assert set(position) == {(t, n) for t, scm in schema.tables.items() for n in scm.nodes}
    for table, scm in schema.tables.items():
        for name, node in scm.nodes.items():
            for tail in node.parents:
                assert position[schema.source(table, tail)] < position[table, name]


def test_influence_flows_child_to_parent_to_other_child():
    tables = {
        "a": SCM({"total": copy(Summary("b", "a_id", "x", "sum"))}, {"total": Column("total")}),
        "b": SCM({"x": Node()}, {"x": Column("x")}),
        "c": SCM({"from_a": copy(Foreign("a_id", "total"))}, {"from_a": Column("from_a")}),
        "d": SCM(
            {"from_b": copy(Foreign("b_id", "x")), "from_c": copy(Foreign("c_id", "from_a"))},
            {"from_b": Column("from_b"), "from_c": Column("from_c")},
        ),
    }
    fkeys = (FK("b", "a_id", "a"), FK("c", "a_id", "a"), FK("d", "b_id", "b"), FK("d", "c_id", "c"))
    schema = Schema(tables, fkeys)
    positions = [
        schema.order.index(location)
        for location in (("b", "x"), ("a", "total"), ("c", "from_a"), ("d", "from_c"))
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


def test_edges_read_through_the_key_they_name():
    buyer_seller = SCM(
        {
            "buyer_value": copy(Foreign("buyer_id", "value")),
            "seller_value": copy(Foreign("seller_id", "value")),
        },
        {"buyer_value": Column("buyer_value"), "seller_value": Column("seller_value")},
    )
    both = SCM(
        {
            "value": Node(),
            "bought": copy(Summary("orders", "buyer_id", "buyer_value", "count")),
            "sold": copy(Summary("orders", "seller_id", "seller_value", "count")),
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
    assert Schema({"t": SCM({"x": Node()}, {"x": Column("x")})}).sample({"t": 5}, seed=0)[
        "t"
    ].shape == (5, 1)
    assert Schema({"t": SCM({}, {})}).sample({"t": 4}, seed=0)["t"].shape == (4, 0)


class BadLink:
    def __init__(self, index):
        self.index = index

    def sample(self, n_child, n_parent, rng):
        return self.index


def test_orphans_and_bad_inputs_surface_instead_of_looking_like_data():
    customers = SCM(
        {"segment": Node(bias=(0.0, 0.0, 0.0), onehot=True, noise=Gumbel()), "value": Node()}, {}
    )
    orders = SCM(
        {
            "segment": copy(Foreign("customer_id", "segment"), dim=3),
            "value": copy(Foreign("customer_id", "value")),
        },
        {
            "segment": Column("segment", "categorical", categories=("a", "b", "c")),
            "amount": Column("value", marginal=LogNormal(), missing=Orphan("customer_id")),
        },
    )
    fkeys = (FK("orders", "customer_id", "customers", nullable=0.3, fill=0.0),)
    schema = Schema({"customers": customers, "orders": orders}, fkeys)
    frames = schema.sample({"customers": 20, "orders": 1000}, seed=0)
    orphan = frames["orders"]["customer_id"].isna()
    assert 0.2 < orphan.mean() < 0.4
    pd.testing.assert_series_equal(frames["orders"]["segment"].isna(), orphan, check_names=False)
    pd.testing.assert_series_equal(frames["orders"]["amount"].isna(), orphan, check_names=False)
    assert (frames["orders"]["amount"].dropna() > 0).all()
    with pytest.raises(ValueError, match="integers"):
        schema.sample({"customers": 20.0, "orders": 10}, seed=0)
    unfilled = (FK("orders", "customer_id", "customers", nullable=0.3),)
    with pytest.raises(ValueError, match="fill"):
        Schema({"customers": customers, "orders": orders}, unfilled).sample(
            {"customers": 20, "orders": 10}, seed=0
        )
    complete = (FK("orders", "customer_id", "customers"),)
    assert (
        Schema({"customers": customers, "orders": orders}, complete)
        .sample({"customers": 20, "orders": 10}, seed=0)["orders"]
        .notna()
        .all()
        .all()
    )
    childless = SCM(
        {**customers.nodes, "mean": copy(Summary("orders", "customer_id", "value", "mean"))}, {}
    )
    with pytest.raises(ValueError, match="fill"):
        Schema({"customers": childless, "orders": orders}, complete).sample(
            {"customers": 20, "orders": 10}, seed=0
        )
    with pytest.raises(ValueError, match="Schema"):
        orders.observe({"segment": np.zeros((2, 3)), "value": np.zeros((2, 1))}, generator(0), 2)
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", TreeLink())
    for bad in (np.zeros(10), np.zeros(9, dtype=int), np.full(10, 20), np.full(10, -2)):
        broken = Schema(
            {"customers": customers, "orders": orders},
            (FK("orders", "customer_id", "customers", BadLink(bad), fill=0.0),),
        )
        with pytest.raises(ValueError, match="link"):
            broken.sample({"customers": 20, "orders": 10}, seed=0)


def test_child_events_follow_their_parent_events():
    customers = SCM(
        {"signup": Node(noise=DEFAULT_CALENDAR)},
        {"signup": Column("signup", "timestamp")},
    )
    orders = SCM(
        {
            "signup": copy(Foreign("customer_id", "signup")),
            "when": Node((LinearEdge("signup"),), noise=Exponential(30 * 24 * 3600.0)),
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
