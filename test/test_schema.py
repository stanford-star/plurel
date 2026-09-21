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
    NearestEdge,
    Node,
    Normal,
    Uniform,
)
from plurel.graph import AGGREGATES, Foreign, Summary
from plurel.links import HSBMLink, RandomLink, TreeLink
from plurel.schema import FK, Schema


def simulate(scm, n, *, seed=None, interventions=None):
    forced = {"t": interventions} if interventions else None
    return Schema({"t": scm}).sample_with_latents({"t": n}, seed=seed, interventions=forced)[1]["t"]


def sample_with_latents(scm, n, *, seed=None, interventions=None):
    forced = {"t": interventions} if interventions else None
    frames, latents = Schema({"t": scm}).sample_with_latents(
        {"t": n}, seed=seed, interventions=forced
    )
    return frames["t"], latents["t"]


def sample(scm, n, *, seed=None, interventions=None):
    return sample_with_latents(scm, n, seed=seed, interventions=interventions)[0]


N = 300
TABLE = np.arange(6.0).reshape(3, 2)
NODES = {
    "y": Node((LinearEdge("x", 2.0), LinearEdge("xz", -0.5)), noise=Normal(std=0.1)),
    "xz": Node((LinearEdge("x"), LinearEdge("z")), op="product", noise=None),
    "x": Node(),
    "z": Node(),
    "h": Node(dim=3),
    "segment": Node((MatrixEdge("h", np.eye(3)),), onehot=True, noise=Gumbel()),
    "cluster": Node((NearestEdge("h", np.eye(3)),), noise=None),
    "embedding": Node((MatrixEdge("segment", TABLE),), noise=None),
    "hidden": Node(
        (MatrixEdge("y", np.array([[0.0, 2.0]])),), bias=(0.0, -1.0), onehot=True, noise=Gumbel()
    ),
}
COLUMNS = {name: Column(name) for name in ("y", "xz", "x", "z")}


@pytest.fixture
def scm():
    return SCM(NODES, COLUMNS)


def test_simulate_evaluates_every_node_in_topological_order(scm):
    latents = simulate(scm, N, seed=0)
    assert set(latents) == set(NODES)
    assert scm.order.index("x") < scm.order.index("xz") < scm.order.index("y")
    for name, node in NODES.items():
        assert latents[name].shape == (N, node.dim)
    np.testing.assert_allclose(latents["xz"], latents["x"] * latents["z"])
    np.testing.assert_array_equal(latents["embedding"], TABLE[latents["segment"].argmax(1)])
    again = simulate(scm, N, seed=0)
    assert all(np.array_equal(latents[name], again[name]) for name in NODES)
    assert not np.array_equal(latents["x"], simulate(scm, N, seed=1)["x"])


def test_interventions_replace_a_node_and_keep_common_random_numbers(scm):
    factual = simulate(scm, N, seed=0)
    counterfactual = simulate(scm, N, seed=0, interventions={"x": 0.0})
    assert not counterfactual["x"].any()
    for name in ("z", "h", "segment"):
        np.testing.assert_array_equal(counterfactual[name], factual[name])
    residuals = [v["y"] - 2.0 * v["x"] + 0.5 * v["x"] * v["z"] for v in (factual, counterfactual)]
    np.testing.assert_allclose(*residuals)
    forced = simulate(scm, N, seed=0, interventions={"segment": np.eye(3)[1]})
    np.testing.assert_array_equal(forced["embedding"], np.tile(TABLE[1], (N, 1)))
    per_row = simulate(scm, N, seed=0, interventions={"x": np.arange(N)})
    np.testing.assert_array_equal(per_row["x"], np.arange(N)[:, None])
    with pytest.raises(ValueError):
        simulate(scm, N, seed=0, interventions={"missing": 1.0})
    with pytest.raises(ValueError):
        simulate(scm, N, seed=0, interventions={"h": np.ones((N, 2))})


def test_construction_rejects_unknown_parents_and_cycles():
    with pytest.raises(ValueError):
        SCM({"y": Node((LinearEdge("x"),))}, {})
    with pytest.raises(ValueError):
        SCM({"a": Node((LinearEdge("b"),)), "b": Node((LinearEdge("a"),))}, {})


def test_simulate_rejects_a_node_that_breaks_its_declared_width():
    scm = SCM({"h": Node(dim=3), "y": Node((LinearEdge("h"),))}, {})
    with pytest.raises(ValueError, match="declared"):
        simulate(scm, N, seed=0)
    overflow = SCM({"x": Node(), "y": Node((LinearEdge("x", np.inf),))}, {})
    with pytest.raises(ValueError, match="non-finite"):
        simulate(overflow, N, seed=0)


def test_sample_observes_columns_from_one_draw(scm):
    frame, latents = sample_with_latents(scm, N, seed=0)
    assert list(frame) == ["y", "xz", "x", "z"] and len(frame) == N
    np.testing.assert_array_equal(frame["x"], latents["x"].ravel())
    assert all(np.array_equal(latents[k], v) for k, v in simulate(scm, N, seed=0).items())
    columns = {
        "amount": Column("y", marginal=Uniform(), missing=0.1),
        "segment": Column("segment", "categorical", categories=("a", "b", "c")),
        "when": Column("z", "timestamp", marginal=DEFAULT_CALENDAR),
    }
    typed = SCM(NODES, columns)
    frame = sample(typed, N, seed=0)
    assert list(frame) == list(columns) and frame["when"].dtype == "datetime64[ns]"
    assert set(frame["segment"]) == {"a", "b", "c"} and 0.05 < frame["amount"].isna().mean() < 0.15
    same = sample(typed, N, seed=0, interventions={"x": 0.0})
    pd.testing.assert_series_equal(same["when"], frame["when"])
    mnar = sample_with_latents(SCM(NODES, {"y": Column("y", missing="hidden")}), N, seed=0)
    frame, latents = mnar
    assert frame["y"].isna().to_numpy().tolist() == (latents["hidden"][:, 1] == 1.0).tolist()
    assert latents["y"][frame["y"].isna()].mean() > latents["y"][frame["y"].notna()].mean()
    with pytest.raises(ValueError):
        SCM(NODES, {"c": Column("missing")})
    with pytest.raises(ValueError):
        SCM(NODES, {"c": Column("y", missing="nobody")})


def test_declared_time_order_is_enforced_on_the_observed_table():
    def table(delay, marginal=None):
        return SCM(
            {
                "placed": Node(noise=DEFAULT_CALENDAR),
                "shipped": Node((LinearEdge("placed"),), noise=delay),
            },
            {
                "placed": Column("placed", "timestamp", marginal=marginal),
                "shipped": Column("shipped", "timestamp", marginal=marginal, after="placed"),
            },
            time_column="placed",
        )

    frame = sample(table(Exponential(3600.0)), N, seed=0)
    assert (frame["shipped"] >= frame["placed"]).all()
    assert table(Exponential(3600.0)).timestamp_nodes == {"placed", "shipped"}
    with pytest.raises(ValueError, match="precedes"):
        sample(table(Normal(std=3600.0)), N, seed=0)
    with pytest.raises(ValueError, match="precedes"):
        sample(table(Exponential(3600.0), marginal=DEFAULT_CALENDAR), N, seed=0)
    for after in ("nothing", "t"):
        with pytest.raises(ValueError, match="another timestamp"):
            SCM({"t": Node()}, {"t": Column("t", "timestamp", after=after)})


ROWS = {"customers": 300, "orders": 2000, "employees": 150}
EMBEDDING = np.arange(9.0).reshape(3, 3)


def copy(dim=1):
    """A node whose value is exactly what its crossing edges bring."""
    return Node(dim=dim, noise=None)


def customers():
    return SCM(
        {
            "segment": Node(bias=(0.0, 0.5, -0.5), onehot=True, noise=Gumbel()),
            "value": Node(),
            "n_orders": copy(),
            "spend": copy(),
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
            "segment": copy(3),
            "value": copy(),
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
            "manager_level": copy(),
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
CROSSINGS = {
    ("customers", "n_orders"): (LinearEdge(Summary("orders", "customer_id", "amount", "count")),),
    ("customers", "spend"): (LinearEdge(Summary("orders", "customer_id", "amount", "sum")),),
    ("orders", "segment"): (LinearEdge(Foreign("customer_id", "segment"), dim=3),),
    ("orders", "value"): (LinearEdge(Foreign("customer_id", "value")),),
    ("employees", "manager_level"): (LinearEdge(Foreign("manager_id", "level")),),
}


def tables():
    return {"customers": customers(), "orders": orders(), "employees": employees()}


@pytest.fixture
def schema():
    return Schema(tables(), FKEYS, CROSSINGS)


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
    with pytest.raises(ValueError, match="belong to the Schema"):
        SCM({"x": Node((LinearEdge(Foreign("k", "y")),))}, {})


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
    with pytest.raises(ValueError):
        Schema(tables(), (FK("orders", "customer_id", "shops"),))
    with pytest.raises(ValueError):
        Schema(tables(), (FK("orders", "amount", "customers"),))
    with pytest.raises(ValueError):
        Schema(tables(), FKEYS + (FK("orders", "customer_id", "customers"),))
    with pytest.raises(ValueError, match="no key column"):
        Schema(tables(), FKEYS[1:], CROSSINGS)
    twice = FKEYS + (FK("orders", "referrer_id", "customers", RandomLink()),)
    referred = {**CROSSINGS, ("orders", "value"): (LinearEdge(Foreign("referrer_id", "value")),)}
    schema = Schema(tables(), twice, referred)
    assert schema.source("orders", Foreign("referrer_id", "value")) == ("customers", "value")
    for bad, message in (
        ({("orders", "value"): (LinearEdge(Foreign("nothing", "value")),)}, "no key column"),
        ({("orders", "value"): (LinearEdge(Foreign("customer_id", "nothing")),)}, "unknown node"),
        (
            {("orders", "value"): (LinearEdge(Summary("orders", "customer_id", "amount", "sum")),)},
            "does not point",
        ),
        ({("orders", "value"): (LinearEdge(("customers", "value")),)}, "not a Foreign"),
        ({("orders", "value"): (LinearEdge("amount"),)}, "across a key"),
        ({("orders", "nothing"): (LinearEdge(Foreign("customer_id", "value")),)}, "unknown node"),
    ):
        with pytest.raises(ValueError, match=message):
            Schema(tables(), FKEYS, {**CROSSINGS, **bad})
    with pytest.raises(ValueError):
        Summary("orders", "customer_id", "amount", "median")
    with pytest.raises(ValueError):
        Summary("orders", "customer_id", "amount", "count", fill=0.0)
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", nullable=1.0)
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", fill=np.inf)
    looped = {
        **CROSSINGS,
        ("customers", "value"): (LinearEdge(Summary("orders", "customer_id", "value", "mean")),),
    }
    with pytest.raises(ValueError, match="acyclic"):
        Schema(tables(), FKEYS, looped)
    schema = Schema(tables(), FKEYS, CROSSINGS)
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
            for parent in node.parents:
                assert position[table, parent] < position[table, name]
    for (table, name), edges in schema.crossings.items():
        for edge in edges:
            assert position[schema.source(table, edge.parent)] < position[table, name]


def test_influence_flows_child_to_parent_to_other_child():
    tables = {
        "a": SCM({"total": copy()}, {"total": Column("total")}),
        "b": SCM({"x": Node()}, {"x": Column("x")}),
        "c": SCM({"from_a": copy()}, {"from_a": Column("from_a")}),
        "d": SCM(
            {"from_b": copy(), "from_c": copy()},
            {"from_b": Column("from_b"), "from_c": Column("from_c")},
        ),
    }
    fkeys = (FK("b", "a_id", "a"), FK("c", "a_id", "a"), FK("d", "b_id", "b"), FK("d", "c_id", "c"))
    crossings = {
        ("a", "total"): (LinearEdge(Summary("b", "a_id", "x", "sum")),),
        ("c", "from_a"): (LinearEdge(Foreign("a_id", "total")),),
        ("d", "from_b"): (LinearEdge(Foreign("b_id", "x")),),
        ("d", "from_c"): (LinearEdge(Foreign("c_id", "from_a")),),
    }
    schema = Schema(tables, fkeys, crossings)
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


def test_edges_read_through_the_key_they_name():
    buyer_seller = SCM(
        {"buyer_value": copy(), "seller_value": copy()},
        {"buyer_value": Column("buyer_value"), "seller_value": Column("seller_value")},
    )
    both = SCM(
        {"value": Node(), "bought": copy(), "sold": copy()},
        {"bought": Column("bought"), "sold": Column("sold")},
    )
    fkeys = (FK("orders", "buyer_id", "customers"), FK("orders", "seller_id", "customers"))
    crossings = {
        ("orders", "buyer_value"): (LinearEdge(Foreign("buyer_id", "value")),),
        ("orders", "seller_value"): (LinearEdge(Foreign("seller_id", "value")),),
        ("customers", "bought"): (
            LinearEdge(Summary("orders", "buyer_id", "buyer_value", "count")),
        ),
        ("customers", "sold"): (
            LinearEdge(Summary("orders", "seller_id", "seller_value", "count")),
        ),
    }
    schema = Schema({"customers": both, "orders": buyer_seller}, fkeys, crossings)
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
        {"segment": copy(3), "value": copy()},
        {
            "segment": Column("segment", "categorical", categories=("a", "b", "c")),
            "amount": Column("value", marginal=LogNormal()),
        },
    )
    crossings = {
        ("orders", "segment"): (LinearEdge(Foreign("customer_id", "segment"), dim=3),),
        ("orders", "value"): (LinearEdge(Foreign("customer_id", "value")),),
    }
    fkeys = (FK("orders", "customer_id", "customers", nullable=0.3, fill=0.0),)
    schema = Schema({"customers": customers, "orders": orders}, fkeys, crossings)
    frames = schema.sample({"customers": 20, "orders": 1000}, seed=0)
    orphan = frames["orders"]["customer_id"].isna()
    assert 0.2 < orphan.mean() < 0.4
    pd.testing.assert_series_equal(frames["orders"]["segment"].isna(), orphan, check_names=False)
    amount = frames["orders"]["amount"].mask(orphan)
    assert amount.isna().to_numpy().tolist() == orphan.tolist() and (amount.dropna() > 0).all()
    with pytest.raises(ValueError, match="integers"):
        schema.sample({"customers": 20.0, "orders": 10}, seed=0)
    unfilled = (FK("orders", "customer_id", "customers", nullable=0.3),)
    with pytest.raises(ValueError, match="fill"):
        Schema({"customers": customers, "orders": orders}, unfilled, crossings).sample(
            {"customers": 20, "orders": 10}, seed=0
        )
    complete = (FK("orders", "customer_id", "customers"),)
    assert (
        Schema({"customers": customers, "orders": orders}, complete, crossings)
        .sample({"customers": 20, "orders": 10}, seed=0)["orders"]
        .notna()
        .all()
        .all()
    )
    childless = SCM({**customers.nodes, "mean": copy()}, {})
    unfilled_summary = {
        **crossings,
        ("customers", "mean"): (LinearEdge(Summary("orders", "customer_id", "value", "mean")),),
    }
    with pytest.raises(ValueError, match="fill"):
        Schema({"customers": childless, "orders": orders}, complete, unfilled_summary).sample(
            {"customers": 20, "orders": 10}, seed=0
        )
    with pytest.raises(ValueError):
        FK("orders", "customer_id", "customers", TreeLink())
    for bad in (np.zeros(10), np.zeros(9, dtype=int), np.full(10, 20), np.full(10, -2)):
        broken = Schema(
            {"customers": customers, "orders": orders},
            (FK("orders", "customer_id", "customers", BadLink(bad), fill=0.0),),
            crossings,
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
            "signup": copy(),
            "when": Node((LinearEdge("signup"),), noise=Exponential(30 * 24 * 3600.0)),
        },
        {"when": Column("when", "timestamp")},
        time_column="when",
    )
    schema = Schema(
        {"customers": customers, "orders": orders},
        (FK("orders", "customer_id", "customers"),),
        {("orders", "signup"): (LinearEdge(Foreign("customer_id", "signup")),)},
    )
    frames = schema.sample({"customers": 50, "orders": 500}, seed=0)
    signup = frames["customers"]["signup"].to_numpy()[frames["orders"]["customer_id"].to_numpy()]
    when = frames["orders"]["when"].to_numpy()
    assert (when >= signup).all() and frames["orders"]["when"].dtype == "datetime64[ns]"
    assert frames["customers"]["signup"].is_monotonic_increasing
