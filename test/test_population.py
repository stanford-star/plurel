import numpy as np
import pytest

from plurel import Gumbel, Table, Uniform
from plurel.columns import Column
from plurel.graph import LinearEdge, LookupEdge, MatrixEdge, Node, ReadEdge
from plurel.population import Grouping, Population, TableMetadata, entity_cut, table_id


def plan() -> Table:
    return Table(
        {
            "cause": Column(Node()),
            "effect": Column(Node((LinearEdge("cause", 1.0),))),
            "hidden": Node(),
            "label": Column(
                Node((LinearEdge("effect", 1.0), LinearEdge("hidden", 1.0))),
                categories=("no", "yes"),
            ),
        }
    )


def population(grouping: Grouping | None = None) -> Population:
    metadata = TableMetadata(
        feature_names=["cause", "effect"],
        feature_types=["numeric", "numeric"],
        target="label",
        plan=plan(),
        realization={"table_id": table_id("case", 3), "seed": 3},
    )
    return Population(metadata=metadata, lanes=(1, 2), render=None, grouping=grouping)


def test_a_hidden_product_node_multiplies_its_parents_and_stays_hidden():
    scm = Table(
        {
            "a": Column(Node()),
            "b": Column(Node()),
            "a_b": Node((LinearEdge("a"), LinearEdge("b")), op="product", noise=None),
            "y": Column(Node((LinearEdge("a"), LinearEdge("a_b", 2.0)), noise=None)),
        }
    )
    a, b = np.array([1.0, -1.0, 0.5]), np.array([0.0, 2.0, -5.0])
    frame, latents = scm.sample_with_latents(3, held={"a": a[:, None], "b": b[:, None]})
    np.testing.assert_allclose(latents["y"][:, 0], a + 2.0 * a * b)
    assert list(frame) == ["a", "b", "y"] and "a_b" in latents


def test_a_lookup_times_a_parent_is_a_per_level_slope():
    table = {
        "segment": Column(Node(), categories=("lo", "hi")),
        "x": Column(Node()),
        "slope": Node(
            (LookupEdge("segment", (-1.0, 2.0), (0.5, 0.5)), LinearEdge("x")),
            op="product",
            noise=None,
        ),
        "y": Column(Node((LinearEdge("slope"),), noise=None)),
    }
    segment, x = np.array([-2.0, -2.0, 2.0, 2.0]), np.array([1.0, 3.0, 1.0, 3.0])
    latents = Table(table).sample_with_latents(
        4, held={"segment": segment[:, None], "x": x[:, None]}
    )[1]
    np.testing.assert_allclose(latents["y"][:, 0], [-1.0, -3.0, 2.0, 6.0])


def test_a_nominal_target_is_a_one_hot_node_over_matrix_edge_scores():
    target = Node(
        (MatrixEdge("x", np.array([[-2.0, 0.0, 2.0]])),),
        bias=(0.0, 1.0, 0.0),
        onehot=True,
        noise=None,
    )
    scm = Table({"x": Column(Node()), "y": Column(target, categories=("a", "b", "c"))})
    latents = scm.sample_with_latents(3, held={"x": np.array([[-3.0], [0.0], [3.0]])})[1]
    assert latents["y"].argmax(1).tolist() == [0, 1, 2]
    frame, latents = scm.sample_with_latents(400, seed=0)
    assert set(frame["y"]) == {"a", "b", "c"}
    scores = latents["x"] @ np.array([[-2.0, 0.0, 2.0]]) + np.array(target.bias)
    assert np.array_equal(latents["y"].argmax(1), scores.argmax(1))
    scored = Node(target.edges, bias=target.bias, onehot=True, noise=Gumbel())
    soft = Table({"x": Node(), "y": Column(scored, categories=("a", "b", "c"))})
    assert 0.0 < (soft.sample_with_latents(4000)[1]["y"].argmax(1) == 1).mean() < 1.0


def test_a_second_reading_of_a_node_is_a_node_of_its_own():
    scm = Table(
        {
            "u": Column(Node()),
            "again": Column(Node(ReadEdge("u"), noise=None)),
            "y": Column(Node(LinearEdge("u", -1.5), noise=None)),
        }
    )
    frame = scm.sample_with_latents(3, held={"u": np.array([[1.0], [2.0], [3.0]])})[0]
    np.testing.assert_allclose(frame["again"], frame["u"])
    np.testing.assert_allclose(frame["y"], [-1.5, -3.0, -4.5])


def test_a_column_reading_a_held_node_records_its_entitys_value():
    plan = Table(
        {
            "u": Node(),
            "x": Column(Node(ReadEdge("u"), noise=None), marginal=Uniform(0.0, 1.0)),
            "raw": Column(Node()),
        }
    )
    grouping = Grouping(
        "shared_causes", ("u",), rows=50.0, spread=0.0, measurement=0.0, column=None
    )
    frame, entities, _ = grouping.draw(plan, 100, np.random.default_rng(0))
    assert frame.groupby(entities)["x"].nunique().max() == 1
    assert frame.groupby(entities)["raw"].nunique().max() > 1
    noisy = Grouping("shared_causes", ("u",), rows=50.0, spread=0.0, measurement=0.5, column=None)
    frame, entities, _ = noisy.draw(plan, 100, np.random.default_rng(0))
    assert frame.groupby(entities)["x"].nunique().max() > 1


def test_a_noiseless_column_of_held_nodes_records_its_entitys_value():
    plan = Table(
        {
            "h": Node(),
            "x": Column(Node(LinearEdge("h"), noise=None), marginal=Uniform()),
            "raw": Column(Node()),
        }
    )
    grouping = Grouping(
        "shared_causes", ("h",), rows=30.0, spread=0.0, measurement=0.0, column=None
    )
    frame, entities, _ = grouping.draw(plan, 300, np.random.default_rng(0))
    assert frame.groupby(entities)["x"].nunique().max() == 1


def test_a_noisy_reading_keeps_the_columns_blanks():
    plan = Table(
        {"u": Column(Node(), marginal=Uniform(0.0, 1.0), missing=0.3), "raw": Column(Node())}
    )
    grouping = Grouping("shared_causes", ("u",), rows=5.0, spread=0.2, measurement=0.3, column=None)
    frame = grouping.draw(plan, 1000, np.random.default_rng(0))[0]
    assert 0.25 < frame["u"].isna().mean() < 0.35 and frame["u"].dropna().between(0.0, 1.0).all()


def test_a_grouping_rejects_sizes_that_mean_nothing():
    for kwargs in ({"rows": 0.0}, {"rows": float("nan")}, {"spread": -1.0}, {"measurement": -1.0}):
        with pytest.raises(ValueError, match="positive"):
            Grouping(**{"mode": "m", "fixed": (), "rows": 5.0, "spread": 0.0, "measurement": 0.0, "column": None, **kwargs})  # fmt: skip


def test_table_metadata_checks_its_feature_types():
    with pytest.raises(ValueError, match="feature_names"):
        TableMetadata(["a", "b"], ["numeric"])
    with pytest.raises(ValueError, match="feature_types"):
        TableMetadata(["a"], ["text"])


def test_a_draw_is_a_pure_function_of_the_lanes_and_its_index():
    first = population().rows(200, 0)
    assert list(first.columns) == ["cause", "effect", "label"]
    assert first.equals(population().rows(200, 0))
    assert not first.equals(population().rows(200, 1))
    assert population().id == "case--s3" and population().features == ["cause", "effect"]


def test_ungrouped_rows_are_their_own_entities():
    assert population().entity_rows(5, 0)[1].tolist() == [0, 1, 2, 3, 4]


def test_a_grouped_draw_holds_the_fixed_nodes_per_entity():
    grouping = Grouping(mode="shared_causes", fixed=("cause",), rows=8.0, spread=0.5, measurement=0.0, column="id")  # fmt: skip
    metadata = TableMetadata(
        feature_names=["id", "cause", "effect"],
        feature_types=["categorical", "numeric", "numeric"],
        target="label",
        plan=plan(),
    )
    grouped = Population(metadata=metadata, lanes=(1, 2), render=None, grouping=grouping)
    frame, entities = grouped.entity_rows(500, 0)
    assert len(frame) == 500 and list(frame.columns)[0] == "id"
    assert np.all(np.diff(entities) >= 0) and entities[-1] > 10
    assert frame.groupby(entities)["cause"].nunique().max() == 1  # the entity's value
    assert frame.groupby(entities)["effect"].nunique().max() > 1  # the row's
    assert frame.groupby(entities)["id"].nunique().max() == 1
    assert frame.groupby("id").ngroups == entities[-1] + 1


def test_entity_sizes_sum_to_the_rows_asked_for():
    grouping = Grouping(mode="replicates", fixed=(), rows=6.0, spread=1.0, measurement=0.0, column=None)  # fmt: skip
    for n in (0, 1, 7, 500):
        assert grouping.sizes(n, np.random.default_rng(n)).sum() == n


def test_an_entity_cut_never_splits_an_entity():
    entities = np.array([0, 0, 1, 1, 1, 2])
    assert entity_cut(entities, 0) == 0
    assert entity_cut(entities, 1) == 2
    assert entity_cut(entities, 3) == 5
    assert entity_cut(entities, 6) == 6
    assert entity_cut(np.arange(4), 2) == 2
    assert entity_cut(np.array([], dtype=int), 3) == 0
