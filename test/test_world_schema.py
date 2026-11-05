from dataclasses import replace

import numpy as np
import pytest

from plurel.world import SHAPE_KINDS, Edge, Node, WorldSchema, shape

CASE = WorldSchema(
    schema_id="ledger_case",
    target="outcome",
    description="Each row is one case.",
    nodes={
        "cause": Node(kind="numeric", units="score", dist="symmetric", scale=(0, 10), desc="a root cause"),
        "other": Node(kind="numeric", units="count", dist="counts", scale=(0, 50), desc="a second root"),
        "mediator": Node(
            kind="numeric", units="percent", scale=(0, 100), noise_scale="low", noise_dist="gaussian",
            parents={"cause": Edge(effect=+1), "other": Edge(effect=-0.5, shape="hinge")}, desc="a mediator",
        ),
        "aftermath": Node(
            kind="numeric", units="amount", scale=(0, 1000), noise_scale="medium", noise_dist="laplace",
            parents={"mediator": Edge(effect=+0.5)}, desc="a consequence of the mediator",
        ),
        "ticket": Node(kind="numeric", units="count", dist="counts", scale=(1000, 9999), desc="a bookkeeping id"),
        "outcome": Node(
            kind="categorical", levels=["no", "yes"], level_probs={"no": 0.7, "yes": 0.3},
            parents={"mediator": Edge(effect=+1), "cause": Edge(effect=-0.5)},
            interactions=[("mediator", "cause")], noise_scale="medium", noise_dist="gaussian", desc="the outcome",
        ),
    },
)  # fmt: skip


def with_nodes(**nodes) -> WorldSchema:
    return replace(CASE, nodes={**CASE.nodes, **nodes})


def with_outcome(**fields) -> WorldSchema:
    return with_nodes(outcome=replace(CASE.nodes["outcome"], **fields))


def test_a_schema_reads_its_graph():
    CASE.validate()
    assert CASE.feature_names() == ["cause", "other", "mediator", "aftermath", "ticket"]
    assert CASE.parent_names("outcome") == ["mediator", "cause"]
    assert CASE.children()["mediator"] == ["aftermath", "outcome"]
    assert CASE.mediators() == ["mediator"]
    assert CASE.isolated_roots() == ["ticket"]
    assert CASE.levels == {"outcome": ["no", "yes"]}
    order = CASE.topo_order()
    assert order.index("cause") < order.index("mediator") < order.index("outcome")


def test_a_parent_may_be_authored_by_its_shorthand():
    node = Node(kind="numeric", units="score", scale=(0, 1), noise_scale="low", noise_dist="gaussian",
                parents={"cause": -0.5, "kind": {"a": 0.2, "b": -0.2}}, desc="a score")  # fmt: skip
    schema = with_nodes(score=node, kind=Node(kind="categorical", levels=["a", "b"], desc="a kind"))
    edges = schema.edges("score")
    assert edges["cause"] == Edge(effect=-0.5)
    assert edges["kind"] == Edge(level_effects={"a": 0.2, "b": -0.2})


def test_a_placed_schema_is_named_by_its_title():
    assert CASE.domain == "ledger_case"
    assert replace(CASE, title="Higher education").domain == "higher education"
    assert replace(CASE, title="Higher education") == CASE  # placement is not identity


def test_a_cycle_and_an_unknown_parent_are_named():
    cycle = with_nodes(cause=replace(CASE.nodes["cause"], parents={"mediator": +1}, dist=None,
                                     noise_scale="low", noise_dist="gaussian"))  # fmt: skip
    with pytest.raises(ValueError, match="cycle"):
        cycle.validate()
    with pytest.raises(ValueError, match="unknown parent"):
        with_outcome(parents={"nobody": +1}).validate()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(cause=replace(CASE.nodes["cause"], dist="counts")), "from its units family"),
        (dict(cause=replace(CASE.nodes["cause"], scale=None)), "authored scale"),
        (dict(mediator=replace(CASE.nodes["mediator"], scale=(0, 200))), "inside \\[0, 100\\]"),
        (dict(cause=replace(CASE.nodes["cause"], rounding=20.0)), "quantum inside the scale"),
        (dict(cause=replace(CASE.nodes["cause"], zero_inflation=1.0)), "share in"),
        (dict(outcome=replace(CASE.nodes["outcome"], missing=0.1)), "never blank"),
        (dict(outcome=replace(CASE.nodes["outcome"], level_probs=None)), "write level_probs"),
        (dict(aftermath=replace(CASE.nodes["aftermath"], mechanism="min")), "two or more parents"),
        (dict(outcome=replace(CASE.nodes["outcome"], mechanism="rules")), "threshold edges only"),
        (dict(mediator=replace(CASE.nodes["mediator"], noise_scale=None)), "noise_scale"),
        (dict(cause=replace(CASE.nodes["cause"], noise_scale="low")), "no structural"),
    ],
)  # fmt: skip
def test_validate_names_the_rule_a_schema_breaks(change, message):
    with pytest.raises(ValueError, match=message):
        with_nodes(**change).validate()


def test_a_hidden_cause_feeds_two_nodes_and_is_never_rendered():
    hidden = Node(kind="numeric", hidden=True, desc="unrecorded frailty")
    nodes = dict(
        frailty=hidden,
        mediator=replace(CASE.nodes["mediator"], parents={**CASE.nodes["mediator"].parents, "frailty": +1}),
        outcome=replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "frailty": +1}),
    )  # fmt: skip
    schema = with_nodes(**nodes)
    schema.validate()
    assert "frailty" not in schema.feature_names() and "frailty" not in schema.column_descriptions
    with pytest.raises(ValueError, match="never rendered"):
        with_nodes(**{**nodes, "frailty": replace(hidden, scale=(0, 1))}).validate()
    with pytest.raises(ValueError, match="two or more nodes"):
        with_nodes(frailty=hidden, mediator=nodes["mediator"]).validate()


def test_a_derived_column_is_a_leaf_over_rendered_numbers():
    ratio = Node(
        kind="numeric", formula=lambda cause, other: cause / (other + 1.0), desc="their ratio"
    )
    schema = with_nodes(ratio=ratio)
    schema.validate()
    assert schema.parent_names("ratio") == ["cause", "other"]
    with pytest.raises(ValueError, match="no children"):
        with_nodes(ratio=ratio, ticket=replace(CASE.nodes["ticket"], parents={"ratio": +1}, dist=None,
                                               noise_scale="low", noise_dist="gaussian")).validate()  # fmt: skip


def test_an_argmax_target_derives_its_levels_from_its_classes():
    scores = {"keep": [("mediator", +1)], "sell": [("cause", +1)], "scrap": [("other", -1)]}
    argmax = with_outcome(
        parents={}, levels=None, class_scores=scores, interactions=[], noise_scale=None, noise_dist=None,
        level_probs={"keep": 0.6, "sell": 0.3, "scrap": 0.1},
    )  # fmt: skip
    argmax.validate()
    assert argmax.levels["outcome"] == ["keep", "sell", "scrap"]
    with pytest.raises(ValueError, match="3\\+ classes"):
        replace(
            argmax,
            nodes={**argmax.nodes, "outcome": replace(argmax.nodes["outcome"], class_scores={"keep": [("cause", 1)]},
                                                      level_probs={"keep": 1.0})},
        ).validate()  # fmt: skip


def test_a_level_regime_needs_a_scalar_parent_to_scale():
    site = Node(kind="categorical", levels=["a", "b"], desc="the site")
    regime = Edge(
        level_effects={"a": -0.2, "b": 0.2}, level_slopes={"a": 0.0, "b": 3.0}, scaled="mediator"
    )
    parents = {**CASE.nodes["outcome"].parents, "site": regime}
    with_nodes(site=site, outcome=replace(CASE.nodes["outcome"], parents=parents)).validate()
    unscaled = {**parents, "site": replace(regime, scaled=None)}
    with pytest.raises(ValueError, match="scaled must name"):
        with_nodes(site=site, outcome=replace(CASE.nodes["outcome"], parents=unscaled)).validate()


@pytest.mark.parametrize("kind", SHAPE_KINDS)
def test_every_shape_class_draws_a_finite_marginal(kind):
    rng = np.random.default_rng(0)
    assert np.isfinite(shape(kind, rng).sample(1000, rng)).all()


def test_shape_classes_keep_their_promises():
    rng = np.random.default_rng(1)
    counts = shape("counts", rng).sample(1000, rng)
    assert np.all(counts >= 0) and np.all(np.mod(counts, 1) == 0)
    inflated = shape("zero_inflated", rng).sample(1000, rng)
    assert (inflated == 0).mean() > 0.1 and (inflated > 0).any()
    skew = lambda x: float(np.mean((x - x.mean()) ** 3)) / float(x.std() ** 3)  # noqa: E731
    assert skew(shape("right", rng).sample(5000, rng)) > 0
    assert skew(shape("left", rng).sample(5000, rng)) < 0
    with pytest.raises(ValueError, match="unknown shape"):
        shape("flat", rng)
