from dataclasses import replace

import numpy as np
import pytest

from plurel import (
    LOW,
    MEDIUM,
    STRONG,
    WEAK,
    Categorical,
    Derived,
    Hidden,
    Noise,
    Numeric,
    Range,
    Scaled,
    Shares,
    Table,
)
from plurel.distributions import SHAPE_KINDS, shape
from plurel.world import validate
from plurel.world.record import ancestors, children, domain, features, level_names, parents

CASE = Table(
    {
        "cause": Numeric(Scaled("score", 0, 10, shape="symmetric"), doc="a root cause"),
        "other": Numeric(Scaled("count", 0, 50, shape="counts"), doc="a second root"),
        "mediator": Numeric(
            Scaled("percent", 0, 100), causes={"cause": STRONG, "other": (-WEAK, "relu")},
            noise=Noise("gaussian", LOW), doc="a mediator",
        ),
        "aftermath": Numeric(
            Scaled("amount", 0, 1000), causes={"mediator": WEAK},
            noise=Noise("laplace", MEDIUM), doc="a consequence of the mediator",
        ),
        "ticket": Numeric(Scaled("count", 1000, 9999, shape="counts"), doc="a bookkeeping id"),
        "outcome": Categorical(
            {"no": 0.7, "yes": 0.3},
            causes={"mediator": STRONG, "cause": -WEAK, ("mediator", "cause"): Range(0.8, 2.5)},
            noise=Noise("gaussian", MEDIUM), doc="the outcome",
        ),
    },
    name="ledger_case",
    target="outcome",
    doc="Each row is one case.",
)  # fmt: skip
SCORES = {
    "keep": {"mediator": Range(1, 3)},
    "sell": {"cause": Range(1, 3)},
    "scrap": {"other": -Range(1, 3)},
}
FRAILTY = Hidden(doc="unrecorded frailty")


def with_entries(**entries) -> Table:
    return Table({**CASE.declared, **entries}, **CASE.metadata)


def changed(name: str, **fields) -> Table:
    return with_entries(**{name: replace(CASE.declared[name], **fields)})


def caused(name: str, **causes):
    entry = CASE.declared[name]
    return replace(entry, causes={**entry.causes, **causes})


def test_a_table_reads_its_graph():
    validate(CASE)
    assert features(CASE) == ["cause", "other", "mediator", "aftermath", "ticket"]
    assert parents(CASE, "outcome") == ["mediator", "cause"]
    assert children(CASE)["mediator"] == ["aftermath", "outcome"]
    assert ancestors(CASE, "outcome") == {"mediator", "cause", "other"}
    assert not children(CASE)["ticket"] and not parents(CASE, "ticket")
    assert level_names(CASE.declared["outcome"]) == ["no", "yes"]
    assert level_names(Categorical(Shares(["a", "b"]))) == ["a", "b"]
    assert CASE.order.index("cause") < CASE.order.index("mediator") < CASE.order.index("outcome")


def test_a_placed_table_is_named_by_its_title():
    assert domain(CASE) == "ledger_case"
    placed = with_entries()
    placed.title = "Higher education"
    assert domain(placed) == "higher education"


def test_a_cycle_and_an_unknown_parent_are_named():
    with pytest.raises(ValueError, match="acyclic"):
        with_entries(cause=Numeric(
            Scaled("score", 0, 10), causes={"mediator": STRONG}, noise=Noise("gaussian", LOW), doc="a cause",
        ))  # fmt: skip
    with pytest.raises(ValueError, match="unknown parents"):
        changed("outcome", causes={"nobody": STRONG})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda: changed("cause", marginal=Scaled("score", 0, 10, shape="counts")), "shape must be one of"),
        (lambda: changed("cause", marginal=None), "names its units and range"),
        (lambda: changed("mediator", marginal=Scaled("percent", 0, 200)), "inside \\[0, 100\\]"),
        (lambda: changed("cause", round=20.0), "round"),
        (lambda: changed("cause", zeros=1.0), "a share"),
        (lambda: changed("outcome", missing=0.1), "never blank"),
        (lambda: changed("outcome", levels=["no", "yes"]), "authors its shares"),
        (lambda: changed("aftermath", how="min"), "two or more causes"),
        (lambda: changed("outcome", how="rules"), "rules read thresholds"),
        (lambda: changed("mediator", noise=None), "authors its noise"),
        (lambda: changed("cause", noise=Noise("gaussian", LOW)), "this one has none"),
    ],
)  # fmt: skip
def test_a_record_names_the_rule_it_breaks(change, message):
    with pytest.raises(ValueError, match=message):
        validate(change())


def test_a_hidden_cause_feeds_two_columns_and_is_never_recorded():
    both = dict(
        mediator=caused("mediator", frailty=STRONG), outcome=caused("outcome", frailty=STRONG)
    )
    schema = with_entries(frailty=FRAILTY, **both)
    validate(schema)
    assert "frailty" not in features(schema)
    with pytest.raises(ValueError, match="two or more columns"):
        validate(with_entries(frailty=FRAILTY, mediator=both["mediator"]))


def test_a_derived_column_is_a_leaf_over_recorded_numbers():
    ratio = Derived(lambda cause, other: cause / (other + 1.0), doc="their ratio")
    schema = with_entries(ratio=ratio)
    validate(schema)
    assert parents(schema, "ratio") == ["cause", "other"]
    with pytest.raises(ValueError, match="unknown parents"):
        with_entries(ratio=ratio, ticket=Numeric(
            Scaled("count", 1000, 9999), causes={"ratio": STRONG}, noise=Noise("gaussian", LOW), doc="a bookkeeping id",
        ))  # fmt: skip


def test_a_scored_target_takes_its_levels_from_its_shares():
    gumbel = Noise("gumbel", Range(0.6, 1.8))
    argmax = with_entries(outcome=Categorical(
        {"keep": 0.6, "sell": 0.3, "scrap": 0.1}, scores=SCORES, noise=gumbel, doc="the outcome",
    ))  # fmt: skip
    validate(argmax)
    assert level_names(argmax.declared["outcome"]) == ["keep", "sell", "scrap"]
    assert parents(argmax, "outcome") == ["mediator", "cause", "other"]
    with pytest.raises(ValueError, match="three or more levels"):
        validate(with_entries(outcome=Categorical(
            {"keep": 0.6, "sell": 0.4}, scores={"keep": {"cause": Range(1, 3)}}, noise=gumbel, doc="the outcome",
        )))  # fmt: skip


def test_a_level_regime_needs_a_level_column_to_set_a_slope():
    site = Categorical(Shares(["a", "b"]), doc="the site")
    causes = CASE.declared["outcome"].causes

    def regime(pair: tuple[str, str]) -> Table:
        slopes = {"site": {"a": -0.2, "b": 0.2}, pair: {"a": 0.0, "b": 3.0}}
        return with_entries(site=site, outcome=replace(CASE.declared["outcome"], causes={**causes, **slopes}))  # fmt: skip

    validate(regime(("site", "mediator")))
    with pytest.raises(ValueError, match="effects per level"):
        regime(("mediator", "site"))


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
