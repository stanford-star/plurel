import hashlib
from dataclasses import replace

import numpy as np
import pytest

from plurel.columns import Column
from plurel.population import entity_cut, table_id
from plurel.schema import SCM
from plurel.world import Edge, Node, WorldSchema, compile_schema, joint_leaves, realize_world
from plurel.world.compiler import ENTITY_EFFECT, sample_bin_probs, vary_shares

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
# with a recorded category among the outcome's causes and a loose categorical consequence of the mediator
GROUPED = replace(CASE, schema_id="grouped_case", nodes={
    **CASE.nodes,
    "outcome": replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "site": {"north": -0.5, "south": 0.5}}),
    "site": Node(kind="categorical", levels=["north", "south"], desc="the site"),
    "ward": Node(kind="categorical", levels=["day", "night"], parents={"mediator": Edge(effect=+0.5)},
                 noise_scale="high", noise_dist="gaussian", desc="the ward"),
})  # fmt: skip
REPLICATES = dict(mode="replicates", rows=8.0, spread=0.5, measurement=0.2, column="subject_id", effect=False)  # fmt: skip


def with_nodes(schema=CASE, schema_id=None, **nodes):
    return replace(schema, schema_id=schema_id or schema.schema_id, nodes={**schema.nodes, **nodes})


def test_a_schema_compiles_to_a_plain_plan():
    plan, target = compile_schema(CASE, np.random.default_rng(0))
    assert isinstance(plan, SCM) and target == "outcome"
    assert all(isinstance(column, Column) for column in plan.columns.values())
    assert set(plan.columns) == set(CASE.feature_names()) | {"outcome"}
    assert "outcome.mediator_cause" in plan.nodes and "outcome.mediator_cause" not in plan.columns


def test_a_table_is_a_pure_function_of_schema_seed_and_draw():
    population = realize_world(CASE, 5)
    frame = population.rows(150, 0)
    stamp = population.metadata.realization
    assert (stamp["schema_id"], stamp["seed"], stamp["table_id"]) == ("ledger_case", 5, table_id("ledger_case", 5))  # fmt: skip
    again = realize_world(CASE, 5)
    assert again.rows(150, 0).equals(frame) and again.metadata.realization == stamp
    assert not population.rows(150, 1).equals(frame)
    assert not realize_world(CASE, 6).rows(150, 0).equals(frame)


def test_the_draws_are_pinned():
    """A change that moves a realization's rows must say so here."""
    frame = realize_world(CASE, 3, target_noise=0.25, share_concentration=100.0).rows(200, 0)
    digest = hashlib.sha256(frame.round(4).to_csv(index=False).encode()).hexdigest()[:16]
    assert digest == "0296cf6feb889282"
    grouped = realize_world(GROUPED, 3, grouping=REPLICATES).rows(64, 0)
    digest = hashlib.sha256(grouped.round(4).to_csv(index=False).encode()).hexdigest()[:16]
    assert digest == "0034260576dd1a34"


def test_the_card_carries_the_authored_text():
    metadata = realize_world(GROUPED, 0).metadata
    assert metadata.feature_names == GROUPED.feature_names() and metadata.target == "outcome"
    assert metadata.feature_types == [GROUPED.nodes[c].kind for c in GROUPED.feature_names()]
    assert metadata.description == "Each row is one case." and metadata.domain == "grouped_case"
    assert metadata.column_descriptions["ward"] == "the ward; categories: day, night"
    assert metadata.category_order["outcome"] == ["no", "yes"]


def test_target_noise_scales_only_the_targets_own_noise():
    authored, quiet = realize_world(CASE, 5), realize_world(CASE, 5, target_noise=0.05)
    a, q = authored.rows(400, 0), quiet.rows(400, 0)
    features = [c for c in a.columns if c != "outcome"]
    assert a[features].equals(q[features]) and not a["outcome"].equals(q["outcome"])
    assert quiet.metadata.realization["target_noise"] == 0.05


def test_level_shares_vary_around_the_authored_value_and_bin_at_fixed_edges():
    def yes(frame):
        return (frame["outcome"].astype(str) == "yes").mean()

    drawn = [yes(realize_world(CASE, seed, share_concentration=100.0).rows(3000, 0)) for seed in range(6)]  # fmt: skip
    assert np.std(drawn) > 0.01 and abs(np.mean(drawn) - 0.3) < 0.1
    population = realize_world(CASE, 3)
    assert yes(population.rows(200, 0)) != yes(population.rows(200, 1))
    assert population.metadata.realization["share_concentration"] is None
    categorical = [c for c in population.plan.columns.values() if c.kind == "categorical"]
    assert categorical and all(c.binning is not None for c in categorical)


def test_shares_are_drawn_and_floored():
    rng = np.random.default_rng(0)
    drawn = sample_bin_probs({"a": ["x", "y", "z"], "one": ["only"]}, rng, concentration=(0.2, 5.0))
    assert set(drawn) == {"a"} and abs(sum(drawn["a"]) - 1.0) < 1e-9
    varied = vary_shares({"a": [0.998, 0.002]}, rng, concentration=100.0)
    assert abs(sum(varied["a"]) - 1.0) < 1e-9 and min(varied["a"]) >= 0.03


def test_a_hidden_cause_is_simulated_and_never_rendered():
    frailty = Node(kind="numeric", hidden=True, desc="unrecorded frailty")
    schema = with_nodes(
        frailty=frailty,
        mediator=replace(CASE.nodes["mediator"], parents={**CASE.nodes["mediator"].parents, "frailty": +1}),
        outcome=replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "frailty": +1}),
    )  # fmt: skip
    population = realize_world(schema, 0)
    assert "frailty" in population.plan.nodes and "frailty" not in population.plan.columns
    frame = population.rows(400, 0)
    assert "frailty" not in frame.columns and list(frame.columns)[-1] == "outcome"


@pytest.mark.parametrize(
    "fields",
    [
        dict(mechanism="max"),
        dict(mechanism="product"),
        dict(mechanism="rules", parents={"mediator": Edge(effect=+1, shape="threshold"), "cause": Edge(effect=-0.5, shape="threshold")}),
        dict(parents={"mediator": Edge(effect=+1), "cause": Edge(effect=-0.5, shape="u")}),
    ],
)  # fmt: skip
def test_every_mechanism_realizes_a_finite_table(fields):
    schema = with_nodes(outcome=replace(CASE.nodes["outcome"], **fields))
    frame = realize_world(schema, 0).rows(200, 0)
    assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
    assert frame["outcome"].astype(str).isin(["no", "yes"]).all()


def test_an_argmax_target_lands_on_its_authored_shares():
    scores = {"keep": [("mediator", +1)], "sell": [("cause", +1)], "scrap": [("other", -1)]}
    schema = with_nodes(outcome=replace(
        CASE.nodes["outcome"], parents={}, levels=None, class_scores=scores, interactions=[],
        level_probs={"keep": 0.6, "sell": 0.3, "scrap": 0.1}, noise_scale=None, noise_dist=None,
    ))  # fmt: skip
    shares = realize_world(schema, 0).rows(4000, 0)["outcome"].value_counts(normalize=True)
    assert abs(shares["keep"] - 0.6) < 0.05 and abs(shares["scrap"] - 0.1) < 0.03


def test_recording_artifacts_shape_the_rendered_columns():
    schema = with_nodes(
        cause=replace(CASE.nodes["cause"], zero_inflation=0.3),
        other=replace(CASE.nodes["other"], censor=(None, 30.0)),
        mediator=replace(CASE.nodes["mediator"], rounding=5.0),
    )
    # seed-pinned: the cap's pile is a property of the drawn counts family (about half of seeds)
    population = realize_world(schema, 1)
    frame = population.rows(500, 0)
    assert 0.25 < (frame["cause"] == 0).mean() < 0.4
    assert frame["other"].max() <= 30 and (frame["other"] == 30).mean() > 0.05
    assert np.all(np.mod(frame["mediator"], 5) == 0)
    assert realize_world(schema, 1).rows(500, 0).equals(frame)


def test_blanks_land_at_their_share_and_where_authored():
    schema = with_nodes(
        other=replace(CASE.nodes["other"], missing=0.2),
        mediator=replace(CASE.nodes["mediator"], missing=0.3, missing_by=("mediator", "high")),
    )
    frame = realize_world(schema, 1).rows(600, 0)
    assert (
        0.15 < frame["other"].isna().mean() < 0.25 and 0.2 < frame["mediator"].isna().mean() < 0.4
    )
    assert frame["mediator"].mean() < realize_world(CASE, 1).rows(600, 0)["mediator"].mean()
    kind = Node(kind="categorical", levels=["a", "b"], missing=0.3, desc="a coded field")
    coded = with_nodes(kind=kind, outcome=replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "kind": {"a": -0.3, "b": 0.3}}))  # fmt: skip
    assert 0.2 < realize_world(coded, 0).rows(600, 0)["kind"].isna().mean() < 0.4


def test_a_derived_column_is_exact_arithmetic_on_the_rendered_values():
    double = Node(
        kind="numeric", formula=lambda cause, other: cause + 2.0 * other, desc="their total"
    )
    frame = realize_world(with_nodes(double=double), 0).rows(100, 0)
    assert np.allclose(frame["double"], frame["cause"] + 2.0 * frame["other"], atol=1e-3)


def test_a_level_regime_changes_a_parents_slope_by_group():
    site = Node(kind="categorical", levels=["a", "b"], level_probs={"a": 0.5, "b": 0.5}, desc="the site")  # fmt: skip
    regime = Edge(level_effects={"a": -0.2, "b": 0.2}, level_slopes={"a": 0.0, "b": 3.0}, scaled="mediator")  # fmt: skip
    schema = with_nodes(site=site, outcome=replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "site": regime}))  # fmt: skip
    frame = realize_world(schema, 0).rows(3000, 0)
    yes = (frame["outcome"] == "yes").astype(float)

    def rank_corr(level):
        rows = frame["site"] == level
        return np.corrcoef(frame["mediator"][rows].rank(), yes[rows].rank())[0, 1]

    assert rank_corr("b") > rank_corr("a") + 0.1


def test_replicate_rows_share_their_entity_class_and_vary_in_their_readings():
    population = realize_world(GROUPED, 0, grouping=REPLICATES)
    assert {"outcome", "mediator", "cause", "other", "site"} <= set(population.grouping.fixed)
    assert not {"aftermath", "ticket", "ward"} & set(population.grouping.fixed)
    frame, entities = population.entity_rows(800, 0)
    by = frame.groupby(entities)
    assert (by["outcome"].nunique() == 1).all() and (by["site"].nunique() == 1).all()
    for reading in ("cause", "mediator"):
        assert 0 < by[reading].std().mean() < 0.5 * by[reading].mean().std()
    assert (by["ticket"].nunique() > 1).mean() > 0.8
    exact, entities = realize_world(GROUPED, 0, grouping={**REPLICATES, "measurement": 0.0}).entity_rows(800, 0)  # fmt: skip
    assert (exact.groupby(entities)["cause"].nunique() == 1).all()


def test_shared_cause_rows_hold_their_entity_roots_and_redraw_the_rest():
    shared = dict(REPLICATES, mode="shared_causes", measurement=0.0, effect=True)
    seed = next(s for s in range(50) if {"site"} == {"site", "cause"} & set(realize_world(GROUPED, s, grouping=shared).grouping.fixed))  # fmt: skip
    population = realize_world(GROUPED, seed, grouping=shared)
    assert ENTITY_EFFECT in population.grouping.fixed
    assert "outcome" in population.metadata.realization["grouping"]["effect_on"]
    frame, entities = population.entity_rows(800, 0)
    by = frame.groupby(entities)
    assert (by["site"].nunique() == 1).all()
    assert (by["cause"].nunique() > 1).mean() > 0.8 and (by["outcome"].nunique() > 1).mean() > 0.2


def test_a_grouped_draw_holds_the_rows_asked_for_under_unique_ids():
    population = realize_world(GROUPED, 0, grouping=dict(REPLICATES, spread=1.0))
    assert population.grouping.sizes(1, np.random.default_rng(0)).tolist() == [1]
    for n in (2, 7, 3000):
        frame, entities = population.entity_rows(n, 0)
        assert len(frame) == len(entities) == n and frame.columns[0] == "subject_id"
        assert entities[0] == 0 and set(np.diff(entities)) <= {0, 1}
        ids = frame["subject_id"].groupby(entities).agg(["nunique", "first"])
        assert (ids["nunique"] == 1).all() and ids["first"].is_unique
        assert ids["first"].str.fullmatch(r"[A-Z]\d{5}").all()
    sizes = np.bincount(entities)
    assert sizes.max() > 4 * np.median(sizes)
    cut = entity_cut(entities, 100)
    assert cut >= 100 and not set(entities[:100]) & set(entities[cut:])
    unnamed = realize_world(GROUPED, 0, grouping=dict(REPLICATES, column=None))
    assert "subject_id" not in unnamed.features
    assert list(unnamed.rows(20, 0).columns) == [*unnamed.features, "outcome"]


def test_a_grouped_tables_description_says_how_its_rows_share_units():
    plain = realize_world(GROUPED, 0)
    assert "text" not in plain.metadata.realization
    for grouping in (REPLICATES, dict(REPLICATES, mode="shared_causes", measurement=0.0, column=None)):  # fmt: skip
        metadata = realize_world(GROUPED, 0, grouping=grouping).metadata
        text = metadata.realization["text"]
        assert metadata.description == f"Each row is one case. {text}"
        assert ("subject_id" in text) == (grouping["column"] is not None)
    assert "unit" in realize_world(GROUPED, 0, grouping=REPLICATES).metadata.column_descriptions["subject_id"]  # fmt: skip


def test_a_joint_target_joins_the_target_and_a_childless_leaf():
    assert joint_leaves(GROUPED) == ["aftermath", "ward"] and joint_leaves(CASE) == ["aftermath"]
    isolated = with_nodes(GROUPED, ward=replace(GROUPED.nodes["ward"], parents={}, noise_scale=None, noise_dist=None))  # fmt: skip
    assert joint_leaves(isolated) == ["aftermath"]
    single, joint = realize_world(GROUPED, 2), realize_world(GROUPED, 2, joint="ward")
    assert joint.target == "outcome_and_ward" and "ward" not in joint.features
    assert joint.metadata.category_order[joint.target] == ["no / day", "no / night", "yes / day", "yes / night"]  # fmt: skip
    a, b = single.rows(300, 0), joint.rows(300, 0)
    assert list(b.columns) == [*joint.features, joint.target]
    assert a[joint.features].equals(b[joint.features])
    assert (b[joint.target].astype(str) == a["outcome"].astype(str) + " / " + a["ward"].astype(str)).all()  # fmt: skip
    stamp = joint.metadata.realization["joint"]
    assert (stamp["part"], stamp["banded"]) == ("ward", False)
    notes = joint.metadata.column_descriptions
    assert not {"outcome", "ward"} & set(notes) and "the ward" in notes[joint.target]
    assert joint.metadata.description == f"{single.metadata.description} {joint.metadata.realization['text']}"  # fmt: skip


def test_a_numeric_leaf_joins_the_target_by_its_median_band():
    blanked = with_nodes(GROUPED, aftermath=replace(GROUPED.nodes["aftermath"], missing=0.3))
    single, joint = realize_world(blanked, 2), realize_world(blanked, 2, joint="aftermath")
    assert joint.metadata.category_order[joint.target] == ["no / low", "no / high", "yes / low", "yes / high"]  # fmt: skip
    a, b = single.rows(400, 0), joint.rows(400, 0)
    assert a[joint.features].equals(b[joint.features]) and b[joint.target].notna().all()
    band = b[joint.target].astype(str).str.split(" / ").str[1]
    assert a["aftermath"][band == "low"].max() < a["aftermath"][band == "high"].min()


def test_a_numeric_band_leaves_the_features_draws_whatever_the_leafs_place():
    early = Node(kind="numeric", units="duration", scale=(0, 60), parents={"cause": Edge(effect=+1)},
                 noise_scale="low", noise_dist="gaussian", desc="a leaf drawn early")  # fmt: skip
    schema = with_nodes(CASE, "early_case", early=early)
    kept = 0
    for seed in range(20):
        joint = realize_world(schema, seed, joint="early")
        if "joint" in joint.metadata.realization:
            kept += 1
            a, b = realize_world(schema, seed).rows(200, 0), joint.rows(200, 0)
            assert a[joint.features].equals(b[joint.features])
    assert kept


def test_a_single_row_has_no_scale_to_render_on():
    population = realize_world(CASE, 0)
    with pytest.raises(ValueError, match="at least two"):
        population.rows(1, 0)
    assert len(population.rows(2, 0)) == 2


def test_rounding_stays_inside_the_scale():
    rounded = replace(CASE.nodes["cause"], scale=(0.4, 10), rounding=1)
    frame = realize_world(with_nodes(CASE, cause=rounded), 0).rows(500, 0)
    assert frame["cause"].min() >= 0.4 - 1e-6


def test_a_category_with_a_lookup_child_keeps_its_shares_and_mechanism():
    phase = Node(kind="categorical", levels=["a", "b"], level_probs={"a": 0.1, "b": 0.9},
                 parents={"cause": Edge(effect=+1)}, noise_scale="low", noise_dist="gaussian", desc="a phase")  # fmt: skip
    after = Node(kind="numeric", units="score", scale=(0, 10), parents={"phase": Edge(level_effects={"a": -1.0, "b": 1.0})},
                 noise_scale="low", noise_dist="gaussian", desc="what follows")  # fmt: skip
    schema = with_nodes(CASE, "phased", phase=phase, after=after)
    for seed in range(5):
        frame = realize_world(schema, seed).rows(4000, 0)
        assert abs((frame["phase"] == "a").mean() - 0.1) < 0.03
        by = frame.groupby("phase", observed=True)["after"].mean()
        assert by["a"] < by["b"]


def test_a_joint_target_must_be_a_joint_leaf():
    for name in ("nope", "cause"):
        with pytest.raises(ValueError, match="joint leaves"):
            realize_world(CASE, 0, joint=name)


def test_one_outlier_does_not_squeeze_a_scale():
    spreads = [realize_world(CASE, seed).rows(3000, 0)["cause"].std() for seed in range(20)]
    span = CASE.nodes["cause"].scale[1] - CASE.nodes["cause"].scale[0]
    assert min(spreads) > 0.08 * span


def test_a_joint_part_too_rare_or_given_away_stays_out():
    rare = with_nodes(GROUPED, "rare_case", ward=replace(GROUPED.nodes["ward"], level_probs={"day": 0.99, "night": 0.01}))  # fmt: skip
    echo = Node(kind="numeric", units="percent", scale=(0, 100), noise_scale="low", noise_dist="gaussian",
                parents={"mediator": Edge(effect=+1)}, desc="an echo of the mediator")  # fmt: skip
    for schema, leaf in ((rare, "ward"), (with_nodes(GROUPED, "echo_case", echo=echo), "echo")):
        population = realize_world(schema, 0, joint=leaf)
        assert population.target == "outcome" and leaf in population.features
        assert "joint" not in population.metadata.realization
