import hashlib
from dataclasses import replace

import numpy as np
import pytest

from plurel import (
    HIGH,
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
from plurel.columns import Column
from plurel.population import entity_cut, table_id
from plurel.world import joint_leaves, realize_world
from plurel.world.realize import ENTITY_EFFECT
from plurel.world.record import features

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


def with_entries(table=CASE, name=None, **entries):
    return Table({**table.declared, **entries}, **{**table.metadata, "name": name or table.name})


def caused(name: str, table=CASE, **causes):
    entry = table.declared[name]
    return replace(entry, causes={**entry.causes, **causes})


# with a recorded category among the outcome's causes and a loose categorical consequence of the mediator
GROUPED = with_entries(
    name="grouped_case",
    outcome=caused("outcome", site={"north": -0.5, "south": 0.5}),
    site=Categorical(Shares(["north", "south"]), doc="the site"),
    ward=Categorical(Shares(["day", "night"]), causes={"mediator": WEAK}, noise=Noise("gaussian", HIGH), doc="the ward"),
)  # fmt: skip
REPLICATES = dict(mode="replicates", rows=8.0, spread=0.5, measurement=0.2, column="subject_id", effect=False)  # fmt: skip


def test_a_record_realizes_to_a_plain_plan():
    plan = realize_world(CASE, 0).plan
    assert isinstance(plan, Table) and plan.target == "outcome" and not plan.family
    assert all(isinstance(column, Column) for column in plan.columns.values())
    assert set(plan.columns) == set(features(CASE)) | {"outcome"}
    assert "outcome.mediator*cause" in plan.nodes and "outcome.mediator*cause" not in plan.columns


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
    assert digest == "98d18f7235e196db"
    grouped = realize_world(GROUPED, 3, grouping=REPLICATES).rows(64, 0)
    digest = hashlib.sha256(grouped.round(4).to_csv(index=False).encode()).hexdigest()[:16]
    assert digest == "f6ac3dbeaf4d611f"


def test_the_card_carries_the_authored_text():
    metadata = realize_world(GROUPED, 0).metadata
    assert metadata.feature_names == features(GROUPED) and metadata.target == "outcome"
    assert metadata.feature_types == ["numeric"] * 5 + ["categorical"] * 2
    assert metadata.description == "Each row is one case." and metadata.domain == "grouped_case"
    assert metadata.column_descriptions["ward"] == "the ward; categories: day, night"
    assert metadata.category_order["outcome"] == ["no", "yes"]


def test_target_noise_scales_only_the_targets_own_noise():
    authored, quiet = realize_world(CASE, 5), realize_world(CASE, 5, target_noise=0.05)
    a, q = authored.rows(400, 0), quiet.rows(400, 0)
    columns = [c for c in a.columns if c != "outcome"]
    assert a[columns].equals(q[columns]) and not a["outcome"].equals(q["outcome"])
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
    drawn = Shares(["x", "y", "z"]).draw(rng)
    assert set(drawn) == {"x", "y", "z"} and abs(sum(drawn.values()) - 1.0) < 1e-9
    varied = Shares({"a": 0.998, "b": 0.002}, 100.0).draw(rng)
    assert abs(sum(varied.values()) - 1.0) < 1e-9 and min(varied.values()) >= 0.03


def test_a_hidden_cause_is_simulated_and_never_recorded():
    schema = with_entries(
        frailty=Hidden(doc="unrecorded frailty"),
        mediator=caused("mediator", frailty=STRONG),
        outcome=caused("outcome", frailty=STRONG),
    )
    population = realize_world(schema, 0)
    assert "frailty" in population.plan.nodes and "frailty" not in population.plan.columns
    frame = population.rows(400, 0)
    assert "frailty" not in frame.columns and list(frame.columns)[-1] == "outcome"


@pytest.mark.parametrize(
    "fields",
    [
        dict(how="max"),
        dict(how="product"),
        dict(how="rules", causes={"mediator": (STRONG, "step"), "cause": (-WEAK, "step"), ("mediator", "cause"): Range(0.8, 2.5)}),
        dict(causes={"mediator": STRONG, "cause": (-WEAK, "square")}),
    ],
)  # fmt: skip
def test_every_mechanism_realizes_a_finite_table(fields):
    schema = with_entries(outcome=replace(CASE.declared["outcome"], **fields))
    frame = realize_world(schema, 0).rows(200, 0)
    assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
    assert frame["outcome"].astype(str).isin(["no", "yes"]).all()


def test_a_scored_target_lands_on_its_authored_shares():
    scores = {"keep": {"mediator": Range(1, 3)}, "sell": {"cause": Range(1, 3)}, "scrap": {"other": -Range(1, 3)}}  # fmt: skip
    schema = with_entries(outcome=Categorical(
        {"keep": 0.6, "sell": 0.3, "scrap": 0.1}, scores=scores, noise=Noise("gumbel", Range(0.6, 1.8)), doc="the outcome",
    ))  # fmt: skip
    shares = realize_world(schema, 0).rows(4000, 0)["outcome"].value_counts(normalize=True)
    assert abs(shares["keep"] - 0.6) < 0.05 and abs(shares["scrap"] - 0.1) < 0.03


def test_recording_artifacts_shape_the_recorded_columns():
    schema = with_entries(
        cause=replace(CASE.declared["cause"], zeros=0.3),
        other=replace(CASE.declared["other"], clip=(None, 30.0)),
        mediator=replace(CASE.declared["mediator"], round=5.0),
    )
    # seed-pinned: the cap's pile is a property of the drawn counts family (about half of seeds)
    population = realize_world(schema, 1)
    frame = population.rows(500, 0)
    assert 0.25 < (frame["cause"] == 0).mean() < 0.4
    assert frame["other"].max() <= 30 and (frame["other"] == 30).mean() > 0.05
    assert np.all(np.mod(frame["mediator"], 5) == 0)
    assert realize_world(schema, 1).rows(500, 0).equals(frame)


def test_blanks_land_at_their_share_and_where_authored():
    schema = with_entries(
        other=replace(CASE.declared["other"], missing=0.2),
        mediator=replace(CASE.declared["mediator"], missing=0.3, missing_by=("mediator", "high")),
    )
    frame = realize_world(schema, 1).rows(600, 0)
    assert (
        0.15 < frame["other"].isna().mean() < 0.25 and 0.2 < frame["mediator"].isna().mean() < 0.4
    )
    assert frame["mediator"].mean() < realize_world(CASE, 1).rows(600, 0)["mediator"].mean()
    kind = Categorical(Shares(["a", "b"]), missing=0.3, doc="a coded field")
    coded = with_entries(kind=kind, outcome=caused("outcome", kind={"a": -0.3, "b": 0.3}))
    assert 0.2 < realize_world(coded, 0).rows(600, 0)["kind"].isna().mean() < 0.4


def test_a_derived_column_is_exact_arithmetic_on_the_recorded_values():
    double = Derived(lambda cause, other: cause + 2.0 * other, doc="their total")
    frame = realize_world(with_entries(double=double), 0).rows(100, 0)
    assert np.allclose(frame["double"], frame["cause"] + 2.0 * frame["other"], atol=1e-3)


def test_a_level_regime_changes_a_parents_slope_by_group():
    site = Categorical({"a": 0.5, "b": 0.5}, doc="the site")
    outcome = CASE.declared["outcome"]
    regime = {"site": {"a": -0.2, "b": 0.2}, ("site", "mediator"): {"a": 0.0, "b": 3.0}}
    schema = with_entries(site=site, outcome=replace(outcome, causes={**outcome.causes, **regime}))
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
    isolated = with_entries(GROUPED, ward=Categorical(Shares(["day", "night"]), doc="the ward"))
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
    blanked = with_entries(GROUPED, aftermath=replace(GROUPED.declared["aftermath"], missing=0.3))
    single, joint = realize_world(blanked, 2), realize_world(blanked, 2, joint="aftermath")
    assert joint.metadata.category_order[joint.target] == ["no / low", "no / high", "yes / low", "yes / high"]  # fmt: skip
    a, b = single.rows(400, 0), joint.rows(400, 0)
    assert a[joint.features].equals(b[joint.features]) and b[joint.target].notna().all()
    band = b[joint.target].astype(str).str.split(" / ").str[1]
    assert a["aftermath"][band == "low"].max() < a["aftermath"][band == "high"].min()


def test_a_numeric_band_leaves_the_features_draws_whatever_the_leafs_place():
    early = Numeric(Scaled("duration", 0, 60), causes={"cause": STRONG}, noise=Noise("gaussian", LOW), doc="a leaf drawn early")  # fmt: skip
    schema = with_entries(CASE, "early_case", early=early)
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
    rounded = Numeric(Scaled("score", 0.4, 10, shape="symmetric"), round=1, doc="a root cause")
    frame = realize_world(with_entries(CASE, cause=rounded), 0).rows(500, 0)
    assert frame["cause"].min() >= 0.4 - 1e-6


def test_a_category_with_a_lookup_child_keeps_its_shares_and_mechanism():
    phase = Categorical({"a": 0.1, "b": 0.9}, causes={"cause": STRONG}, noise=Noise("gaussian", LOW), doc="a phase")  # fmt: skip
    after = Numeric(Scaled("score", 0, 10), causes={"phase": (Range(0.7, 1.6), {"a": -1.0, "b": 1.0})},
                    noise=Noise("gaussian", LOW), doc="what follows")  # fmt: skip
    schema = with_entries(CASE, "phased", phase=phase, after=after)
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
    span = CASE.declared["cause"].marginal.high - CASE.declared["cause"].marginal.low
    assert min(spreads) > 0.08 * span


def test_a_joint_part_too_rare_or_given_away_stays_out():
    rare = with_entries(GROUPED, "rare_case", ward=replace(GROUPED.declared["ward"], levels={"day": 0.99, "night": 0.01}))  # fmt: skip
    echo = Numeric(Scaled("percent", 0, 100), causes={"mediator": STRONG}, noise=Noise("gaussian", LOW), doc="an echo of the mediator")  # fmt: skip
    for schema, leaf in ((rare, "ward"), (with_entries(GROUPED, "echo_case", echo=echo), "echo")):
        population = realize_world(schema, 0, joint=leaf)
        assert population.target == "outcome" and leaf in population.features
        assert "joint" not in population.metadata.realization
