import numpy as np
import pandas as pd
import pytest

from plurel import (
    HIGH,
    LOW,
    MEDIUM,
    STRONG,
    WEAK,
    ByLevel,
    Categorical,
    Derived,
    ForeignKey,
    Hidden,
    Noise,
    Normal,
    Numeric,
    PrimaryKey,
    Range,
    Real,
    Scaled,
    Schema,
    Shares,
    Summary,
    Table,
)
from plurel.population import Grouping, Population, TableMetadata


def spearman(a, b) -> float:
    return pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)), method="spearman")


def family() -> Table:
    return Table(
        {
            "cover": Hidden(),
            "age": Numeric(Scaled("duration", 0, 40, shape="right"), round=1),
            "kind": Categorical({"a": 0.7, "b": 0.3}),
            "wait": Numeric(
                Scaled("duration", 0, 120), causes={"cover": -STRONG}, noise=Noise("skewed", LOW)
            ),
            "cost": Numeric(
                Scaled("amount", 5, 500),
                causes={"age": WEAK, "kind": (Range(0.7, 1.6), {"b": 2.0})},
                noise=MEDIUM,
                missing=0.1,
            ),
            "total": Derived(lambda wait, cost: wait + cost),
            "outcome": Categorical(
                {"yes": 0.2, "no": 0.8},
                causes={"cost": (STRONG, "tanh"), "cover": WEAK},
                noise=Noise("laplace", HIGH),
            ),
        },
        target="outcome",
        doc="Each row is one visit.",
    )


def weight(table: Table, node: str, parent: str) -> float:
    return next(e.weight for e in table.nodes[node].edges if e.parent == parent)


def test_a_family_samples_only_through_a_seed():
    with pytest.raises(ValueError, match="realize"):
        family().sample(10, seed=0)


def test_a_seed_draws_magnitudes_within_their_ranges_and_repeats():
    tables = [family().realize(seed) for seed in range(6)]
    weights = [weight(t, "wait", "cover") for t in tables]
    assert len(set(weights)) == 6
    assert all(-3.0 <= w <= -1.6 for w in weights)
    frame = tables[0].sample(500, seed=1)
    assert frame.equals(family().realize(0).sample(500, seed=1))
    assert not frame.equals(tables[1].sample(500, seed=1))


def test_a_realized_table_holds_its_authored_shape():
    frame = family().realize(3).sample(8000, seed=0)
    assert list(frame.columns) == ["age", "kind", "wait", "cost", "total", "outcome"]
    assert (frame.outcome == "yes").mean() == pytest.approx(0.2, abs=0.03)
    assert frame.age.between(0, 40).all() and (frame.age % 1 == 0).all()
    assert frame.wait.between(0, 120).all()
    assert frame.cost.dropna().between(5, 500).all()
    assert frame.cost.isna().mean() == pytest.approx(0.1, abs=0.02)
    assert (frame.total.isna() == frame.cost.isna()).all()
    both = frame.dropna(subset="cost")
    assert np.allclose(both.total, both.wait + both.cost, atol=1e-5)


def test_adding_a_column_moves_no_other_draw():
    declared = family().declared
    grown = Table({**declared, "extra": Numeric(Scaled("score", 0, 1))}, target="outcome")
    assert weight(family().realize(5), "wait", "cover") == weight(grown.realize(5), "wait", "cover")


def test_realized_shares_follow_the_rows_and_small_draws_work():
    table = Table(
        {"x": Numeric(), "k": Categorical({"a": 0.5, "b": 0.5}, causes={"x": 1.0}, noise=0.2)}
    ).realize(0)
    assert table.sample(1, seed=0).shape == (1, 2)
    counts = {(table.sample(40, seed=s).k == "a").sum() for s in range(12)}
    assert len(counts) > 1


def test_a_stated_strength_means_what_it_says():
    def tied(strength: float) -> float:
        frame = Table(
            {"x": Numeric(Normal()), "y": Numeric(Normal(), causes={"x": strength}, noise=1.0)}
        ).sample(5000, seed=0)
        return spearman(frame.x, frame.y)

    assert tied(3.0) > tied(1.0) > tied(0.3) > 0.2


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Scaled("minutes", 0, 1), "units"),
        (lambda: Scaled("duration", 0, 1, shape="left"), "shape"),
        (lambda: Scaled("duration", 5, 5), "higher"),
        (lambda: Scaled("duration", np.nan, 5), "finite"),
        (lambda: Scaled("percent", 0, 120), "percent"),
        (lambda: Scaled("signed", 0, 5), "spans zero"),
        (lambda: Scaled("amount", -1, 5), "negative"),
        (lambda: Noise("cauchy", 1.0).draw(np.random.default_rng(0)), "unknown noise"),
        (lambda: Categorical({"a": 0.5, "b": 0.5}, causes={"x": 1.0}, scores={"a": {}}), "scores"),
        (lambda: Categorical(["a", "b"], scores={"a": {"x": 1.0}}), "shares"),
        (lambda: Table({"x": Numeric(Normal(), how="mean", causes={"y": 1.0})}), "how must"),
        (lambda: Table({"x": Numeric(Normal(), clip=(None, None))}), "clip"),
        (lambda: Table({"x": Numeric(Normal(), zeros=1.5)}), "zeros"),
        (lambda: Table({"x": Numeric(Normal(), missing=0.1, missing_by=("x", "up"))}), "high or low"),
        (lambda: Table({"x": Numeric(Normal(), missing_by=("x", "high"))}), "missing rate"),
        (lambda: Table({"x": Numeric(Normal(), missing=0.1, missing_by=("y", "high"))}), "numeric"),
        (lambda: Table({"d": Derived(lambda y: y)}), "numeric"),
        (lambda: Table({"d": Derived(lambda: 1.0)}), "at least one"),
        (lambda: Table({"x": Numeric(Normal()), "d": Derived(lambda x: x), "e": Derived(lambda d: d)}), "numeric"),
        (lambda: Table({"x": Numeric(Normal()), "d": Derived(lambda x: x), "y": Numeric(causes={"d": 1.0})}), "unknown parents"),
        (lambda: Table({"x": Numeric(ByLevel("k", {"a": (0, 1)}))}), "categorical"),
        (lambda: Table({"k": Categorical(["a", "b"]), "x": Numeric(ByLevel("k", {"a": (0, 1)}))}), "one range"),
        (lambda: Table({"k": Categorical(["a", "b"]), "y": Numeric(causes={"k": 1.0})}), "no order"),
        (lambda: Table({"k": Categorical(["a", "b"]), "s": Categorical({"p": 0.5, "q": 0.5}, scores={"p": {"k": 1.0}})}), "no numbers"),
        (lambda: Table({"x": Numeric(Normal())}, target="y"), "target"),
        (lambda: Numeric(Scaled("score", 0, 10), round=20), "no step"),
        (lambda: Numeric(Scaled("score", 0, 10), clip=(None, 11)), "inside the range"),
        (lambda: Table({"x": Numeric(Normal())}).sample(1, seed=0), "at least two"),
        (lambda: Numeric(ByLevel("k", {"a": (0, 1)}), round=1), "renders last"),
        (lambda: Numeric(ByLevel("k", {"a": (0, 1)}), causes={"k": {"a": 1.0}}), "itself"),
        (lambda: Categorical({"a": 0.5, "b": 0.5}, scores={"a": {}}, how="max"), "take the place"),
        (lambda: Categorical({"a": 0.5, "b": 0.5}, scores={"a": {}}, ordered=True), "take the place"),
        (lambda: Categorical({"a": 0.5, "b": 0.5}, scores={"a": {}}, noise=Noise("laplace", 1.0)), "Gumbel"),
        (lambda: Categorical({"a": 0.5, "b": 0.5}, scores={"a": {}}, noise=Normal()), "Gumbel"),
        (lambda: Table({"x": Numeric(Normal()), "k": Categorical({"a": 0.5, "b": 0.7}, scores={"a": {"x": 1.0}})}), "summing to one"),
        (lambda: Table({"p_id": ForeignKey("p"), "x": Numeric(causes={("p_id.a", "p_id.b"): 1.0})}), "crosses"),
    ],
)  # fmt: skip
def test_a_mistake_raises_when_the_table_is_built(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_an_ordered_level_reads_by_weight_and_by_level():
    frame = Table(
        {
            "x": Numeric(Normal()),
            "k": Categorical(["a", "b", "c"], causes={"x": 1.0}, noise=0.0, ordered=True),
            "up": Numeric(causes={"k": 1.0}, noise=0.0),
            "top": Numeric(causes={"k": {"c": 5.0}}, noise=0.0),
        }
    ).sample(3000, seed=0)
    assert spearman(frame.x, frame.up) > 0.99
    assert frame.groupby("k", observed=True).top.mean().round(6).to_dict() == {
        "a": 0,
        "b": 0,
        "c": 5,
    }


def test_a_pair_multiplies_two_causes_and_rules_threshold_them():
    frame = Table(
        {
            "a": Numeric(),
            "b": Numeric(),
            "product": Numeric(causes={("a", "b"): 2.0}, noise=0.0),
            "both": Numeric(causes={("a", "b"): 1.0}, noise=0.0, how="rules"),
        }
    ).sample(2000, seed=0)
    assert np.allclose(frame["product"], 2.0 * frame.a * frame.b)
    assert frame.both.nunique() == 2
    assert (frame.both > 0).eq((frame.a > 0) == (frame.b > 0)).all()


def test_a_level_sets_the_slope_of_a_number():
    frame = Table(
        {
            "k": Categorical(["flat", "steep"]),
            "x": Numeric(Normal()),
            "y": Numeric(causes={("k", "x"): {"steep": 3.0}}, noise=0.0),
        }
    ).sample(2000, seed=0)
    flat, steep = frame[frame.k == "flat"], frame[frame.k == "steep"]
    assert (flat.y == 0).all()
    assert spearman(steep.x, steep.y) > 0.999


@pytest.mark.parametrize(("how", "reduce"), [("max", np.maximum), ("min", np.minimum)])
def test_how_combines_causes(how, reduce):
    frame = Table(
        {
            "a": Numeric(),
            "b": Numeric(),
            "y": Numeric(causes={"a": 1.0, "b": 1.0}, noise=0.0, how=how),
        }
    ).sample(1000, seed=0)
    assert np.allclose(frame.y, reduce(frame.a, frame.b))


def test_records_floor_cap_and_blank_by_a_column():
    frame = Table(
        {
            "x": Numeric(Normal()),
            "z": Numeric(Scaled("amount", 3, 90), zeros=0.3),
            "capped": Numeric(causes={"x": 1.0}, noise=0.0, clip=(None, 0.5)),
            "shy": Numeric(Normal(), missing=0.2, missing_by=("x", "high")),
        }
    ).realize(0).sample(10000, seed=0)  # fmt: skip
    assert (frame.z == 3).mean() >= 0.3
    assert frame.capped.max() == 0.5 and frame.capped.min() < -1.0
    assert frame.shy.isna().mean() == pytest.approx(0.2, abs=0.02)
    assert frame.x[frame.shy.isna()].mean() > 0.5


@pytest.mark.parametrize("rate", [0.2, 0.5, 0.93])
def test_blanks_by_a_column_keep_their_share_at_any_rate(rate):
    frame = Table(
        {"x": Numeric(), "y": Numeric(Normal(), missing=rate, missing_by=("x", "low"))}
    ).sample(20000, seed=0)
    assert frame.y.isna().mean() == pytest.approx(rate, abs=0.01)
    assert frame.x[frame.y.isna()].mean() < frame.x[frame.y.notna()].mean()


def test_a_one_sided_clip_keeps_the_other_limit():
    frame = Table(
        {"x": Numeric(Scaled("score", 0.4, 10, shape="symmetric"), round=1, clip=(None, 8.0))}
    ).realize(0).sample(4000, seed=0)  # fmt: skip
    assert frame.x.min() >= 0.4 and frame.x.max() <= 8.0


def test_each_level_takes_its_range():
    table = Table(
        {
            "k": Categorical({"small": 0.5, "big": 0.5}),
            "x": Numeric(ByLevel("k", {"small": (1, 10), "big": (100, 1000)}), noise=0.5),
        }
    )
    assert table.family
    frame = table.realize(0).sample(4000, seed=0)
    assert frame.x[frame.k == "small"].between(1, 10).all()
    assert frame.x[frame.k == "big"].between(100, 1000).all()


def test_scores_are_fit_to_their_shares():
    wanted = {"a": 0.6, "b": 0.3, "c": 0.1}
    table = Table(
        {
            "x": Numeric(Normal()),
            "k": Categorical(wanted, scores={"a": {"x": 2.0}, "c": {"x": -Range(1, 3)}}, noise=0.0),
        }
    )
    assert table.family
    frame = table.realize(0).sample(8000, seed=0)
    shares = frame.k.value_counts(normalize=True)
    for level, share in wanted.items():
        assert shares[level] == pytest.approx(share, abs=0.03)
    means = frame.groupby("k", observed=True).x.mean()
    assert means["a"] > means["b"] > means["c"]


def test_shares_vary_around_the_authored_and_keep_every_level():
    table = Table({"k": Categorical(Shares({"a": 0.9, "b": 0.09, "c": 0.01}, concentration=20.0))})
    shares = [
        table.realize(s).sample(4000, seed=0).k.value_counts(normalize=True) for s in range(8)
    ]
    assert len({round(s["a"], 3) for s in shares}) > 1
    assert all(s["c"] > 0.02 for s in shares)
    assert all(abs(s["a"] - 0.9) < 0.3 for s in shares)


def test_an_authored_database_realizes_across_its_keys():
    schema = Schema(
        {
            "cafes": Table(
                {
                    "id": PrimaryKey(),
                    "size": Numeric(Scaled("count", 1, 30, shape="counts")),
                    "visits": Hidden(causes={Summary("visits", "cafe_id"): WEAK}, noise=0.1),
                    "kind": Categorical(Shares(["city", "town"])),
                }
            ),
            "visits": Table(
                {
                    "id": PrimaryKey(),
                    "cafe_id": ForeignKey("cafes"),
                    "wait": Numeric(
                        Scaled("duration", 0, 120),
                        causes={"cafe_id.size": -STRONG, "cafe_id.kind": (WEAK, {"town": 1.0})},
                        noise=LOW,
                    ),
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="realize"):
        schema.sample({"cafes": 5, "visits": 5}, seed=0)
    frames = schema.realize(1).sample({"cafes": 50, "visits": 3000}, seed=0)
    joined = frames["visits"].merge(frames["cafes"], left_on="cafe_id", right_on="id")
    assert joined.wait.between(0, 120).all()
    assert spearman(joined.wait, joined["size"]) < -0.5
    assert (
        schema.realize(1)
        .sample({"cafes": 50, "visits": 3000}, seed=0)["visits"]
        .equals(frames["visits"])
    )


def test_a_family_may_generate_on_real_rows():
    real = pd.DataFrame({"age": np.arange(300.0), "city": np.resize(["Boston", "Lima"], 300)})
    table = Table.from_frame(
        real,
        extra={
            "spend": Numeric(
                Scaled("amount", 0, 100), causes={"age": STRONG, "city": {"Lima": 1.0}}, noise=LOW
            )
        },
    )
    assert table.family
    frame = table.realize(0).sample(seed=0)
    assert frame.age.equals(real.age) and frame.city.equals(real.city)
    assert spearman(frame.age, frame.spend) > 0.8


def test_grouped_rows_recompute_what_they_derive():
    plan = Table(
        {
            "x": Numeric(Scaled("score", 0, 10)),
            "k": Categorical({"a": 0.5, "b": 0.5}, causes={"x": 1.0}, noise=0.2),
            "y": Numeric(ByLevel("k", {"a": (0, 1), "b": (5, 6)}), causes={"x": 0.5}, noise=0.2),
            "d": Derived(lambda x, y: x + y),
        },
        target="k",
    ).realize(0)
    grouping = Grouping(
        "replicates", ("x", "k"), rows=4.0, spread=0.3, measurement=0.0, column=None
    )
    population = Population(
        TableMetadata(["x", "y", "d"], ["numeric"] * 3, target="k", plan=plan),
        lanes=(1,),
        render=None,
        grouping=grouping,
    )
    frame, entities = population.entity_rows(400, 0)
    assert np.allclose(frame.d, frame.x + frame.y, atol=1e-5)
    assert frame.groupby(entities).x.nunique().max() == 1
    assert frame.y[frame.k == "b"].between(5, 6).all()


def test_real_columns_stay_unranged():
    assert not Table({"x": Real(np.arange(3.0))}).family
    assert Table({"x": Real(np.arange(3.0))}).realize(0).sample(seed=0).x.tolist() == [
        0.0,
        1.0,
        2.0,
    ]
