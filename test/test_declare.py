import numpy as np
import pandas as pd
import pytest

from plurel import (
    Categorical,
    ForeignKey,
    Hidden,
    LogNormal,
    Normal,
    Numeric,
    Poisson,
    PrimaryKey,
    Real,
    Schema,
    Summary,
    Table,
    Timestamp,
    Uniform,
    create_database,
)


def spearman(a, b) -> float:
    return pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)), method="spearman")


def customers_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(300)],
            "city": np.resize(["Boston", "Austin", "Lima"], 300),
            "tenure": np.arange(300.0),
            "joined": pd.date_range("2020-01-01", periods=300),
        }
    )


def test_causes_set_direction_shape_and_noise():
    frame = Table(
        {
            "x": Numeric(Uniform(18.0, 80.0)),
            "up": Numeric(LogNormal(), causes={"x": 1.0}, noise=0.1),
            "down": Numeric(LogNormal(), causes={"x": -1.0}, noise=0.1),
            "bent": Numeric(causes={"x": (1.0, "square")}, noise=0.0),
            "loose": Numeric(LogNormal(), causes={"x": 1.0}, noise=3.0),
        }
    ).sample(4000, seed=0)
    assert frame.x.between(18.0, 80.0).all() and (frame.up > 0).all()
    assert spearman(frame.x, frame.up) > 0.9 and spearman(frame.x, frame.down) < -0.9
    assert 0.1 < spearman(frame.x, frame.loose) < 0.6
    assert abs(spearman(frame.x, frame.bent)) < 0.1


def test_a_column_without_a_marginal_is_its_weighted_sum():
    frame, latents = Table(
        {
            "x": Hidden(),
            "y": Hidden(),
            "total": Numeric(causes={"x": 2.0, "y": -1.0}, noise=0.0),
        }
    ).sample_with_latents(50, seed=0)
    assert np.allclose(frame.total, 2.0 * latents["x"][:, 0] - latents["y"][:, 0])
    assert set(frame) == {"total"}


def test_categorical_shares_order_and_level_effects():
    frame = Table(
        {
            "age": Numeric(Normal()),
            "tier": Categorical({"low": 0.5, "mid": 0.3, "top": 0.2}, causes={"age": 1.0}),
            "after_rank": Numeric(causes={"tier": 1.0}, noise=0.0),
            "after_top": Numeric(causes={"tier": {"top": 5.0}}, noise=0.0),
        }
    ).sample(5000, seed=0)
    shares = frame.tier.value_counts(normalize=True)
    assert shares["low"] == pytest.approx(0.5, abs=0.01)
    assert shares["top"] == pytest.approx(0.2, abs=0.01)
    means = frame.groupby("tier", observed=True).age.mean()
    assert means["low"] < means["mid"] < means["top"]
    ranks = frame.groupby("tier", observed=True).after_rank.mean()
    assert ranks["low"] < ranks["mid"] < ranks["top"]
    assert (frame.after_top[frame.tier == "top"] == 5.0).all()
    assert (frame.after_top[frame.tier != "top"] == 0.0).all()


def test_clip_round_and_missing_apply_to_the_column():
    frame = Table(
        {
            "age": Numeric(Normal(50.0, 30.0), clip=(18.0, 95.0), round=1.0, missing=0.1),
            "count": Numeric(Poisson(4.0)),
        }
    ).sample(4000, seed=0)
    seen = frame.age.dropna()
    assert seen.between(18.0, 95.0).all() and (seen % 1 == 0).all()
    assert frame.age.isna().mean() == pytest.approx(0.1, abs=0.02)


@pytest.mark.parametrize(
    "table, match",
    [
        ({"x": Categorical("abc")}, "not one string"),
        ({"x": Numeric(causes={"y": 1.0})}, "unknown parents"),
        ({"x": Numeric(noise=-1.0)}, "standard deviation"),
        ({"y": Hidden(), "x": Numeric(causes={"y": {"a": 1.0}})}, "categorical parent"),
        (
            {"y": Categorical(("a", "b")), "x": Numeric(causes={"y": {"z": 1.0}})},
            "unknown levels",
        ),
        ({"y": Real(["a", "b"]), "x": Numeric(causes={"y": 1.0})}, "in no order"),
        ({"x": Numeric(causes={Summary("t", "k"): {"a": 1.0}})}, "a summary is a number"),
        ({"x": Real([1.0, 2.0]), "y": Real([1.0])}, "row count"),
        ({"x": "y"}, "a key, a column"),
    ],
)
def test_a_table_rejects_what_it_cannot_mean(table, match):
    with pytest.raises(ValueError, match=match):
        Table(table)


def test_keys_cross_tables_by_dotted_name():
    schema = Schema(
        {
            "customers": Table(
                {
                    "id": PrimaryKey(),
                    "spend": Numeric(LogNormal()),
                    "plan": Categorical(("free", "paid")),
                }
            ),
            "orders": Table(
                {
                    "id": PrimaryKey(),
                    "customer_id": ForeignKey("customers"),
                    "time": Timestamp("2022-01-01", "2023-01-01"),
                    "amount": Numeric(
                        LogNormal(),
                        causes={"customer_id.spend": 1.0, "customer_id.plan": {"paid": 2.0}},
                        noise=0.2,
                    ),
                },
                time_column="time",
            ),
        }
    )
    frames = schema.sample({"customers": 300, "orders": 3000}, seed=0)
    joined = frames["orders"].merge(frames["customers"], left_on="customer_id", right_on="id")
    assert frames["orders"].time.dt.year.eq(2022).all()
    paid = joined.plan == "paid"
    assert spearman(joined.spend[paid], joined.amount[paid]) > 0.5
    assert joined.amount[paid].mean() > joined.amount[~paid].mean()
    assert list(frames["orders"]) == ["id", "customer_id", "time", "amount"]


def test_a_bare_count_summary_is_exact():
    frames = Schema(
        {
            "d": Table(
                {"id": PrimaryKey(), "size": Numeric(causes={Summary("e", "d_id"): 1.0}, noise=0.0)}
            ),
            "e": Table({"id": PrimaryKey(), "d_id": ForeignKey("d")}),
        }
    ).sample({"d": 5, "e": 60}, seed=0)
    counts = frames["e"].d_id.value_counts().reindex(range(5), fill_value=0).to_numpy()
    assert (frames["d"]["size"].to_numpy() == counts).all()


def test_a_null_key_gets_no_level_effect():
    frames = Schema(
        {
            "p": Table({"id": PrimaryKey(), "kind": Categorical(("a", "b"))}),
            "c": Table(
                {
                    "id": PrimaryKey(),
                    "p_id": ForeignKey("p", nullable=0.3),
                    "x": Numeric(causes={"p_id.kind": {"a": 1.0, "b": 2.0}}, noise=0.0),
                }
            ),
        }
    ).sample({"p": 50, "c": 500}, seed=0)
    child = frames["c"]
    assert (child.x[child.p_id.isna()] == 0.0).all()
    assert (child.x[child.p_id.notna()] >= 1.0).all()


def test_real_rows_stay_and_children_read_them():
    real = customers_frame()
    schema = Schema(
        {
            "customers": Table.from_frame(real, primary_key="customer_id"),
            "orders": Table(
                {
                    "id": PrimaryKey(),
                    "customer_id": ForeignKey("customers"),
                    "amount": Numeric(
                        LogNormal(), causes={"customer_id.city": {"Boston": 2.0}}, noise=0.2
                    ),
                    "late": Numeric(causes={"customer_id.tenure": 1.0}, noise=0.0),
                }
            ),
        }
    )
    frames = schema.sample({"orders": 2000}, seed=0)
    pd.testing.assert_frame_equal(frames["customers"], real)
    orders = frames["orders"]
    assert orders.customer_id.isin(real.customer_id).all()
    joined = orders.merge(real, on="customer_id")
    boston = joined.city == "Boston"
    assert joined.amount[boston].mean() > 2.0 * joined.amount[~boston].mean()
    assert spearman(joined.tenure, joined.late) > 0.99
    with pytest.raises(ValueError, match="given rows"):
        schema.sample({"customers": 10, "orders": 5})


def test_generated_columns_join_real_rows():
    real = customers_frame()
    table = Table.from_frame(
        real,
        primary_key="customer_id",
        extra={
            "churned": Categorical(("no", "yes"), causes={"tenure": -1.0, "city": {"Lima": 1.0}})
        },
    )
    frame = table.sample(seed=0)
    pd.testing.assert_frame_equal(frame[real.columns], real)
    churned = frame.churned == "yes"
    assert frame.tenure[churned].mean() < frame.tenure[~churned].mean()
    assert (frame.city[churned] == "Lima").mean() > (frame.city[~churned] == "Lima").mean()
    with pytest.raises(ValueError, match="no columns"):
        Table.from_frame(real, primary_key="nope")


def test_real_children_keep_their_keys_and_blanks():
    parents = customers_frame()
    orders = pd.DataFrame(
        {
            "order_id": range(6),
            "customer_id": ["C0", "C1", None, "C1", "C2", "C0"],
            "time": pd.date_range("2024-01-01", periods=6),
        }
    )
    schema = Schema(
        {
            "customers": Table.from_frame(parents, primary_key="customer_id"),
            "orders": Table.from_frame(
                orders,
                primary_key="order_id",
                foreign_keys={"customer_id": "customers"},
                time_column="time",
                extra={"value": Numeric(causes={"customer_id.city": {"Boston": 1.0}}, noise=0.0)},
            ),
        }
    )
    frames = schema.sample(seed=0)
    pd.testing.assert_series_equal(frames["orders"].customer_id, orders.customer_id)
    assert frames["orders"].value.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    db = create_database(schema, frames).table_dict
    assert db["customers"].df.customer_id.tolist() == list(range(300))
    assert db["orders"].df.customer_id.isna().sum() == 1
    assert db["orders"].df.customer_id.dropna().tolist() == [0, 1, 1, 2, 0]

    missing = orders.assign(customer_id=["C0", "nope", None, "C1", "C2", "C0"])
    with pytest.raises(ValueError, match="values its parent lacks"):
        Schema(
            {
                "customers": Table.from_frame(parents, primary_key="customer_id"),
                "orders": Table.from_frame(missing, foreign_keys={"customer_id": "customers"}),
            }
        ).sample(seed=0)


def test_real_keys_must_be_unique_and_complete():
    with pytest.raises(ValueError, match="unique and present"):
        PrimaryKey(["a", "a"])
    with pytest.raises(ValueError, match="unique and present"):
        PrimaryKey(["a", None])
    with pytest.raises(ValueError, match="own blanks"):
        ForeignKey("p", values=["a"], nullable=0.1)


def test_positional_real_children_of_a_generated_parent():
    keys = pd.Series([0, 1, None, 2], dtype="Int64")
    frames = Schema(
        {
            "p": Table({"id": PrimaryKey(), "x": Numeric(Normal())}),
            "c": Table.from_frame(pd.DataFrame({"p_id": keys}), foreign_keys={"p_id": "p"}),
        }
    ).sample({"p": 5}, seed=0)
    assert frames["c"].p_id.isna().tolist() == [False, False, True, False]
    assert frames["c"].p_id.dropna().tolist() == [0, 1, 2]
