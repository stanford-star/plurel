import numpy as np
import pandas as pd
import pytest

from plurel import (
    Categorical,
    Derived,
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
            "tier": Categorical(
                {"low": 0.5, "mid": 0.3, "top": 0.2}, causes={"age": 1.0}, ordered=True
            ),
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
        ({"x": Numeric(causes={"y": 1.0})}, "unknown parents"),
        ({"y": Hidden(), "x": Numeric(causes={"y": 1.0}, noise=-1.0)}, "standard deviation"),
        ({"y": Hidden(), "x": Numeric(causes={"y": {"a": 1.0}})}, "categorical parent"),
        (
            {"y": Categorical(("a", "b")), "x": Numeric(causes={"y": {"z": 1.0}})},
            "unknown levels",
        ),
        ({"y": Real(["a", "b"]), "x": Numeric(causes={"y": 1.0})}, "in no order"),
        ({"x": Numeric(causes={Summary("t", "k"): {"a": 1.0}})}, "a summary is a number"),
        ({"x": Real([1.0, 2.0]), "y": Real([1.0])}, "row count"),
        ({"x": "y"}, "a key, a column"),
        ({"x": Numeric(noise=0.1)}, "has none"),
        ({"y": Hidden(), "x": Numeric(causes={"y": (1.0, "nope")})}, "transform must be"),
        ({"x": Numeric({"a": 1.0})}, "must be a distribution"),
        ({"x": Numeric(clip=(float("nan"), 1.0))}, "clip"),
        ({"x": Numeric(round=float("nan"))}, "round"),
        ({"zip.code": Real([1.0, 2.0]), "zip": Real(["a", "b"])}, "for itself"),
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


def test_one_blank_keeps_a_real_columns_signal():
    x = np.random.default_rng(0).normal(size=2000)
    x[0] = np.nan
    frame = Table({"x": Real(x), "y": Numeric(causes={"x": 1.0}, noise=0.1)}).sample(seed=0)
    assert np.isnan(frame.x[0]) and spearman(frame.x[1:], frame.y[1:]) > 0.9


def test_large_integer_keys_survive_blanks():
    base = 2**60
    parents = pd.DataFrame({"id": pd.array([base + 1, base + 2, base + 3], dtype="Int64")})
    children = pd.DataFrame({"p_id": pd.array([base + 3, None, base + 1], dtype="Int64")})
    schema = Schema(
        {
            "p": Table.from_frame(parents, primary_key="id"),
            "c": Table.from_frame(children, foreign_keys={"p_id": "p"}),
        }
    )
    frames = schema.sample(seed=0)
    assert frames["c"].p_id[0] == base + 3 and frames["c"].p_id[2] == base + 1
    assert pd.isna(frames["c"].p_id[1])
    keys = create_database(schema, frames).table_dict["c"].df.p_id
    assert keys[0] == 2 and pd.isna(keys[1]) and keys[2] == 0

    generated = Table({"id": PrimaryKey(), "p_id": ForeignKey("p", nullable=0.3)})
    drawn = Schema({"p": schema.tables["p"], "c": generated}).sample({"c": 100}, seed=0)["c"]
    assert drawn.p_id.isna().any() and drawn.p_id.dropna().isin(parents.id).all()


def real_schema():
    parents = pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    children = pd.DataFrame({"p_id": ["c", "a", "b", "a"], "y": [4.0, 1.0, 3.0, 2.0]})
    return Schema(
        {
            "p": Table.from_frame(parents, primary_key="id"),
            "c": Table.from_frame(children, foreign_keys={"p_id": "p"}),
        }
    )


def test_a_database_maps_real_keys_within_the_frames_given():
    schema = real_schema()
    frames = schema.sample(seed=0)
    resorted = {**frames, "c": frames["c"].sort_values("y")}
    db = create_database(schema, resorted).table_dict
    assert db["c"].df.p_id.tolist() == [0, 0, 1, 2] and db["c"].df.y.tolist() == [1, 2, 3, 4]

    parents = {**frames, "p": frames["p"].sort_values("x", ascending=False)}
    db = create_database(schema, parents).table_dict
    assert db["p"].df.x.iloc[db["c"].df.p_id.astype(int)].tolist() == [3.0, 1.0, 2.0, 1.0]

    dropped = {**frames, "p": frames["p"].iloc[1:]}
    with pytest.raises(ValueError, match="values its parent lacks"):
        create_database(schema, dropped)


def test_from_frame_keeps_what_it_is_told_to_keep():
    frame = pd.DataFrame({"id": [1, 2], "a": [3.0, 4.0]})
    with pytest.raises(ValueError, match="both the primary"):
        Table.from_frame(frame, primary_key="id", foreign_keys={"id": "p"})
    with pytest.raises(ValueError, match="replace the real"):
        Table.from_frame(frame, extra={"a": Numeric()})


def test_real_levels_have_no_order_to_summarize():
    with pytest.raises(ValueError, match="in no order"):
        Schema(
            {
                "p": Table(
                    {
                        "id": PrimaryKey(),
                        "x": Numeric(causes={Summary("c", "p_id", "city", "sum"): 1.0}),
                    }
                ),
                "c": Table({"p_id": ForeignKey("p"), "city": Real(["a", "b"])}),
            }
        )


def test_rounding_lands_on_the_step():
    frame = Table({"x": Numeric(Uniform(0.0, 1.0), round=0.01)}).sample(500, seed=0)
    assert (frame.x == frame.x.round(2)).all()


def test_one_string_is_no_levels():
    with pytest.raises(ValueError, match="not one string"):
        Categorical("abc")


def test_real_rows_take_table_metadata_and_a_task_target():
    real = pd.DataFrame({"id": ["A", "B", "C"], "age": [20, 40, 60]})
    table = Table.from_frame(
        real,
        primary_key="id",
        extra={"churn": Categorical({"no": 0.8, "yes": 0.2}, causes={"age": -0.8})},
        target="churn",
        name="customers",
        doc="One customer per row",
        sources="a CRM export",
        docs={"age": "Age in years", "churn": "Left within a year"},
    )
    realized = table.realize(0)
    for each in (table, realized):
        assert (each.name, each.target, each.doc, each.sources) == (
            "customers",
            "churn",
            "One customer per row",
            "a CRM export",
        )
        assert each.docs == {"age": "Age in years", "churn": "Left within a year"}
    frame = realized.sample(seed=0)
    assert frame.id.tolist() == ["A", "B", "C"] and frame.age.tolist() == [20, 40, 60]
    described = realized.describe()
    assert described["target"] == "churn" and described["columns"]["age"]["doc"] == "Age in years"
    assert Table.from_frame(real, primary_key="id", target="age").target == "age"
    with pytest.raises(ValueError, match="target 'id' is a key"):
        Table.from_frame(real, primary_key="id", target="id")


def test_derived_columns_bind_parameters_to_any_column_name():
    real = pd.DataFrame({"Unit Price": [10.0, 20.0, None], "Quantity": [2.0, 3.0, 4.0]})
    total = Derived(
        lambda price, quantity: price * quantity,
        columns={"price": "Unit Price", "quantity": "Quantity"},
    )
    table = Table.from_frame(
        real,
        extra={
            "Total": total,
            "With tax": Derived(lambda total: total * 1.1, columns={"total": "Total"}),
        },
    )
    frame = table.sample(seed=0)
    assert list(frame) == ["Unit Price", "Quantity", "Total", "With tax"]
    np.testing.assert_array_equal(frame.Total, [20.0, 60.0, np.nan])
    np.testing.assert_allclose(frame["With tax"], [22.0, 66.0, np.nan])
    assert table.describe()["columns"]["Total"]["inputs"] == ["Unit Price", "Quantity"]
    with pytest.raises(ValueError, match=r"binds \['cost'\]"):
        Derived(lambda price: price, columns={"cost": "Unit Price"})
    with pytest.raises(ValueError, match=r"reads Price \(as price\)"):
        Table.from_frame(
            real, extra={"T": Derived(lambda price: price, columns={"price": "Price"})}
        )
    assert Derived(lambda a: a, "what it is").doc == "what it is"
    with pytest.raises(ValueError, match="cycle"):
        Table(
            {
                "a": Derived(lambda x: x, columns={"x": "b"}),
                "b": Derived(lambda y: y, columns={"y": "a"}),
            }
        )
