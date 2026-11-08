import numpy as np
import pandas as pd
import pytest

from plurel import (
    Categorical,
    Derived,
    Exponential,
    ForeignKey,
    Numeric,
    PrimaryKey,
    Recorded,
    Schema,
    Summary,
    Table,
    Timestamp,
    Uniform,
)


def test_a_recorded_cause_reads_what_the_frame_holds():
    table = Table(
        {
            "x": Numeric(Uniform(100, 200), round=5, clip=(110, None), missing=0.1),
            "y": Numeric(causes={Recorded("x", fill=0.0): 2.0}, noise=0),
            "share": Derived(lambda x, y: x / (y + 1.0)),
            "z": Numeric(causes={Recorded("y"): 1.0, Recorded("share", fill=-1.0): 1.0}, noise=0),
        }
    )
    frame = table.sample(500, seed=0)
    np.testing.assert_array_equal(frame.y, 2 * frame.x.fillna(0.0))
    np.testing.assert_allclose(frame.z, frame.y + frame.share.fillna(-1.0))
    assert frame.x.isna().any() and (frame.x.dropna() % 5 == 0).all() and frame.x.min() >= 110


def test_noise_on_recorded_causes_is_in_their_units():
    table = Table(
        {"x": Numeric(Uniform(100, 200)), "y": Numeric(causes={Recorded("x"): 2.0}, noise=5.0)}
    )
    frame = table.sample(20000, seed=0)
    residual = frame.y - 2 * frame.x
    assert abs(residual.mean()) < 0.2 and abs(residual.std() - 5.0) < 0.1
    assert table.sample(50, seed=1).equals(table.sample(50, seed=1))


def test_recorded_summaries_agree_with_pandas():
    hows = {"sum": None, "mean": -1.0, "min": -1.0, "max": -1.0, "count": None}
    teams = Table(
        {"id": PrimaryKey()}
        | {
            how: Numeric(
                causes={Summary("staff", "team_id", Recorded("pay"), how, fill): 1.0}, noise=0
            )
            for how, fill in hows.items()
        }
    )
    staff = Table(
        {
            "team_id": ForeignKey("teams", nullable=0.1),
            "pay": Numeric(Uniform(100, 200), missing=0.3),
        }
    )
    frames = Schema({"teams": teams, "staff": staff}).sample({"teams": 40, "staff": 60}, seed=0)
    groups = frames["staff"].groupby("team_id").pay
    for how, fill in hows.items():
        expected = getattr(groups, how)().reindex(range(40)).fillna(0.0 if fill is None else fill)
        np.testing.assert_array_equal(frames["teams"][how], expected.to_numpy(dtype=float))
    assert (frames["teams"]["count"] == 0).any() and (frames["teams"]["mean"] == -1.0).any()


def test_a_delay_after_a_recorded_time_never_precedes_it():
    joined = pd.Series(pd.date_range("2020-01-01", periods=50, freq="17h"))
    people = Table.from_frame(
        pd.DataFrame({"id": range(50), "joined": joined}),
        primary_key="id",
        extra={"signup": Timestamp("2020-01-01", "2021-01-01")},
    )
    week = Exponential(7 * 86400)
    orders = Table(
        {
            "person_id": ForeignKey("people"),
            "placed": Timestamp(causes={Recorded("person_id.signup"): 1.0}, noise=week),
            "shipped": Timestamp(causes={Recorded("placed"): 1.0}, noise=week),
            "welcome": Timestamp(causes={Recorded("person_id.joined"): 1.0}, noise=week),
        },
        time_column="placed",
    )
    schema = Schema({"people": people, "orders": orders})
    for seed in range(5):
        frames = schema.sample({"orders": 2000}, seed=seed)
        rows = frames["orders"].merge(frames["people"], left_on="person_id", right_on="id")
        assert (rows.placed > rows.signup).all() and (rows.shipped > rows.placed).all()
        assert (rows.welcome > rows.joined).all()
        assert pd.Timedelta(days=3) < (rows.placed - rows.signup).median() < pd.Timedelta(days=7)


def test_rows_without_a_recorded_value_need_a_fill():
    people = Table({"id": PrimaryKey(), "age": Numeric(Uniform(20, 60), missing=0.2)})

    def orders(cause):
        return Table(
            {"person_id": ForeignKey("people", nullable=0.2), "y": Numeric(causes={cause: 1.0})}
        )

    rows = {"people": 50, "orders": 200}
    with pytest.raises(ValueError, match="give Recorded a fill"):
        Schema({"people": people, "orders": orders(Recorded("person_id.age"))}).sample(rows, seed=0)
    filled = Schema({"people": people, "orders": orders(Recorded("person_id.age", fill=-100.0))})
    frames = filled.sample(rows, seed=0)
    unread = frames["orders"].merge(
        frames["people"], how="left", left_on="person_id", right_on="id"
    )
    assert (frames["orders"].y[unread.age.isna().to_numpy()] < -90).all()
    with pytest.raises(ValueError, match="give Recorded a fill"):
        Table({"x": Numeric(missing=0.5), "y": Numeric(causes={Recorded("x"): 1.0})}).sample(
            9, seed=0
        )


@pytest.mark.parametrize(
    "table, match",
    [
        (
            {"c": Categorical(("a", "b")), "y": Numeric(causes={Recorded("c"): 1.0})},
            "Recorded reads",
        ),
        ({"x": Numeric(), "y": Numeric(causes={Recorded("x"): {"a": 1.0}})}, "a weight"),
        ({"x": Numeric(), "y": Numeric(causes={("x", Recorded("x")): 1.0})}, "a product"),
        (
            {"x": Numeric(), "y": Numeric(causes={Recorded("x"): 1.0}, how="max")},
            "recorded value adds",
        ),
        ({"x": Numeric(causes={Recorded("x"): 1.0})}, "acyclic"),
        (
            {"x": Numeric(causes={"y": 1.0}), "y": Numeric(causes={Recorded("x"): 1.0})},
            "acyclic",
        ),
        (
            {
                "x": Numeric(missing=0.2, missing_by=("y", "high")),
                "y": Numeric(causes={Recorded("x", fill=0.0): 1.0}),
            },
            "acyclic",
        ),
    ],
)
def test_a_table_rejects_recorded_reads_it_cannot_mean(table, match):
    with pytest.raises(ValueError, match=match):
        Table(table)


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: Summary("staff", "team_id", Recorded("pay", fill=0.0), "sum"), "skips blanks"),
        (lambda: Timestamp("2020-01-01"), "both a start and an end"),
        (lambda: Timestamp(), "give some"),
        (lambda: Recorded("x", fill=float("inf")), "finite"),
    ],
)
def test_declarations_reject_recorded_misuse(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_a_schema_checks_what_its_recorded_reads_cross():
    parents = Table(
        {
            "id": PrimaryKey(),
            "c": Categorical(("a", "b")),
            "t": Timestamp("2020-01-01", "2021-01-01"),
        }
    )
    reads_level = Table({"p_id": ForeignKey("p"), "y": Numeric(causes={Recorded("p_id.c"): 1.0})})
    with pytest.raises(ValueError, match="Recorded reads"):
        Schema({"p": parents, "k": reads_level})
    events = Table(
        {"p_id": ForeignKey("p"), "t": Timestamp("2020-01-01", "2021-01-01")}, time_column="t"
    )
    later = Table(
        {
            "id": PrimaryKey(),
            "last": Numeric(causes={Summary("k", "p_id", Recorded("t"), "max", 0.0): 1.0}),
        }
    )
    with pytest.raises(ValueError, match="later events"):
        Schema({"p": later, "k": events})
    ring = {
        "p": Table(
            {"id": PrimaryKey(), "x": Numeric(causes={Summary("k", "p_id", Recorded("y")): 1.0})}
        ),
        "k": Table({"p_id": ForeignKey("p"), "y": Numeric(causes={Recorded("p_id.x"): 1.0})}),
    }
    with pytest.raises(ValueError, match="acyclic"):
        Schema(ring)
