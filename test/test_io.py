import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database
from relbench.base import Table as RelbenchTable
from relbench.base.table import is_time_sorted

from plurel import Exponential, ForeignKey, Normal, PrimaryKey, Schema, Table, Uniform
from plurel.columns import DEFAULT_CALENDAR, Column
from plurel.distributions import Gumbel
from plurel.graph import Foreign, LinearEdge, Node
from plurel.io import create_database, read_database, split_timestamps, write_database
from plurel.links import HSBMLink, TreeLink

ROWS = {"customers": 40, "orders": 300}


def customers(key=True):
    return Table(
        {
            **({"customer_id": PrimaryKey()} if key else {}),
            "segment": Column(
                Node(bias=(0.0, 0.0, 0.0), onehot=True, noise=Gumbel()), categories=("a", "b", "c")
            ),
            "spend": Column(Node()),
        }
    )


def orders(key=True, time_column="when"):
    return Table(
        {
            **({"order_id": PrimaryKey()} if key else {}),
            "when": Column(Node(), "timestamp", marginal=DEFAULT_CALENDAR),
            "amount": Column(Node((LinearEdge("when"),), noise=Normal(std=0.5))),
            "value": Column(Node(LinearEdge(Foreign("customer_id", "spend")), noise=None)),
            "customer_id": ForeignKey("customers", HSBMLink((2,), (2,)), nullable=0.1, fill=0.0),
        },
        time_column=time_column,
    )


@pytest.fixture
def schema():
    return Schema({"customers": customers(), "orders": orders()})


def test_a_key_may_be_named_self_over_a_temporal_parent():
    parent = Table(
        {"id": PrimaryKey(), "t": Column(Node(), "timestamp", marginal=DEFAULT_CALENDAR)},
        time_column="t",
    )
    child = Table({"self": ForeignKey("parent"), "x": Column(Node())})
    schema = Schema({"parent": parent, "child": child})
    frames = schema.sample({"parent": 10, "child": 40}, seed=0)
    db = create_database(schema, frames)
    assert db.table_dict["child"].df["self"].between(0, 9).all()


def test_tables_declare_their_keys_and_time():
    assert orders().pkey_column == "order_id" and orders().time_column == "when"
    assert orders(key=False, time_column=None).pkey_column is None
    with pytest.raises(ValueError, match="timestamp"):
        orders(time_column="amount")
    with pytest.raises(ValueError, match="timestamp"):
        orders(time_column="nothing")
    with pytest.raises(ValueError, match="one key"):
        Table({"x": Node(), "a": PrimaryKey(), "b": PrimaryKey()})


def test_database_wraps_sampled_tables_with_relbench_metadata(schema):
    frames = schema.sample(ROWS, seed=0)
    assert list(frames["orders"]) == ["order_id", "when", "amount", "value", "customer_id"]
    db = create_database(schema, frames)
    assert set(db.table_dict) == set(ROWS)
    customer_table, order_table = db.table_dict["customers"], db.table_dict["orders"]
    assert customer_table.pkey_col == "customer_id" and customer_table.time_col is None
    assert customer_table.fkey_col_to_pkey_table == {}
    assert order_table.pkey_col == "order_id" and order_table.time_col == "when"
    assert order_table.fkey_col_to_pkey_table == {"customer_id": "customers"}
    np.testing.assert_array_equal(customer_table.df["customer_id"], np.arange(ROWS["customers"]))
    linked = order_table.df["customer_id"].dropna().astype(int)
    assert linked.isin(customer_table.df["customer_id"]).all()
    assert order_table.df["customer_id"].isna().any()
    assert db.min_timestamp == order_table.df["when"].min()
    assert db.max_timestamp == order_table.df["when"].max()
    expected = frames["orders"].sort_values("when", kind="stable").reset_index(drop=True)
    expected["order_id"] = np.arange(len(expected))
    pd.testing.assert_frame_equal(order_table.df, expected)
    events = Schema({"customers": customers(), "orders": orders(key=False)})
    event_table = create_database(events, events.sample(ROWS, seed=0)).table_dict["orders"]
    assert event_table.pkey_col is None and "order_id" not in event_table.df
    unkeyed = Schema({"customers": customers(key=False), "orders": orders()})
    with pytest.raises(ValueError, match="primary key"):
        create_database(unkeyed, unkeyed.sample(ROWS, seed=0))
    with pytest.raises(ValueError):
        create_database(schema, {"customers": frames["customers"]})
    with pytest.raises(ValueError, match="lacks"):
        create_database(schema, {**frames, "orders": frames["orders"].drop(columns="order_id")})
    with pytest.raises(ValueError, match="datetime"):
        create_database(schema, {**frames, "orders": frames["orders"].assign(when=0.0)})


def test_write_and_read_round_trip(schema, tmp_path):
    db = create_database(schema, schema.sample(ROWS, seed=0))
    split = db.min_timestamp + (db.max_timestamp - db.min_timestamp) * 0.7
    path = write_database(
        db,
        tmp_path / "synthetic",
        name="synthetic",
        val_timestamp=split,
        test_timestamp=db.max_timestamp,
    )
    assert (path / "manifest.yaml").exists()
    assert {p.name for p in (path / "db").iterdir()} == {"customers.parquet", "orders.parquet"}
    loaded = read_database(path)
    for name, table in db.table_dict.items():
        again = loaded.table_dict[name]
        assert (again.pkey_col, again.time_col) == (table.pkey_col, table.time_col)
        assert again.fkey_col_to_pkey_table == table.fkey_col_to_pkey_table
        pd.testing.assert_frame_equal(again.df, table.df, check_categorical=False)
    order_keys = loaded.table_dict["orders"].df["customer_id"]
    assert order_keys.isna().sum() == db.table_dict["orders"].df["customer_id"].isna().sum()
    with pytest.raises(FileExistsError):
        write_database(
            db, path, name="synthetic", val_timestamp=split, test_timestamp=db.max_timestamp
        )
    smaller = create_database(schema, schema.sample({"customers": 40, "orders": 100}, seed=1))
    (path / "db" / "stale.parquet").touch()
    write_database(
        smaller,
        path,
        name="synthetic",
        val_timestamp=split,
        test_timestamp=db.max_timestamp,
        overwrite=True,
    )
    assert {p.name for p in (path / "db").iterdir()} == {"customers.parquet", "orders.parquet"}
    assert len(read_database(path).table_dict["orders"].df) == 100
    with pytest.raises(ValueError, match="val_timestamp"):
        write_database(
            db, tmp_path / "other", name="x", val_timestamp=db.max_timestamp, test_timestamp=split
        )
    shuffled = db.table_dict["orders"].df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    unordered = Database({"orders": RelbenchTable(shuffled, {}, "order_id", "when")})
    with pytest.raises(ValueError, match="time order"):
        write_database(
            unordered, tmp_path / "bad", name="x", val_timestamp=split, test_timestamp=split
        )


def test_a_home_path_is_expanded(schema, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    db = create_database(schema, schema.sample(ROWS, seed=0))
    split = db.max_timestamp
    write_database(db, "~/made/here", name="x", val_timestamp=split, test_timestamp=split)
    assert (tmp_path / "made" / "here" / "manifest.yaml").exists() and not (tmp_path / "~").exists()
    assert set(read_database("~/made/here").table_dict) == set(db.table_dict)


def test_database_puts_temporal_tables_in_time_order_with_keys_as_positions():
    seconds = Uniform(DEFAULT_CALENDAR.start.timestamp(), DEFAULT_CALENDAR.end.timestamp())
    customers = Table(
        {
            "id": PrimaryKey(),
            "signup": Column(Node(noise=seconds), "timestamp"),
            "value": Column(Node()),
        },
        time_column="signup",
    )
    orders = Table(
        {
            "id": PrimaryKey(),
            "customer_id": ForeignKey("customers"),
            "signup": Node(LinearEdge(Foreign("customer_id", "signup")), noise=None),
            "when": Column(
                Node((LinearEdge("signup"),), noise=Exponential(3600.0)), "timestamp", missing=0.1
            ),
            "value": Column(Node(LinearEdge(Foreign("customer_id", "value")), noise=None)),
            "amount": Column(Node((LinearEdge("value"),), noise=Normal(std=0.1))),
        },
        time_column="when",
    )
    employees = Table(
        {
            "id": PrimaryKey(),
            "joined": Column(Node(), "timestamp", marginal=DEFAULT_CALENDAR),
            "level": Column(Node()),
            "manager_level": Column(Node(LinearEdge(Foreign("manager_id", "level")), noise=None)),
            "manager_id": ForeignKey("employees", TreeLink(roots=0.2), fill=0.0),
        },
        time_column="joined",
    )
    schema = Schema({"customers": customers, "orders": orders, "employees": employees})
    rows = {"customers": 100, "orders": 1000, "employees": 80}
    frames = schema.sample(rows, seed=0)
    assert not frames["customers"]["signup"].is_monotonic_increasing
    tables = {name: table.df for name, table in create_database(schema, frames).table_dict.items()}
    for name, scm in schema.tables.items():
        assert is_time_sorted(tables[name][scm.time_column])
        np.testing.assert_array_equal(tables[name]["id"], np.arange(rows[name]))
    keys = tables["orders"]["customer_id"].to_numpy(dtype=int)
    when = tables["orders"]["when"]
    known = when.notna().to_numpy()
    assert 0 < (~known).sum() and known[: known.sum()].all()
    signup = tables["customers"]["signup"].to_numpy()[keys]
    assert (when.to_numpy()[known] >= signup[known]).all()
    np.testing.assert_array_equal(
        tables["orders"]["value"], tables["customers"]["value"].to_numpy()[keys]
    )
    manager = tables["employees"]["manager_id"]
    roots = manager.isna().to_numpy()
    bosses = manager.to_numpy(dtype=float, na_value=-1).astype(int)
    level = tables["employees"]["level"].to_numpy()
    np.testing.assert_array_equal(
        tables["employees"]["manager_level"].to_numpy()[~roots], level[bosses[~roots]]
    )
    current = bosses
    for _ in range(rows["employees"]):
        current = np.where(current >= 0, bosses[current], -1)
    assert (current == -1).all()


def test_split_timestamps_follow_the_pooled_time_columns(schema):
    db = create_database(schema, schema.sample(ROWS, seed=0))
    val, test = split_timestamps(db, 0.2, 0.1)
    when = db.table_dict["orders"].df["when"]
    assert val < test and abs((when < val).mean() - 0.7) < 0.02
    assert abs((when < test).mean() - 0.9) < 0.02
    with pytest.raises(ValueError, match="timestamped"):
        split_timestamps(Database({"customers": db.table_dict["customers"]}))
    for bad in ((0.0, 0.1), (0.5, 0.5), (-0.1, 0.2)):
        with pytest.raises(ValueError, match="share"):
            split_timestamps(db, *bad)


def test_importing_plurel_leaves_relbench_unloaded_until_a_database_function_is_asked_for():
    code = (
        "import sys, plurel\n"
        "assert 'relbench' not in sys.modules\n"
        "assert callable(plurel.create_database) and 'relbench' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
