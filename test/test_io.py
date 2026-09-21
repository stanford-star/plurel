import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table
from relbench.base.table import is_time_sorted

from plurel import (
    DEFAULT_CALENDAR,
    SCM,
    Column,
    Exponential,
    LinearEdge,
    Node,
    Normal,
    Uniform,
)
from plurel.distributions import Gumbel
from plurel.graph import Foreign, Summary
from plurel.io import create_database, read_database, split_timestamps, write_database
from plurel.links import HSBMLink, TreeLink
from plurel.schema import FK, Schema

ROWS = {"customers": 40, "orders": 300}


def customers(key=True):
    columns = {
        "customer_id": Column(kind="key"),
        "segment": Column("segment", "categorical", categories=("a", "b", "c")),
        "spend": Column("spend"),
    }
    if not key:
        del columns["customer_id"]
    return SCM(
        {
            "segment": Node(bias=(0.0, 0.0, 0.0), onehot=True, noise=Gumbel()),
            "spend": Node(
                (LinearEdge(Summary("orders", "customer_id", "amount", "sum")),), noise=None
            ),
        },
        columns,
    )


def orders(key=True, time_column="when"):
    columns = {
        "order_id": Column(kind="key"),
        "when": Column("when", "timestamp", marginal=DEFAULT_CALENDAR),
        "amount": Column("amount"),
        "value": Column("value"),
    }
    if not key:
        del columns["order_id"]
    return SCM(
        {
            "when": Node(),
            "value": Node((LinearEdge(Foreign("customer_id", "spend")),), noise=None),
            "amount": Node((LinearEdge("when"),), noise=Normal(std=0.5)),
        },
        columns,
        time_column=time_column,
    )


FKEYS = (FK("orders", "customer_id", "customers", HSBMLink((2,), (2,)), nullable=0.1, fill=0.0),)


@pytest.fixture
def schema():
    return Schema({"customers": customers(), "orders": orders()}, FKEYS)


def test_tables_declare_their_keys_and_time():
    assert orders().pkey_column == "order_id" and orders().time_column == "when"
    assert orders(key=False, time_column=None).pkey_column is None
    with pytest.raises(ValueError, match="timestamp"):
        orders(time_column="amount")
    with pytest.raises(ValueError, match="timestamp"):
        orders(time_column="nothing")
    with pytest.raises(ValueError, match="one key"):
        SCM({"x": Node()}, {"a": Column(kind="key"), "b": Column(kind="key")})


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
    events = Schema({"customers": customers(), "orders": orders(key=False)}, FKEYS)
    event_table = create_database(events, events.sample(ROWS, seed=0)).table_dict["orders"]
    assert event_table.pkey_col is None and "order_id" not in event_table.df
    unkeyed = Schema({"customers": customers(key=False), "orders": orders()}, FKEYS)
    with pytest.raises(ValueError, match="primary key"):
        create_database(unkeyed, unkeyed.sample(ROWS, seed=0))
    with pytest.raises(ValueError, match="collides"):
        Schema(
            {"customers": customers(), "orders": orders()}, (FK("orders", "order_id", "customers"),)
        )
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
    unordered = Database({"orders": Table(shuffled, {}, "order_id", "when")})
    with pytest.raises(ValueError, match="time order"):
        write_database(
            unordered, tmp_path / "bad", name="x", val_timestamp=split, test_timestamp=split
        )


def test_database_puts_temporal_tables_in_time_order_with_keys_as_positions():
    seconds = Uniform(DEFAULT_CALENDAR.start.timestamp(), DEFAULT_CALENDAR.end.timestamp())
    customers = SCM(
        {
            "signup": Node(noise=seconds),
            "value": Node(),
            "n_orders": Node(
                (LinearEdge(Summary("orders", "customer_id", "amount", "count")),), noise=None
            ),
        },
        {
            "id": Column(kind="key"),
            "signup": Column("signup", "timestamp"),
            "value": Column("value"),
            "n_orders": Column("n_orders"),
        },
        time_column="signup",
    )
    orders = SCM(
        {
            "signup": Node((LinearEdge(Foreign("customer_id", "signup")),), noise=None),
            "when": Node((LinearEdge("signup"),), noise=Exponential(3600.0)),
            "value": Node((LinearEdge(Foreign("customer_id", "value")),), noise=None),
            "amount": Node((LinearEdge("value"),), noise=Normal(std=0.1)),
        },
        {
            "id": Column(kind="key"),
            "when": Column("when", "timestamp", missing=0.1),
            "value": Column("value"),
            "amount": Column("amount"),
        },
        time_column="when",
    )
    employees = SCM(
        {
            "joined": Node(),
            "level": Node(),
            "manager_level": Node((LinearEdge(Foreign("manager_id", "level")),), noise=None),
        },
        {
            "id": Column(kind="key"),
            "joined": Column("joined", "timestamp", marginal=DEFAULT_CALENDAR),
            "level": Column("level"),
            "manager_level": Column("manager_level"),
        },
        time_column="joined",
    )
    fkeys = (
        FK("orders", "customer_id", "customers"),
        FK("employees", "manager_id", "employees", TreeLink(roots=0.2), fill=0.0),
    )
    schema = Schema({"customers": customers, "orders": orders, "employees": employees}, fkeys)
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
    np.testing.assert_array_equal(tables["customers"]["n_orders"], np.bincount(keys, minlength=100))
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
