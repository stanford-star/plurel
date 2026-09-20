import numpy as np
import pandas as pd
import pytest

from plurel import DEFAULT_CALENDAR, SCM, Column, Combine, LinearEffect, Normal, Root, Softmax
from plurel.io import create_database, read_database, write_database
from plurel.links import HSBMLink
from plurel.schema import FK, Port, Schema

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
            "segment": Softmax(biases=(0.0, 0.0, 0.0)),
            "spend": Port("orders", "amount", aggregate="sum"),
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
            "when": Root(),
            "value": Port("customers", "spend", fill=np.nan),
            "amount": Combine((LinearEffect("when"),), noise=Normal(std=0.5)),
        },
        columns,
        time_column=time_column,
    )


FKEYS = (FK("orders", "customer_id", "customers", HSBMLink((2,), (2,)), nullable=0.1),)


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
        SCM({"x": Root()}, {"a": Column(kind="key"), "b": Column(kind="key")})


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
    pd.testing.assert_frame_equal(order_table.df, frames["orders"])
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
    shuffled = frames["orders"].sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="time order"):
        create_database(schema, {**frames, "orders": shuffled})
    gap = frames["orders"].copy()
    gap.loc[0, "when"] = pd.NaT
    with pytest.raises(ValueError, match="time order"):
        create_database(schema, {**frames, "orders": gap})


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
