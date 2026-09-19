import numpy as np
import pandas as pd
import pytest

from plurel import DEFAULT_CALENDAR, SCM, Column, Combine, LinearEffect, Normal, Root, Softmax
from plurel.io import database, read, write
from plurel.links import HSBMLink
from plurel.schema import FK, Port, Schema

ROWS = {"customers": 40, "orders": 300}
CUSTOMER_COLUMNS = {
    "segment": Column("segment", "categorical", categories=("a", "b", "c")),
    "spend": Column("spend"),
}
ORDER_COLUMNS = {
    "when": Column("when", marginal=DEFAULT_CALENDAR),
    "amount": Column("amount"),
    "value": Column("value"),
}


def customers(pkey="customer_id"):
    return SCM(
        {
            "segment": Softmax(biases=(0.0, 0.0, 0.0)),
            "spend": Port("orders", "amount", aggregate="sum"),
        },
        CUSTOMER_COLUMNS,
        pkey_column=pkey,
    )


def orders(pkey="order_id", time="when"):
    return SCM(
        {
            "when": Root(),
            "value": Port("customers", "spend", fill=np.nan),
            "amount": Combine((LinearEffect("when"),), noise=Normal(std=0.5)),
        },
        ORDER_COLUMNS,
        pkey_column=pkey,
        time_column=time,
    )


FKEYS = (FK("orders", "customer_id", "customers", HSBMLink((2,), (2,)), nullable=0.1),)


@pytest.fixture
def schema():
    return Schema({"customers": customers(), "orders": orders()}, FKEYS)


def test_tables_declare_their_keys_and_time():
    with pytest.raises(ValueError, match="Calendar"):
        orders(time="amount")
    with pytest.raises(ValueError, match="Calendar"):
        orders(time="nothing")
    with pytest.raises(ValueError, match="collides"):
        orders(pkey="amount")
    assert orders(pkey=None, time=None).pkey_column is None


def test_database_adds_primary_keys_and_relbench_metadata(schema):
    frames = schema.sample(ROWS, seed=0)
    db = database(schema, frames)
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
    pd.testing.assert_frame_equal(order_table.df.drop(columns="order_id"), frames["orders"])
    events = Schema({"customers": customers(), "orders": orders(pkey=None)}, FKEYS)
    event_table = database(events, events.sample(ROWS, seed=0)).table_dict["orders"]
    assert event_table.pkey_col is None and "order_id" not in event_table.df
    unkeyed = Schema({"customers": customers(pkey=None), "orders": orders()}, FKEYS)
    with pytest.raises(ValueError, match="primary key"):
        database(unkeyed, unkeyed.sample(ROWS, seed=0))
    clashing = Schema({"customers": customers(), "orders": orders(pkey="customer_id")}, FKEYS)
    with pytest.raises(ValueError, match="already"):
        database(clashing, clashing.sample(ROWS, seed=0))
    with pytest.raises(ValueError):
        database(schema, {"customers": frames["customers"]})


def test_write_and_read_round_trip(schema, tmp_path):
    db = database(schema, schema.sample(ROWS, seed=0))
    split = db.min_timestamp + (db.max_timestamp - db.min_timestamp) * 0.7
    path = write(
        db,
        tmp_path / "synthetic",
        name="synthetic",
        val_timestamp=split,
        test_timestamp=db.max_timestamp,
    )
    assert (path / "manifest.yaml").exists()
    assert {p.name for p in (path / "db").iterdir()} == {"customers.parquet", "orders.parquet"}
    loaded = read(path)
    for name, table in db.table_dict.items():
        again = loaded.table_dict[name]
        assert (again.pkey_col, again.time_col) == (table.pkey_col, table.time_col)
        assert again.fkey_col_to_pkey_table == table.fkey_col_to_pkey_table
        pd.testing.assert_frame_equal(again.df, table.df, check_categorical=False)
    order_keys = loaded.table_dict["orders"].df["customer_id"]
    assert order_keys.isna().sum() == db.table_dict["orders"].df["customer_id"].isna().sum()
