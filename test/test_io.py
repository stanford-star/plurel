import numpy as np
import pandas as pd
import pytest

from plurel import DEFAULT_CALENDAR, SCM, Column, Combine, LinearEffect, Normal, Root, Softmax
from plurel.io import database, read, write
from plurel.links import HSBMLink
from plurel.schema import FK, Port, Schema

ROWS = {"customers": 40, "orders": 300}
PKEYS = {"customers": "customer_id", "orders": "order_id"}


@pytest.fixture
def schema():
    customers = SCM(
        {
            "segment": Softmax(biases=(0.0, 0.0, 0.0)),
            "spend": Port("orders", "amount", aggregate="sum"),
        },
        {
            "segment": Column("segment", "categorical", categories=("a", "b", "c")),
            "spend": Column("spend"),
        },
    )
    orders = SCM(
        {
            "when": Root(),
            "value": Port("customers", "spend", fill=np.nan),
            "amount": Combine((LinearEffect("when"),), noise=Normal(std=0.5)),
        },
        {
            "when": Column("when", marginal=DEFAULT_CALENDAR),
            "amount": Column("amount"),
            "value": Column("value"),
        },
    )
    return Schema(
        {"customers": customers, "orders": orders},
        (FK("orders", "customer_id", "customers", HSBMLink((2,), (2,)), nullable=0.1),),
    )


def test_database_adds_primary_keys_and_relbench_metadata(schema):
    frames = schema.sample(ROWS, seed=0)
    db = database(schema, frames, PKEYS, times={"orders": "when"})
    assert set(db.table_dict) == set(ROWS)
    customers, orders = db.table_dict["customers"], db.table_dict["orders"]
    assert customers.pkey_col == "customer_id" and customers.time_col is None
    assert customers.fkey_col_to_pkey_table == {}
    assert orders.pkey_col == "order_id" and orders.time_col == "when"
    assert orders.fkey_col_to_pkey_table == {"customer_id": "customers"}
    np.testing.assert_array_equal(customers.df["customer_id"], np.arange(ROWS["customers"]))
    linked = orders.df["customer_id"].dropna().astype(int)
    assert linked.isin(customers.df["customer_id"]).all() and orders.df["customer_id"].isna().any()
    assert db.min_timestamp == orders.df["when"].min()
    assert db.max_timestamp == orders.df["when"].max()
    pd.testing.assert_frame_equal(orders.df.drop(columns="order_id"), frames["orders"])
    events = database(schema, frames, {"customers": "customer_id"})
    assert events.table_dict["orders"].pkey_col is None
    assert "order_id" not in events.table_dict["orders"].df
    for bad in (
        {"pkeys": PKEYS, "times": {"orders": "amount"}},
        {"pkeys": PKEYS, "times": {"shops": "when"}},
        {"pkeys": {"customers": "spend", "orders": "order_id"}},
        {"pkeys": {"orders": "order_id"}},
    ):
        with pytest.raises(ValueError):
            database(schema, frames, **bad)
    with pytest.raises(ValueError):
        database(schema, {"customers": frames["customers"]}, PKEYS)


def test_write_and_read_round_trip(schema, tmp_path):
    db = database(schema, schema.sample(ROWS, seed=0), PKEYS, times={"orders": "when"})
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
    orders = loaded.table_dict["orders"].df["customer_id"]
    assert orders.isna().sum() == db.table_dict["orders"].df["customer_id"].isna().sum()
