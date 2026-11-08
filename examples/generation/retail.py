"""Static retail orders and their lines, around retained customers and products.

The mechanisms are authored, not fit: the dataset shows how the declarations compose, not how real
shops behave. Run it with `python examples/generation/retail.py`; `--scale` grows the customers.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from plurel import (
    Categorical,
    Counts,
    Derived,
    ForeignKey,
    Numeric,
    PrimaryKey,
    Range,
    Recorded,
    Schema,
    Summary,
    Table,
    Uniform,
    create_database,
    read_database,
    write_database,
)


def fixtures(scale: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small synthetic stand-ins for a retained customer table and a price list."""
    rng = np.random.default_rng(0)
    n = 40 * scale
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "segment": rng.choice(["consumer", "business"], n, p=[0.8, 0.2]),
            "tenure_years": rng.integers(0, 15, n).astype(float),
        }
    )
    products = pd.DataFrame(
        {
            "product_id": [f"P{i:03d}" for i in range(24)],
            "price": np.round(rng.uniform(2, 200, 24), 2),
        }
    )
    return customers, products


def build(customers: pd.DataFrame, products: pd.DataFrame) -> Schema:
    customer_table = Table.from_frame(
        customers,
        primary_key="customer_id",
        doc="one retained customer per row",
        docs={"tenure_years": "whole years since the first purchase"},
    )
    product_table = Table.from_frame(
        products, primary_key="product_id", docs={"price": "list price in dollars"}
    )
    orders = Table(
        {
            "order_id": PrimaryKey(),
            # how many orders a customer places: about 2.5, more with tenure (a latent effect,
            # so a weight, not orders per year), at most 10; realizing draws the mean
            "customer_id": ForeignKey(
                "customers",
                counts=Counts(Range(2.0, 3.0), causes={"tenure_years": 0.5}, most=10),
            ),
            "channel": Categorical({"web": 0.6, "store": 0.3, "phone": 0.1}),
            # exact: the sum of the recorded line totals, as a groupby would give
            "total": Numeric(
                causes={Summary("lines", "order_id", Recorded("line_total"), "sum"): 1.0},
                noise=0,
                doc="sum of the order's line totals",
            ),
            # the target reads the total's latent: larger orders are returned more often
            "returned": Categorical({"no": 0.9, "yes": 0.1}, causes={"total": 1.0}),
        },
        name="orders",
        doc="one order per row, a static snapshot",
        target="returned",
    )
    lines = Table(
        {
            # every order has 1 to 8 lines
            "order_id": ForeignKey("orders", counts=Counts(3.0, least=1, most=8)),
            # no product twice in one order, cheaper ones preferred
            "product_id": ForeignKey(
                "products", unique="order_id", prefer={Recorded("price"): -0.01}
            ),
            "quantity": Numeric(Uniform(1, 5), round=1),
            # exact: the product's recorded price, copied in dollars
            "unit_price": Numeric(causes={Recorded("product_id.price"): 1.0}, noise=0),
            "line_total": Derived(lambda unit_price, quantity: unit_price * quantity),
        },
        name="lines",
        doc="one order line per row",
    )
    return Schema(
        {"customers": customer_table, "products": product_table, "orders": orders, "lines": lines}
    )


def check(frames: dict[str, pd.DataFrame], customers: pd.DataFrame, products: pd.DataFrame) -> None:
    """The invariants the declarations promise."""
    orders, lines = frames["orders"], frames["lines"]
    assert frames["customers"].equals(customers) and frames["products"].equals(products)
    assert orders.customer_id.isin(customers.customer_id).all()
    assert (
        lines.order_id.isin(orders.order_id).all()
        and lines.product_id.isin(products.product_id).all()
    )
    per_order = lines.order_id.value_counts().reindex(orders.order_id, fill_value=0)
    assert per_order.between(1, 8).all(), "every order has 1 to 8 lines"
    assert not lines.duplicated(["order_id", "product_id"]).any(), "no product twice in an order"
    priced = lines.merge(products, on="product_id")
    np.testing.assert_array_equal(priced.unit_price, priced.price)
    np.testing.assert_allclose(lines.line_total, np.round(lines.unit_price * lines.quantity, 6))
    totals = lines.groupby("order_id").line_total.sum().reindex(orders.order_id)
    np.testing.assert_allclose(orders.total, totals.to_numpy(), rtol=1e-12)
    assert set(orders.returned) <= {"no", "yes"}


def main(mechanism_seed: int, data_seed: int, scale: int, out: Path | None) -> None:
    customers, products = fixtures(scale)
    schema = build(customers, products)
    generator = schema.realize(mechanism_seed)
    print(generator.explain())
    # the retained tables keep their rows and the counted ones take what their counts give
    frames = generator.sample(seed=data_seed)
    check(frames, customers, products)
    db = create_database(generator, frames)
    with tempfile.TemporaryDirectory() as scratch:
        path = write_database(db, (out or Path(scratch)) / "retail", name="retail", overwrite=True)
        loaded = read_database(path)
    for name, table in db.table_dict.items():
        assert len(loaded.table_dict[name].df) == len(table.df)
        assert loaded.table_dict[name].fkey_col_to_pkey_table == table.fkey_col_to_pkey_table
    sizes = {name: len(frame) for name, frame in frames.items()}
    print(f"retail: {sizes}; every invariant holds and the database round-trips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mechanism-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--out", type=Path, help="keep the saved database here")
    args = parser.parse_args()
    main(args.mechanism_seed, args.data_seed, args.scale, args.out)
