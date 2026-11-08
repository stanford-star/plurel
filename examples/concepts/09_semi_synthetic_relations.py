"""Retain real keys across two tables, then attach generated rows to their actual key values.

Sampling preserves external identifiers. A generated dependent has its own row count, while
retained parents and children keep theirs. Only export converts keys to row positions.
"""

import numpy as np
import pandas as pd

from plurel import Counts, ForeignKey, Numeric, Recorded, Schema, Table

customers = pd.DataFrame({"id": ["C-A", "C-B"], "discount": [0.05, 0.1]})
orders = pd.DataFrame({"id": ["O-X", "O-Y", "O-Z"], "customer_id": ["C-B", "C-A", "C-B"]})
schema = Schema(
    {
        "customers": Table.from_frame(customers, primary_key="id"),
        "orders": Table.from_frame(
            orders,
            primary_key="id",
            foreign_keys={"customer_id": "customers"},
            extra={"discount": Numeric(causes={Recorded("customer_id.discount"): 1}, noise=0)},
        ),
        "contacts": Table({"order_id": ForeignKey("orders", counts=Counts(2, least=1, most=3))}),
    }
)
frames = schema.sample(seed=0)
pd.testing.assert_frame_equal(frames["customers"], customers)
pd.testing.assert_frame_equal(frames["orders"][list(orders)], orders)
expected = orders.customer_id.map(customers.set_index("id").discount)
np.testing.assert_allclose(frames["orders"].discount, expected)
assert frames["contacts"].order_id.isin(orders.id).all()
counts = frames["contacts"].order_id.value_counts().reindex(orders.id, fill_value=0)
assert counts.between(1, 3).all()
print(frames["orders"])
print("semi_synthetic_relations: retained real keys and generated 1–3 contacts per order")
