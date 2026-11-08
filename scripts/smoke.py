"""Check an installed plurel from outside the source checkout: the public API imports without
relbench, a small database generates, and it saves and loads back through relbench."""

import sys
import tempfile
from pathlib import Path

import plurel

if Path(plurel.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "plurel"):
    sys.exit(f"plurel imported from the checkout at {plurel.__file__}, not the installed wheel")
if "relbench" in sys.modules:
    sys.exit("importing plurel loaded relbench, which only the database functions need")
missing = [name for name in plurel.__all__ if not hasattr(plurel, name)]
if missing:
    sys.exit(f"public names missing: {missing}")

from plurel import (  # noqa: E402
    Categorical,
    ForeignKey,
    Numeric,
    PrimaryKey,
    Schema,
    Table,
    Timestamp,
    Uniform,
    create_database,
    read_database,
    split_timestamps,
    write_database,
)

schema = Schema(
    {
        "customers": Table({"id": PrimaryKey(), "segment": Categorical(("a", "b"))}),
        "orders": Table(
            {
                "customer_id": ForeignKey("customers"),
                "when": Timestamp("2020-01-01", "2021-01-01"),
                "amount": Numeric(Uniform(1, 100), causes={"customer_id.segment": {"b": 1.0}}),
            },
            time_column="when",
        ),
    }
)
db = create_database(schema, schema.sample({"customers": 20, "orders": 200}, seed=0))
val, test = split_timestamps(db)
with tempfile.TemporaryDirectory() as scratch:
    path = write_database(
        db, Path(scratch) / "shop", name="shop", val_timestamp=val, test_timestamp=test
    )
    loaded = read_database(path)
    if len(loaded.table_dict["orders"].df) != 200:
        sys.exit("the saved database did not load back whole")
print(f"plurel {plurel.__file__} works without the checkout")
