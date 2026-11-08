"""Build event times with explicit causal delays, then choose a temporal export split.

A bounded timestamp uses a calendar marginal. A spanless timestamp reads recorded epoch
seconds plus noise; a positive delay ensures ordering. A positive latent weight alone does not.
"""

import tempfile
from pathlib import Path

import pandas as pd

from plurel import (
    Exponential,
    ForeignKey,
    PrimaryKey,
    Recorded,
    Schema,
    Table,
    Timestamp,
    create_database,
    read_database,
    split_timestamps,
    write_database,
)

schema = Schema(
    {
        "customers": Table(
            {"id": PrimaryKey(), "joined": Timestamp("2020-01-01", "2021-01-01")},
            time_column="joined",
        ),
        "orders": Table(
            {
                "id": PrimaryKey(),
                "customer_id": ForeignKey("customers"),
                "placed": Timestamp(
                    causes={Recorded("customer_id.joined"): 1}, noise=Exponential(30 * 86400)
                ),
                "shipped": Timestamp(causes={Recorded("placed"): 1}, noise=Exponential(2 * 86400)),
            },
            time_column="placed",
        ),
    }
)
frames = schema.sample({"customers": 20, "orders": 200}, seed=0)
joined = frames["orders"].customer_id.map(frames["customers"].set_index("id").joined)
assert (frames["orders"].placed >= joined).all()
assert (frames["orders"].shipped >= frames["orders"].placed).all()
db = create_database(schema, frames)  # sort event tables and remap positional keys together
val_timestamp, test_timestamp = split_timestamps(db)
with tempfile.TemporaryDirectory() as folder:
    path = write_database(
        db,
        Path(folder) / "events",
        name="events",
        val_timestamp=val_timestamp,
        test_timestamp=test_timestamp,
    )
    restored = read_database(path)  # loads the full database, including rows after the split
for name, original in db.table_dict.items():
    pd.testing.assert_frame_equal(original.df, restored.table_dict[name].df)
print("Split timestamps:", val_timestamp, test_timestamp)
print("temporal_data: positive delays preserve event order through temporal export")
