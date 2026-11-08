"""Export frames to a static relational database without inventing timestamps.

The relbench representation uses positional keys. Compare round trips with create_database's
output, not the original external identifiers. describe() is inspectable metadata, not a loader
that reconstructs executable generators or arbitrary callables.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

import json
import tempfile
from pathlib import Path

import pandas as pd

from plurel import ForeignKey, Schema, Table, create_database, read_database, write_database

people = pd.DataFrame({"id": ["A-100", "B-200"], "age": [25, 40]})
schema = Schema(
    {
        "people": Table.from_frame(people, primary_key="id"),
        "visits": Table({"person_id": ForeignKey("people")}),
    }
)
frames = schema.sample({"visits": 10}, seed=0)
assert frames["visits"].person_id.isin(people.id).all()
db = create_database(schema, frames)
assert db.table_dict["people"].df.id.tolist() == [0, 1]
assert db.table_dict["visits"].df.person_id.isin([0, 1]).all()
assert frames["people"].id.tolist() == people.id.tolist(), "export keeps the sampled frames intact"

with tempfile.TemporaryDirectory() as folder:
    path = write_database(db, Path(folder) / "static", name="static")
    restored = read_database(path)
    # A caller may save this description for inspection; it does not serialize the generator.
    Path(folder, "description.json").write_text(json.dumps(schema.describe(), indent=2))
    try:
        write_database(db, path, name="static")
    except FileExistsError:
        print("Existing output is protected; replacing it requires overwrite=True")
    else:
        raise AssertionError("existing output should be protected")
for name, original in db.table_dict.items():
    pd.testing.assert_frame_equal(original.df, restored.table_dict[name].df)
    assert original.fkey_col_to_pkey_table == restored.table_dict[name].fkey_col_to_pkey_table
print("static_export: positional keys round-trip without a temporal split")
