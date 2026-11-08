"""Calibration itself must respect the schema's relational cardinality constraints.

Default calibration sizes cannot infer every per-group capacity. Give feasible calibration
rows when needed; subsequent sampling still gets its own row counts and constraints.
"""

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

import pandas as pd

from plurel import Categorical, ForeignKey, Schema, Table

schema = Schema(
    {
        "rooms": Table.from_frame(pd.DataFrame({"id": ["A", "B", "C"]}), primary_key="id"),
        "meetings": Table(
            {
                "slot": Categorical(("morning", "afternoon")),
                "room_id": ForeignKey("rooms", unique="slot"),
                "kind": Categorical(("workshop", "lecture")),
            }
        ),
    }
)
# Three rooms across two slots cannot hold the default thousands of calibration meetings.
try:
    schema.realize(seed=0)
except ValueError as error:
    assert "realize(seed, rows)" in str(error)
    print("Expected calibration failure:", error)
else:
    raise AssertionError("an infeasible calibration draw should fail")

generator = schema.realize(seed=0, rows={"meetings": 4})
frames = generator.sample({"meetings": 2}, seed=10)  # calibration did not fix the output size
assert len(frames["meetings"]) == 2
assert not frames["meetings"].duplicated(["slot", "room_id"]).any()
print(frames["meetings"])
print("calibration_sizes: feasible calibration rows and sampling rows are separate choices")
