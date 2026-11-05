# An authored table, end to end

A small catalog with one schema: an item brought to a volunteer repair cafe, and how the visit ended.

```bash
python examples/world/generate.py
```

| Path | What it is |
|---|---|
| `plans/community_services.yml` | The sector's plan: its scope, its tree and the records to author |
| `catalog/<sector>/<division>/<group>/item_visits.py` | The schema, placed by its path |
| `catalog/__init__.py` | Loads the catalog against its plans |
| `world_knowledge/community_services.json` | The text written for the schema, keyed by its id |
| `generate.py` | Checks the schema, draws rows, sets an intervention and lints the text |

The outcome's shares follow the [2022 RepairMonitor report](https://www.repaircafe.org/wp-content/uploads/2023/04/Factsheet_RepairMonitor_2022_EN.pdf) of the Repair Cafe International Foundation. The columns and their dependencies are illustrative.
