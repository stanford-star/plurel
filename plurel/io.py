from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.base import Database, Table
from relbench.load import load_dataset
from relbench.manifest import DatasetManifest, TableSpec

from plurel.schema import Schema


def database(
    schema: Schema,
    frames: Mapping[str, pd.DataFrame],
    pkeys: Mapping[str, str],
    times: Mapping[str, str] | None = None,
) -> Database:
    pkeys, times = dict(pkeys), dict(times or {})
    if set(frames) != set(schema.tables):
        raise ValueError("one frame per table of the schema")
    if unknown := (set(pkeys) | set(times)) - set(schema.tables):
        raise ValueError(f"keys or time columns for unknown tables {sorted(unknown)}")
    if missing := {fk.parent for fk in schema.fkeys} - set(pkeys):
        raise ValueError(
            f"tables referenced by a foreign key need a primary key: {sorted(missing)}"
        )
    tables = {}
    for name, frame in frames.items():
        pkey, time = pkeys.get(name), times.get(name)
        if pkey in frame:
            raise ValueError(f"table {name!r} already has a column {pkey!r}")
        if time is not None and not pd.api.types.is_datetime64_any_dtype(frame.get(time)):
            raise ValueError(f"time column {time!r} of {name!r} must be a datetime column")
        df = frame.reset_index(drop=True)
        if pkey is not None:
            df = pd.concat([pd.Series(np.arange(len(frame)), name=pkey), df], axis=1)
        fkeys = {fk.column: fk.parent for fk in schema.fkeys if fk.table == name}
        tables[name] = Table(df, fkeys, pkey_col=pkey, time_col=time)
    return Database(tables)


def write(
    db: Database,
    path: str | Path,
    *,
    name: str,
    val_timestamp: pd.Timestamp,
    test_timestamp: pd.Timestamp,
    description: str | None = None,
) -> Path:
    path = Path(path)
    (path / "db").mkdir(parents=True, exist_ok=True)
    for table_name, table in db.table_dict.items():
        table.df.to_parquet(path / "db" / f"{table_name}.parquet", index=False)
    manifest = DatasetManifest(
        name=name,
        val_timestamp=str(val_timestamp),
        test_timestamp=str(test_timestamp),
        description=description,
        tables={
            table_name: TableSpec(
                pkey=table.pkey_col,
                time_col=table.time_col,
                fkeys=dict(table.fkey_col_to_pkey_table),
            )
            for table_name, table in db.table_dict.items()
        },
    )
    manifest.save(path / "manifest.yaml")
    return path


def read(path: str | Path) -> Database:
    return load_dataset(Path(path)).get_db(upto_test_timestamp=False)
