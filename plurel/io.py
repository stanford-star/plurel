from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from relbench.base import Database, Table
from relbench.load import load_dataset
from relbench.manifest import DatasetManifest, TableSpec

from plurel.schema import Schema


def database(schema: Schema, frames: Mapping[str, pd.DataFrame]) -> Database:
    if set(frames) != set(schema.tables):
        raise ValueError("one frame per table of the schema")
    if missing := {
        fk.parent for fk in schema.fkeys if schema.tables[fk.parent].pkey_column is None
    }:
        raise ValueError(
            f"tables referenced by a foreign key need a primary key: {sorted(missing)}"
        )
    tables = {}
    for name, frame in frames.items():
        scm = schema.tables[name]
        fkeys = {fk.column: fk.parent for fk in schema.fkeys if fk.table == name}
        table = Table(frame.reset_index(drop=True), fkeys, scm.pkey_column, scm.time_column)
        tables[name] = table
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
