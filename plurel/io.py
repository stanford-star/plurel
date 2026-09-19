import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from relbench.base import Database, Table
from relbench.load import load_dataset
from relbench.manifest import DatasetManifest, TableSpec

from plurel.schema import Schema


def create_database(schema: Schema, frames: Mapping[str, pd.DataFrame]) -> Database:
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
        declared = {scm.pkey_column, scm.time_column} - {None}
        if not declared <= set(frame):
            raise ValueError(f"frame of {name!r} lacks its declared key or time column")
        if scm.time_column and not pd.api.types.is_datetime64_any_dtype(frame[scm.time_column]):
            raise ValueError(f"time column of {name!r} is not a datetime column")
        fkeys = {fk.column: fk.parent for fk in schema.fkeys if fk.table == name}
        table = Table(frame.reset_index(drop=True), fkeys, scm.pkey_column, scm.time_column)
        tables[name] = table
    return Database(tables)


def write_database(
    db: Database,
    path: str | Path,
    *,
    name: str,
    val_timestamp: pd.Timestamp,
    test_timestamp: pd.Timestamp,
    description: str | None = None,
    overwrite: bool = False,
) -> Path:
    if val_timestamp > test_timestamp:
        raise ValueError("val_timestamp must not be after test_timestamp")
    for table_name, table in db.table_dict.items():
        for column in table.df.columns[table.df.dtypes == "category"]:
            if not pd.api.types.is_string_dtype(table.df[column].cat.categories):
                raise ValueError(
                    f"categorical column {column!r} of {table_name!r} needs string categories to"
                    " survive parquet"
                )
    path = Path(path)
    if (path / "manifest.yaml").exists():
        if not overwrite:
            raise FileExistsError(f"{path} already holds a dataset; pass overwrite=True")
        shutil.rmtree(path / "db")
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


def read_database(path: str | Path) -> Database:
    return load_dataset(Path(path)).get_db(upto_test_timestamp=False)
