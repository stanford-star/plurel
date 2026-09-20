import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.base import Database, Table
from relbench.base.table import is_time_sorted
from relbench.load import load_dataset
from relbench.manifest import DatasetManifest, TableSpec

from plurel.schema import Schema


def order_by_time(schema: Schema, frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Put temporal tables in time order with keys as row positions, the order a time cut keeps."""
    frames, positions = dict(frames), {}
    for name, scm in schema.tables.items():
        if scm.time_column is None:
            continue
        order = np.argsort(frames[name][scm.time_column].to_numpy(), kind="stable")
        frames[name] = frames[name].iloc[order].reset_index(drop=True)
        if scm.pkey_column is not None:
            frames[name][scm.pkey_column] = np.arange(len(order))
        positions[name] = np.argsort(order)
    for fk in schema.fkeys:
        if fk.parent in positions:
            keys = frames[fk.table][fk.column]
            known = keys.notna().to_numpy()
            moved = keys.copy()
            moved[known] = positions[fk.parent][keys[known].to_numpy(dtype=int)]
            frames[fk.table] = frames[fk.table].assign(**{fk.column: moved})
    return frames


def create_database(schema: Schema, frames: Mapping[str, pd.DataFrame]) -> Database:
    if set(frames) != set(schema.tables):
        raise ValueError("one frame per table of the schema")
    if missing := {
        fk.parent for fk in schema.fkeys if schema.tables[fk.parent].pkey_column is None
    }:
        raise ValueError(
            f"tables referenced by a foreign key need a primary key: {sorted(missing)}"
        )
    for name, frame in frames.items():
        scm = schema.tables[name]
        declared = {scm.pkey_column, scm.time_column} - {None}
        if not declared <= set(frame):
            raise ValueError(f"frame of {name!r} lacks its declared key or time column")
        if scm.time_column and not pd.api.types.is_datetime64_any_dtype(frame[scm.time_column]):
            raise ValueError(f"time column of {name!r} is not a datetime column")
    frames = order_by_time(schema, frames)
    tables = {
        name: Table(
            frame.reset_index(drop=True),
            {fk.column: fk.parent for fk in schema.fkeys if fk.table == name},
            schema.tables[name].pkey_column,
            schema.tables[name].time_column,
        )
        for name, frame in frames.items()
    }
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
        if table.time_col and not is_time_sorted(table.df[table.time_col]):
            raise ValueError(f"rows of {table_name!r} must be in time order, missing times last")
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
        shutil.rmtree(path / "db", ignore_errors=True)
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
