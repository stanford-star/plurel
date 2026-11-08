import errno
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from relbench.base import Database, Table
from relbench.base.table import is_time_sorted
from relbench.load import load_dataset
from relbench.manifest import DatasetManifest, TableSpec

from plurel.schema import Schema


def positional_keys(schema: Schema, frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Given key values as the row positions a database's keys are, within the frames as given."""
    out = dict(frames)
    for (table, column), fk in schema.keys.items():
        if schema.parent_keys(fk) is not None:
            keys = frames[table][column]
            parent = frames[fk.parent][schema.tables[fk.parent].pkey_column]
            indices = pd.Index(parent).get_indexer(keys)
            if ((indices < 0) & keys.notna().to_numpy()).any():
                raise ValueError(f"key {column!r} of {table!r} holds values its parent lacks")
            positions = pd.Series(indices, index=keys.index, dtype="Int64").mask(indices < 0)
            out[table] = out[table].assign(**{column: positions})
    for name, scm in schema.tables.items():
        if scm.pkey_column is not None and scm.columns[scm.pkey_column].values is not None:
            out[name] = out[name].assign(**{scm.pkey_column: np.arange(len(out[name]))})
    return out


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
    for (table, column), fk in schema.keys.items():
        if fk.parent in positions:
            keys = frames[table][column]
            known = keys.notna().to_numpy()
            moved = keys.copy()
            moved[known] = positions[fk.parent][keys[known].to_numpy(dtype=int)]
            frames[table] = frames[table].copy(deep=False)
            frames[table][column] = moved
    return frames


def create_database(schema: Schema, frames: Mapping[str, pd.DataFrame]) -> Database:
    if set(frames) != set(schema.tables):
        raise ValueError("one frame per table of the schema")
    if missing := {
        fk.parent for fk in schema.keys.values() if schema.tables[fk.parent].pkey_column is None
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
    frames = order_by_time(schema, positional_keys(schema, frames))
    tables = {
        name: Table(
            frame.reset_index(drop=True),
            {column: fk.parent for column, fk in schema.tables[name].fkeys.items()},
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
    val_timestamp: pd.Timestamp | None = None,
    test_timestamp: pd.Timestamp | None = None,
    description: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write every file into a hidden sibling of `path`, check it, then rename it into place.

    `val_timestamp` and `test_timestamp` give a temporal split, both or neither: a dataset saved
    without one, such as a static one, records them as `NaT`, no split, and loads all its rows.

    A symlinked `path` is written where it points. `path` holds a complete dataset or none. A
    failed or interrupted write keeps the previous dataset. A killed one leaves a hidden
    `.<name>.*.staging` sibling, or, if killed while replacing, the previous dataset in a
    `.<name>.*.retired` sibling. Of writers racing for one path, each either publishes a complete
    dataset or raises FileExistsError.
    """
    if (val_timestamp is None) != (test_timestamp is None):
        raise ValueError("give both val_timestamp and test_timestamp, or neither for no split")
    if val_timestamp is not None and val_timestamp > test_timestamp:
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
    path = Path(path).expanduser()
    target = path.resolve()
    if target == target.parent:
        raise ValueError(f"{path} is a filesystem root, which cannot hold a dataset")
    if holds_dataset(target):
        if not overwrite:
            raise FileExistsError(f"{path} already holds a dataset; pass overwrite=True")
    elif target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"{path} holds files but no dataset, so it is not replaced")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.staging")
    try:
        stage(db, staging, name, val_timestamp, test_timestamp, description)
        publish(staging, target, overwrite)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path


def holds_dataset(path: Path) -> bool:
    return (path / "manifest.yaml").is_file()


def stage(
    db: Database,
    staging: Path,
    name: str,
    val_timestamp: pd.Timestamp | None,
    test_timestamp: pd.Timestamp | None,
    description: str | None,
) -> None:
    (staging / "db").mkdir(parents=True)
    for table_name, table in db.table_dict.items():
        table.df.to_parquet(staging / "db" / f"{table_name}.parquet", index=False)
    manifest = DatasetManifest(
        name=name,
        val_timestamp=str(pd.Timestamp(val_timestamp)),
        test_timestamp=str(pd.Timestamp(test_timestamp)),
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
    manifest.save(staging / "manifest.yaml")
    saved = DatasetManifest.load(staging / "manifest.yaml")
    for table_name, table in db.table_dict.items():
        written = pq.read_metadata(staging / "db" / f"{table_name}.parquet")
        columns = [str(column) for column in table.df.columns]
        if (
            table_name not in saved.tables
            or written.num_rows != len(table.df)
            or written.schema.names != columns
        ):
            raise OSError(f"table {table_name!r} did not survive writing to {staging}")


def publish(staging: Path, path: Path, overwrite: bool) -> None:
    """Rename `staging` onto `path`, keeping a replaced dataset aside until the rename lands."""
    retired = staging.with_suffix(".retired")
    try:
        if overwrite and holds_dataset(path):
            path.rename(retired)
        elif path.is_dir():
            path.rmdir()
        staging.rename(path)
    except BaseException as error:
        taken = path.exists()
        if not taken and retired.exists():
            retired.rename(path)
        if taken and isinstance(error, OSError) and error.errno in (errno.EEXIST, errno.ENOTEMPTY):
            if holds_dataset(path):
                shutil.rmtree(retired, ignore_errors=True)
            raise FileExistsError(f"another writer took {path} first") from error
        raise
    shutil.rmtree(retired, ignore_errors=True)


def read_database(path: str | Path) -> Database:
    """The saved database, every row of it, whether or not it was saved with a split."""
    return load_dataset(Path(path).expanduser()).get_db(upto_test_timestamp=False)


def split_timestamps(
    db: Database, val_share: float = 0.1, test_share: float = 0.1
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Timestamps leaving `val_share` then `test_share` of the timestamped rows after them."""
    if min(val_share, test_share) <= 0.0 or val_share + test_share >= 1.0:
        raise ValueError("val_share and test_share must be positive and sum to less than one")
    times = [
        table.df[table.time_col].dropna() for table in db.table_dict.values() if table.time_col
    ]
    pooled = pd.concat(times, ignore_index=True) if times else pd.Series(dtype="datetime64[ns]")
    if pooled.empty:
        raise ValueError("a time split needs timestamped rows")
    return pooled.quantile(1.0 - val_share - test_share), pooled.quantile(1.0 - test_share)
