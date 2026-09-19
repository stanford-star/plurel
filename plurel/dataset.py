from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from relbench.base import Database, Dataset, Table
from relbench.load import load_dataset
from relbench.manifest import DatasetManifest, TableSpec

from plurel.config import Config
from plurel.schema import RandomSchemaGraphBuilder, SQLSchemaGraphBuilder
from plurel.scm import SCM
from plurel.utils import TableType, set_random_seed


def _standardize(x):
    return (x - x.mean()) / max(x.std(), 1e-8)


def _bounded_z(x, max_abs: float = 10.0):
    return np.clip(_standardize(x), -max_abs, max_abs)


def _threshold(x):
    z = _standardize(x)
    return np.maximum(z - np.quantile(z, np.random.uniform(0.4, 0.95)), 0)


def _square(x):
    z = _bounded_z(x)
    return np.sign(z) * np.square(z)


def _expm1(x):
    z = _bounded_z(x)
    return np.sign(z) * np.expm1(np.abs(z))


def _extreme_outlier(x):
    """Standardize, then scale a random 0.5-2% of rows by 15-30x."""
    z = _standardize(x)
    n = len(z)
    n_extreme = max(1, int(n * np.random.uniform(0.005, 0.02)))
    idx = np.random.choice(n, n_extreme, replace=False)
    factor = np.random.uniform(15.0, 30.0)
    out = z.copy()
    out[idx] = out[idx] * factor
    return out


COLUMN_TRANSFORM_REGISTRY: dict[str, callable] = {
    "identity": lambda x: x,
    "rank_uniform": lambda x: (np.argsort(np.argsort(x)) + 1) / (len(x) + 1),
    "log": lambda x: np.sign(x) * np.log1p(np.abs(x)),
    "sqrt": lambda x: np.sign(x) * np.sqrt(np.abs(x)),
    "standardize": _standardize,
    "threshold": _threshold,
    "square": _square,
    "expm1": _expm1,
    "extreme_outlier": _extreme_outlier,
}


class SyntheticDataset(Dataset):
    """
    A synthetic relbench compatible dataset.

    Args:
        seed (int): Random seed for reproducibility of dataset generation.
        config (Config): Config object with constants and sampling choices.
    """

    def __init__(self, seed: int, config: Config):
        self.seed = seed
        self.config = config
        set_random_seed(self.seed)
        self.initialize_timestamps()
        self.cache_dir = self.config.cache_dir

    def initialize_timestamps(self):
        start_timestamp = self.config.database_params.min_timestamp
        end_timestamp = self.config.database_params.max_timestamp
        total_days = (end_timestamp - start_timestamp).days

        days = np.sort(np.random.choice(total_days, 2, replace=False))
        self.min_timestamp = start_timestamp + pd.Timedelta(days=int(days[0]))
        self.max_timestamp = start_timestamp + pd.Timedelta(days=int(days[1]))

        timestamps = pd.date_range(start=self.min_timestamp, end=self.max_timestamp)
        val_start_idx = int(len(timestamps) * self.config.val_split)
        test_start_idx = int(len(timestamps) * self.config.test_split)
        self.val_timestamp = timestamps[val_start_idx]
        self.test_timestamp = timestamps[test_start_idx]

    def get_db(self, upto_test_timestamp=True) -> Database:
        """Build the database, or load it from ``cache_dir`` if already generated.

        With a ``cache_dir`` the dataset is written directly in relbench-3.0.0
        format: a self-describing dir with manifest.yaml next to
        db/<table>.parquet, loadable with ``relbench.load.load_dataset``.
        """
        if self.cache_dir is None:
            return super().get_db(upto_test_timestamp)

        cache_dir = Path(self.cache_dir).expanduser()
        if (cache_dir / "manifest.yaml").exists():
            return load_dataset(cache_dir).get_db(upto_test_timestamp)

        db = super().get_db(upto_test_timestamp)
        db_dir = cache_dir / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        for table_name, table in db.table_dict.items():
            table.df.to_parquet(db_dir / f"{table_name}.parquet", index=False)
        self.write_manifest(db)
        return db

    def write_manifest(self, db: Database, name: str | None = None):
        cache_dir = Path(self.cache_dir).expanduser()
        manifest_path = cache_dir / "manifest.yaml"
        if manifest_path.exists():
            return manifest_path

        manifest = DatasetManifest(
            name=name or cache_dir.name,
            val_timestamp=str(self.val_timestamp),
            test_timestamp=str(self.test_timestamp),
            description=f"PluRel synthetic relational database (seed {self.seed}).",
            tables={
                table_name: TableSpec(
                    pkey=table.pkey_col,
                    time_col=table.time_col,
                    fkeys=dict(table.fkey_col_to_pkey_table),
                )
                for table_name, table in db.table_dict.items()
            },
        )
        manifest.save(manifest_path)
        return manifest_path

    def _get_random_dag_table_relationships(self, num_tables: int):
        """
        Each table will have the following attributes:
        ```py
        {
            "columns":dict[col_name -> {
                "stype": stype,
                "categories": list[str] | None
            }],
            "pkey_col": str | None,
            "fkey_col_to_pkey_table": dict[str, str],
        }
        ```
        """
        builder = RandomSchemaGraphBuilder(
            config=self.config, num_tables=num_tables, seed=self.seed
        )
        return builder.build_graph()

    def _get_schema_file_table_relationships(self, schema_file: str):
        builder = SQLSchemaGraphBuilder(sql_file=schema_file)
        builder.load_schema()
        return builder.build_graph()

    def configure_table_relationships(
        self, num_tables: int | None = None, schema_file: str | None = None
    ) -> nx.DiGraph:
        """
        Define the primary -> foreign key relationships between tables as a DAG.

        Args:
            num_tables (int): Number of tables for the layout based on random DAGs.
            schema_file (str): A predefined SQL based schema file.
        """
        assert num_tables or schema_file, "either `num_tables` or `schema_file` must not be None"
        # higher preference to schema file
        if schema_file:
            table_relationships = self._get_schema_file_table_relationships(schema_file=schema_file)
        elif num_tables:
            table_relationships = self._get_random_dag_table_relationships(num_tables=num_tables)

        if table_relationships.number_of_nodes() > 1:
            activity_tables = [
                table
                for table in table_relationships.nodes
                if table_relationships.out_degree(table) == 0
            ]
        else:
            activity_tables = []
        for table in table_relationships.nodes:
            if table in activity_tables:
                table_type = TableType.Activity
            else:
                table_type = TableType.Entity

            table_relationships.nodes[table]["type"] = table_type
            table_relationships.nodes[table]["num_rows"] = self.get_num_rows(table_type=table_type)
        return table_relationships

    def get_num_rows(self, table_type):
        if table_type == TableType.Entity:
            num_rows = self.config.database_params.num_rows_entity_table_choices.sample_uniform()
        elif table_type == TableType.Activity:
            num_rows = self.config.database_params.num_rows_activity_table_choices.sample_uniform()
        return num_rows

    def apply_col_transforms(self, df, pkey_col, fkey_cols):
        for col_name, _type in df.dtypes.items():
            if col_name not in [pkey_col, *fkey_cols, "date"] and _type in [float]:
                col_transform_name = (
                    self.config.database_params.col_transform_choices.sample_uniform()
                )
                col_transform = COLUMN_TRANSFORM_REGISTRY[col_transform_name]
                df[col_name] = col_transform(df[col_name].values).astype(float)
        return df

    def apply_zero_inflation(self, df, pkey_col, fkey_cols):
        """Clip float columns below a sampled quantile to zero."""
        prob = self.config.database_params.zero_inflation_col_prob
        if prob <= 0.0:
            return df
        for col_name, _type in df.dtypes.items():
            if col_name in [pkey_col, *fkey_cols, "date"] or _type not in [float]:
                continue
            if np.random.rand() >= prob:
                continue
            q = float(self.config.database_params.zero_inflation_quantile_choices.sample_uniform())
            values = df[col_name].values
            threshold = np.quantile(values, q)
            df[col_name] = np.maximum(values - threshold, 0).astype(float)
        return df

    def implant_nan(self, df, pkey_col, fkey_cols):
        nan_perc = self.config.database_params.column_nan_perc_choices.sample_uniform()
        num_nan_cells = int(np.floor(nan_perc * len(df)))
        for col_name, _type in df.dtypes.items():
            if col_name not in [pkey_col, *fkey_cols] and _type in [float]:
                nan_cells_idx = np.random.choice(df.index, size=num_nan_cells, replace=False)
                df.loc[nan_cells_idx, col_name] = np.nan
        return df

    def binarize_columns(self, df, feature_columns, pkey_col, fkey_cols):
        """Collapse int categoricals to ``value > 0`` and threshold float
        columns at a sampled quantile; runs before ``implant_nan``."""
        binarize = self.config.scm_params.binarize_int_categoricals
        bool_prob = self.config.database_params.bool_col_prob
        for col_name, _type in df.dtypes.items():
            if col_name in [pkey_col, *fkey_cols]:
                continue
            if _type in [int] and binarize:
                categories = feature_columns[col_name]["categories"]
                if not categories or type(categories[0]) != int:
                    continue
                df[col_name] = (df[col_name] > 0).astype(bool)
                if len(df[col_name].unique()) == 1:
                    df.drop(columns=[col_name], inplace=True)
            elif _type in [float] and bool_prob > 0.0 and np.random.rand() < bool_prob:
                q = float(
                    self.config.database_params.bool_threshold_quantile_choices.sample_uniform()
                )
                values = df[col_name].values
                threshold = np.quantile(values, q)
                df[col_name] = (values > threshold).astype(bool)
        return df

    def materialize_string_categoricals(self, df, feature_columns, pkey_col, fkey_cols):
        """Map int-coded categorical columns to their string labels."""
        for col_name, _type in df.dtypes.items():
            if col_name in [pkey_col, *fkey_cols] or _type not in [int]:
                continue
            categories = feature_columns[col_name]["categories"]
            if not categories or type(categories[0]) != str:
                continue
            df[col_name] = df[col_name].map(lambda i: categories[i])
        return df

    def make_db(self) -> Database:
        set_random_seed(self.seed)
        num_tables = self.config.database_params.num_tables_choices.sample_uniform()
        self.table_relationships = self.configure_table_relationships(
            num_tables=num_tables, schema_file=self.config.schema_file
        )
        topological_gens = list(nx.topological_generations(self.table_relationships))
        table_name_to_scm = {}

        total_gens = len(topological_gens)
        for gen_idx, gen in enumerate(topological_gens):
            for table_id in gen:
                table_type = self.table_relationships.nodes[table_id]["type"]
                table_name = self.table_relationships.nodes[table_id]["name"]
                num_rows = self.table_relationships.nodes[table_id]["num_rows"]
                columns = self.table_relationships.nodes[table_id]["columns"]
                pkey_col = self.table_relationships.nodes[table_id]["pkey_col"]
                fkey_col_to_pkey_table = self.table_relationships.nodes[table_id][
                    "fkey_col_to_pkey_table"
                ]
                feature_columns = {
                    col: col_info
                    for col, col_info in columns.items()
                    if col != pkey_col and col not in fkey_col_to_pkey_table.keys()
                }

                child_table_names = []
                for child_table_id in sorted(list(self.table_relationships.successors(table_id))):
                    child_table_names.append(self.table_relationships.nodes[child_table_id]["name"])

                foreign_scm_info = {
                    foreign_table_name: table_name_to_scm[foreign_table_name]
                    for foreign_table_name in fkey_col_to_pkey_table.values()
                }
                scm = SCM(
                    table_name=table_name,
                    child_table_names=child_table_names,
                    feature_columns=feature_columns,
                    pkey_col=pkey_col,
                    fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                    foreign_scm_info=foreign_scm_info,
                    scm_params=self.config.scm_params,
                    dag_params=self.config.dag_params,
                )
                if table_type == TableType.Activity:
                    min_timestamp, max_timestamp = (
                        self.min_timestamp,
                        self.max_timestamp,
                    )
                else:
                    min_timestamp, max_timestamp = None, None
                scm.generate_df(
                    num_rows=num_rows,
                    table_type=table_type,
                    min_timestamp=min_timestamp,
                    max_timestamp=max_timestamp,
                )
                table_name_to_scm[table_name] = scm

        table_dict = {}

        for table_id in self.table_relationships.nodes:
            table_name = self.table_relationships.nodes[table_id]["name"]
            df = table_name_to_scm[table_name].df
            pkey_col = self.table_relationships.nodes[table_id]["pkey_col"]
            fkey_col_to_pkey_table = self.table_relationships.nodes[table_id][
                "fkey_col_to_pkey_table"
            ]
            columns = self.table_relationships.nodes[table_id]["columns"]
            feature_columns = {
                col: col_info
                for col, col_info in columns.items()
                if col != pkey_col and col not in fkey_col_to_pkey_table.keys()
            }
            ########## post-processing ###############
            df = self.apply_col_transforms(
                df=df, pkey_col=pkey_col, fkey_cols=list(fkey_col_to_pkey_table.keys())
            )
            df = self.apply_zero_inflation(
                df=df, pkey_col=pkey_col, fkey_cols=list(fkey_col_to_pkey_table.keys())
            )
            df = self.binarize_columns(
                df=df,
                feature_columns=feature_columns,
                pkey_col=pkey_col,
                fkey_cols=list(fkey_col_to_pkey_table.keys()),
            )
            df = self.materialize_string_categoricals(
                df=df,
                feature_columns=feature_columns,
                pkey_col=pkey_col,
                fkey_cols=list(fkey_col_to_pkey_table.keys()),
            )
            df = self.implant_nan(
                df=df, pkey_col=pkey_col, fkey_cols=list(fkey_col_to_pkey_table.keys())
            )
            ##########################################
            time_col = "date" if "date" in df.columns else None
            table_dict[table_name] = Table(
                df=df,
                time_col=time_col,
                pkey_col=pkey_col,
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
            )

        return Database(table_dict)
