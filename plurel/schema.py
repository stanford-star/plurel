import networkx as nx
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    String,
    Time,
    create_engine,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.exc import SQLAlchemyError
from torch_frame import stype

from plurel.config import Config
from plurel.dag import DAG_REGISTRY


class RandomSchemaGraphBuilder:
    """
    Builds a random relational schema graph based on configuration parameters.

    Generates table structures with primary keys, foreign keys, and feature columns
    using random DAG layouts.

    Args:
        config: Configuration object with database and SCM parameters.
        num_tables: Number of tables to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(self, config: Config, num_tables: int, seed: int):
        self.config = config
        self.num_tables = num_tables
        self.seed = seed

    def build_graph(self) -> nx.DiGraph:
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
        dag_class = self.config.database_params.table_layout_choices.sample_uniform()
        dag = DAG_REGISTRY[dag_class](
            num_nodes=self.num_tables, dag_params=self.config.dag_params, seed=self.seed
        )
        G = dag.graph

        for table_id in G.nodes:
            G.nodes[table_id]["name"] = f"table_{table_id}"

        for table_id in G.nodes:
            # most tables are narrow with few being wide
            num_cols = self.config.database_params.num_cols_choices.sample_pl()
            feature_cols = [f"feature_{idx}" for idx in range(num_cols)]
            pkey_col = "row_idx"
            fkey_col_to_pkey_table = {
                f"foreign_row_{idx}": G.nodes[parent_table_id]["name"]
                for idx, parent_table_id in enumerate(sorted(list(G.predecessors(table_id))))
            }
            fkey_cols = list(fkey_col_to_pkey_table.keys())

            columns = {}
            for col in [pkey_col, *fkey_cols]:
                _stype = stype.categorical
                columns[col] = {
                    "_stype": stype.categorical,
                    "categories": None,  # since these are pk/fk
                }

            for feature_col in feature_cols:
                _stype = self.config.scm_params.col_stype_choices.sample_uniform()
                if _stype == stype.categorical:
                    num_categories = self.config.scm_params.num_categories_choices.sample()
                    categories = list(range(num_categories))
                else:
                    categories = None
                columns[feature_col] = {
                    "_stype": _stype,
                    "categories": categories,
                }

            metadata = {
                "columns": columns,
                "pkey_col": pkey_col,
                "fkey_col_to_pkey_table": fkey_col_to_pkey_table,
            }
            for k, v in metadata.items():
                G.nodes[table_id][k] = v

        return G


class SQLSchemaGraphBuilder:
    """
    Builds a relational schema graph from an SQL schema file.

    Parses CREATE TABLE statements and extracts table structures including
    columns, primary keys, foreign keys, and data types.

    Args:
        sql_file: Path to the SQL schema file.
    """

    def __init__(self, sql_file: str):
        self.sql_file = sql_file
        self.metadata = MetaData()
        self.tables = {}

    def load_schema(self):
        sql_content = open(self.sql_file).read()
        engine = create_engine("sqlite:///:memory:")

        try:
            with engine.begin() as conn:
                for stmt in sql_content.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
        except SQLAlchemyError as e:
            raise ValueError(f"Failed to execute SQL schema: {e}")

        self.metadata.reflect(bind=engine)

        for table_name, table in self.metadata.tables.items():
            self.tables[table_name] = self._build_table_ir(table)

    def _map_torch_frame_type(self, col):
        """
        Returns:
            {
                "_stype": stype,
                "categories": list[str] | None
            }
        """
        t = col.type

        enum_vals = self._get_enum_values(col)
        if enum_vals is not None:
            return {
                "_stype": stype.categorical,
                "categories": enum_vals,
            }
        if col.primary_key or col.foreign_keys:
            return {
                "_stype": stype.categorical,
                "categories": None,
            }
        if isinstance(t, Boolean):
            return {
                "_stype": stype.categorical,
                "categories": [0, 1],
            }
        if isinstance(t, Float | Numeric | Integer | BigInteger):
            return {
                "_stype": stype.numerical,
                "categories": None,
            }

        if isinstance(t, Date | DateTime | Time | String):
            raise ValueError(f"column: {col.name} of type: {t} is not supported")

    def _get_enum_values(self, col):
        t = col.type

        if isinstance(t, SAEnum):
            return list(t.enums)

        # --- SQLite-style CHECK constraint: col IN (...) ---
        for constraint in col.table.constraints:
            if not hasattr(constraint, "sqltext"):
                continue
            sql = str(constraint.sqltext).lower()
            name = col.name.lower()
            if f"{name} in" in sql:
                inside = sql.split("in", 1)[1]
                inside = inside.strip().lstrip("(").rstrip(")")
                vals = [v.strip().strip("'\"") for v in inside.split(",")]
                return vals

        return None

    def _build_table_ir(self, table):
        """
        Returns:
        {
            "columns":dict[col_name -> {
                "_stype": stype,
                "categories": list[str] | None
            }]
            "pkey_col": str | None,
            "fkey_col_to_pkey_table": dict[str, str],
        }

        Example:
        ```py
        {
            "columns": {
                "status": {
                    "_stype": stype.categorical,
                    "categories": ["open", "closed", "pending"]
                },
                "age": {
                    "_stype": stype.numerical,
                    "categories": None
                }
            },
            "pkey_col": "...",
            "fkey_col_to_pkey_table": {...},
        }
        ```
        """

        # ---------- Columns ----------
        columns = {}
        for c in table.columns:
            col_info = self._map_torch_frame_type(c)
            columns[c.name] = col_info

        # ---------- Primary key ----------
        pkeys = [c.name for c in table.primary_key.columns]
        if len(pkeys) > 1:
            raise ValueError(
                f"Composite primary keys not supported in simplified IR: {table.name} -> {pkeys}"
            )
        pkey_col = pkeys[0] if pkeys else None

        # ---------- Foreign keys ----------
        fkey_map = {}
        for fk in table.foreign_keys:
            local_col = fk.parent.name
            ref_table = fk.column.table.name
            fkey_map[local_col] = ref_table

        return {
            "columns": columns,
            "pkey_col": pkey_col,
            "fkey_col_to_pkey_table": fkey_map,
        }

    def build_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph from schema."""
        if not self.tables:
            raise ValueError("Schema not loaded. Call load_schema() first.")

        G = nx.DiGraph()

        # Add nodes
        for table_name, table_info in self.tables.items():
            G.add_node(table_name, **{"name": table_name, **table_info})

        # Add edges: referenced_table -> current_table
        for table_name, table_info in self.tables.items():
            for _, ref_table in table_info["fkey_col_to_pkey_table"].items():
                G.add_edge(ref_table, table_name)

        return G

    def draw_graph(self, G: nx.DiGraph, filepath: str):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        plt.title("Database Schema Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath)
