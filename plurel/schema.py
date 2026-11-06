from collections.abc import Hashable, Mapping
from dataclasses import replace
from graphlib import CycleError, TopologicalSorter
from operator import index as as_int

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.declare import CODE, Declaration, ForeignKey, PrimaryKey, Real
from plurel.graph import AGGREGATES, COMPLETE, Edge, Foreign, LevelEdge, LinearEdge, Node, Summary
from plurel.links import TreeLink
from plurel.random import Seed, generator, lane

Links = dict[tuple[str, str], np.ndarray]


def checked(name: str, node: Node, latent: np.ndarray, n: int) -> np.ndarray:
    if latent.shape != (n, node.dim):
        raise ValueError(f"{name!r} produced {latent.shape}, declared {(n, node.dim)}")
    if not np.isfinite(latent).all():
        raise ValueError(f"{name!r} produced non-finite values")
    return latent


def bind(edge: Edge, parent: Node | Column | None) -> Edge:
    """The edge lined up with the column it reads: its levels, and no order a real one lacks."""
    categorical = isinstance(parent, Column) and parent.kind == "categorical"
    if isinstance(edge, LevelEdge):
        if not categorical:
            raise ValueError(f"effects per level need a categorical parent, not {edge.parent!r}")
        return edge.bound(parent.categories)
    if categorical and isinstance(edge, LinearEdge) and parent.values is not None:
        raise ValueError(f"{edge.parent!r} holds real levels in no order: give effects per level")
    return edge


def topological[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as error:
        raise ValueError("nodes must form a directed acyclic graph") from error


class Table:
    """A table: a dict of its keys, its columns (`Numeric`, `Categorical`, `Timestamp`, `Real`) and
    the `Hidden` causes they read. The lower-level `Node` and `Column` are accepted too."""

    def __init__(self, table: Mapping[str, object], *, time_column: str | None = None) -> None:
        keys = {name for name, entry in table.items() if isinstance(entry, ForeignKey)}
        self.table: dict[str, Node | Column | ForeignKey | PrimaryKey] = {}
        for name, entry in table.items():
            if isinstance(entry, Declaration):
                entries = entry.entries(name, keys)
                if clash := (set(entries) - {name}) & set(table):
                    raise ValueError(f"{name!r} needs the names {sorted(clash)} for itself")
                self.table.update(entries)
            elif isinstance(entry, Node | Column | ForeignKey | PrimaryKey):
                self.table[name] = entry
            else:
                raise ValueError(f"{name!r} must be a key, a column or a hidden cause")
        self.bind_local()
        self.fkeys = {name: e for name, e in self.table.items() if isinstance(e, ForeignKey)}
        self.columns = {
            name: e for name, e in self.table.items() if isinstance(e, Column | PrimaryKey)
        }
        self.nodes = {
            name: entry.node if isinstance(entry, Column) else entry
            for name, entry in self.table.items()
            if isinstance(entry, Node | Column)
        }
        self.hidden = tuple(name for name in self.nodes if name not in self.columns)
        for child, node in self.nodes.items():
            if unknown := set(node.parents) - set(self.nodes):
                raise ValueError(f"{child!r} refers to unknown parents {sorted(unknown)}")
        self.order = topological({name: node.parents for name, node in self.nodes.items()})
        for name, column in self.columns.items():
            if isinstance(column.missing, str) and (
                column.missing not in self.nodes
                or self.nodes[column.missing].dim != 2
                or not self.nodes[column.missing].onehot
            ):
                raise ValueError(f"column {name!r} needs a two-class node {column.missing!r}")
        kinds = {name: column.kind for name, column in self.columns.items()}
        keys = [name for name, kind in kinds.items() if kind == "key"]
        if len(keys) > 1:
            raise ValueError("a table has at most one key column")
        if time_column is not None and kinds.get(time_column) != "timestamp":
            raise ValueError(f"time column {time_column!r} must be a timestamp column")
        self.pkey_column = keys[0] if keys else None
        self.time_column = time_column
        self.real = {
            name: column.real_latent
            for name, column in self.columns.items()
            if isinstance(column, Column) and column.values is not None
        }
        sizes = {
            len(e.values) for e in self.table.values() if getattr(e, "values", None) is not None
        }
        if len(sizes) > 1:
            raise ValueError(f"given values disagree on the row count: {sorted(sizes)}")
        self.size = sizes.pop() if sizes else None

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        primary_key: str | None = None,
        foreign_keys: Mapping[str, str] | None = None,
        time_column: str | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> "Table":
        """A table of real rows, kept as they are; `foreign_keys` maps a column to the table it
        points at, and `extra` adds generated columns that may read the real ones."""
        foreign_keys, extra = dict(foreign_keys or {}), dict(extra or {})
        if primary_key in foreign_keys:
            raise ValueError(f"{primary_key!r} cannot be both the primary and a foreign key")
        if taken := set(extra) & set(frame):
            raise ValueError(f"extra would replace the real columns {sorted(taken)}")
        if unknown := (({primary_key} - {None}) | set(foreign_keys)) - set(frame):
            raise ValueError(f"frame has no columns {sorted(unknown)}")
        entries: dict[str, object] = {}
        for name in frame:
            if name == primary_key:
                entries[name] = PrimaryKey(frame[name])
            elif name in foreign_keys:
                entries[name] = ForeignKey(foreign_keys[name], values=frame[name], fill=0.0)
            else:
                entries[name] = Real(frame[name])
        return cls({**entries, **extra}, time_column=time_column)

    def bind_local(self) -> None:
        """Line the edges up with the columns of this table they read."""

        def bound(node: Node) -> Node:
            edges = []
            for edge in node.edges:
                if isinstance(edge.parent, str):
                    edge = bind(edge, self.table.get(edge.parent.removesuffix(CODE)))
                edges.append(edge)
            return replace(node, edges=tuple(edges))

        for name, entry in self.table.items():
            if isinstance(entry, Column):
                self.table[name] = replace(entry, node=bound(entry.node))
            elif isinstance(entry, Node):
                self.table[name] = bound(entry)

    def sample(self, n: int | None = None, *, seed: Seed = None) -> pd.DataFrame:
        return self.sample_with_latents(n, seed=seed)[0]

    def sample_with_latents(
        self,
        n: int | None = None,
        *,
        seed: Seed = None,
        held: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """This table alone, through a one-table Schema; `held` nodes take the given values."""
        if self.fkeys:
            raise ValueError(f"keys {sorted(self.fkeys)} need the tables of a Schema")
        frames, latents = Schema({"table": self}).sample_with_latents(
            {} if n is None else {"table": n}, seed=seed, held={"table": held} if held else None
        )
        return frames["table"], latents["table"]


class Schema:
    """The plan of a database: its tables, joined by the foreign keys they hold and the edges
    that read across them. It realizes every node in one topological order across tables."""

    def __init__(self, tables: Mapping[str, Table]) -> None:
        self.tables = dict(tables)
        keys = {
            (table, column): fk
            for table, scm in self.tables.items()
            for column, fk in scm.fkeys.items()
        }
        for (table, column), fk in keys.items():
            if fk.parent not in self.tables:
                raise ValueError(f"foreign key {column!r} refers to unknown table {fk.parent!r}")
            if isinstance(fk.link, TreeLink) and table != fk.parent:
                raise ValueError("a tree link needs a self-referential foreign key")
        self.keys = keys
        self.eventful = {t for t, scm in self.tables.items() if scm.time_column is not None}
        for _ in self.tables:
            self.eventful |= {t for (t, _c), fk in keys.items() if fk.parent in self.eventful}
        self.crossings = {
            (table, name): tuple(self.bound(table, edge) for edge in node.across)
            for table, scm in self.tables.items()
            for name, node in scm.nodes.items()
            if node.across
        }
        parents: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
        for table, scm in self.tables.items():
            for name, node in scm.nodes.items():
                parents[table, name] = tuple((table, parent) for parent in node.parents)
        for (table, name), edges in self.crossings.items():
            sources = (self.source(table, edge.parent) for edge in edges)
            parents[table, name] += tuple(source for source in sources if source[1] is not None)
        self.order = topological(parents)

    def bound(self, table: str, edge: Edge) -> Edge:
        """A crossing edge lined up with the column it reads."""
        tail = edge.parent
        if isinstance(tail, Summary) and tail.node is not None:
            self.key(tail.table, tail.key)
            return bind(edge, self.tables[tail.table].table.get(tail.node))
        if not isinstance(tail, Foreign):
            return edge
        origin = self.tables[self.key(table, tail.key).parent]
        return bind(edge, origin.table.get(edge.parent.node.removesuffix(CODE)))

    def key(self, table: str, column: str) -> ForeignKey:
        if (table, column) not in self.keys:
            raise ValueError(f"{table!r} has no key column {column!r}")
        return self.keys[table, column]

    def source(self, table: str, tail: Hashable) -> tuple[str, str | None]:
        """The node a crossing edge of `table` reads through its tail; a bare count reads none."""
        if isinstance(tail, Foreign):
            origin, node = self.key(table, tail.key).parent, tail.node
        elif isinstance(tail, Summary):
            if self.key(tail.table, tail.key).parent != table:
                raise ValueError(f"key {tail.key!r} of {tail.table!r} does not point at {table!r}")
            if tail.table in self.eventful:
                raise ValueError(f"a summary of {tail.table!r} would read later events")
            origin, node = tail.table, tail.node
        else:
            raise ValueError(f"{tail!r} is not a Foreign or a Summary")
        if node is not None and node not in self.tables[origin].nodes:
            raise ValueError(f"unknown node {origin}.{node}")
        return origin, node

    def links(self, rows: Mapping[str, int], root: int) -> Links:
        links = {}
        for (table, column), fk in self.keys.items():
            if fk.values is not None:
                links[table, column] = self.real_link(table, column, fk, rows[fk.parent])
                continue
            stream = lane(root, "link", table, column)
            drawn = np.asarray(fk.link.sample(rows[table], rows[fk.parent], stream))
            if drawn.shape != (rows[table],) or not np.issubdtype(drawn.dtype, np.integer):
                raise ValueError(f"link {column!r} must return one integer per child row")
            if len(drawn) and (drawn.min() < -1 or drawn.max() >= rows[fk.parent]):
                raise ValueError(f"link {column!r} points outside the parent table")
            indices = drawn.astype(np.int64)
            if fk.nullable:
                indices[stream.random(len(indices)) < fk.nullable] = -1
            links[table, column] = indices
        return links

    def real_link(self, table: str, column: str, fk: ForeignKey, n_parent: int) -> np.ndarray:
        """Row positions in the parent of given key values; a blank points at nothing."""
        given = self.parent_keys(fk)
        indices = (pd.RangeIndex(n_parent) if given is None else pd.Index(given)).get_indexer(
            fk.values
        )
        if ((indices < 0) & fk.values.notna().to_numpy()).any():
            raise ValueError(f"key {column!r} of {table!r} holds values its parent lacks")
        return indices.astype(np.int64)

    def parent_keys(self, fk: ForeignKey) -> pd.Series | None:
        """The parent's given key values, None when its key is its row positions."""
        parent = self.tables[fk.parent]
        return None if parent.pkey_column is None else parent.columns[parent.pkey_column].values

    def resolve(
        self,
        table: str,
        tail: Foreign | Summary,
        rows: Mapping[str, int],
        latents: dict[str, dict[str, np.ndarray]],
        links: Links,
    ) -> np.ndarray:
        """The array a crossing tail contributes to `table`, one row per row of the table."""
        n = rows[table]
        if isinstance(tail, Foreign):
            fk = self.keys[table, tail.key]
            indices = links[table, tail.key]
            source = latents[fk.parent][tail.node]
            linked = indices >= 0
            if linked.all():
                return source.take(indices, axis=0).astype(float, copy=False)
            fill = -1.0 if tail.node.endswith(CODE) else fk.fill  # no level, so no effect
            if fill is None:
                raise ValueError(f"key {tail.key!r} of {table!r} has null rows; set its fill")
            out = np.full((n, source.shape[1]), float(fill))
            out[linked] = source[indices[linked]]
            return out
        indices = links[tail.table, tail.key]
        linked = indices >= 0
        values = np.empty((int(linked.sum()), 1))
        if tail.node is not None:
            values = latents[tail.table][tail.node][linked]
        out = AGGREGATES[tail.how](values, indices[linked], n)
        empty = np.bincount(indices[linked], minlength=n) == 0
        if empty.any() and tail.how not in COMPLETE:
            if tail.fill is None:
                raise ValueError(
                    f"summary of {tail.table}.{tail.node} met rows without children; set fill"
                )
            out[empty] = tail.fill
        return out

    def propagate(
        self,
        rows: Mapping[str, int],
        links: Links,
        root: int,
        held: Mapping[str, Mapping[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Evaluate every node of every table once, in topological order, each with its own
        noise stream: a held node takes its values, any other its edges
        plus noise."""
        latents: dict[str, dict[str, np.ndarray]] = {table: {} for table in self.tables}
        for table, name in self.order:
            node, n = self.tables[table].nodes[name], rows[table]
            if name in held.get(table, {}):
                latents[table][name] = checked(name, node, held[table][name], n)
                continue
            exogenous, across = node.sample_noise(n, lane(root, "noise", table, name)), None
            for edge in self.crossings.get((table, name), ()):
                term = edge.apply(self.resolve(table, edge.parent, rows, latents, links))
                across = term if across is None else across + term
            parents = {parent: latents[table][parent] for parent in node.parents}
            value = node.evaluate(parents, exogenous, across)
            latents[table][name] = checked(name, node, value, n)
        return latents

    def observe(
        self,
        rows: Mapping[str, int],
        latents: dict[str, dict[str, np.ndarray]],
        links: Links,
        root: int,
    ) -> dict[str, pd.DataFrame]:
        """Turn latents into one frame per table: its columns, then its key columns."""
        frames = {}
        for table, scm in self.tables.items():
            n = rows[table]
            observed = {
                name: column.observe(name, latents[table], lane(root, "column", table, name), n)
                for name, column in scm.columns.items()
            }
            frame = pd.DataFrame(observed, index=range(n))
            for column, fk in scm.fkeys.items():
                indices, given = links[table, column], self.parent_keys(fk)
                if given is None:
                    frame[column] = pd.Series(indices, dtype="Int64").mask(indices < 0)
                else:
                    keys = given.astype("Int64") if pd.api.types.is_integer_dtype(given) else given
                    frame[column] = keys.take(indices).reset_index(drop=True).mask(indices < 0)
            frames[table] = frame[[name for name in scm.table if name in frame]]
        return frames

    def sample(
        self,
        rows: Mapping[str, int] | None = None,
        *,
        seed: Seed = None,
    ) -> dict[str, pd.DataFrame]:
        return self.sample_with_latents(rows, seed=seed)[0]

    def sample_with_latents(
        self,
        rows: Mapping[str, int] | None = None,
        *,
        seed: Seed = None,
        held: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:
        """One frame and the latents per table; a table of given values has their row count, and
        `held` nodes take the given `(n, dim)` values."""
        rows = dict(rows or {})
        for table, scm in self.tables.items():
            if scm.size is not None and rows.setdefault(table, scm.size) != scm.size:
                raise ValueError(f"{table!r} holds {scm.size} given rows, not {rows[table]}")
        try:
            rows = {table: as_int(n) for table, n in rows.items()}
        except TypeError as error:
            raise ValueError("row counts must be integers") from error
        if set(rows) != set(self.tables) or min(rows.values(), default=0) < 0:
            raise ValueError("rows must give a non-negative count for every table")
        held = dict(held or {})
        for table, nodes in held.items():
            if table not in self.tables:
                raise ValueError(f"held values for unknown table {table!r}")
            if unknown := set(nodes) - set(self.tables[table].nodes):
                raise ValueError(f"held values for unknown nodes {sorted(unknown)} of {table!r}")
        root = int(generator(seed).integers(2**63))
        links = self.links(rows, root)
        for table, scm in self.tables.items():
            if scm.real:
                held[table] = {**scm.real, **held.get(table, {})}
        latents = self.propagate(rows, links, root, held)
        return self.observe(rows, latents, links, root), latents
