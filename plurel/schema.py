from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from operator import index as as_int

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.graph import AGGREGATES, COMPLETE, Edge, Foreign, Node, Summary
from plurel.links import Link, RandomLink, TreeLink
from plurel.random import Seed, generator

Links = dict[tuple[str, str], np.ndarray]


def checked(name: str, node: Node, latent: np.ndarray, n: int) -> np.ndarray:
    if latent.shape != (n, node.dim):
        raise ValueError(f"{name!r} produced {latent.shape}, declared {(n, node.dim)}")
    if not np.isfinite(latent).all():
        raise ValueError(f"{name!r} produced non-finite values")
    return latent


@dataclass(frozen=True)
class FK:
    """A key column of `table` pointing at rows of `parent`; `fill` is what edges crossing it
    read on rows whose key is null."""

    table: str
    column: str
    parent: str
    link: Link = RandomLink()
    nullable: float = 0.0
    fill: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.nullable < 1.0:
            raise ValueError("nullable must be in [0, 1)")
        if isinstance(self.link, TreeLink) and self.table != self.parent:
            raise ValueError("a tree link needs a self-referential foreign key")
        if self.fill is not None and not np.isfinite(self.fill):
            raise ValueError("fill must be finite")


def topological[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as error:
        raise ValueError("nodes must form a directed acyclic graph") from error


class SCM:
    """A table: each name holds a hidden Node or an observed Column with its Node. Edges that
    cross keys belong to the Schema."""

    def __init__(
        self, table: Mapping[str, Node | Column], *, time_column: str | None = None
    ) -> None:
        self.table = dict(table)
        for name, entry in self.table.items():
            if not isinstance(entry, Node | Column):
                raise ValueError(f"{name!r} must be a Node or a Column")
        self.columns = {name: e for name, e in self.table.items() if isinstance(e, Column)}
        self.nodes = {
            name: entry.node if isinstance(entry, Column) else entry
            for name, entry in self.table.items()
            if isinstance(entry, Node) or entry.kind != "key"
        }
        self.hidden = tuple(name for name in self.nodes if name not in self.columns)
        for child, node in self.nodes.items():
            if not all(isinstance(parent, str) for parent in node.parents):
                raise ValueError(f"{child!r} reads across a key; such edges belong to the Schema")
            if unknown := set(node.parents) - set(self.nodes):
                raise ValueError(f"{child!r} refers to unknown parents {sorted(unknown)}")
        self.order = topological({name: node.parents for name, node in self.nodes.items()})
        for name, column in self.columns.items():
            if isinstance(column.missing, str) and column.missing not in self.nodes:
                raise ValueError(f"column {name!r} refers to unknown node {column.missing!r}")
        kinds = {name: column.kind for name, column in self.columns.items()}
        keys = [name for name, kind in kinds.items() if kind == "key"]
        if len(keys) > 1:
            raise ValueError("a table has at most one key column")
        if time_column is not None and kinds.get(time_column) != "timestamp":
            raise ValueError(f"time column {time_column!r} must be a timestamp column")
        self.pkey_column = keys[0] if keys else None
        self.time_column = time_column

    def sample(self, n: int, *, seed: Seed = None) -> pd.DataFrame:
        return self.sample_with_latents(n, seed=seed)[0]

    def sample_with_latents(
        self, n: int, *, seed: Seed = None, held: Mapping[str, np.ndarray] | None = None
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """This table alone, through a one-table Schema; `held` nodes take the given values."""
        frames, latents = Schema({"table": self}).sample_with_latents(
            {"table": n}, seed=seed, held={"table": held} if held else None
        )
        return frames["table"], latents["table"]


Crossings = Mapping[tuple[str, str], Edge | tuple[Edge, ...]]


class Schema:
    """The plan of a database: its tables, the keys between them, and the edges that cross
    keys, `crossings`, mapping a (table, node) to edges whose tails are `Foreign` or `Summary`;
    their contributions join the node's signal. The Schema realizes every node in one
    topological order across tables."""

    def __init__(
        self,
        tables: Mapping[str, SCM],
        fkeys: tuple[FK, ...] = (),
        crossings: Crossings | None = None,
    ) -> None:
        self.tables = dict(tables)
        self.fkeys = tuple(fkeys)
        for fk in self.fkeys:
            if fk.table not in self.tables or fk.parent not in self.tables:
                raise ValueError(f"foreign key {fk.column!r} refers to unknown tables")
            if fk.column in self.tables[fk.table].columns:
                raise ValueError(
                    f"foreign key {fk.column!r} collides with a column of {fk.table!r}"
                )
        self.keys = {(fk.table, fk.column): fk for fk in self.fkeys}
        if len(self.keys) != len(self.fkeys):
            raise ValueError("foreign key columns must be unique per table")
        self.crossings = {
            node: (edges,) if isinstance(edges, Edge) else tuple(edges)
            for node, edges in (crossings or {}).items()
        }
        parents: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
        for table, scm in self.tables.items():
            for name, node in scm.nodes.items():
                parents[table, name] = tuple((table, parent) for parent in node.parents)
        for (table, name), edges in self.crossings.items():
            if table not in self.tables or name not in self.tables[table].nodes:
                raise ValueError(f"crossing edges refer to unknown node {table}.{name}")
            if any(isinstance(edge.parent, str) for edge in edges):
                raise ValueError(f"crossing edges into {table}.{name} must read across a key")
            parents[table, name] += tuple(self.source(table, edge.parent) for edge in edges)
        self.order = topological(parents)

    def key(self, table: str, column: str) -> FK:
        if (table, column) not in self.keys:
            raise ValueError(f"{table!r} has no key column {column!r}")
        return self.keys[table, column]

    def source(self, table: str, tail: Hashable) -> tuple[str, str]:
        """The node a crossing edge of `table` reads through its tail."""
        if isinstance(tail, Foreign):
            origin, node = self.key(table, tail.key).parent, tail.node
        elif isinstance(tail, Summary):
            if self.key(tail.table, tail.key).parent != table:
                raise ValueError(f"key {tail.key!r} of {tail.table!r} does not point at {table!r}")
            origin, node = tail.table, tail.node
        else:
            raise ValueError(f"{tail!r} is not a Foreign or a Summary")
        if node not in self.tables[origin].nodes:
            raise ValueError(f"unknown node {origin}.{node}")
        return origin, node

    def links(self, rows: Mapping[str, int], rng: np.random.Generator) -> Links:
        links = {}
        for fk, stream in zip(self.fkeys, rng.spawn(len(self.fkeys))):
            drawn = np.asarray(fk.link.sample(rows[fk.table], rows[fk.parent], stream))
            if drawn.shape != (rows[fk.table],) or not np.issubdtype(drawn.dtype, np.integer):
                raise ValueError(f"link {fk.column!r} must return one integer per child row")
            if len(drawn) and (drawn.min() < -1 or drawn.max() >= rows[fk.parent]):
                raise ValueError(f"link {fk.column!r} points outside the parent table")
            indices = drawn.astype(np.int64)
            if fk.nullable:
                indices[stream.random(len(indices)) < fk.nullable] = -1
            links[fk.table, fk.column] = indices
        return links

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
            out = np.empty((n, source.shape[1]))
            out[linked] = source[indices[linked]]
            if not linked.all():
                if fk.fill is None:
                    raise ValueError(f"key {tail.key!r} of {table!r} has null rows; set its fill")
                out[~linked] = fk.fill
            return out
        indices = links[tail.table, tail.key]
        linked = indices >= 0
        out = AGGREGATES[tail.how](latents[tail.table][tail.node][linked], indices[linked], n)
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
        rng: np.random.Generator,
        held: Mapping[str, Mapping[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Evaluate every node of every table once, in topological order, each with its own
        noise stream unless it only reads: a held node takes its values, any other its local
        edges plus its crossing edges plus noise."""
        latents: dict[str, dict[str, np.ndarray]] = {table: {} for table in self.tables}
        drawn = [key for key in self.order if not self.tables[key[0]].nodes[key[1]].reads]
        streams = dict(zip(drawn, rng.spawn(len(drawn))))
        for table, name in self.order:
            node, n = self.tables[table].nodes[name], rows[table]
            if name in held.get(table, {}):
                latents[table][name] = checked(name, node, held[table][name], n)
                continue
            exogenous, across = node.sample_noise(n, streams.get((table, name))), None
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
        rng: np.random.Generator,
    ) -> dict[str, pd.DataFrame]:
        """Turn latents into one frame per table: its columns, then its key columns."""
        frames = {}
        for (table, scm), stream in zip(self.tables.items(), rng.spawn(len(self.tables))):
            n = rows[table]
            observed = {
                name: column.observe(name, latents[table], stream, n)
                for name, column in scm.columns.items()
            }
            frame = pd.DataFrame(observed, index=range(n))
            for fk in self.fkeys:
                if fk.table == table:
                    indices = links[table, fk.column]
                    frame[fk.column] = pd.Series(indices, dtype="Int64").mask(indices < 0)
            frames[table] = frame
        return frames

    def sample(
        self,
        rows: Mapping[str, int],
        *,
        seed: Seed = None,
    ) -> dict[str, pd.DataFrame]:
        return self.sample_with_latents(rows, seed=seed)[0]

    def sample_with_latents(
        self,
        rows: Mapping[str, int],
        *,
        seed: Seed = None,
        held: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:
        """One frame and the latents per table; `held` nodes take the given `(n, dim)` values."""
        try:
            rows = {table: as_int(n) for table, n in rows.items()}
        except TypeError as error:
            raise ValueError("row counts must be integers") from error
        if set(rows) != set(self.tables) or min(rows.values(), default=0) < 0:
            raise ValueError("rows must give a non-negative count for every table")
        held = dict(held or {})
        for table, nodes in held.items():
            if unknown := set(nodes) - set(self.tables[table].nodes):
                raise ValueError(f"held values for unknown nodes {sorted(unknown)} of {table!r}")
        linking, noise, observation = generator(seed).spawn(3)
        links = self.links(rows, linking)
        latents = self.propagate(rows, links, noise, held)
        return self.observe(rows, latents, links, observation), latents
