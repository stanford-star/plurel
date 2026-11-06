from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from operator import index as as_int

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.graph import AGGREGATES, COMPLETE, Foreign, Node, Summary
from plurel.links import Link, RandomLink, TreeLink
from plurel.random import Seed, generator, lane

Links = dict[tuple[str, str], np.ndarray]


def checked(name: str, node: Node, latent: np.ndarray, n: int) -> np.ndarray:
    if latent.shape != (n, node.dim):
        raise ValueError(f"{name!r} produced {latent.shape}, declared {(n, node.dim)}")
    if not np.isfinite(latent).all():
        raise ValueError(f"{name!r} produced non-finite values")
    return latent


@dataclass(frozen=True)
class FK:
    """A foreign key column pointing at rows of `parent`; `fill` is what edges crossing it read
    on rows whose key is null."""

    parent: str
    link: Link = RandomLink()
    nullable: float = 0.0
    fill: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.nullable < 1.0:
            raise ValueError("nullable must be in [0, 1)")
        if self.fill is not None and not np.isfinite(self.fill):
            raise ValueError("fill must be finite")


def topological[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as error:
        raise ValueError("nodes must form a directed acyclic graph") from error


class SCM:
    """A table: each name holds a hidden Node, an observed Column with its Node, or an FK."""

    def __init__(
        self, table: Mapping[str, Node | Column | FK], *, time_column: str | None = None
    ) -> None:
        self.table = dict(table)
        for name, entry in self.table.items():
            if not isinstance(entry, Node | Column | FK):
                raise ValueError(f"{name!r} must be a Node, a Column or an FK")
        self.fkeys = {name: e for name, e in self.table.items() if isinstance(e, FK)}
        self.columns = {name: e for name, e in self.table.items() if isinstance(e, Column)}
        self.nodes = {
            name: entry.node if isinstance(entry, Column) else entry
            for name, entry in self.table.items()
            if isinstance(entry, Node) or name in self.columns and entry.kind != "key"
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

    def sample(self, n: int, *, seed: Seed = None) -> pd.DataFrame:
        return self.sample_with_latents(n, seed=seed)[0]

    def sample_with_latents(
        self, n: int, *, seed: Seed = None, held: Mapping[str, np.ndarray] | None = None
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """This table alone, through a one-table Schema; `held` nodes take the given values."""
        if self.fkeys:
            raise ValueError(f"keys {sorted(self.fkeys)} need the tables of a Schema")
        frames, latents = Schema({"table": self}).sample_with_latents(
            {"table": n}, seed=seed, held={"table": held} if held else None
        )
        return frames["table"], latents["table"]


class Schema:
    """The plan of a database: its tables, joined by the foreign keys they hold and the edges
    that read across them. It realizes every node in one topological order across tables."""

    def __init__(self, tables: Mapping[str, SCM]) -> None:
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
            (table, name): node.across
            for table, scm in self.tables.items()
            for name, node in scm.nodes.items()
            if node.across
        }
        parents: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
        for table, scm in self.tables.items():
            for name, node in scm.nodes.items():
                parents[table, name] = tuple((table, parent) for parent in node.parents)
        for (table, name), edges in self.crossings.items():
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
            if tail.table in self.eventful:
                raise ValueError(f"a summary of {tail.table!r} would read later events")
            origin, node = tail.table, tail.node
        else:
            raise ValueError(f"{tail!r} is not a Foreign or a Summary")
        if node not in self.tables[origin].nodes:
            raise ValueError(f"unknown node {origin}.{node}")
        return origin, node

    def links(self, rows: Mapping[str, int], root: int) -> Links:
        links = {}
        for (table, column), fk in self.keys.items():
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
            if fk.fill is None:
                raise ValueError(f"key {tail.key!r} of {table!r} has null rows; set its fill")
            out = np.full((n, source.shape[1]), float(fk.fill))
            out[linked] = source[indices[linked]]
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
            for column in scm.fkeys:
                indices = links[table, column]
                frame[column] = pd.Series(indices, dtype="Int64").mask(indices < 0)
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
            if table not in self.tables:
                raise ValueError(f"held values for unknown table {table!r}")
            if unknown := set(nodes) - set(self.tables[table].nodes):
                raise ValueError(f"held values for unknown nodes {sorted(unknown)} of {table!r}")
        root = int(generator(seed).integers(2**63))
        links = self.links(rows, root)
        latents = self.propagate(rows, links, root, held)
        return self.observe(rows, latents, links, root), latents
