from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import replace
from graphlib import CycleError, TopologicalSorter
from operator import index as as_int

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.declare import (
    CODE,
    Declaration,
    Derived,
    ForeignKey,
    PrimaryKey,
    Real,
)
from plurel.fit import fitted
from plurel.graph import (
    AGGREGATES,
    COMPLETE,
    Edge,
    Foreign,
    LevelEdge,
    LinearEdge,
    Node,
    Recorded,
    Summary,
)
from plurel.links import TreeLink
from plurel.random import Seed, generator, lane

Links = dict[tuple[str, str], np.ndarray]
SPAN = (pd.Timestamp.min.value // 10**9 + 1, pd.Timestamp.max.value // 10**9)  # epoch seconds


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
    if categorical and isinstance(edge, LinearEdge) and not parent.ordered:
        raise ValueError(f"{edge.parent!r} holds levels in no order: give effects per level")
    return edge


def topological[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as error:
        raise ValueError(
            "nodes and the records they read must form a directed acyclic graph, but"
            f" {error.args[1]} read each other in a cycle"
        ) from error


def numbers(recorded: pd.Series, where: str) -> np.ndarray:
    """A record as floats, a time as epoch seconds and a blank as NaN."""
    if getattr(recorded.dtype, "tz", None) is not None:
        raise ValueError(
            f"{where} holds times in a time zone, which a time read from it would lose: give it"
            " naive times, e.g. converted with .dt.tz_convert(None)"
        )
    if pd.api.types.is_datetime64_any_dtype(recorded):
        seconds = recorded.to_numpy("datetime64[ns]").astype(np.int64) / 1e9
        return np.where(recorded.isna().to_numpy(), np.nan, seconds)
    return recorded.to_numpy(dtype=float, na_value=np.nan)


def filled(values: np.ndarray, tail: Recorded, where: str) -> np.ndarray:
    """Recorded values with the tail's fill where there are none."""
    blank = np.isnan(values)
    if not blank.any():
        return values
    if tail.fill is None:
        raise ValueError(
            f"{where} has rows without a recorded value, blank or with a key pointing at nothing:"
            " give Recorded a fill"
        )
    return np.where(blank, tail.fill, values)


class Table:
    """A table: a dict of its keys, its columns (`Numeric`, `Categorical`, `Timestamp`, `Real`,
    `Derived`) and the `Hidden` causes they read, as latents or, `Recorded`, as recorded values.
    The lower-level `Node` and `Column` are accepted too. A table holding ranges is a family:
    `realize(seed)` draws one table of it. `doc`, `target` and `sources` say what the table
    records, what it is read for and where its numbers come from; `docs` holds each entry's description, from its `doc` or given, and survives realizing."""

    def __init__(
        self,
        table: Mapping[str, object],
        *,
        time_column: str | None = None,
        name: str | None = None,
        doc: str | None = None,
        target: str | None = None,
        sources: str | None = None,
        docs: Mapping[str, str] | None = None,
    ) -> None:
        self.declared = dict(table)
        self.name, self.doc, self.target, self.sources = name, doc, target, sources
        given = {k: e.doc for k, e in self.declared.items() if getattr(e, "doc", None)}
        given |= dict(docs or {})
        if unknown := set(given) - set(self.declared):
            raise ValueError(f"docs name no entries of the table: {sorted(unknown)}")
        self.docs = {name: given[name] for name in self.declared if name in given}
        self.family = any(isinstance(e, Declaration) and e.family for e in self.declared.values())
        table = self.draws(0)
        keys = {name for name, entry in table.items() if isinstance(entry, ForeignKey)}
        self.table: dict[str, Node | Column | ForeignKey | PrimaryKey | Derived] = {}
        for name, entry in table.items():
            if isinstance(entry, Declaration):
                entries = entry.entries(name, keys)
                if clash := (set(entries) - {name}) & set(table):
                    raise ValueError(f"{name!r} needs the names {sorted(clash)} for itself")
                self.table.update(entries)
            elif isinstance(entry, Node | Column | ForeignKey | PrimaryKey | Derived):
                self.table[name] = entry
            else:
                raise ValueError(f"{name!r} must be a key, a column or a hidden cause")
        self.bind_local()
        self.fkeys = {name: e for name, e in self.table.items() if isinstance(e, ForeignKey)}
        self.derived = {name: e for name, e in self.table.items() if isinstance(e, Derived)}
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
        for latent in self.real.values():
            latent.setflags(write=False)
        sizes = {
            len(e.values) for e in self.table.values() if getattr(e, "values", None) is not None
        }
        if len(sizes) > 1:
            raise ValueError(f"given values disagree on the row count: {sorted(sizes)}")
        self.size = sizes.pop() if sizes else None
        self.readable = {
            *self.derived,
            *(c for c, kind in kinds.items() if kind in ("numeric", "timestamp")),
        }
        self.check_records(table, kinds)
        self.plan()
        for name, column in self.columns.items():
            if getattr(column, "fit_shares", None) and name in self.crossing:
                raise ValueError(
                    f"{name!r} is scored and depends on a key: no fit holds its shares"
                )

    def plan(self) -> None:
        """The graph of the nodes and of the records, which read nodes and other records; the
        order of its nodes; and what depends on a key."""
        graph: dict[str | Recorded, tuple[str | Recorded, ...]] = {
            name: (
                *node.parents,
                *(e.parent.vertex for e in node.across if isinstance(e.parent, Recorded)),
            )
            for name, node in self.nodes.items()
        }
        graph |= {Recorded(name): self.inputs(name) for name in (*self.columns, *self.derived)}
        order = topological(graph)
        self.graph, self.order = graph, tuple(v for v in order if isinstance(v, str))
        self.crossing: set[str | Recorded] = set()
        for vertex in order:
            node = self.nodes.get(vertex)
            keyed = node is not None and any(
                isinstance(e.parent, Foreign | Summary) for e in node.edges
            )
            if keyed or self.crossing.intersection(graph.get(vertex, ())):
                self.crossing.add(vertex)

    def inputs(self, name: str) -> tuple[str | Recorded, ...]:
        """What recording a column reads: its node and the nodes its blanks and ranges follow, or
        the records a derived column computes from."""
        if name in self.derived:
            return tuple(Recorded(c) for c in self.derived[name].inputs)
        column = self.columns[name]
        if isinstance(column, PrimaryKey):
            return ()
        missing = (column.missing,) if isinstance(column.missing, str) else ()
        by = (column.missing_by[0],) if column.missing_by else ()
        level = (column.by_level[0] + CODE,) if column.by_level else ()
        return (name, *missing, *by, *level)

    def check_records(self, table: Mapping[str, object], kinds: Mapping[str, str]) -> None:
        """The columns a table's records read by name."""
        for name, column in self.columns.items():
            by = getattr(column, "missing_by", None)
            if by is not None and kinds.get(by[0]) != "numeric":
                raise ValueError(f"{name!r} is blank by {by[0]!r}, which is no numeric column")
        for name, derived in self.derived.items():
            readable = {*self.derived, *(c for c, kind in kinds.items() if kind == "numeric")}
            if unknown := [c for c in derived.inputs if c not in readable]:
                raise ValueError(
                    f"{name!r} reads {unknown}, which are no numeric or derived columns"
                )
        for name, node in self.nodes.items():
            for edge in node.across:
                if isinstance(edge.parent, Recorded) and edge.parent.column not in self.readable:
                    raise ValueError(
                        f"{name!r} reads {edge.parent.column!r} as recorded: Recorded reads a"
                        " numeric, timestamp or derived column of the table"
                    )
        for name, (parent, ranges) in self.by_level.items():
            if kinds.get(parent) != "categorical":
                raise ValueError(f"{name!r} takes ranges from {parent!r}: no categorical")
            if {level for level, *_ in ranges} != set(self.columns[parent].categories):
                raise ValueError(f"{name!r} needs one range per level of {parent!r}")
        numbers = {*self.hidden, *(name for name, kind in kinds.items() if kind == "numeric")}
        for name, entry in table.items():
            for scored in (getattr(entry, "scores", None) or {}).values():
                if unknown := set(scored) - numbers:
                    raise ValueError(f"{name!r} scores {sorted(unknown)}: no numbers of its table")
        if self.target is not None:
            recorded = {*self.derived, *(c for c, kind in kinds.items() if kind != "key")}
            if self.target not in recorded:
                why = (
                    "a hidden cause no frame records"
                    if self.target in self.nodes
                    else ("a key" if self.target in self.table else "no entry of the table")
                )
                raise ValueError(f"target {self.target!r} is {why}: it must be a recorded column")

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "time_column": self.time_column,
            "name": self.name,
            "doc": self.doc,
            "target": self.target,
            "sources": self.sources,
            "docs": self.docs,
        }

    @property
    def by_level(self) -> dict[str, tuple[str, tuple[tuple[object, float, float], ...]]]:
        """The columns placed on a range per level of a categorical, by that parent and the ranges."""
        return {
            name: column.by_level
            for name, column in self.columns.items()
            if getattr(column, "by_level", None) is not None
        }

    def draws(self, root: int, table: str | None = None) -> dict[str, object]:
        """Every entry as one seed draws it, each from its own stream named by the table and the
        entry: adding a column moves no other. The table is named by `table`, else its own name."""
        table = table or self.name or "table"
        return {
            name: entry.draw(lane(root, "realize", table, name))
            if isinstance(entry, Declaration)
            else entry
            for name, entry in self.declared.items()
        }

    def like(self, entries: Mapping[str, object]) -> "Table":
        """A table of other entries under this one's metadata."""
        return Table(entries, **self.metadata)

    def drawn(self, root: int, table: str | None = None) -> "Table":
        """The table one seed draws, compiled, so nothing in it is left to draw; itself when it
        holds nothing to draw."""
        return self.like(self.like(self.draws(root, table)).table) if self.family else self

    def realize(self, seed: int) -> "Table":
        """One table of this family: every range drawn from `seed`, then its level cuts and scores
        fit to one draw of its rows. A table draws by its name, so a named table realizes the same
        alone and in a Schema; one with nothing left to draw is only fit again."""
        return Schema({self.name or "table": self}).realize(seed).tables[self.name or "table"]

    def record(
        self,
        name: str,
        latents: Mapping[str, np.ndarray],
        records: Mapping[str, pd.Series],
        rng: np.random.Generator,
        n: int,
    ) -> pd.Series:
        """What a column records, a drawn time in epoch seconds: a derived column computes it from
        the records it reads, any other observes its latent, then takes its range by level. A
        marginal over causes is taken in the order of a draw's rows, and one row has none."""
        if name in self.derived:
            return pd.Series(self.derived[name].compute(records))
        column = self.columns[name]
        if isinstance(column, PrimaryKey):
            return column.observe(name, latents, rng, n)
        if n == 1 and column.marginal is not None and (column.node.parents or column.node.across):
            raise ValueError("a marginal follows its causes in a draw's order: draw at least two")
        recorded = column.record(name, latents, rng, n)
        if column.kind == "timestamp" and column.values is None:
            seconds = recorded.to_numpy(dtype=float)
            if ((seconds < SPAN[0]) | (seconds > SPAN[1])).any():
                raise ValueError(
                    f"{name!r} records times outside {pd.Timestamp.min} to {pd.Timestamp.max}:"
                    " lower the weights or noise on the times it reads"
                )
        if column.by_level is None:
            return recorded
        parent = column.by_level[0]
        codes, levels = latents[parent + CODE][:, 0], self.columns[parent].categories
        return pd.Series(column.placed_by_level(recorded.to_numpy(dtype=float), codes, levels))

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
            {} if n is None else {"table": n},
            seed=seed,
            held={"table": held} if held else None,
        )
        return frames["table"], latents["table"]


class Schema:
    """The plan of a database: its tables, joined by the foreign keys they hold and the edges
    that read across them. It realizes every node, and records every column, in one topological
    order across tables, so a cause may read what another column records. A
    table's draws follow its name, else its key here, so a named table draws the same under any
    key."""

    def __init__(self, tables: Mapping[str, Table]) -> None:
        self.tables = dict(tables)
        self.identity = {table: scm.name or table for table, scm in self.tables.items()}
        named = Counter(self.identity.values())
        if shared := sorted(t for t, name in self.identity.items() if named[name] > 1):
            raise ValueError(
                f"tables {shared} share a name, so they would share their draws: give each table"
                " its own name, e.g. a copy as Table(entries, name=...)"
            )
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
        parents: dict[tuple[str, Hashable], tuple[tuple[str, Hashable], ...]] = {}
        for table, scm in self.tables.items():
            for vertex, reads in scm.graph.items():
                parents[table, vertex] = tuple((table, read) for read in reads)
        self.read: set[tuple[str, str]] = set()
        for (table, name), edges in self.crossings.items():
            sources = tuple(self.source(table, edge.parent) for edge in edges)
            parents[table, name] += tuple(source for source in sources if source[1] is not None)
            self.read |= {(t, read.column) for t, read in sources if isinstance(read, Recorded)}
        self.order = topological(parents)

    def bound(self, table: str, edge: Edge) -> Edge:
        """A crossing edge lined up with the column it reads; a record is a number to weigh."""
        origin, read = self.source(table, edge.parent)
        if read is None or isinstance(read, Recorded):
            return edge
        read = read if isinstance(edge.parent, Summary) else read.removesuffix(CODE)
        return bind(edge, self.tables[origin].table.get(read))

    def key(self, table: str, column: str) -> ForeignKey:
        if (table, column) not in self.keys:
            raise ValueError(f"{table!r} has no key column {column!r}")
        return self.keys[table, column]

    def source(self, table: str, tail: Hashable) -> tuple[str, str | Recorded | None]:
        """The node or record an edge of `table` reads through its tail; a bare count reads none."""
        if isinstance(tail, Recorded):
            return table, tail.vertex
        if isinstance(tail, Foreign):
            origin, read = self.key(table, tail.key).parent, tail.node
        elif isinstance(tail, Summary):
            if self.key(tail.table, tail.key).parent != table:
                raise ValueError(f"key {tail.key!r} of {tail.table!r} does not point at {table!r}")
            if tail.table in self.eventful:
                raise ValueError(f"a summary of {tail.table!r} would read later events")
            origin, read = tail.table, tail.node
        else:
            raise ValueError(f"{tail!r} is not a Foreign, a Summary or a Recorded")
        if isinstance(read, Recorded):
            if read.column not in self.tables[origin].readable:
                raise ValueError(
                    f"{origin}.{read.column} is read as recorded: Recorded reads a numeric,"
                    " timestamp or derived column"
                )
            return origin, read.vertex
        if read is not None and read not in self.tables[origin].nodes:
            raise ValueError(f"unknown node {origin}.{read}")
        return origin, read

    def links(self, rows: Mapping[str, int], root: int) -> Links:
        links = {}
        for (table, column), fk in self.keys.items():
            if fk.values is not None:
                links[table, column] = self.real_link(table, column, fk, rows[fk.parent])
                continue
            stream = lane(root, "link", self.identity[table], column)
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
        tail: Foreign | Summary | Recorded,
        rows: Mapping[str, int],
        latents: dict[str, dict[str, np.ndarray]],
        recorded: dict[str, dict[str, np.ndarray]],
        links: Links,
    ) -> np.ndarray:
        """The array a tail the schema resolves contributes to `table`, one row per row of it."""
        n = rows[table]
        if isinstance(tail, Recorded):
            return filled(recorded[table][tail.column], tail, f"{table}.{tail.column}")[:, None]
        if isinstance(tail, Foreign):
            fk = self.keys[table, tail.key]
            indices = links[table, tail.key]
            if isinstance(tail.node, Recorded):
                values, linked = np.full(n, np.nan), indices >= 0
                values[linked] = recorded[fk.parent][tail.node.column][indices[linked]]
                where = f"{tail.key}.{tail.node.column} of {table!r}"
                return filled(values, tail.node, where)[:, None]
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
        if isinstance(tail.node, Recorded):
            read = recorded[tail.table][tail.node.column]
            linked &= ~np.isnan(read)
            values = read[linked][:, None]
        elif tail.node is None:
            values = np.empty((int(linked.sum()), 1))
        else:
            values = latents[tail.table][tail.node][linked]
        out = AGGREGATES[tail.how](values, indices[linked], n)
        empty = np.bincount(indices[linked], minlength=n) == 0
        if empty.any() and tail.how not in COMPLETE:
            if tail.fill is None:
                read = getattr(tail.node, "column", tail.node)
                raise ValueError(
                    f"summary of {tail.table}.{read} met rows with nothing to aggregate; set fill"
                )
            out[empty] = tail.fill
        return out

    def propagate(
        self,
        rows: Mapping[str, int],
        links: Links,
        root: int,
        held: Mapping[str, Mapping[str, np.ndarray]],
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, pd.Series]]]:
        """Evaluate every node of every table once and record every column, in topological order,
        each with its own stream: a held node takes its values, any other its edges plus noise."""
        latents: dict[str, dict[str, np.ndarray]] = {table: {} for table in self.tables}
        records: dict[str, dict[str, pd.Series]] = {table: {} for table in self.tables}
        recorded: dict[str, dict[str, np.ndarray]] = {table: {} for table in self.tables}
        for table, name in self.order:
            scm, n = self.tables[table], rows[table]
            if isinstance(name, Recorded):
                column = name.column
                stream = lane(root, "column", self.identity[table], column)
                records[table][column] = scm.record(
                    column, latents[table], records[table], stream, n
                )
                if (table, column) in self.read:
                    recorded[table][column] = numbers(records[table][column], f"{table}.{column}")
                continue
            node = scm.nodes[name]
            if name in held.get(table, {}):
                latents[table][name] = checked(name, node, held[table][name], n)
                continue
            exogenous, across = (
                node.sample_noise(n, lane(root, "noise", self.identity[table], name)),
                None,
            )
            for edge in self.crossings.get((table, name), ()):
                term = edge.apply(self.resolve(table, edge.parent, rows, latents, recorded, links))
                across = term if across is None else across + term
            parents = {parent: latents[table][parent] for parent in node.parents}
            value = node.evaluate(parents, exogenous, across)
            latents[table][name] = checked(name, node, value, n)
        return latents, records

    def observe(
        self,
        rows: Mapping[str, int],
        records: dict[str, dict[str, pd.Series]],
        links: Links,
    ) -> dict[str, pd.DataFrame]:
        """One frame per table of its records, in the order the table declares them, with its key
        columns."""
        frames = {}
        for table, scm in self.tables.items():
            n = rows[table]
            framed = {
                name: c.framed(records[table][name])
                if isinstance(c, Column)
                else records[table][name]
                for name, c in scm.columns.items()
            }
            frame = pd.DataFrame(framed, index=range(n))
            for column, fk in scm.fkeys.items():
                indices, given = links[table, column], self.parent_keys(fk)
                if given is None:
                    frame[column] = pd.Series(indices, dtype="Int64").mask(indices < 0)
                else:
                    keys = given.astype("Int64") if pd.api.types.is_integer_dtype(given) else given
                    frame[column] = keys.take(indices).reset_index(drop=True).mask(indices < 0)
            for name in scm.derived:
                frame[name] = records[table][name]
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
        if family := [table for table, scm in self.tables.items() if scm.family]:
            raise ValueError(f"{family} hold ranges: draw one table with realize(seed)")
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
        latents, records = self.propagate(rows, links, root, held)
        return self.observe(rows, records, links), latents

    def realize(self, seed: int) -> "Schema":
        """One database of this family: see `Table.realize`."""
        drawn = {t: scm.drawn(seed, self.identity[t]) for t, scm in self.tables.items()}
        return Schema(drawn).fitted(seed)

    def fitted(self, seed: int) -> "Schema":
        """Scores and level cuts fit to one draw: see `plurel.fit.fitted`."""
        return fitted(self, seed)
