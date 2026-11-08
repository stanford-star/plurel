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
    key_node,
)
from plurel.describe import describe_schema, describe_table, explain
from plurel.draw import Linked, Run
from plurel.fit import fitted
from plurel.graph import (
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

Entry = Declaration | ForeignKey | PrimaryKey | Derived | Node | Column  # what a table holds
Vertex = tuple[str, Hashable]
SPAN = (pd.Timestamp.min.value // 10**9 + 1, pd.Timestamp.max.value // 10**9)  # epoch seconds


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


def bound_local(node: Node, entries: Mapping[str, object]) -> Node:
    """The node with each edge lined up with the entry of its table it reads by name."""
    edges = (
        bind(edge, entries.get(edge.parent.removesuffix(CODE)))
        if isinstance(edge.parent, str)
        else edge
        for edge in node.edges
    )
    return replace(node, edges=tuple(edges))


def topological[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as error:
        raise ValueError(
            "nodes and the records they read must form a directed acyclic graph, but"
            f" {error.args[1]} read each other in a cycle"
        ) from error


class Table:
    """A table: a dict of its keys, its columns (`Numeric`, `Categorical`, `Timestamp`, `Real`,
    `Derived`) and the `Hidden` causes they read, as latents or, `Recorded`, as recorded values.
    The lower-level `Node` and `Column` are accepted too. A table holding ranges is a family:
    `realize(seed)` draws one table of it. `doc`, `target` and `sources` say what the table
    records, what it is read for and where its numbers come from; `docs` holds each entry's description, from its `doc` or given, and survives realizing."""

    def __init__(
        self,
        table: Mapping[str, "Entry"],
        *,
        time_column: str | None = None,
        name: str | None = None,
        doc: str | None = None,
        target: str | None = None,
        sources: str | None = None,
        docs: Mapping[str, str] | None = None,
    ) -> None:
        self.declared = dict(table)
        self.authored = self.declared
        self.name, self.doc, self.target, self.sources = name, doc, target, sources
        given = {k: e.doc for k, e in self.declared.items() if getattr(e, "doc", None)}
        given |= dict(docs or {})
        if unknown := set(given) - set(self.declared):
            raise ValueError(f"docs name no entries of the table: {sorted(unknown)}")
        self.docs = {name: given[name] for name in self.declared if name in given}
        self.family = any(
            isinstance(e, Declaration | ForeignKey) and e.family for e in self.declared.values()
        )
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
        counted = [name for name, fk in self.fkeys.items() if fk.counts is not None]
        if len(counted) > 1:
            raise ValueError(f"keys {counted} would each set the table's size: give one counts")
        if counted and self.size is not None:
            raise ValueError(f"{counted[0]!r} counts rows of a table whose rows are given")
        self.counted = counted[0] if counted else None
        for name, fk in self.fkeys.items():
            for column in (*fk.match, *fk.together):
                if (
                    column == name
                    or column not in self.fkeys
                    and kinds.get(column) != "categorical"
                ):
                    raise ValueError(
                        f"{name!r} reads {column!r}: a key matches, and is unique, on a categorical"
                        " or another key of its table"
                    )
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
            self.check_reads(node, f"{name!r}")
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

    def check_reads(self, node: Node, reader: str) -> None:
        """That `reader`, a column of this table or a key reading it, reads only this table's
        columns: nodes by name, and numbers, times and derived values as recorded."""
        if unknown := set(node.parents) - set(self.nodes):
            raise ValueError(
                f"{reader} refers to unknown parents {sorted(unknown)}, no columns of the table"
            )
        for edge in node.across:
            if isinstance(edge.parent, Recorded):
                self.check_recorded(edge.parent.column, reader)

    def check_recorded(self, column: str, reader: str) -> None:
        """That `reader` may read `column` of this table as recorded: a number, a time or a
        derived value."""
        if column in self.readable:
            return
        entry = self.table.get(column)
        if entry is None:
            why = "no column of the table"
        elif isinstance(entry, ForeignKey):
            why = "a foreign key"
        elif isinstance(entry, Node):
            why = "a hidden cause"
        else:
            why = f"a {entry.kind} column"
        raise ValueError(
            f"{reader} reads {column!r} as recorded, which is {why}: Recorded reads a numeric,"
            " timestamp or derived column"
        )

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
            or isinstance(entry, ForeignKey)
            and (entry.counts is not None or bool(entry.prefer))
            else entry
            for name, entry in self.declared.items()
        }

    def like(self, entries: Mapping[str, object]) -> "Table":
        """A table of other entries under this one's metadata and authored entries."""
        table = Table(entries, **self.metadata)
        table.authored = self.authored
        return table

    def drawn(self, root: int, table: str | None = None) -> "Table":
        """The table one seed draws, compiled, so nothing in it is left to draw; itself when it
        holds nothing to draw."""
        return self.like(self.like(self.draws(root, table)).table) if self.family else self

    def realize(self, seed: int, rows: int | None = None) -> "Table":
        """One table of this family: every range drawn from `seed`, then its level cuts and scores
        fit to one draw of its rows, `rows` of them if given. A table draws by its name, so a named
        table realizes the same alone and in a Schema; one with nothing left to draw is only fit
        again."""
        name = self.name or "table"
        given = None if rows is None else {name: rows}
        return Schema({name: self}).realize(seed, given).tables[name]

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

    def describe(self) -> dict[str, object]:
        """This table as resolved, as plain data: see `plurel.describe`."""
        return describe_table(self)

    def explain(self) -> str:
        """This table as resolved, as readable lines."""
        return explain(self.describe())

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
        for name, entry in self.table.items():
            if isinstance(entry, Column):
                self.table[name] = replace(entry, node=bound_local(entry.node, self.table))
            elif isinstance(entry, Node):
                self.table[name] = bound_local(entry, self.table)

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
        self.sized = {t: scm.counted for t, scm in self.tables.items() if scm.counted}
        self.choosers = {
            (table, column): chooser
            for (table, column), fk in keys.items()
            if (chooser := self.chooser(table, column, fk)) is not None
        }
        parents: dict[Vertex, tuple[Vertex, ...]] = {}
        for table, scm in self.tables.items():
            for vertex, reads in scm.graph.items():
                parents[table, vertex] = tuple((table, read) for read in reads)
        resolved: list[Vertex] = []
        for (table, name), edges in self.crossings.items():
            parents[table, name] += (found := self.reads(table, edges))
            resolved += found
        for (table, column), (_, edges) in self.choosers.items():
            resolved += self.reads(keys[table, column].parent, edges)
        self.read = {(t, read.column) for t, read in resolved if isinstance(read, Recorded)}
        for (table, column), fk in keys.items():
            parents[table, Linked(column)] = self.link_reads(table, column, fk)
        for table in self.sized:
            for vertex in self.tables[table].graph:
                parents[table, vertex] += tuple(self.size(table))
        self.order = topological(parents)

    def chooser(
        self, table: str, column: str, fk: ForeignKey
    ) -> tuple[Node, tuple[Edge, ...]] | None:
        """The node weighing a key's parents, read from the parent's columns, and the edges the
        schema resolves for it; None for a key that reads no column of its parent."""
        if fk.counts is not None and fk.parent == table:
            raise ValueError(f"{column!r} would count rows of its own table")
        self.check_match(table, column, fk)
        causes, noise = (fk.counts.causes, fk.counts.noise) if fk.counts else (fk.prefer, 0)
        parent = self.tables[fk.parent]
        node = key_node(causes, noise, parent.fkeys)
        if node is None:
            return None
        what = "counts its rows" if fk.counts else "weighs its parents"
        try:
            parent.check_reads(node, "the key")
            node = bound_local(node, parent.table)
            return node, tuple(self.bound(fk.parent, edge) for edge in node.across)
        except ValueError as error:
            raise ValueError(f"{column!r} of {table!r} {what} in {fk.parent!r}: {error}") from error

    def link_reads(self, table: str, column: str, fk: ForeignKey) -> tuple[Vertex, ...]:
        """What a key's links wait on: the sizes of both tables, the columns it matches and what
        its weights read."""
        reads = [] if column == self.sized.get(table) else self.size(table)
        reads += self.size(fk.parent)
        for mine, theirs in fk.match.items():
            reads += [self.matched_vertex(table, mine), self.matched_vertex(fk.parent, theirs)]
        reads += [self.matched_vertex(table, column) for column in fk.together]
        if (table, column) in self.choosers:
            node, edges = self.choosers[table, column]
            reads += [(fk.parent, parent) for parent in node.parents]
            reads += self.reads(fk.parent, edges)
        return tuple(reads)

    def size(self, table: str) -> list[Vertex]:
        """The vertex a table's size waits on: the key that counts its rows, if one does."""
        return [(table, Linked(self.sized[table]))] if table in self.sized else []

    def reads(self, table: str, edges: tuple[Edge, ...]) -> tuple[Vertex, ...]:
        """What edges a node of `table` resolves read: nodes, records and the links they cross."""
        out: list[Vertex] = []
        for edge in edges:
            origin, read = self.source(table, edge.parent)
            if read is not None:
                out.append((origin, read))
            if isinstance(edge.parent, Foreign):
                out.append((table, Linked(edge.parent.key)))
            elif isinstance(edge.parent, Summary):
                out.append((edge.parent.table, Linked(edge.parent.key)))
        return tuple(out)

    def matched_vertex(self, table: str, column: str) -> Vertex:
        """What a match reads of a column: a key's links or a categorical's record."""
        keyed = column in self.tables[table].fkeys
        return table, Linked(column) if keyed else Recorded(column)

    def check_match(self, table: str, column: str, fk: ForeignKey) -> None:
        """A match pairs two categoricals, or two keys pointing at the same table."""
        parent = self.tables[fk.parent]
        for mine, theirs in fk.match.items():
            if mine in self.tables[table].fkeys:
                pointed = self.keys[table, mine].parent
                if theirs not in parent.fkeys or parent.fkeys[theirs].parent != pointed:
                    raise ValueError(
                        f"{column!r} matches key {mine!r} to {fk.parent}.{theirs}, which is no key"
                        f" of {pointed!r}"
                    )
            elif getattr(parent.columns.get(theirs), "kind", None) != "categorical":
                raise ValueError(
                    f"{column!r} matches {mine!r} to {fk.parent}.{theirs}, which is no categorical"
                )

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
            self.tables[origin].check_recorded(read.column, f"a cause of {table!r} across a key")
            return origin, read.vertex
        if read is not None and read not in self.tables[origin].nodes:
            raise ValueError(f"unknown node {origin}.{read}")
        return origin, read

    def parent_keys(self, fk: ForeignKey) -> pd.Series | None:
        """The parent's given key values, None when its key is its row positions."""
        parent = self.tables[fk.parent]
        return None if parent.pkey_column is None else parent.columns[parent.pkey_column].values

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
        counted = set(self.sized)
        if not set(self.tables) - counted <= set(rows) <= set(self.tables):
            raise ValueError(
                "rows must give a non-negative count for every table, which a table whose key"
                " counts its rows may leave out"
            )
        if min(rows.values(), default=0) < 0:
            raise ValueError("rows must give a non-negative count for every table")
        held = dict(held or {})
        for table, nodes in held.items():
            if table not in self.tables:
                raise ValueError(f"held values for unknown table {table!r}")
            if unknown := set(nodes) - set(self.tables[table].nodes):
                raise ValueError(f"held values for unknown nodes {sorted(unknown)} of {table!r}")
        root = int(generator(seed).integers(2**63))
        for table, scm in self.tables.items():
            if scm.real:
                held[table] = {**scm.real, **held.get(table, {})}
        run = Run(self, rows, root).propagate(held)
        return run.frames(), run.latents

    def describe(self) -> dict[str, object]:
        """Every table as resolved, as plain data: see `plurel.describe`."""
        return describe_schema(self)

    def explain(self) -> str:
        """Every table as resolved, as readable lines."""
        return explain(self.describe())

    def realize(self, seed: int, rows: Mapping[str, int] | None = None) -> "Schema":
        """One database of this family: see `Table.realize`; `rows` sets the calibration draw's
        row counts, see `plurel.fit.fitted`."""
        drawn = {t: scm.drawn(seed, self.identity[t]) for t, scm in self.tables.items()}
        return Schema(drawn).fitted(seed, rows)

    def fitted(self, seed: int, rows: Mapping[str, int] | None = None) -> "Schema":
        """Scores and level cuts fit to one draw: see `plurel.fit.fitted`."""
        return fitted(self, seed, rows)
