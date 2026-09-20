from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from operator import index as as_int

import numpy as np
import pandas as pd

from plurel.links import Link, RandomLink, TreeLink
from plurel.random import Seed, generator
from plurel.scm import SCM, Interventions, generations

Location = tuple[str, str]
Links = dict[Location, np.ndarray]

COMPLETE = ("count", "sum")


def _sum(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, values.shape[1]))
    np.add.at(out, indices, values)
    return out


def _mean(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    count = np.bincount(indices, minlength=n)[:, None]
    total = _sum(values, indices, n)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _extreme(op: np.ufunc, start: float) -> Callable[..., np.ndarray]:
    def aggregate(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
        out = np.full((n, values.shape[1]), start)
        op.at(out, indices, values)
        out[np.bincount(indices, minlength=n) == 0] = np.nan
        return out

    return aggregate


AGGREGATES: dict[str, Callable[..., np.ndarray]] = {
    "count": lambda values, indices, n: np.bincount(indices, minlength=n)[:, None].astype(float),
    "sum": lambda values, indices, n: _sum(values, indices, n),
    "mean": _mean,
    "max": _extreme(np.maximum, -np.inf),
    "min": _extreme(np.minimum, np.inf),
}


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


@dataclass(frozen=True)
class Foreign:
    """Tail of an edge crossing a key: `node` of the row that `key` points at."""

    key: str
    node: str


@dataclass(frozen=True)
class Summary:
    """Tail of an edge aggregating, per row, the rows of `table` that point at it through
    `key`; `fill` is read where no row does, needed for mean, max and min."""

    table: str
    key: str
    node: str
    how: str
    fill: float | None = None

    def __post_init__(self) -> None:
        if self.how not in AGGREGATES:
            raise ValueError(f"how must be one of {tuple(AGGREGATES)}")
        if self.fill is not None and (self.how in COMPLETE or not np.isfinite(self.fill)):
            raise ValueError("fill is a finite value for mean, max or min only")


@dataclass(frozen=True)
class Orphan:
    """Missingness marker for a column: the rows whose `key` is null."""

    key: str


@dataclass(frozen=True)
class Childless:
    """Missingness marker for a column: the rows no row of `table` points at through `key`."""

    table: str
    key: str


class Schema:
    def __init__(self, tables: Mapping[str, SCM], fkeys: tuple[FK, ...] = ()) -> None:
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
        parents: dict[Location, tuple[Location, ...]] = {}
        for table, scm in self.tables.items():
            for name, node in scm.nodes.items():
                parents[table, name] = tuple(self.source(table, tail) for tail in node.parents)
            for column in scm.columns.values():
                if column.kind != "key" and not isinstance(column.missing, int | float | str):
                    self.check_marker(table, column.missing)
        self.generations = generations(parents)
        self.order = tuple(location for generation in self.generations for location in generation)

    def key(self, table: str, column: str) -> FK:
        if (table, column) not in self.keys:
            raise ValueError(f"{table!r} has no key column {column!r}")
        return self.keys[table, column]

    def source(self, table: str, tail: Hashable) -> Location:
        """The node an edge tail of `table` reads, local or across a key."""
        if isinstance(tail, str):
            origin, node = table, tail
        elif isinstance(tail, Foreign):
            origin, node = self.key(table, tail.key).parent, tail.node
        elif isinstance(tail, Summary):
            if self.key(tail.table, tail.key).parent != table:
                raise ValueError(f"key {tail.key!r} of {tail.table!r} does not point at {table!r}")
            origin, node = tail.table, tail.node
        else:
            raise ValueError(f"{tail!r} is not a node name, a Foreign or a Summary")
        if node not in self.tables[origin].nodes:
            raise ValueError(f"unknown node {origin}.{node}")
        return origin, node

    def check_marker(self, table: str, marker: Hashable) -> None:
        if isinstance(marker, Orphan):
            self.key(table, marker.key)
        elif isinstance(marker, Childless):
            if self.key(marker.table, marker.key).parent != table:
                raise ValueError(
                    f"key {marker.key!r} of {marker.table!r} does not point at {table!r}"
                )
        else:
            raise ValueError(f"{marker!r} is not a missingness rate, node or marker")

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

    def indicator(
        self, table: str, marker: Orphan | Childless, rows: Mapping[str, int], links: Links
    ) -> np.ndarray:
        """A two-class indicator, class one where the marker says the value is missing."""
        if isinstance(marker, Orphan):
            missing = links[table, marker.key] < 0
        else:
            indices = links[marker.table, marker.key]
            missing = np.bincount(indices[indices >= 0], minlength=rows[table]) == 0
        return np.eye(2)[missing.astype(int)]

    def evaluate(
        self,
        location: Location,
        rows: Mapping[str, int],
        latents: dict[str, dict[str, np.ndarray]],
        links: Links,
        stream: np.random.Generator,
        interventions: Mapping[str, Interventions],
    ) -> np.ndarray:
        table, name = location
        scm, forced = self.tables[table], interventions.get(table, {})
        inputs = {}
        if name not in forced:
            for tail in scm.nodes[name].parents:
                if not isinstance(tail, str):
                    inputs[tail] = self.resolve(table, tail, rows, latents, links)
        return scm.evaluate(name, rows[table], {**latents[table], **inputs}, stream, forced)

    def propagate(
        self,
        rows: Mapping[str, int],
        links: Links,
        rng: np.random.Generator,
        interventions: Mapping[str, Interventions],
    ) -> dict[str, dict[str, np.ndarray]]:
        streams = dict(zip(self.order, rng.spawn(len(self.order))))
        latents: dict[str, dict[str, np.ndarray]] = {table: {} for table in self.tables}
        for generation in self.generations:
            for location in generation:
                latents[location[0]][location[1]] = self.evaluate(
                    location, rows, latents, links, streams[location], interventions
                )
        return latents

    def observe(
        self,
        rows: Mapping[str, int],
        latents: dict[str, dict[str, np.ndarray]],
        links: Links,
        rng: np.random.Generator,
    ) -> dict[str, pd.DataFrame]:
        frames = {}
        for (table, scm), stream in zip(self.tables.items(), rng.spawn(len(self.tables))):
            markers = {
                column.missing: self.indicator(table, column.missing, rows, links)
                for column in scm.columns.values()
                if column.kind != "key" and not isinstance(column.missing, int | float | str)
            }
            frame = scm.observe({**latents[table], **markers}, stream, rows[table])
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
        interventions: Mapping[str, Interventions] | None = None,
    ) -> dict[str, pd.DataFrame]:
        return self.sample_with_latents(rows, seed=seed, interventions=interventions)[0]

    def sample_with_latents(
        self,
        rows: Mapping[str, int],
        *,
        seed: Seed = None,
        interventions: Mapping[str, Interventions] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:
        try:
            rows = {table: as_int(n) for table, n in rows.items()}
        except TypeError as error:
            raise ValueError("row counts must be integers") from error
        if set(rows) != set(self.tables) or min(rows.values(), default=0) < 0:
            raise ValueError("rows must give a non-negative count for every table")
        interventions = dict(interventions or {})
        if unknown := set(interventions) - set(self.tables):
            raise ValueError(f"interventions on unknown tables {sorted(unknown)}")
        for table, nodes in interventions.items():
            if unknown := set(nodes) - set(self.tables[table].nodes):
                raise ValueError(f"interventions on unknown nodes {sorted(unknown)} of {table!r}")
        linking, noise, observation = generator(seed).spawn(3)
        links = self.links(rows, linking)
        latents = self.propagate(rows, links, noise, interventions)
        return self.observe(rows, latents, links, observation), latents
