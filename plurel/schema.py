from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import index as as_index

import numpy as np
import pandas as pd

from plurel.links import Link, RandomLink, TreeLink
from plurel.mechanisms import Mechanism
from plurel.random import Seed, generator
from plurel.scm import SCM, Interventions, generations

Node = tuple[str, str]
Links = dict[Node, np.ndarray]


def _sum(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, values.shape[1]))
    np.add.at(out, index, values)
    return out


def _mean(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    count = np.bincount(index, minlength=n)[:, None]
    total = _sum(values, index, n)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _extreme(op: np.ufunc, start: float) -> Callable[..., np.ndarray]:
    def aggregate(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
        out = np.full((n, values.shape[1]), start)
        op.at(out, index, values)
        out[np.bincount(index, minlength=n) == 0] = np.nan
        return out

    return aggregate


AGGREGATES: dict[str, Callable[..., np.ndarray]] = {
    "count": lambda values, index, n: np.bincount(index, minlength=n)[:, None].astype(float),
    "sum": lambda values, index, n: _sum(values, index, n),
    "mean": _mean,
    "max": _extreme(np.maximum, -np.inf),
    "min": _extreme(np.minimum, np.inf),
}
COMPLETE = ("count", "sum")


@dataclass(frozen=True)
class FK:
    table: str
    column: str
    parent: str
    link: Link = RandomLink()
    nullable: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.nullable < 1.0:
            raise ValueError("nullable must be in [0, 1)")
        if isinstance(self.link, TreeLink) and self.table != self.parent:
            raise ValueError("a tree link needs a self-referential foreign key")


@dataclass(frozen=True)
class Port(Mechanism):
    table: str
    node: str
    via: str | None = None
    aggregate: str | None = None
    fill: float | None = None
    dim: int = 1

    def __post_init__(self) -> None:
        if self.aggregate is not None and self.aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {tuple(AGGREGATES)}")
        if self.aggregate == "count" and self.dim != 1:
            raise ValueError("a count has one dimension")

    def evaluate(self, latents: dict[str, np.ndarray], exogenous: np.ndarray) -> np.ndarray:
        return exogenous

    def resolve(self, source: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
        linked = index >= 0
        if self.aggregate is not None:
            out = AGGREGATES[self.aggregate](source[linked], index[linked], n)
            empty = np.bincount(index[linked], minlength=n) == 0
        else:
            out = np.empty((n, self.dim))
            out[linked] = source[index[linked]]
            empty = ~linked
        if empty.any() and self.aggregate not in COMPLETE:
            if self.fill is None:
                raise ValueError(
                    f"port {self.table}.{self.node} met rows without a match; set fill"
                )
            out[empty] = self.fill
        return out


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
        if len({(fk.table, fk.column) for fk in self.fkeys}) != len(self.fkeys):
            raise ValueError("foreign key columns must be unique per table")
        self.ports = {
            (table, name): (mechanism, self._fkey(table, name, mechanism))
            for table, scm in self.tables.items()
            for name, mechanism in scm.mechanisms.items()
            if isinstance(mechanism, Port)
        }
        parents: dict[Node, tuple[Node, ...]] = {}
        for table, scm in self.tables.items():
            for name, mechanism in scm.mechanisms.items():
                local = tuple((table, parent) for parent in mechanism.parents)
                source = ((mechanism.table, mechanism.node),) if isinstance(mechanism, Port) else ()
                parents[table, name] = local + source
        self.generations = generations(parents)
        self.order = tuple(node for generation in self.generations for node in generation)

    def _fkey(self, table: str, name: str, port: Port) -> FK:
        if port.table not in self.tables or port.node not in self.tables[port.table].mechanisms:
            raise ValueError(f"port {name!r} refers to unknown node {port.table}.{port.node}")
        child, parent = (port.table, table) if port.aggregate else (table, port.table)
        matches = [
            fk
            for fk in self.fkeys
            if (fk.table, fk.parent) == (child, parent) and port.via in (None, fk.column)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"port {name!r} needs exactly one foreign key from {child} to {parent}"
            )
        source = self.tables[port.table].mechanisms[port.node].dim
        if port.dim != (1 if port.aggregate == "count" else source):
            raise ValueError(f"port {name!r} declares dim {port.dim}, source has {source}")
        return matches[0]

    def links(self, rows: Mapping[str, int], rng: np.random.Generator) -> Links:
        links = {}
        for fk, stream in zip(self.fkeys, rng.spawn(len(self.fkeys))):
            drawn = np.asarray(fk.link.sample(rows[fk.table], rows[fk.parent], stream))
            if drawn.shape != (rows[fk.table],) or not np.issubdtype(drawn.dtype, np.integer):
                raise ValueError(f"link {fk.column!r} must return one integer per child row")
            if len(drawn) and (drawn.min() < -1 or drawn.max() >= rows[fk.parent]):
                raise ValueError(f"link {fk.column!r} points outside the parent table")
            index = drawn.astype(np.int64)
            if fk.nullable:
                index[stream.random(len(index)) < fk.nullable] = -1
            links[fk.table, fk.column] = index
        return links

    def evaluate(
        self,
        node: Node,
        rows: Mapping[str, int],
        latents: dict[str, dict[str, np.ndarray]],
        links: Links,
        stream: np.random.Generator,
        interventions: Mapping[str, Interventions],
    ) -> np.ndarray:
        table, name = node
        if node in self.ports and name not in interventions.get(table, {}):
            port, fk = self.ports[node]
            source = latents[port.table][port.node]
            index = links[fk.table, fk.column]
            latent = port.resolve(source, index, rows[table])
            if latent.shape != (rows[table], port.dim):
                raise ValueError(
                    f"{name!r} produced {latent.shape}, declared {(rows[table], port.dim)}"
                )
            return latent
        scm = self.tables[table]
        return scm.evaluate(name, rows[table], latents[table], stream, interventions.get(table, {}))

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
            for node in generation:
                latents[node[0]][node[1]] = self.evaluate(
                    node, rows, latents, links, streams[node], interventions
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
            frame = scm.observe(latents[table], stream, rows[table])
            for fk in self.fkeys:
                if fk.table == table:
                    index = links[table, fk.column]
                    frame[fk.column] = pd.Series(index, dtype="Int64").mask(index < 0)
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
            rows = {table: as_index(n) for table, n in rows.items()}
        except TypeError as error:
            raise ValueError("row counts must be integers") from error
        if set(rows) != set(self.tables) or min(rows.values(), default=0) < 0:
            raise ValueError("rows must give a non-negative count for every table")
        interventions = dict(interventions or {})
        if unknown := set(interventions) - set(self.tables):
            raise ValueError(f"interventions on unknown tables {sorted(unknown)}")
        for table, nodes in interventions.items():
            if unknown := set(nodes) - set(self.tables[table].mechanisms):
                raise ValueError(f"interventions on unknown nodes {sorted(unknown)} of {table!r}")
        linking, noise, observation = generator(seed).spawn(3)
        links = self.links(rows, linking)
        latents = self.propagate(rows, links, noise, interventions)
        return self.observe(rows, latents, links, observation), latents
