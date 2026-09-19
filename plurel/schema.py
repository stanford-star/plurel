from collections.abc import Callable, Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

import numpy as np
import pandas as pd

from plurel.links import Link, RandomLink
from plurel.mechanisms import Mechanism
from plurel.random import Seed, generator
from plurel.scm import SCM, Interventions, checked, intervention

Node = tuple[str, str]


def _sum(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, values.shape[1]))
    np.add.at(out, index, values)
    return out


def _mean(values: np.ndarray, index: np.ndarray, n: int, fill: float) -> np.ndarray:
    count = np.bincount(index, minlength=n)[:, None]
    total = _sum(values, index, n)
    return np.divide(total, count, out=np.full_like(total, fill), where=count > 0)


def _extreme(op: np.ufunc, start: float) -> Callable[..., np.ndarray]:
    def aggregate(values: np.ndarray, index: np.ndarray, n: int, fill: float) -> np.ndarray:
        out = np.full((n, values.shape[1]), start)
        op.at(out, index, values)
        out[np.bincount(index, minlength=n) == 0] = fill
        return out

    return aggregate


AGGREGATES: dict[str, Callable[..., np.ndarray]] = {
    "count": lambda values, index, n, fill: np.bincount(index, minlength=n)[:, None].astype(float),
    "sum": lambda values, index, n, fill: _sum(values, index, n),
    "mean": _mean,
    "max": _extreme(np.maximum, -np.inf),
    "min": _extreme(np.minimum, np.inf),
}


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


@dataclass(frozen=True)
class Port(Mechanism):
    table: str
    node: str
    via: str | None = None
    aggregate: str | None = None
    fill: float = 0.0
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
            return AGGREGATES[self.aggregate](source[linked], index[linked], n, self.fill)
        out = np.full((n, self.dim), self.fill)
        out[linked] = source[index[linked]]
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
        parents: dict[Node, list[Node]] = {}
        for table, scm in self.tables.items():
            for name, mechanism in scm.mechanisms.items():
                parents[table, name] = [(table, parent) for parent in mechanism.parents]
                if isinstance(mechanism, Port):
                    parents[table, name].append((mechanism.table, mechanism.node))
        try:
            self.order = tuple(TopologicalSorter(parents).static_order())
        except CycleError as error:
            raise ValueError("nodes must form a directed acyclic graph across tables") from error

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
        if set(rows) != set(self.tables) or min(rows.values(), default=0) < 0:
            raise ValueError("rows must give a non-negative count for every table")
        interventions = dict(interventions or {})
        if unknown := set(interventions) - set(self.tables):
            raise ValueError(f"interventions on unknown tables {sorted(unknown)}")
        rng = generator(seed)
        links = {}
        for fk in self.fkeys:
            index = fk.link.sample(rows[fk.table], rows[fk.parent], rng)
            if fk.nullable:
                index[rng.random(len(index)) < fk.nullable] = -1
            links[fk] = index
        exogenous = {
            node: self.tables[node[0]].mechanisms[node[1]].sample_noise(rows[node[0]], rng)
            for node in self.order
        }
        latents: dict[str, dict[str, np.ndarray]] = {table: {} for table in self.tables}
        for table, name in self.order:
            mechanism = self.tables[table].mechanisms[name]
            n = rows[table]
            if name in interventions.get(table, {}):
                latents[table][name] = intervention(interventions[table][name], n, mechanism.dim)
            elif (table, name) in self.ports:
                port, fk = self.ports[table, name]
                source = latents[port.table][port.node]
                latents[table][name] = checked(name, port, port.resolve(source, links[fk], n), n)
            else:
                parents = {parent: latents[table][parent] for parent in mechanism.parents}
                latent = mechanism.evaluate(parents, exogenous[table, name])
                latents[table][name] = checked(name, mechanism, latent, n)
        frames = {}
        for table, scm in self.tables.items():
            frame = scm.observe(latents[table], rng, rows[table])
            for fk in self.fkeys:
                if fk.table == table:
                    frame[fk.column] = pd.Series(links[fk], dtype="Int64").mask(links[fk] < 0)
            frames[table] = frame
        return frames, latents
