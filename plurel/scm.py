from collections.abc import Hashable, Mapping
from graphlib import CycleError, TopologicalSorter

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.mechanisms import Mechanism
from plurel.random import Seed, generator

Interventions = Mapping[str, float | np.ndarray]


def intervention(value: float | np.ndarray, n: int, dim: int) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape == (n,):
        value = value[:, None]
    try:
        return np.array(np.broadcast_to(value, (n, dim)))
    except ValueError as error:
        raise ValueError(f"intervention of shape {value.shape} does not fit {(n, dim)}") from error


def checked(name: str, mechanism: Mechanism, latent: np.ndarray, n: int) -> np.ndarray:
    if latent.shape != (n, mechanism.dim):
        raise ValueError(f"{name!r} produced {latent.shape}, declared {(n, mechanism.dim)}")
    if not np.isfinite(latent).all():
        raise ValueError(f"{name!r} produced non-finite values")
    return latent


def generations[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[tuple[T, ...], ...]:
    sorter = TopologicalSorter(parents)
    try:
        sorter.prepare()
    except CycleError as error:
        raise ValueError("nodes must form a directed acyclic graph") from error
    levels = []
    while sorter.is_active():
        ready = tuple(sorter.get_ready())
        levels.append(ready)
        sorter.done(*ready)
    return tuple(levels)


class SCM:
    def __init__(
        self,
        mechanisms: Mapping[str, Mechanism],
        columns: Mapping[str, Column],
        time_column: str | None = None,
    ) -> None:
        self.mechanisms = dict(mechanisms)
        for child, mechanism in self.mechanisms.items():
            unknown = set(mechanism.parents) - set(self.mechanisms)
            if unknown:
                raise ValueError(f"{child!r} refers to unknown parents {sorted(unknown)}")
        parents = {name: mechanism.parents for name, mechanism in self.mechanisms.items()}
        self.generations = generations(parents)
        self.order = tuple(name for generation in self.generations for name in generation)
        self.columns = dict(columns)
        for name, column in self.columns.items():
            if column.kind == "key":
                continue
            nodes = (
                {column.node, column.missing} if isinstance(column.missing, str) else {column.node}
            )
            if unknown := nodes - set(self.mechanisms):
                raise ValueError(f"column {name!r} refers to unknown nodes {sorted(unknown)}")
        kinds = {name: column.kind for name, column in self.columns.items()}
        for name, column in self.columns.items():
            if column.after is not None and (
                column.after == name or kinds.get(column.after) != "timestamp"
            ):
                raise ValueError(f"column {name!r} must come after another timestamp column")
        keys = [name for name, kind in kinds.items() if kind == "key"]
        if len(keys) > 1:
            raise ValueError("a table has at most one key column")
        if time_column is not None and kinds.get(time_column) != "timestamp":
            raise ValueError(f"time column {time_column!r} must be a timestamp column")
        self.pkey_column = keys[0] if keys else None
        self.time_column = time_column
        self.timestamp_nodes = frozenset(
            column.node for column in self.columns.values() if column.kind == "timestamp"
        )

    def evaluate(
        self,
        name: str,
        n: int,
        latents: dict[str, np.ndarray],
        stream: np.random.Generator,
        interventions: Interventions,
    ) -> np.ndarray:
        mechanism = self.mechanisms[name]
        if name in interventions:
            return intervention(interventions[name], n, mechanism.dim)
        parents = {parent: latents[parent] for parent in mechanism.parents}
        exogenous = mechanism.sample_noise(n, stream)
        return checked(name, mechanism, mechanism.evaluate(parents, exogenous), n)

    def simulate(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> dict[str, np.ndarray]:
        interventions = dict(interventions or {})
        if unknown := set(interventions) - set(self.mechanisms):
            raise ValueError(f"interventions on unknown nodes {sorted(unknown)}")
        streams = dict(zip(self.order, generator(seed).spawn(len(self.order))))
        latents: dict[str, np.ndarray] = {}
        for generation in self.generations:
            for name in generation:
                latents[name] = self.evaluate(name, n, latents, streams[name], interventions)
        return latents

    def observe(
        self, latents: dict[str, np.ndarray], rng: np.random.Generator, n: int
    ) -> pd.DataFrame:
        observed = {name: column.observe(latents, rng, n) for name, column in self.columns.items()}
        frame = pd.DataFrame(observed, index=range(n))
        for name, column in self.columns.items():
            if column.after is not None and (frame[name] < frame[column.after]).any():
                raise ValueError(f"column {name!r} precedes {column.after!r} on some rows")
        return frame

    def sample(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> pd.DataFrame:
        return self.sample_with_latents(n, seed=seed, interventions=interventions)[0]

    def sample_with_latents(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        root = generator(seed)
        latents = self.simulate(n, seed=root, interventions=interventions)
        return self.observe(latents, root.spawn(1)[0], n), latents
