from collections.abc import Hashable, Mapping
from graphlib import CycleError, TopologicalSorter

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.mechanisms import Node
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


def checked(name: str, node: Node, latent: np.ndarray, n: int) -> np.ndarray:
    if latent.shape != (n, node.dim):
        raise ValueError(f"{name!r} produced {latent.shape}, declared {(n, node.dim)}")
    if not np.isfinite(latent).all():
        raise ValueError(f"{name!r} produced non-finite values")
    return latent


def topological[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as error:
        raise ValueError("nodes must form a directed acyclic graph") from error


class SCM:
    """A table's DAG. Edge tails that cross keys are `inputs` a Schema supplies; a table with
    inputs samples only through a Schema."""

    def __init__(
        self,
        nodes: Mapping[str, Node],
        columns: Mapping[str, Column],
        time_column: str | None = None,
    ) -> None:
        self.nodes = dict(nodes)
        inputs: set[Hashable] = set()
        for child, node in self.nodes.items():
            local = {parent for parent in node.parents if isinstance(parent, str)}
            if unknown := local - set(self.nodes):
                raise ValueError(f"{child!r} refers to unknown parents {sorted(unknown)}")
            inputs |= set(node.parents) - local
        parents = {
            name: tuple(parent for parent in node.parents if isinstance(parent, str))
            for name, node in self.nodes.items()
        }
        self.order = topological(parents)
        self.columns = dict(columns)
        for name, column in self.columns.items():
            if column.kind == "key":
                continue
            needed = {column.node}
            if isinstance(column.missing, str):
                needed.add(column.missing)
            if unknown := needed - set(self.nodes):
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
        self.inputs = frozenset(inputs)

    def evaluate(
        self,
        name: str,
        n: int,
        latents: Mapping[Hashable, np.ndarray],
        stream: np.random.Generator,
        interventions: Interventions,
    ) -> np.ndarray:
        node = self.nodes[name]
        if name in interventions:
            return intervention(interventions[name], n, node.dim)
        parents = {parent: latents[parent] for parent in node.parents}
        exogenous = node.sample_noise(n, stream)
        return checked(name, node, node.evaluate(parents, exogenous), n)

    def simulate(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> dict[str, np.ndarray]:
        if self.inputs:
            raise ValueError("the table reads through keys; sample it through a Schema")
        interventions = dict(interventions or {})
        if unknown := set(interventions) - set(self.nodes):
            raise ValueError(f"interventions on unknown nodes {sorted(unknown)}")
        latents: dict[str, np.ndarray] = {}
        for name, stream in zip(self.order, generator(seed).spawn(len(self.order))):
            latents[name] = self.evaluate(name, n, latents, stream, interventions)
        return latents

    def observe(
        self, latents: Mapping[Hashable, np.ndarray], rng: np.random.Generator, n: int
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
