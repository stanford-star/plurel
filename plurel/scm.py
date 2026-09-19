from collections.abc import Mapping
from graphlib import CycleError, TopologicalSorter

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.mechanisms import Mechanism
from plurel.random import Seed, generator

Interventions = Mapping[str, float | np.ndarray]


def _intervention(value: float | np.ndarray, n: int, dim: int) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape == (n,):
        value = value[:, None]
    return np.array(np.broadcast_to(value, (n, dim)))


class SCM:
    def __init__(self, mechanisms: Mapping[str, Mechanism], columns: Mapping[str, Column]) -> None:
        self.mechanisms = dict(mechanisms)
        for child, mechanism in self.mechanisms.items():
            unknown = set(mechanism.parents) - set(self.mechanisms)
            if unknown:
                raise ValueError(f"{child!r} refers to unknown parents {sorted(unknown)}")
        parents = {name: mechanism.parents for name, mechanism in self.mechanisms.items()}
        try:
            self.order = tuple(TopologicalSorter(parents).static_order())
        except CycleError as error:
            raise ValueError("mechanisms must form a directed acyclic graph") from error
        self.columns = dict(columns)
        for name, column in self.columns.items():
            nodes = (
                {column.node, column.missing} if isinstance(column.missing, str) else {column.node}
            )
            if unknown := nodes - set(self.mechanisms):
                raise ValueError(f"column {name!r} refers to unknown nodes {sorted(unknown)}")

    def simulate(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> dict[str, np.ndarray]:
        unknown = set(interventions or ()) - set(self.mechanisms)
        if unknown:
            raise ValueError(f"interventions on unknown nodes {sorted(unknown)}")
        rng = generator(seed)
        exogenous = {name: self.mechanisms[name].sample_noise(n, rng) for name in self.order}
        latents: dict[str, np.ndarray] = {}
        for name in self.order:
            mechanism = self.mechanisms[name]
            if interventions and name in interventions:
                latents[name] = _intervention(interventions[name], n, mechanism.dim)
                continue
            parents = {parent: latents[parent] for parent in mechanism.parents}
            latent = mechanism.evaluate(parents, exogenous[name])
            if latent.shape != (n, mechanism.dim):
                raise ValueError(f"{name!r} produced {latent.shape}, declared {(n, mechanism.dim)}")
            latents[name] = latent
        return latents

    def sample(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> pd.DataFrame:
        return self.sample_with_latents(n, seed=seed, interventions=interventions)[0]

    def sample_with_latents(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        rng = generator(seed)
        latents = self.simulate(n, seed=rng, interventions=interventions)
        observed = {name: column.observe(latents, rng) for name, column in self.columns.items()}
        return pd.DataFrame(observed, index=range(n)), latents
