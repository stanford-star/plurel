from collections.abc import Mapping
from graphlib import CycleError, TopologicalSorter

import numpy as np

from plurel.mechanisms import Mechanism
from plurel.random import Seed, generator

Interventions = Mapping[str, float | np.ndarray]


def _intervention(value: float | np.ndarray, n: int, dim: int) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape == (n,):
        value = value[:, None]
    return np.array(np.broadcast_to(value, (n, dim)))


class SCM:
    def __init__(self, mechanisms: Mapping[str, Mechanism]) -> None:
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

    def simulate(
        self, n: int, *, seed: Seed = None, interventions: Interventions | None = None
    ) -> dict[str, np.ndarray]:
        unknown = set(interventions or ()) - set(self.mechanisms)
        if unknown:
            raise ValueError(f"interventions on unknown nodes {sorted(unknown)}")
        rng = generator(seed)
        exogenous = {name: self.mechanisms[name].sample_noise(n, rng) for name in self.order}
        values: dict[str, np.ndarray] = {}
        for name in self.order:
            mechanism = self.mechanisms[name]
            if interventions and name in interventions:
                values[name] = _intervention(interventions[name], n, mechanism.dim)
                continue
            parents = {parent: values[parent] for parent in mechanism.parents}
            value = mechanism.evaluate(parents, exogenous[name])
            if value.shape != (n, mechanism.dim):
                raise ValueError(f"{name!r} produced {value.shape}, declared {(n, mechanism.dim)}")
            values[name] = value
        return values
