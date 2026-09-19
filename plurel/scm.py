from collections.abc import Mapping

import networkx as nx
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
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.mechanisms)
        for child, mechanism in self.mechanisms.items():
            for parent in mechanism.parents:
                if parent not in self.mechanisms:
                    raise ValueError(f"{child!r} refers to unknown parent {parent!r}")
                self.graph.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("mechanisms must form a directed acyclic graph")
        self.order = tuple(nx.topological_sort(self.graph))

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
