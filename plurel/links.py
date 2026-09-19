from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from plurel.distributions import Distribution

CHUNK_BYTES = 100_000_000


@runtime_checkable
class Link(Protocol):
    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray: ...


def clusters(n: int, hierarchy: tuple[int, ...]) -> np.ndarray:
    base = np.arange(n) // int(np.ceil(n / np.prod(hierarchy)))
    strides = np.cumprod((1, *hierarchy[:0:-1]))[::-1]
    return (base[:, None] // strides[None, :]) % np.asarray(hierarchy)[None, :]


@dataclass(frozen=True)
class RandomLink:
    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, n_parent, n_child)


@dataclass(frozen=True)
class HSBMLink:
    parent_hierarchy: tuple[int, ...] = (1,)
    child_hierarchy: tuple[int, ...] = (1,)
    within: float = 0.9
    between: tuple[float, float] = (0.001, 0.002)
    attractiveness: Distribution | None = None
    inactive: float = 0.0

    def __post_init__(self) -> None:
        if len(self.parent_hierarchy) != len(self.child_hierarchy):
            raise ValueError("one cluster count per level on both sides")
        if not 0.0 <= self.inactive < 1.0:
            raise ValueError("inactive must be in [0, 1)")

    def affinity(self, parent: int, child: int, rng: np.random.Generator) -> np.ndarray:
        affinity = rng.uniform(*self.between, (parent, child))
        index = np.arange(max(parent, child))
        affinity[index % parent, index % child] = self.within
        return affinity

    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        parent_clusters = clusters(n_parent, self.parent_hierarchy)
        child_clusters = clusters(n_child, self.child_hierarchy)
        levels = [
            np.log(self.affinity(parent, child, rng))
            for parent, child in zip(self.parent_hierarchy, self.child_hierarchy)
        ]
        log_weight = np.zeros(n_parent)
        if self.attractiveness is not None:
            weight = self.attractiveness.sample(n_parent, rng)
            log_weight += np.log(np.maximum(weight, np.finfo(float).tiny))
        inactive = rng.permutation(n_parent)[: min(round(n_parent * self.inactive), n_parent - 1)]
        log_weight[inactive] = -np.inf
        parents = np.empty(n_child, dtype=np.int64)
        chunk = max(1, min(n_child, CHUNK_BYTES // (8 * n_parent)))
        for start in range(0, n_child, chunk):
            stop = min(start + chunk, n_child)
            log_p = log_weight[:, None] + sum(
                level[
                    parent_clusters[:, index][:, None], child_clusters[start:stop, index][None, :]
                ]
                for index, level in enumerate(levels)
            )
            cdf = np.cumsum(np.exp(log_p - log_p.max(axis=0, keepdims=True)), axis=0)
            draws = rng.uniform(0.0, 1.0, (1, stop - start)) * cdf[-1]
            parents[start:stop] = (cdf >= draws).argmax(axis=0)
        return parents


@dataclass(frozen=True)
class ForestLink:
    roots: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.roots <= 1.0:
            raise ValueError("roots must be in (0, 1]")

    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        if n_child != n_parent:
            raise ValueError("a forest links a table to itself")
        parents = np.floor(rng.uniform(0.0, 1.0, n_child) * np.arange(n_child)).astype(np.int64)
        roots = rng.random(n_child) < self.roots
        roots[:1] = True
        return np.where(roots, -1, parents)


LINKS: dict[str, type] = {
    "random": RandomLink,
    "hsbm": HSBMLink,
    "forest": ForestLink,
}
