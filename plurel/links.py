from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from plurel.distributions import Distribution

CHUNK_BYTES = 100_000_000


@runtime_checkable
class Link(Protocol):
    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray: ...


def _check_sizes(n_child: int, n_parent: int) -> None:
    if n_child < 0 or n_parent < 0:
        raise ValueError("table sizes must be non-negative")
    if n_child and not n_parent:
        raise ValueError("no parent rows to link to")


def _check_counts(counts: tuple[int, ...]) -> None:
    if not counts or min(counts) < 1:
        raise ValueError("at least one cluster per level")


def clusters(n: int, counts: tuple[int, ...], shares: np.ndarray | None = None) -> np.ndarray:
    _check_counts(counts)
    shares = np.ones(int(np.prod(counts))) if shares is None else np.asarray(shares, dtype=float)
    if len(shares) != np.prod(counts) or not (np.isfinite(shares) & (shares > 0)).all():
        raise ValueError("one positive finite share per base cluster")
    k = len(shares)
    if n < k:
        raise ValueError("fewer rows than base clusters")
    bounds = np.round(np.cumsum(shares) / shares.sum() * (n - k)) + np.arange(1, k + 1)
    base = np.searchsorted(bounds, np.arange(n), side="right")
    strides = np.cumprod((1, *counts[:0:-1]))[::-1]
    return (base[:, None] // strides[None, :]) % np.asarray(counts)[None, :]


@dataclass(frozen=True)
class RandomLink:
    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        _check_sizes(n_child, n_parent)
        return rng.integers(0, max(n_parent, 1), n_child)


@dataclass(frozen=True)
class HSBMLink:
    parent_clusters: tuple[int, ...] = (1,)
    child_clusters: tuple[int, ...] = (1,)
    within: float = 0.9
    between: tuple[float, float] = (0.001, 0.002)
    cluster_weights: Distribution | None = None
    attractiveness: Distribution | None = None
    inactive: float = 0.0

    def __post_init__(self) -> None:
        _check_counts(self.parent_clusters)
        _check_counts(self.child_clusters)
        if len(self.parent_clusters) != len(self.child_clusters):
            raise ValueError("one cluster count per level on both sides")
        if self.within <= 0 or not 0 < self.between[0] <= self.between[1]:
            raise ValueError("within and between affinities must be positive, between ascending")
        if not 0.0 <= self.inactive < 1.0:
            raise ValueError("inactive must be in [0, 1)")

    def affinity(self, parent: int, child: int, rng: np.random.Generator) -> np.ndarray:
        affinity = rng.uniform(*self.between, (parent, child))
        index = np.arange(max(parent, child))
        affinity[index % parent, index % child] = self.within
        return affinity

    def shares(self, counts: tuple[int, ...], rng: np.random.Generator) -> np.ndarray | None:
        if self.cluster_weights is None:
            return None
        return self.cluster_weights.sample(int(np.prod(counts)), rng)

    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        _check_sizes(n_child, n_parent)
        if n_child == 0:
            return np.empty(0, dtype=np.int64)
        parent_clusters = clusters(
            n_parent, self.parent_clusters, self.shares(self.parent_clusters, rng)
        )
        child_clusters = clusters(
            n_child, self.child_clusters, self.shares(self.child_clusters, rng)
        )
        levels = [
            np.log(self.affinity(parent, child, rng))
            for parent, child in zip(self.parent_clusters, self.child_clusters)
        ]
        log_weight = np.zeros(n_parent)
        if self.attractiveness is not None:
            weight = self.attractiveness.sample(n_parent, rng)
            if not (np.isfinite(weight).all() and weight.min() >= 0 and weight.max() > 0):
                raise ValueError(
                    "attractiveness must draw finite non-negative weights, some positive"
                )
            with np.errstate(divide="ignore"):
                log_weight += np.log(weight)
        n_inactive = round(n_parent * self.inactive)
        if n_inactive >= n_parent:
            raise ValueError("inactive share leaves no active parent")
        log_weight[rng.permutation(n_parent)[:n_inactive]] = -np.inf
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
            parents[start:stop] = (cdf > draws).argmax(axis=0)
        return parents


@dataclass(frozen=True)
class TreeLink:
    roots: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.roots <= 1.0:
            raise ValueError("roots must be in (0, 1]")

    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        _check_sizes(n_child, n_parent)
        if n_child != n_parent:
            raise ValueError("a tree links a table to itself")
        parents = np.floor(rng.uniform(0.0, 1.0, n_child) * np.arange(n_child)).astype(np.int64)
        roots = rng.random(n_child) < self.roots
        roots[:1] = True
        return np.where(roots, -1, parents)


LINKS: dict[str, type] = {
    "random": RandomLink,
    "hsbm": HSBMLink,
    "tree": TreeLink,
}
