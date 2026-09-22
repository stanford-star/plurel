from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from plurel.distributions import Distribution, Uniform


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
    cumulative = np.cumsum(shares)
    bounds = np.round(cumulative / cumulative[-1] * (n - k)) + np.arange(1, k + 1)
    base = np.searchsorted(bounds, np.arange(n), side="right")
    strides = np.cumprod((1, *counts[:0:-1]))[::-1]
    return (base[:, None] // strides[None, :]) % np.asarray(counts)[None, :]


@dataclass(frozen=True)
class RandomLink:
    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        _check_sizes(n_child, n_parent)
        if n_child == 0:
            return np.empty(0, dtype=np.int64)
        return rng.integers(0, n_parent, n_child)


@dataclass(frozen=True)
class HSBMLink:
    parent_clusters: tuple[int, ...] = (1,)
    child_clusters: tuple[int, ...] = (1,)
    within: float | Distribution = 0.9
    between: Distribution = Uniform(0.001, 0.002)
    cluster_weights: Distribution | None = None
    popularity: Distribution | None = None
    inactive: float = 0.0

    def __post_init__(self) -> None:
        _check_counts(self.parent_clusters)
        _check_counts(self.child_clusters)
        if len(self.parent_clusters) != len(self.child_clusters):
            raise ValueError("one cluster count per level on both sides")
        if not isinstance(self.within, Distribution) and self.within <= 0:
            raise ValueError("within affinity must be positive")
        if not 0.0 <= self.inactive < 1.0:
            raise ValueError("inactive must be in [0, 1)")

    def labels(self, n: int, counts: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.cluster_weights is None:
            return clusters(n, counts)
        return clusters(n, counts, self.cluster_weights.sample(int(np.prod(counts)), rng))

    def block_affinity(self, parent: int, child: int, rng: np.random.Generator) -> np.ndarray:
        block_affinity = self.between.sample(parent * child, rng).reshape(parent, child)
        index = np.arange(max(parent, child))
        within = self.within
        if isinstance(within, Distribution):
            within = within.sample(len(index), rng)
        block_affinity[index % parent, index % child] = within
        if not (np.isfinite(block_affinity).all() and block_affinity.min() > 0):
            raise ValueError("affinities must draw finite positive values")
        return block_affinity

    def log_weights(self, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        log_weight = np.zeros(n_parent)
        if self.popularity is not None:
            weight = self.popularity.sample(n_parent, rng)
            if not (np.isfinite(weight).all() and weight.min() >= 0 and weight.max() > 0):
                raise ValueError("popularity must draw finite non-negative weights, some positive")
            with np.errstate(divide="ignore"):
                log_weight += np.log(weight)
        n_inactive = round(n_parent * self.inactive)
        if n_inactive >= n_parent:
            raise ValueError("inactive share leaves no active parent")
        log_weight[rng.permutation(n_parent)[:n_inactive]] = -np.inf
        if not np.isfinite(log_weight).any():
            raise ValueError("no active parent with positive weight")
        return log_weight

    def sample(self, n_child: int, n_parent: int, rng: np.random.Generator) -> np.ndarray:
        _check_sizes(n_child, n_parent)
        if n_child == 0:
            return np.empty(0, dtype=np.int64)
        parent_labels = self.labels(n_parent, self.parent_clusters, rng)
        child_labels = self.labels(n_child, self.child_clusters, rng)
        levels = [
            np.log(self.block_affinity(parent, child, rng))
            for parent, child in zip(self.parent_clusters, self.child_clusters)
        ]
        log_weight = self.log_weights(n_parent, rng)
        groups, membership = np.unique(child_labels, axis=0, return_inverse=True)
        order = np.argsort(membership.reshape(-1), kind="stable")
        bounds = np.searchsorted(membership.reshape(-1)[order], np.arange(len(groups) + 1))
        draws = rng.uniform(0.0, 1.0, n_child)
        parents = np.empty(n_child, dtype=np.int64)
        for group, label in enumerate(groups):
            log_p = log_weight + sum(
                level[parent_labels[:, index], label[index]] for index, level in enumerate(levels)
            )
            cdf = np.cumsum(np.exp(log_p - log_p.max()))
            members = order[bounds[group] : bounds[group + 1]]
            parents[members] = np.searchsorted(cdf[:-1], draws[members] * cdf[-1], side="right")
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
