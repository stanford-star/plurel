from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

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
        shares = None
        if self.cluster_weights is not None:
            shares = self.cluster_weights.sample(int(np.prod(counts)), rng)
        # rows follow time, and labels must not; the same order for a table's every key keeps
        # its blocks shared across them
        return clusters(n, counts, shares)[np.random.default_rng(n).permutation(n)]

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
        codes, membership = np.unique(
            np.ravel_multi_index(child_labels.T, self.child_clusters), return_inverse=True
        )
        groups = np.stack(np.unravel_index(codes, self.child_clusters), axis=1)
        order = np.argsort(membership, kind="stable")
        bounds = np.searchsorted(membership[order], np.arange(len(groups) + 1))
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


def matched(
    child: np.ndarray, parent: np.ndarray, log_weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """For each child row, a parent row of the same group drawn by weight, each group's weights
    taken relative to its heaviest so none vanishes: a group is an integer, -1 for none, and -1 is
    returned where no parent of its group has weight."""
    eligible = np.flatnonzero((parent >= 0) & np.isfinite(log_weights))
    order = eligible[np.argsort(parent[eligible], kind="stable")]
    groups = parent[order]
    draws = rng.random(len(child))
    out = np.full(len(child), -1, dtype=np.int64)
    if not len(order):
        return out
    heads = np.flatnonzero(np.diff(groups, prepend=-2))
    heaviest = np.maximum.reduceat(log_weights[order], heads)
    segment = np.repeat(np.arange(len(heads)), np.diff(np.append(heads, len(order))))
    cumulative = np.cumsum(np.exp(log_weights[order] - heaviest[segment]))
    start = np.searchsorted(groups, child, side="left")
    end = np.searchsorted(groups, child, side="right")
    found = (child >= 0) & (end > start)
    start, end = start[found], end[found]
    before = np.where(start > 0, cumulative[np.maximum(start - 1, 0)], 0.0)
    target = before + draws[found] * (cumulative[end - 1] - before)
    pick = np.clip(np.searchsorted(cumulative, target, side="right"), start, end - 1)
    out[found] = order[pick]
    return out


def counted(
    weights: np.ndarray,
    mean: float,
    total: int | None,
    rng: np.random.Generator,
    most: int | None = None,
    least: int = 0,
) -> np.ndarray:
    """The parent row of each child row, the rows in random order: a given `total` split among the
    parents in proportion to their weights, or else each parent's Poisson count around `mean`
    times its weight over the average weight. Every parent takes from `least` to `most`: a
    Poisson count clipped to them, or `least` each and the rest split, its excess over `most`
    dealt again among the parents with room."""
    if not len(weights):
        if total:
            raise ValueError("no parent rows to give children to")
        return np.empty(0, dtype=np.int64)
    share = weights / weights.sum()
    if total is None:
        counts = np.clip(rng.poisson(mean * len(weights) * share), least, most)
    else:
        if total < least * len(weights):
            raise ValueError(
                f"{total} rows cannot give {len(weights)} parents at least {least} each"
            )
        weighed = int((share > 0).sum())  # a parent of no weight takes `least` rows and no more
        if most is not None and total > least * len(weights) + (most - least) * weighed:
            raise ValueError(
                f"{total} rows cannot go to {len(weights)} parents at most {most} each, of which"
                f" {len(weights) - weighed} have no weight and take only {least}"
                if weighed < len(weights)
                else f"{total} rows cannot go to {weighed} parents at most {most} each"
            )
        counts = least + rng.multinomial(total - least * len(weights), share)
        while most is not None and (excess := int(np.maximum(counts - most, 0).sum())):
            counts = np.minimum(counts, most)
            room = share * (counts < most)
            counts += rng.multinomial(excess, room / room.sum())
    return rng.permutation(np.repeat(np.arange(len(weights)), counts))


DISTINCT_ROUNDS = 8  # redraws of repeated rows before a group is drawn whole, without replacement


def joint_codes(columns: list[np.ndarray], n: int) -> np.ndarray:
    """One integer per row, equal where every column agrees: -1 where any is blank (NaN or -1)."""
    joint = np.zeros(n, dtype=np.int64)
    for values in columns:
        codes = pd.factorize(values)[0]
        joint = np.where((joint < 0) | (codes < 0), -1, joint * (codes.max(initial=0) + 1) + codes)
        joint = np.where(joint < 0, -1, pd.factorize(joint)[0])
    return joint


def repeats(group: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """The rows whose pick an earlier row of their group already holds; group -1 is no group."""
    rows = np.flatnonzero((group >= 0) & (picks >= 0))
    rows = rows[np.lexsort((picks[rows], group[rows]))]
    same = (group[rows][1:] == group[rows][:-1]) & (picks[rows][1:] == picks[rows][:-1])
    out = np.zeros(len(group), dtype=bool)
    out[rows[1:][same]] = True
    return out


def distinct(
    child: np.ndarray,
    parent: np.ndarray,
    log_weights: np.ndarray,
    together: np.ndarray,
    rng: np.random.Generator,
    leave: bool = False,
) -> np.ndarray:
    """`matched`, with no two rows of one `together` group (-1: none) on the same parent. Repeats
    are redrawn a few rounds; a group still repeating, or with more rows than parents, is then
    drawn whole, without replacement, by Gumbel keys, so a full group costs its pool once. Rows a
    group has no parent left for point at nothing if `leave`, and raise otherwise."""
    blank = [np.where(codes < 0, np.nan, codes) for codes in (child, together)]
    group = joint_codes(blank, len(child))
    eligible = np.flatnonzero((parent >= 0) & np.isfinite(log_weights))
    pool = eligible[np.argsort(parent[eligible], kind="stable")]
    size = np.bincount(
        parent[pool], minlength=max(child.max(initial=-1), parent.max(initial=-1)) + 1
    )
    demand = np.bincount(group[group >= 0], minlength=group.max(initial=-1) + 1)
    first = np.full(len(demand), -1)
    first[group[group >= 0][::-1]] = np.flatnonzero(group >= 0)[::-1]
    supply = size[child[first]] if len(demand) else np.empty(0, dtype=np.int64)
    short = demand > supply
    if short.any() and not leave:
        worst = int(np.flatnonzero(short)[0])
        raise ValueError(
            f"{demand[worst]} rows that must not repeat a parent share only {supply[worst]}"
            " parents to point at"
        )
    picks = matched(child, parent, log_weights, rng)
    open_ = np.ones(len(group), dtype=bool)
    open_[group >= 0] = ~short[group[group >= 0]]
    for _ in range(DISTINCT_ROUNDS):
        clash = repeats(group, picks) & open_
        if not clash.any():
            break
        picks[clash] = matched(child[clash], parent, log_weights, rng)
    hard = np.union1d(np.unique(group[repeats(group, picks)]), np.flatnonzero(short))
    if not len(hard):
        return picks
    lengths, take = supply[hard], np.minimum(demand[hard], supply[hard])
    starts = np.searchsorted(parent[pool], child[first[hard]], side="left")
    owner = np.repeat(hard, lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    candidates = pool[np.repeat(starts, lengths) + offset]
    keys = log_weights[candidates] + rng.gumbel(size=len(candidates))
    candidates = candidates[np.lexsort((-keys, owner))]
    chosen = candidates[offset < np.repeat(take, lengths)]
    rows = np.flatnonzero(np.isin(group, hard))
    rows = rows[np.lexsort((rng.random(len(rows)), group[rows]))]
    rank = np.arange(len(rows)) - np.repeat(np.cumsum(demand[hard]) - demand[hard], demand[hard])
    served = rank < np.repeat(take, demand[hard])
    picks[rows] = -1
    picks[rows[served]] = chosen
    return picks
