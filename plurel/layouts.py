import heapq
from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

Parents = tuple[tuple[int, ...], ...]
Edges = list[tuple[int, int]]


@runtime_checkable
class Layout(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> Parents: ...


def _check_size(n: int) -> None:
    if n < 0:
        raise ValueError("node count must be non-negative")


def _components(n: int, edges: Edges) -> np.ndarray:
    label = np.arange(n)
    for u, v in edges:
        old, new = label[u], label[v]
        if old != new:
            label[label == old] = new
    return label


def _connected(n: int, edges: Edges, rng: np.random.Generator) -> Edges:
    label = _components(n, edges)
    groups = [np.flatnonzero(label == value) for value in np.unique(label)]
    bridges = [
        (int(rng.choice(left)), int(rng.choice(right))) for left, right in zip(groups, groups[1:])
    ]
    return edges + bridges


def _parents(n: int, edges: Edges) -> Parents:
    parents = [set() for _ in range(n)]
    for u, v in edges:
        parents[max(u, v)].add(min(u, v))
    return tuple(tuple(sorted(node)) for node in parents)


def _pairs(mask: np.ndarray) -> Edges:
    return [(int(u), int(v)) for u, v in zip(*np.nonzero(np.triu(mask, k=1)))]


def _tree(n: int, rng: np.random.Generator) -> Edges:
    sequence = rng.integers(0, n, n - 2)
    degree = np.bincount(sequence, minlength=n) + 1
    leaves = [int(node) for node in np.flatnonzero(degree == 1)]
    heapq.heapify(leaves)
    edges = []
    for node in sequence:
        edges.append((heapq.heappop(leaves), int(node)))
        degree[node] -= 1
        if degree[node] == 1:
            heapq.heappush(leaves, int(node))
    edges.append((heapq.heappop(leaves), heapq.heappop(leaves)))
    return edges


def _rooted(n: int, edges: Edges, rng: np.random.Generator, reverse: bool) -> Parents:
    neighbours = [[] for _ in range(n)]
    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)
    root = int(rng.integers(n))
    order, above, queue = [root], {root: None}, deque([root])
    while queue:
        node = queue.popleft()
        for other in neighbours[node]:
            if other not in above:
                above[other] = node
                order.append(other)
                queue.append(other)
    if reverse:
        order.reverse()
    position = {node: index for index, node in enumerate(order)}
    parents = [[] for _ in range(n)]
    for child, parent in above.items():
        if parent is not None:
            source, target = (child, parent) if reverse else (parent, child)
            parents[position[target]].append(position[source])
    return tuple(tuple(sorted(node)) for node in parents)


@dataclass(frozen=True)
class ErdosRenyi:
    p: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must be in [0, 1]")

    def sample(self, n: int, rng: np.random.Generator) -> Parents:
        _check_size(n)
        return _parents(n, _connected(n, _pairs(rng.random((n, n)) < self.p), rng))


@dataclass(frozen=True)
class BarabasiAlbert:
    m: int = 2
    sink_dropout: float = 0.4

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError("m must be at least one")
        if not 0.0 <= self.sink_dropout <= 1.0:
            raise ValueError("sink_dropout must be in [0, 1]")

    def sample(self, n: int, rng: np.random.Generator) -> Parents:
        _check_size(n)
        m = min(self.m, max(n - 1, 0))
        targets, repeated, edges = set(range(m)), [], []
        for node in range(m, n):
            edges += [(node, target) for target in targets]
            repeated += [node] * m + list(targets)
            targets = set()
            while len(targets) < m:
                targets.add(int(rng.choice(repeated)))
        labels = rng.permutation(n)
        parents = [set(node) for node in _parents(n, [(labels[u], labels[v]) for u, v in edges])]
        children = np.zeros(n, dtype=int)
        for own in parents:
            children[list(own)] += 1
        for node, own in enumerate(parents):
            if len(own) > 1 and children[node] == 0 and rng.random() < self.sink_dropout:
                own.remove(int(rng.choice(sorted(own))))
        return tuple(tuple(sorted(node)) for node in parents)


@dataclass(frozen=True)
class RandomTree:
    def sample(self, n: int, rng: np.random.Generator) -> Parents:
        _check_size(n)
        return _rooted(n, _tree(n, rng), rng, reverse=False) if n > 1 else ((),) * n


@dataclass(frozen=True)
class ReverseRandomTree:
    def sample(self, n: int, rng: np.random.Generator) -> Parents:
        _check_size(n)
        return _rooted(n, _tree(n, rng), rng, reverse=True) if n > 1 else ((),) * n


@dataclass(frozen=True)
class WattsStrogatz:
    k: int = 2
    rewire: float = 0.2

    def __post_init__(self) -> None:
        if self.k < 2 or self.k % 2:
            raise ValueError("k must be an even number of neighbours, at least two")
        if not 0.0 <= self.rewire <= 1.0:
            raise ValueError("rewire must be in [0, 1]")

    def sample(self, n: int, rng: np.random.Generator) -> Parents:
        _check_size(n)
        k = min(self.k, n)
        lattice = [(u, (u + step) % n) for step in range(1, k // 2 + 1) for u in range(n)]
        edges = {(min(u, v), max(u, v)) for u, v in lattice if u != v}
        for u, v in lattice:
            key = (min(u, v), max(u, v))
            if u != v and key in edges and rng.random() < self.rewire:
                candidates = [w for w in range(n) if w != u and (min(u, w), max(u, w)) not in edges]
                if candidates:
                    edges.remove(key)
                    w = int(rng.choice(candidates))
                    edges.add((min(u, w), max(u, w)))
        return _parents(n, _connected(n, sorted(edges), rng))


@dataclass(frozen=True)
class RandomCauchy:
    offset: float = 0.0

    def sample(self, n: int, rng: np.random.Generator) -> Parents:
        _check_size(n)
        shift = self.offset + rng.standard_cauchy()
        shift = shift + rng.standard_cauchy(n)[:, None] + rng.standard_cauchy(n)
        probability = 1.0 / (1.0 + np.exp(-np.clip(shift, -500.0, 500.0)))
        return _parents(n, _connected(n, _pairs(rng.random((n, n)) < probability), rng))


@dataclass(frozen=True)
class Layered:
    depth: int = 6
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least one")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def sample(self, n: int, rng: np.random.Generator) -> Parents:
        _check_size(n)
        if n == 0:
            return ()
        depth = min(self.depth, n)
        sizes = np.bincount(rng.integers(0, depth, n - depth), minlength=depth) + 1
        layers = np.split(np.arange(n), np.cumsum(sizes)[:-1])
        edges = []
        for sources, targets in zip(layers, layers[1:]):
            kept = rng.random((len(sources), len(targets))) >= self.dropout
            for v in range(len(targets)):
                if not kept[:, v].any():
                    kept[rng.integers(len(sources)), v] = True
            for u in range(len(sources)):
                if not kept[u].any():
                    kept[u, rng.integers(len(targets))] = True
            edges += [(int(sources[u]), int(targets[v])) for u, v in zip(*np.nonzero(kept))]
        return _parents(n, edges)


LAYOUTS: dict[str, type] = {
    "erdos_renyi": ErdosRenyi,
    "barabasi_albert": BarabasiAlbert,
    "random_tree": RandomTree,
    "reverse_random_tree": ReverseRandomTree,
    "watts_strogatz": WattsStrogatz,
    "random_cauchy": RandomCauchy,
    "layered": Layered,
}
