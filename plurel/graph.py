from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from functools import reduce

import numpy as np

from plurel.distributions import Distribution, Normal

Function = str | Callable[[np.ndarray], np.ndarray]

TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0) - 0.4,
    "square": lambda x: x**2 - 1.0,
    "step": lambda x: np.where(x > 0.0, 0.8, -0.8),
    "cube": lambda x: np.clip(x, -3.0, 3.0) ** 3 / 9.0,
    "exp": lambda x: np.exp(np.clip(x, -2.5, 2.5)),
    "sigmoid": lambda x: np.tanh(x / 2.0) / 2.0,
    "softplus": lambda x: np.logaddexp(0.0, x) - 0.7,
    "abs": lambda x: np.abs(x) - 0.8,
    "sin": np.sin,
    "log": lambda x: np.sign(x) * np.log1p(np.abs(x)),
}
TRANSFORM_NAMES = tuple(TRANSFORMS)

QUADRATIC_CHUNK = 65536  # rows a quadratic edge expands at once: its product is features^2 wide
COMPOUND_SPAN = 2.5  # the compounding sum is clipped here, so the product stays within decades

REDUCTIONS: dict[str, Callable[[list[np.ndarray]], np.ndarray]] = {
    "sum": lambda terms: reduce(np.add, terms),
    "product": lambda terms: reduce(np.multiply, terms),
    "max": lambda terms: reduce(np.maximum, terms),
    "min": lambda terms: reduce(np.minimum, terms),
    "logsumexp": lambda terms: reduce(np.logaddexp, terms),
    "concat": lambda terms: np.concatenate(terms, axis=1),
    # compounding: the exponential of the clipped sum, centred at zero
    "compound": lambda terms: np.expm1(
        np.clip(reduce(np.add, terms), -COMPOUND_SPAN, COMPOUND_SPAN)
    ),
}


def apply_transform(transform: Function, x: np.ndarray) -> np.ndarray:
    return transform(x) if callable(transform) else TRANSFORMS[transform](x)


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero mean and unit deviation per column; a constant column is zero."""
    if not len(x):
        return x
    centered = x - x.mean(0)
    deviation = centered.std(0)
    return centered / np.where(deviation > 0.0, deviation, 1.0)


def level_edges(latent: np.ndarray, probabilities: tuple[float, ...]) -> np.ndarray:
    """The latent's own quantiles at the cumulative shares; a single row has none to speak of."""
    if latent.size == 1:
        raise ValueError("levels are quantiles of a draw's rows: draw at least two")
    cuts = np.clip(np.cumsum(probabilities)[:-1], 0.0, 1.0)
    return np.quantile(latent, cuts) if latent.size else np.array([])


def bin_levels(latent: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each value's level; a value on an edge takes the lower one."""
    return np.digitize(latent, edges, right=True)


def _check_probabilities(probabilities: tuple[float, ...] | None, size: int) -> None:
    if probabilities is None:
        return
    if len(probabilities) != size:
        raise ValueError("one probability per level")
    if min(probabilities) <= 0 or not np.isclose(sum(probabilities), 1.0):
        raise ValueError("probabilities must be positive and sum to one")


def _uniform(size: int) -> tuple[float, ...]:
    return (1.0 / size,) * size


def _draw(distribution: Distribution, n: int, rng: np.random.Generator, dim: int) -> np.ndarray:
    return np.stack([distribution.sample(n, rng) for _ in range(dim)], axis=1)


@dataclass(frozen=True)
class Edge:
    """One parent's contribution to a node; the tail is a node name or a crossing reference."""

    parent: Hashable
    dim = 1

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearEdge(Edge):
    weight: float = 1.0
    transform: Function = "identity"
    dim: int = 1

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.weight * apply_transform(self.transform, x)


@dataclass(frozen=True)
class ReadEdge(Edge):
    """A parent's block as it is, or one slot of it."""

    dim: int = 1
    slot: int | None = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x if self.slot is None else x[:, [self.slot]]


@dataclass(frozen=True)
class LookupEdge(Edge):
    values: tuple[float, ...]
    probabilities: tuple[float, ...] | None = None
    edges: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError("at least two level values")
        _check_probabilities(self.probabilities, len(self.values))
        if self.edges is not None and len(self.edges) != len(self.values) - 1:
            raise ValueError("edges are one fewer than the values")

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != 1:
            raise ValueError("a lookup reads a one-dimensional parent")
        edges = self.edges
        if edges is None:
            edges = level_edges(x, self.probabilities or _uniform(len(self.values)))
        return np.asarray(self.values)[bin_levels(x, edges)]


@dataclass(frozen=True)
class LevelEdge(Edge):
    """An effect per level of a categorical parent's code node, zero for a level it leaves out or
    a row with none; `bound` lines the effects up with the parent's levels."""

    effects: tuple[tuple[object, float], ...]
    values: tuple[float, ...] = ()

    def bound(self, levels: tuple[object, ...]) -> "LevelEdge":
        given = dict(self.effects)
        if unknown := set(given) - set(levels):
            raise ValueError(f"effects name unknown levels {sorted(map(str, unknown))}")
        return replace(self, values=tuple(float(given.get(level, 0.0)) for level in levels))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.append(self.values, 0.0)[x.astype(np.int64)]


@dataclass(frozen=True)
class MatrixEdge(Edge):
    matrix: np.ndarray

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x @ self.matrix


@dataclass(frozen=True)
class NearestEdge(Edge):
    centers: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.centers)

    def apply(self, x: np.ndarray) -> np.ndarray:
        distances = (self.centers**2).sum(1) - 2.0 * x @ self.centers.T
        return np.eye(self.dim)[distances.argmin(1)]


@dataclass(frozen=True)
class MLPEdge(Edge):
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...] | None = None
    activations: tuple[Function, ...] | None = None

    def __post_init__(self) -> None:
        if not self.weights or any(
            a.shape[1] != b.shape[0] for a, b in zip(self.weights, self.weights[1:])
        ):
            raise ValueError("weights must chain")
        if self.activations is not None and len(self.activations) != len(self.weights) + 1:
            raise ValueError("one activation before each layer and one after the last")

    @property
    def dim(self) -> int:
        return self.weights[-1].shape[1]

    def apply(self, x: np.ndarray) -> np.ndarray:
        depth = len(self.weights)
        activations = self.activations or ("identity",) * (depth + 1)
        biases = self.biases or (0.0,) * depth
        h = apply_transform(activations[0], x)
        for weight, bias, activation in zip(self.weights, biases, activations[1:]):
            h = apply_transform(activation, h @ weight + bias)
        return h


@dataclass(frozen=True)
class TreeEdge(Edge):
    split_dims: np.ndarray
    split_points: np.ndarray
    leaves: np.ndarray

    def __post_init__(self) -> None:
        trees, depth = self.split_dims.shape
        if self.split_points.shape != (trees, depth) or self.leaves.shape[:2] != (trees, 2**depth):
            raise ValueError("one split point per split and 2**depth leaves per tree")

    @property
    def dim(self) -> int:
        return self.leaves.shape[2]

    def apply(self, x: np.ndarray) -> np.ndarray:
        trees, depth = self.split_dims.shape
        out = np.zeros((len(x), self.dim))
        for tree in range(trees):
            index = np.zeros(len(x), dtype=np.int64)
            for level in range(depth):
                split = x[:, self.split_dims[tree, level]] > self.split_points[tree, level]
                index |= split << level
            out += self.leaves[tree].take(index, axis=0)
        out /= trees
        return out


@dataclass(frozen=True)
class FourierEdge(Edge):
    frequencies: np.ndarray
    phases: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if not self.frequencies.shape[1] == len(self.phases) == self.weights.shape[0]:
            raise ValueError("one phase and one weight row per frequency")

    @property
    def dim(self) -> int:
        return self.weights.shape[1]

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x @ self.frequencies + self.phases) @ self.weights


@dataclass(frozen=True)
class QuadraticEdge(Edge):
    tensor: np.ndarray

    def __post_init__(self) -> None:
        if self.tensor.ndim != 3 or self.tensor.shape[1] != self.tensor.shape[2]:
            raise ValueError("tensor must be (dim, features + 1, features + 1)")

    @property
    def dim(self) -> int:
        return self.tensor.shape[0]

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        tensor = self.tensor.transpose(1, 0, 2).reshape(x.shape[1], -1)
        out = np.empty((len(x), self.dim))
        for start in range(0, len(x), QUADRATIC_CHUNK):
            rows = x[start : start + QUADRATIC_CHUNK]
            mixed = (rows @ tensor).reshape(len(rows), self.dim, x.shape[1])
            mixed *= rows[:, None, :]
            out[start : start + QUADRATIC_CHUNK] = mixed.sum(-1)
        return out


@dataclass(frozen=True)
class Node:
    """The one node type: a reduction over per-parent edges, plus bias and noise.

    An edge whose tail is a `Foreign`, a `Summary` or a `Recorded` is resolved by the schema; such
    terms add to the signal whatever the reduction. Without edges the node is a root whose value is its noise,
    `dim` wide. With `onehot`
    the node emits the one-hot argmax of its scores; with Gumbel noise that samples the
    class from the softmax of the scores. With `standardize` the signal, the reduction plus
    any terms across keys, is standardized per dimension before noise and bias are added, so
    the noise is relative to a unit-scale signal.
    """

    edges: Edge | tuple[Edge, ...] = ()
    op: str = "sum"
    bias: tuple[float, ...] | None = None
    onehot: bool = False
    dim: int | None = None
    noise: Distribution | None = field(default_factory=Normal, kw_only=True)
    standardize: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.edges, Edge):
            object.__setattr__(self, "edges", (self.edges,))
        if isinstance(self.op, Edge):
            raise ValueError("several edges go in one tuple: Node((first, second))")
        if self.op not in REDUCTIONS:
            raise ValueError(f"op must be one of {tuple(REDUCTIONS)}")
        dims = [edge.dim for edge in self.local]
        if dims:
            derived = sum(dims) if self.op == "concat" else max(dims)
        elif self.bias:
            derived = len(self.bias)
        elif self.dim is None:
            derived = max((edge.dim for edge in self.across), default=1)
        else:
            derived = self.dim
        if self.dim is None:
            object.__setattr__(self, "dim", derived)
        elif self.dim != derived:
            raise ValueError(f"dim {self.dim} does not match the edges or bias, {derived}")
        if self.bias is not None and len(self.bias) != self.dim:
            raise ValueError("one bias per dimension")
        if self.onehot and self.dim < 2:
            raise ValueError("a one-hot node needs at least two classes")

    @property
    def local(self) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if isinstance(edge.parent, str))

    @property
    def across(self) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if not isinstance(edge.parent, str))

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(edge.parent for edge in self.local))

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _draw(self.noise, n, rng, self.dim) if self.noise else np.zeros((n, self.dim))

    def evaluate(
        self,
        latents: Mapping[Hashable, np.ndarray],
        exogenous: np.ndarray,
        across: np.ndarray | None = None,
    ) -> np.ndarray:
        """The value from the parents' latents, the exogenous term and, if given, the summed
        terms read across keys."""
        terms = [edge.apply(latents[edge.parent]) for edge in self.local]
        signal = REDUCTIONS[self.op](terms) if terms else np.zeros_like(exogenous)
        if across is not None:
            signal = signal + across
        if self.standardize:
            signal = standardize(signal)
        value = signal + exogenous
        if self.bias is not None:
            value = value + np.asarray(self.bias)
        return np.eye(self.dim)[value.argmax(1)] if self.onehot else value


COMPLETE = ("count", "sum")


def _sum(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    out = np.empty((n, values.shape[1]))
    for d in range(values.shape[1]):
        out[:, d] = np.bincount(indices, values[:, d], minlength=n)
    return out


def _mean(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    count = np.bincount(indices, minlength=n)[:, None]
    total = _sum(values, indices, n)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _extreme(op: np.ufunc, start: float) -> Callable[..., np.ndarray]:
    def aggregate(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
        out = np.full((n, values.shape[1]), start)
        op.at(out, indices, values)
        out[np.bincount(indices, minlength=n) == 0] = np.nan
        return out

    return aggregate


AGGREGATES: dict[str, Callable[..., np.ndarray]] = {
    "count": lambda values, indices, n: np.bincount(indices, minlength=n)[:, None].astype(float),
    "sum": lambda values, indices, n: _sum(values, indices, n),
    "mean": _mean,
    "max": _extreme(np.maximum, -np.inf),
    "min": _extreme(np.minimum, np.inf),
}


@dataclass(frozen=True)
class Recorded:
    """Tail of an edge reading what a column's frame holds, in its units, instead of its latent: a
    number, a derived value, or a timestamp in epoch seconds. `fill` is read where there is none, a
    blank or a key pointing at nothing; without one such a row raises."""

    column: str
    fill: float | None = None

    def __post_init__(self) -> None:
        if self.fill is not None and not np.isfinite(self.fill):
            raise ValueError("fill must be finite")

    @property
    def vertex(self) -> "Recorded":
        """The recorded column, whatever is read where it has none."""
        return Recorded(self.column)


@dataclass(frozen=True)
class Foreign:
    """Tail of an edge crossing a key: `node` of the row that `key` points at, or what it records."""

    key: str
    node: str | Recorded


@dataclass(frozen=True)
class Summary:
    """Tail of an edge aggregating, per row, the rows of `table` that point at it through `key`:
    how many, or the `how` of their `node`'s latents. A `Recorded` node aggregates what the rows
    record, skipping blanks as pandas does, so a count is of the recorded rows. `fill` is read where
    nothing is aggregated, needed for mean, max and min."""

    table: str
    key: str
    node: str | Recorded | None = None
    how: str = "count"
    fill: float | None = None

    def __post_init__(self) -> None:
        if self.how not in AGGREGATES:
            raise ValueError(f"how must be one of {tuple(AGGREGATES)}")
        if self.node is None and self.how != "count":
            raise ValueError(f"a {self.how} needs a node to read")
        if self.fill is not None and (self.how in COMPLETE or not np.isfinite(self.fill)):
            raise ValueError("fill is a finite value for mean, max or min only")
        if isinstance(self.node, Recorded) and self.node.fill is not None:
            raise ValueError("a summary skips blanks: give the summary its fill, not the column")
