from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from statistics import NormalDist

import numpy as np

from plurel.distributions import Distribution, Gumbel, Normal

Function = str | Callable[[np.ndarray], np.ndarray]

TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0) - 0.4,
    "square": lambda x: x**2 - 1.0,
    "step": lambda x: np.where(x > 0.0, 0.8, -0.8),
    "cube": lambda x: np.clip(x, -3.0, 3.0) ** 3 / 9.0,
    "exp": lambda x: np.exp(np.clip(x, -2.5, 2.5)),
}
TRANSFORM_NAMES = tuple(TRANSFORMS)

REDUCTIONS: dict[str, Callable[[list[np.ndarray]], np.ndarray]] = {
    "sum": lambda terms: reduce(np.add, terms),
    "product": lambda terms: reduce(np.multiply, terms),
    "max": lambda terms: reduce(np.maximum, terms),
    "min": lambda terms: reduce(np.minimum, terms),
    "logsumexp": lambda terms: reduce(np.logaddexp, terms),
    "concat": lambda terms: np.concatenate(terms, axis=1),
}


def apply_transform(transform: Function, x: np.ndarray) -> np.ndarray:
    return transform(x) if callable(transform) else TRANSFORMS[transform](x)


def _normal_edges(probabilities: tuple[float, ...]) -> np.ndarray:
    cuts = np.cumsum(probabilities)[:-1]
    return np.asarray([NormalDist().inv_cdf(float(np.clip(c, 1e-6, 1 - 1e-6))) for c in cuts])


def bin_levels(latent: np.ndarray, probabilities: tuple[float, ...]) -> np.ndarray:
    return np.digitize(latent, _normal_edges(probabilities))


def nested_logits(
    allowed: tuple[tuple[int, ...], ...], probabilities: tuple[float, ...]
) -> np.ndarray:
    mask = np.zeros((len(allowed), len(probabilities)))
    for code, subset in enumerate(allowed):
        mask[code, list(subset)] = 1.0
    return np.log(np.maximum(mask * np.asarray(probabilities), np.finfo(float).tiny))


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
class Effect:
    parent: str
    dim = 1

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearEffect(Effect):
    weight: float = 1.0
    transform: Function = "linear"
    dim: int = 1

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.weight * apply_transform(self.transform, x)


@dataclass(frozen=True)
class LookupEffect(Effect):
    values: tuple[float, ...]
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError("at least two level values")
        _check_probabilities(self.probabilities, len(self.values))

    def apply(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.probabilities or _uniform(len(self.values))
        return np.asarray(self.values)[bin_levels(x, probabilities)]


@dataclass(frozen=True)
class MatrixEffect(Effect):
    matrix: np.ndarray

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x @ self.matrix


@dataclass(frozen=True)
class NearestEffect(Effect):
    centers: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.centers)

    def apply(self, x: np.ndarray) -> np.ndarray:
        distances = (self.centers**2).sum(1) - 2.0 * x @ self.centers.T
        return np.eye(self.dim)[distances.argmin(1)]


@dataclass(frozen=True)
class MLPEffect(Effect):
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
        activations = self.activations or ("linear",) * (depth + 1)
        biases = self.biases or (0.0,) * depth
        h = apply_transform(activations[0], x)
        for weight, bias, activation in zip(self.weights, biases, activations[1:]):
            h = apply_transform(activation, h @ weight + bias)
        return h


@dataclass(frozen=True)
class TreeEffect(Effect):
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
        sides = x[:, self.split_dims] > self.split_points
        index = sides @ (2 ** np.arange(self.split_dims.shape[1]))
        return self.leaves[np.arange(len(self.leaves)), index].mean(1)


@dataclass(frozen=True)
class FourierEffect(Effect):
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
class QuadraticEffect(Effect):
    tensor: np.ndarray

    def __post_init__(self) -> None:
        if self.tensor.ndim != 3 or self.tensor.shape[1] != self.tensor.shape[2]:
            raise ValueError("tensor must be (dim, features + 1, features + 1)")

    @property
    def dim(self) -> int:
        return self.tensor.shape[0]

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        return np.einsum("oij,ni,nj->no", self.tensor, x, x)


@dataclass(frozen=True)
class Mechanism:
    noise: Distribution | None = field(default_factory=Normal, kw_only=True)
    dim = 1
    parents = ()

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _draw(self.noise, n, rng, self.dim) if self.noise else np.zeros((n, self.dim))

    def evaluate(self, values: dict[str, np.ndarray], exogenous: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class Root(Mechanism):
    dim: int = 1

    def evaluate(self, values: dict[str, np.ndarray], exogenous: np.ndarray) -> np.ndarray:
        return exogenous


@dataclass(frozen=True)
class Combine(Mechanism):
    effects: tuple[Effect, ...] = ()
    op: str = "sum"

    def __post_init__(self) -> None:
        if self.op not in REDUCTIONS:
            raise ValueError(f"op must be one of {tuple(REDUCTIONS)}")

    @property
    def dim(self) -> int:
        dims = [effect.dim for effect in self.effects]
        return sum(dims) if self.op == "concat" else max(dims, default=1)

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(effect.parent for effect in self.effects))

    def evaluate(self, values: dict[str, np.ndarray], exogenous: np.ndarray) -> np.ndarray:
        terms = [effect.apply(values[effect.parent]) for effect in self.effects]
        return REDUCTIONS[self.op](terms) + exogenous if terms else exogenous


@dataclass(frozen=True)
class Softmax(Combine):
    biases: tuple[float, ...] | None = None
    noise: Distribution = field(default_factory=Gumbel, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim < 2:
            raise ValueError("at least two classes")
        if self.biases is not None and len(self.biases) != self.dim:
            raise ValueError("one bias per class")

    @property
    def dim(self) -> int:
        return super().dim if self.effects else len(self.biases or ())

    def evaluate(self, values: dict[str, np.ndarray], exogenous: np.ndarray) -> np.ndarray:
        scores = super().evaluate(values, exogenous) + np.asarray(self.biases or 0.0)
        return np.eye(self.dim)[scores.argmax(1)]


EFFECTS: dict[str, type] = {
    "linear": LinearEffect,
    "lookup": LookupEffect,
    "matrix": MatrixEffect,
    "nearest": NearestEffect,
    "mlp": MLPEffect,
    "tree": TreeEffect,
    "fourier": FourierEffect,
    "quadratic": QuadraticEffect,
}

MECHANISMS: dict[str, type] = {
    "root": Root,
    "combine": Combine,
    "softmax": Softmax,
}
