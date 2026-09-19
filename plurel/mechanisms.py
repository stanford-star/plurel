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
    dim = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearEffect(Effect):
    weight: float = 1.0
    transform: Function = "linear"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.weight * apply_transform(self.transform, x)


@dataclass(frozen=True)
class LookupEffect(Effect):
    values: tuple[float, ...]
    probabilities: tuple[float, ...] | None = None
    dim = 1

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
    dim: int | None = None

    def __post_init__(self) -> None:
        if self.op not in REDUCTIONS:
            raise ValueError(f"op must be one of {tuple(REDUCTIONS)}")
        if self.dim is None:
            dims = [effect.dim or 1 for effect in self.effects]
            object.__setattr__(
                self, "dim", sum(dims) if self.op == "concat" else max(dims, default=1)
            )

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

    def evaluate(self, values: dict[str, np.ndarray], exogenous: np.ndarray) -> np.ndarray:
        scores = super().evaluate(values, exogenous) + np.asarray(self.biases or 0.0)
        return np.eye(self.dim)[scores.argmax(1)]


EFFECTS: dict[str, type] = {
    "linear": LinearEffect,
    "lookup": LookupEffect,
    "matrix": MatrixEffect,
    "nearest": NearestEffect,
}

MECHANISMS: dict[str, type] = {
    "root": Root,
    "combine": Combine,
    "softmax": Softmax,
}
