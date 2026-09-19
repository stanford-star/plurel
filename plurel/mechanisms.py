from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from plurel.distributions import Distribution, Normal

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

REDUCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sum": lambda terms: terms.sum(0),
    "max": lambda terms: terms.max(0),
    "min": lambda terms: terms.min(0),
    "product": lambda terms: terms.prod(0),
}


def apply_transform(transform: Function, values: np.ndarray) -> np.ndarray:
    return transform(values) if callable(transform) else TRANSFORMS[transform](values)


def _normal_edges(probabilities: tuple[float, ...]) -> np.ndarray:
    cuts = np.cumsum(probabilities)[:-1]
    return np.asarray([NormalDist().inv_cdf(float(np.clip(c, 1e-6, 1 - 1e-6))) for c in cuts])


def bin_levels(latent: np.ndarray, probabilities: tuple[float, ...]) -> np.ndarray:
    return np.digitize(latent, _normal_edges(probabilities))


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

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearEffect(Effect):
    weight: float
    transform: Function = "linear"

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        return self.weight * apply_transform(self.transform, values[self.parent])


@dataclass(frozen=True)
class LookupEffect(Effect):
    values: tuple[float, ...]
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError("at least two level values")
        _check_probabilities(self.probabilities, len(self.values))

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        probabilities = self.probabilities or _uniform(len(self.values))
        return np.asarray(self.values)[bin_levels(values[self.parent], probabilities)]


@dataclass(frozen=True)
class Mechanism:
    noise: Distribution | None = field(default_factory=Normal, kw_only=True)
    dim = 1
    parents = ()

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _draw(self.noise, n, rng, self.dim) if self.noise else np.zeros((n, 0))

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class Root(Mechanism):
    dim: int = 1

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return noise


@dataclass(frozen=True)
class Combine(Mechanism):
    effects: tuple[Effect, ...] = ()
    op: str = "sum"

    def __post_init__(self) -> None:
        if self.op not in REDUCTIONS:
            raise ValueError(f"op must be one of {tuple(REDUCTIONS)}")

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(effect.parent for effect in self.effects))

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        terms = [effect.evaluate(parents) for effect in self.effects] or [np.zeros_like(noise)]
        return REDUCTIONS[self.op](np.stack(terms)) + noise


MECHANISMS: dict[str, type] = {
    "root": Root,
    "combine": Combine,
}
