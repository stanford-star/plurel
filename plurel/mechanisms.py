from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from statistics import NormalDist
from typing import Protocol, runtime_checkable

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
}
TRANSFORM_NAMES = tuple(TRANSFORMS)


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


@runtime_checkable
class Mechanism(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def parents(self) -> tuple[str, ...]: ...

    @property
    def mean_parents(self) -> tuple[str, ...]: ...

    @property
    def noise_parents(self) -> tuple[str, ...]: ...

    @property
    def interaction_pairs(self) -> tuple[tuple[str, str], ...]: ...

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray: ...


class Structure:
    @property
    def mean_parents(self) -> tuple[str, ...]:
        return self.parents

    @property
    def noise_parents(self) -> tuple[str, ...]:
        return ()

    @property
    def interaction_pairs(self) -> tuple[tuple[str, str], ...]:
        return ()


@dataclass(frozen=True)
class Effect:
    parent: str
    weight: float
    transform: Function = "linear"

    @property
    def parents(self) -> tuple[str, ...]:
        return (self.parent,)

    @property
    def key(self) -> str:
        return self.parent

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        return self.weight * apply_transform(self.transform, values[self.parent])


@dataclass(frozen=True)
class LookupEffect:
    parent: str
    values: tuple[float, ...]
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError("at least two level values")
        _check_probabilities(self.probabilities, len(self.values))

    @property
    def parents(self) -> tuple[str, ...]:
        return (self.parent,)

    @property
    def key(self) -> str:
        return self.parent

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        probabilities = self.probabilities or _uniform(len(self.values))
        return np.asarray(self.values)[bin_levels(values[self.parent], probabilities)]


@dataclass(frozen=True)
class ProductEffect:
    parents: tuple[str, str]
    weight: float

    def __post_init__(self) -> None:
        if len(set(self.parents)) != 2:
            raise ValueError("two distinct parents")

    @property
    def key(self) -> tuple[str, str]:
        return self.parents

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        left, right = self.parents
        return self.weight * values[left] * values[right]


@dataclass(frozen=True)
class LookupScaleEffect:
    selector: str
    scaled: str
    scales: tuple[float, ...]
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.scales) < 2 or self.selector == self.scaled:
            raise ValueError("at least two scales over two distinct parents")
        _check_probabilities(self.probabilities, len(self.scales))

    @property
    def parents(self) -> tuple[str, ...]:
        return (self.selector, self.scaled)

    @property
    def key(self) -> tuple[str, str]:
        return (self.selector, self.scaled)

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        probabilities = self.probabilities or _uniform(len(self.scales))
        levels = bin_levels(values[self.selector], probabilities)
        return np.asarray(self.scales)[levels] * values[self.scaled]


@dataclass(frozen=True)
class TransformedProductEffect:
    parents: tuple[str, ...]
    weight: float
    transform: Function = "linear"

    def __post_init__(self) -> None:
        if len(set(self.parents)) != len(self.parents) or len(self.parents) < 2:
            raise ValueError("two or more distinct parents")

    @property
    def key(self) -> tuple[str, ...]:
        return self.parents

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        terms = [apply_transform(self.transform, values[parent]) for parent in self.parents]
        return self.weight * np.prod(terms, axis=0)


MeanEffect = Effect | LookupEffect | ProductEffect | LookupScaleEffect | TransformedProductEffect


@dataclass(frozen=True)
class Noise:
    std: float = 1.0
    distribution: Distribution = field(default_factory=Normal)
    scale_effects: tuple[Effect, ...] = ()
    clip: float = 3.0

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError("std must be non-negative")

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(effect.parent for effect in self.scale_effects))

    def sample(self, n: int, rng: np.random.Generator, dim: int = 1) -> np.ndarray:
        return np.stack([self.distribution.sample(n, rng) for _ in range(dim)], axis=1)

    def apply(self, values: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        log_scale = sum((effect.evaluate(values) for effect in self.scale_effects), 0.0)
        return self.std * np.exp(np.clip(log_scale, -self.clip, self.clip)) * noise


@dataclass(frozen=True)
class Root(Structure):
    distribution: Distribution = field(default_factory=Normal)
    dim: int = 1

    @property
    def parents(self) -> tuple[str, ...]:
        return ()

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.stack([self.distribution.sample(n, rng) for _ in range(self.dim)], axis=1)

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return noise


@dataclass(frozen=True)
class Additive(Structure):
    effects: tuple[MeanEffect, ...] = ()
    noise: Noise = field(default_factory=Noise)

    @property
    def dim(self) -> int:
        return 1

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.mean_parents + self.noise_parents))

    @property
    def mean_parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(parent for effect in self.effects for parent in effect.parents))

    @property
    def noise_parents(self) -> tuple[str, ...]:
        return self.noise.parents

    @property
    def interaction_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs = [pair for effect in self.effects for pair in combinations(effect.parents, 2)]
        return tuple(dict.fromkeys(pairs))

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.noise.sample(n, rng)

    def contributions(
        self, parents: dict[str, np.ndarray]
    ) -> dict[str | tuple[str, ...], np.ndarray]:
        out: dict[str | tuple[str, ...], np.ndarray] = {}
        for effect in self.effects:
            out[effect.key] = out.get(effect.key, 0.0) + effect.evaluate(parents)
        return out

    def terms(self, parents: dict[str, np.ndarray], n: int) -> np.ndarray:
        values = list(self.contributions(parents).values())
        return np.stack(values) if values else np.zeros((1, n, 1))

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return self.terms(parents, len(noise)).sum(0) + self.noise.apply(parents, noise)


@dataclass(frozen=True)
class Aggregate(Additive):
    op: str = "max"

    def __post_init__(self) -> None:
        if self.op not in ("max", "min"):
            raise ValueError("op must be max or min")

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        terms = self.terms(parents, len(noise))
        mean = terms.max(0) if self.op == "max" else terms.min(0)
        return mean + self.noise.apply(parents, noise)


@dataclass(frozen=True)
class Multiplicative(Additive):
    span: float = 2.5

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        total = np.clip(self.terms(parents, len(noise)).sum(0), -self.span, self.span)
        return np.expm1(total) + self.noise.apply(parents, noise)


MECHANISMS: dict[str, type] = {
    "root": Root,
    "additive": Additive,
    "aggregate": Aggregate,
    "multiplicative": Multiplicative,
}
