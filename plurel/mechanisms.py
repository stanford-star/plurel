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
}
TRANSFORM_NAMES = tuple(TRANSFORMS)

REDUCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sum": lambda terms: terms.sum(0),
    "max": lambda terms: terms.max(0),
    "min": lambda terms: terms.min(0),
    "product": lambda terms: np.expm1(np.clip(terms.sum(0), -2.5, 2.5)),
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


class Effect:
    parents: tuple[str, ...]

    @property
    def key(self) -> str | tuple[str, ...]:
        return self.parents[0] if len(self.parents) == 1 else self.parents

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearEffect(Effect):
    parent: str
    weight: float
    transform: Function = "linear"

    @property
    def parents(self) -> tuple[str, ...]:
        return (self.parent,)

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        return self.weight * apply_transform(self.transform, values[self.parent])


@dataclass(frozen=True)
class LookupEffect(Effect):
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

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        probabilities = self.probabilities or _uniform(len(self.values))
        return np.asarray(self.values)[bin_levels(values[self.parent], probabilities)]


@dataclass(frozen=True)
class ProductEffect(Effect):
    parents: tuple[str, str]
    weight: float

    def __post_init__(self) -> None:
        if len(set(self.parents)) != 2:
            raise ValueError("two distinct parents")

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        left, right = self.parents
        return self.weight * values[left] * values[right]


@dataclass(frozen=True)
class LookupScaleEffect(Effect):
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

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        probabilities = self.probabilities or _uniform(len(self.scales))
        levels = bin_levels(values[self.selector], probabilities)
        return np.asarray(self.scales)[levels] * values[self.scaled]


@dataclass(frozen=True)
class TransformedProductEffect(Effect):
    parents: tuple[str, ...]
    weight: float
    transform: Function = "linear"

    def __post_init__(self) -> None:
        if len(set(self.parents)) != len(self.parents) or len(self.parents) < 2:
            raise ValueError("two or more distinct parents")

    def evaluate(self, values: dict[str, np.ndarray]) -> np.ndarray:
        terms = [apply_transform(self.transform, values[parent]) for parent in self.parents]
        return self.weight * np.prod(terms, axis=0)


@dataclass(frozen=True)
class Noise:
    distribution: Distribution = field(default_factory=Normal)
    scale_effects: tuple[Effect, ...] = ()
    clip: float = 3.0

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p for effect in self.scale_effects for p in effect.parents))

    def sample(self, n: int, rng: np.random.Generator, dim: int) -> np.ndarray:
        return np.stack([self.distribution.sample(n, rng) for _ in range(dim)], axis=1)

    def apply(self, values: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        log_scale = sum((effect.evaluate(values) for effect in self.scale_effects), 0.0)
        return np.exp(np.clip(log_scale, -self.clip, self.clip)) * noise


@dataclass(frozen=True)
class Mechanism:
    noise: Noise | None = field(default_factory=Noise, kw_only=True)
    dim = 1
    parents = ()

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.noise.sample(n, rng, self.dim) if self.noise else np.zeros((n, 0))

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
        effects = (*self.effects, *self.noise.scale_effects)
        return tuple(dict.fromkeys(p for effect in effects for p in effect.parents))

    def contributions(
        self, parents: dict[str, np.ndarray]
    ) -> dict[str | tuple[str, ...], np.ndarray]:
        out: dict[str | tuple[str, ...], np.ndarray] = {}
        for effect in self.effects:
            out[effect.key] = out.get(effect.key, 0.0) + effect.evaluate(parents)
        return out

    def evaluate(self, parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        terms = list(self.contributions(parents).values()) or [np.zeros_like(noise)]
        return REDUCTIONS[self.op](np.stack(terms)) + self.noise.apply(parents, noise)


MECHANISMS: dict[str, type] = {
    "root": Root,
    "combine": Combine,
}
