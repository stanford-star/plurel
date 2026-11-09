import hashlib
from dataclasses import dataclass, replace

import numpy as np

Seed = int | np.random.Generator | None


def generator(seed: Seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def lane(root: int, *names: str) -> np.random.Generator:
    """The stream of a named thing: it depends on the root and the names alone, so adding a node,
    a column or a key moves no other draw."""
    keys = (hashlib.blake2b(name.encode(), digest_size=8).digest() for name in names)
    return np.random.default_rng([root, *(int.from_bytes(key) for key in keys)])


def _open_unit(rng: np.random.Generator) -> float:
    return float(rng.integers(1, 2**53) / 2**53)


@dataclass(frozen=True)
class Choices:
    values: tuple
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("at least one value to choose from")
        if self.weights is not None and (
            len(self.weights) != len(self.values) or min(self.weights) < 0 or not sum(self.weights)
        ):
            raise ValueError("one non-negative weight per value, not all zero")

    def draw(self, rng: np.random.Generator):
        weights = None if self.weights is None else np.asarray(self.weights) / sum(self.weights)
        return self.values[rng.choice(len(self.values), p=weights)]

    def without(self, values: tuple) -> "Choices":
        keep = [k for k, value in enumerate(self.values) if value not in values]
        weights = None if self.weights is None else tuple(self.weights[k] for k in keep)
        return Choices(tuple(self.values[k] for k in keep), weights)

    def warp(self, rng: np.random.Generator) -> "Choices":
        base = np.ones(len(self.values)) if self.weights is None else np.asarray(self.weights)
        preference = rng.dirichlet(np.full(len(self.values), META_CONCENTRATION.draw(rng)))
        return Choices(self.values, tuple(float(w) for w in base * preference))


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    shape: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("low must not exceed high")

    def unit(self, rng: np.random.Generator) -> float:
        return float(rng.random() if self.shape is None else rng.beta(*self.shape))

    def draw(self, rng: np.random.Generator) -> float:
        return self.low + (self.high - self.low) * self.unit(rng)

    def __neg__(self) -> "Range":
        return replace(self, low=-self.high, high=-self.low)

    def warp(self, rng: np.random.Generator) -> "Range":
        """A preference within the range. Warping a warped range draws the new preference near
        the old one, so preferences nest, a table's within its database's: from the old shape,
        made at least as concentrated as a uniform, which never pushes them to the range's ends."""
        if self.shape is None:
            location = _open_unit(rng)
        else:
            a, b = self.shape
            location = float(rng.beta(*np.asarray(self.shape) * max(1.0, 2.0 / (a + b))))
            location = min(max(location, 2**-53), 1 - 2**-53)  # a beta draw may land on 0 or 1
        concentration = META_CONCENTRATION.draw(rng)
        shape = (location * concentration, (1.0 - location) * concentration)
        return replace(self, shape=shape)


@dataclass(frozen=True)
class LogRange(Range):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.low <= 0:
            raise ValueError("a log range must be positive")

    def draw(self, rng: np.random.Generator) -> float:
        return float(np.exp(np.log(self.low) + np.log(self.high / self.low) * self.unit(rng)))


@dataclass(frozen=True)
class IntegersRange(Range):
    def draw(self, rng: np.random.Generator) -> int:
        return min(int(self.low + (self.high + 1 - self.low) * self.unit(rng)), int(self.high))


@dataclass(frozen=True)
class LogIntegersRange(LogRange):
    def draw(self, rng: np.random.Generator) -> int:
        span = np.log((self.high + 1) / self.low)
        return min(int(np.exp(np.log(self.low) + span * self.unit(rng))), int(self.high))


META_CONCENTRATION = LogRange(0.1, 10_000.0)
