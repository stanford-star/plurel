from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

BUSINESS_HOURS = (0.05,) * 7 + (0.5, 0.5) + (1.0,) * 9 + (0.5,) * 4 + (0.1,) * 2


@runtime_checkable
class Distribution(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...


@dataclass(frozen=True)
class Normal:
    mean: float = 0.0
    std: float = 1.0

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError("std must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, n)


@dataclass(frozen=True)
class Uniform:
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError("low must be below high")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, n)


@dataclass(frozen=True)
class Beta:
    alpha: float = 1.0
    beta: float = 1.0
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")
        if self.low >= self.high:
            raise ValueError("low must be below high")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.low + (self.high - self.low) * rng.beta(self.alpha, self.beta, n)


@dataclass(frozen=True)
class LogNormal:
    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mean, self.sigma, n)


@dataclass(frozen=True)
class Exponential:
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(self.scale, n)


@dataclass(frozen=True)
class Pareto:
    alpha: float = 2.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.scale <= 0:
            raise ValueError("alpha and scale must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.scale * rng.pareto(self.alpha, n)


@dataclass(frozen=True)
class Poisson:
    lam: float = 1.0

    def __post_init__(self) -> None:
        if self.lam <= 0:
            raise ValueError("lam must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(self.lam, n).astype(float)


@dataclass(frozen=True)
class Mixture:
    components: tuple[Distribution, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.components) < 2:
            raise ValueError("a mixture needs at least two components")
        if self.weights is not None:
            if len(self.weights) != len(self.components):
                raise ValueError("one weight per component")
            if any(w <= 0 for w in self.weights) or not np.isclose(sum(self.weights), 1.0):
                raise ValueError("weights must be positive and sum to one")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        assignment = rng.choice(len(self.components), n, p=self.weights)
        out = np.empty(n)
        for index, component in enumerate(self.components):
            mask = assignment == index
            if mask.any():
                out[mask] = component.sample(int(mask.sum()), rng)
        return out


@dataclass(frozen=True)
class Trend:
    alpha: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

    def values(self, t: np.ndarray) -> np.ndarray:
        return self.scale * t**self.alpha * np.exp(max(0.0, self.alpha - 5.0) * t)


@dataclass(frozen=True)
class Cycle:
    periods: float = 1.0
    scale: float = 1.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.periods <= 0:
            raise ValueError("periods must be positive")

    def values(self, t: np.ndarray) -> np.ndarray:
        return self.scale * np.sin(2.0 * np.pi * self.periods * t + self.phase)


@dataclass(frozen=True)
class AutoRegressive:
    rho: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if self.scale < 0:
            raise ValueError("scale must be non-negative")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        innovations = rng.normal(0.0, self.scale, n)
        if self.rho == 0.0:
            return innovations
        out = np.empty(n)
        state = 0.0
        for index in range(n):
            state = self.rho * state + innovations[index]
            out[index] = state
        return out


@dataclass(frozen=True)
class TimeSeries:
    trend: Trend = field(default_factory=Trend)
    cycle: Cycle = field(default_factory=Cycle)
    noise: AutoRegressive = field(default_factory=AutoRegressive)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n)
        return self.trend.values(t) + self.cycle.values(t) + self.noise.sample(n, rng)


@dataclass(frozen=True)
class Calendar:
    start: pd.Timestamp
    end: pd.Timestamp
    weekday_weights: tuple[float, ...] = (1.0,) * 7
    hour_weights: tuple[float, ...] = BUSINESS_HOURS
    oversampling: int = 5

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if len(self.weekday_weights) != 7 or len(self.hour_weights) != 24:
            raise ValueError("seven weekday weights and twenty-four hour weights")
        if min(self.weekday_weights) < 0 or min(self.hour_weights) < 0:
            raise ValueError("weights must be non-negative")
        if self.oversampling < 1:
            raise ValueError("oversampling must be at least one")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        span = (self.end - self.start).total_seconds()
        candidates = max(n * self.oversampling, n + 1)
        offsets = rng.uniform(0.0, span, candidates)
        stamps = pd.DatetimeIndex(self.start + pd.to_timedelta(offsets, unit="s"))
        weights = (
            np.asarray(self.weekday_weights)[stamps.weekday]
            * np.asarray(self.hour_weights)[stamps.hour]
        )
        chosen = rng.choice(candidates, n, replace=False, p=weights / weights.sum())
        epoch = self.start.timestamp() + offsets[chosen]
        return np.sort(epoch)


DISTRIBUTIONS: dict[str, type] = {
    "normal": Normal,
    "uniform": Uniform,
    "beta": Beta,
    "lognormal": LogNormal,
    "exponential": Exponential,
    "pareto": Pareto,
    "poisson": Poisson,
    "mixture": Mixture,
    "time_series": TimeSeries,
    "calendar": Calendar,
}
