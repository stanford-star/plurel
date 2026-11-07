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
        if self.std < 0:
            raise ValueError("std must be non-negative")

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
    """Lomax form, ``scale * pareto(alpha)``: the support starts at zero, not at ``scale``."""

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
class Gumbel:
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gumbel(self.loc, self.scale, n)


@dataclass(frozen=True)
class Laplace:
    """Symmetric with heavier tails than the normal: a double exponential."""

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.laplace(self.loc, self.scale, n)


@dataclass(frozen=True)
class StudentT:
    """Symmetric with power-law tails: the fewer the degrees of freedom, the more outliers."""

    df: float = 5.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.df <= 0 or self.scale <= 0:
            raise ValueError("df and scale must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.scale * rng.standard_t(self.df, n)


@dataclass(frozen=True)
class Gamma:
    """Positive and right-skewed, from exponential-like at shape 1 to near-normal at large shapes;
    the chi-square family is its shape-and-scale special case."""

    shape: float = 2.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("shape and scale must be positive")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(self.shape, self.scale, n)


@dataclass(frozen=True)
class NegativeBinomial:
    """Overdispersed counts: failures before ``successes`` successes at probability ``p``; the
    geometric family is its one-success case, the Poisson its limit."""

    successes: float = 2.0
    p: float = 0.5

    def __post_init__(self) -> None:
        if self.successes <= 0 or not 0.0 < self.p <= 1.0:
            raise ValueError("successes must be positive and p in (0, 1]")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.negative_binomial(self.successes, self.p, n).astype(float)


@dataclass(frozen=True)
class Affine:
    """``loc + scale * distribution``: a location and scale on any distribution; a negative scale
    mirrors it, which turns a right skew into a left one."""

    distribution: Distribution
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale == 0:
            raise ValueError("scale must be non-zero")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.loc + self.scale * self.distribution.sample(n, rng)


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
        draws = np.stack([component.sample(n, rng) for component in self.components])
        return draws[assignment, np.arange(n)]


@dataclass(frozen=True)
class Trend:
    alpha: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

    def values(self, t: np.ndarray) -> np.ndarray:
        return self.scale * t**self.alpha


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
        """The AR(1) recursion as a scan over blocks short enough for the powers of `rho` to
        stay finite."""
        noise = rng.normal(0.0, self.scale, n)
        if not self.rho or not n:
            return noise
        block = max(1, min(n, int(250.0 / -np.log10(self.rho))))
        powers = self.rho ** np.arange(1, block + 1)
        out, carry = np.empty(n), 0.0
        for start in range(0, n, block):
            steps = powers[: min(block, n - start)]
            out[start : start + block] = steps * (
                np.cumsum(noise[start : start + block] / steps) + carry
            )
            carry = out[start + len(steps) - 1]
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

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if len(self.weekday_weights) != 7 or len(self.hour_weights) != 24:
            raise ValueError("seven weekday weights and twenty-four hour weights")
        for weights in (self.weekday_weights, self.hour_weights):
            if min(weights) < 0 or max(weights) <= 0:
                raise ValueError("weights must be non-negative with a positive entry")

    def weights(self, seconds: np.ndarray) -> np.ndarray:
        """The weight of each epoch second: its weekday's times its hour's."""
        whole = np.floor(seconds).astype(np.int64)
        weekday = np.asarray(self.weekday_weights)[(whole // 86400 + 3) % 7]
        return weekday * np.asarray(self.hour_weights)[whole % 86400 // 3600]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sorted epoch seconds, uniform over the span and thinned by the weights: a draw is
        kept with probability its weight over the largest weight."""
        if n == 0:
            return np.empty(0)
        start, end = self.start.timestamp(), self.end.timestamp()
        if end - start < 8 * 86400.0:
            cells = np.arange(np.floor(start / 3600.0), np.ceil(end / 3600.0)) * 3600.0
            if not self.weights(np.maximum(cells, start)).any():
                raise ValueError("the calendar puts no weight on its span")
        weekday, hour = np.asarray(self.weekday_weights), np.asarray(self.hour_weights)
        top, typical = weekday.max() * hour.max(), weekday.mean() * hour.mean()
        kept, count = [], 0
        while count < n:
            draws = int(1.1 * (n - count) * top / typical) + 16
            seconds = rng.uniform(start, end, draws)
            kept.append(seconds[rng.uniform(0.0, top, draws) < self.weights(seconds)])
            count += len(kept[-1])
        return np.sort(np.concatenate(kept)[:n])


CENTRAL = (0.005, 0.995)  # the quantiles a placed range spans


def record_step(low: float, high: float) -> float:
    """The power of ten a value on `[low, high]` is recorded to: about a 500th of the span."""
    return float(10.0 ** np.floor(np.log10((high - low) / 500.0)))


def placed(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """The values moved so their central 99% spans `[low, high]`, recorded to its step."""
    ends = np.quantile(values, CENTRAL)
    unit = (values - ends[0]) / (ends[1] - ends[0] or 1.0)
    step = record_step(low, high)
    return np.clip(np.round((low + unit * (high - low)) / step) * step, low, high)


@dataclass(frozen=True)
class Bounded:
    """A distribution moved by `loc` and `scale` onto `[low, high]`, recorded to a `step`."""

    base: Distribution
    loc: float
    scale: float
    low: float
    high: float
    step: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        moved = self.loc + self.scale * self.base.sample(n, rng)
        return np.clip(np.round(moved / self.step) * self.step, self.low, self.high)


SHAPE_KINDS = ("symmetric", "right", "left", "bimodal", "multimodal", "counts", "zero_inflated")


def _pick(rng: np.random.Generator, *families):
    return families[int(rng.integers(len(families)))]()


def shape(kind: str, rng: np.random.Generator) -> Distribution:
    """One plurel distribution of shape class ``kind``, its family and parameters drawn."""
    u = rng.uniform
    if kind == "symmetric":
        return _pick(
            rng,
            lambda: Normal(0.0, u(0.3, 4.0)),
            lambda: Uniform(-(w := u(0.5, 6.0)), w),
            lambda: Laplace(0.0, u(0.4, 2.5)),
            lambda: StudentT(u(2.2, 15.0), u(0.4, 1.8)),
        )
    if kind in ("right", "left"):
        tail = _pick(
            rng,
            lambda: LogNormal(0.0, u(0.4, 2.5)),
            lambda: Gamma(u(1.1, 5.0), u(0.4, 2.5)),
            lambda: Exponential(u(0.4, 2.5)),
            lambda: Pareto(u(1.2, 6.0)),
            lambda: Gumbel(0.0, u(0.4, 2.0)),
        )
        return Affine(tail, scale=-1.0) if kind == "left" else tail
    if kind in ("bimodal", "multimodal"):
        # unequal component weights and per-component spreads, jittered centers: real mixtures
        # are a dominant mode plus minorities, not evenly spaced equal humps
        modes = 2 if kind == "bimodal" else int(rng.integers(3, 6))
        span = u(2.5, 7.0)
        centers = np.linspace(-span / 2, span / 2, modes) + u(-0.15, 0.15, modes) * span
        weights = np.clip(rng.dirichlet(np.full(modes, u(0.8, 4.0))), 0.05, None)
        spreads = u(0.25, 1.2, modes)
        components = tuple(
            Normal(float(c), float(s)) for c, s in zip(centers, spreads, strict=True)
        )
        return Mixture(components, tuple(float(w) for w in weights / weights.sum()))
    if kind == "counts":
        # integer-valued measured quantities (events, items): right-skewed, tie-bearing,
        # non-negative, the dominant real numeric column shape
        return _pick(
            rng,
            lambda: Poisson(u(0.5, 80.0)),
            lambda: NegativeBinomial(u(0.8, 8.0), u(0.1, 0.8)),
        )
    if kind == "zero_inflated":
        # a point mass at zero plus a right tail (spend, claims, counts of rare events)
        zeros = float(u(0.15, 0.85))
        tail = _pick(
            rng,
            lambda: Exponential(u(0.4, 2.5)),
            lambda: LogNormal(0.0, u(0.4, 1.5)),
            lambda: Gamma(u(1.0, 4.0), u(0.4, 2.0)),
        )
        return Mixture((Normal(0.0, 0.0), tail), (zeros, 1.0 - zeros))
    raise ValueError(f"unknown shape {kind!r}; expected one of {SHAPE_KINDS}")
