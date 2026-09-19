from dataclasses import dataclass

import numpy as np

from plurel.columns import DEFAULT_CALENDAR, Column
from plurel.distributions import Calendar, Exponential, LogNormal, Mixture, Normal, Pareto, Uniform
from plurel.layouts import BarabasiAlbert, Layered, RandomCauchy
from plurel.mechanisms import (
    TRANSFORM_NAMES,
    Combine,
    Effect,
    FourierEffect,
    LinearEffect,
    MatrixEffect,
    Mechanism,
    MLPEffect,
    QuadraticEffect,
    Root,
    Softmax,
    TreeEffect,
)
from plurel.random import Seed, generator
from plurel.scm import SCM

ACTIVATIONS = tuple(name for name in TRANSFORM_NAMES if name != "identity")


def _linear(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    return LinearEffect(parent, float(rng.normal()), str(rng.choice(TRANSFORM_NAMES)), dim=d_in)


def _matrix(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    return MatrixEffect(parent, rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, d_out)))


def _mlp(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    hidden = prior.hidden.draw(rng)
    weights = (
        rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, hidden)),
        rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, d_out)),
    )
    biases = (rng.normal(0.0, 0.5, hidden), np.zeros(d_out))
    return MLPEffect(
        parent, weights, biases, ("identity", str(rng.choice(ACTIVATIONS)), "identity")
    )


def _tree(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    trees, depth = prior.trees.draw(rng), prior.depth.draw(rng)
    splits = rng.integers(0, d_in, (trees, depth))
    return TreeEffect(
        parent, splits, rng.normal(size=(trees, depth)), rng.normal(size=(trees, 2**depth, d_out))
    )


def _fourier(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    frequencies = rng.normal(size=(d_in, prior.frequencies)) * rng.uniform(0.5, 3.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, prior.frequencies)
    weights = rng.normal(0.0, 1.0 / np.sqrt(prior.frequencies), (prior.frequencies, d_out))
    return FourierEffect(parent, frequencies, phases, weights)


def _quadratic(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    scale = 1.0 / (np.sqrt(d_in) * (d_in + 1))
    return QuadraticEffect(parent, rng.normal(0.0, scale, (d_out, d_in + 1, d_in + 1)))


BUILDERS = {
    "linear": _linear,
    "matrix": _matrix,
    "mlp": _mlp,
    "tree": _tree,
    "fourier": _fourier,
    "quadratic": _quadratic,
}
FAMILIES = tuple(BUILDERS)
PRESERVING = ("linear",)


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


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("low must not exceed high")

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class LogRange(Range):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.low <= 0:
            raise ValueError("a log range must be positive")

    def draw(self, rng: np.random.Generator) -> float:
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


@dataclass(frozen=True)
class IntegersRange(Range):
    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high + 1))


@dataclass(frozen=True)
class LogIntegersRange(LogRange):
    def draw(self, rng: np.random.Generator) -> int:
        return min(
            int(np.exp(rng.uniform(np.log(self.low), np.log(self.high + 1)))), int(self.high)
        )


@dataclass(frozen=True)
class TablePrior:
    nodes: Range = LogIntegersRange(3, 16)
    layouts: Choices = Choices(
        (RandomCauchy(), RandomCauchy(2.0), BarabasiAlbert(2), Layered(3, 0.2))
    )
    width: Range = LogIntegersRange(1, 4)
    categorical: float = 0.3
    classes: Range = IntegersRange(2, 8)
    families: Choices = Choices(FAMILIES)
    ops: Choices = Choices(("sum", "product", "max", "logsumexp"), (6.0, 1.0, 1.0, 1.0))
    noise: Range = LogRange(0.01, 0.5)
    root_noise: Choices = Choices(
        (Normal(), Uniform(-1.7, 1.7), Mixture((Normal(-1.5, 0.5), Normal(1.5, 0.5))))
    )
    hidden: Range = LogIntegersRange(2, 16)
    trees: Range = LogIntegersRange(1, 8)
    depth: Range = IntegersRange(1, 4)
    frequencies: int = 16
    columns: Range = IntegersRange(3, 12)
    marginals: Choices = Choices(
        (None, Uniform(), LogNormal(), Pareto(2.0), Exponential()), (3.0, 1.0, 1.0, 1.0, 1.0)
    )
    binned: float = 0.2
    missing: Range = Range(0.01, 0.1)
    missing_share: float = 0.3
    timestamp: float = 0.5
    calendar: Calendar = DEFAULT_CALENDAR

    def realize(self, seed: Seed = None) -> SCM:
        rng = generator(seed)
        n = self.nodes.draw(rng)
        parents = self.layouts.draw(rng).sample(n, rng)
        categorical = rng.random(n) < self.categorical
        dims = [
            self.classes.draw(rng) if categorical[i] else self.width.draw(rng) for i in range(n)
        ]
        mechanisms = {
            f"n{i}": self.mechanism(parents[i], dims, i, categorical[i], rng) for i in range(n)
        }
        columns = {"id": Column(kind="key")}
        feature_nodes = rng.permutation(n)[: rng.integers(1, n + 1)]
        for c in range(self.columns.draw(rng)):
            i = int(rng.choice(feature_nodes))
            columns[f"col{c}"] = self.column(i, dims[i], categorical[i], rng)
        time_column = None
        if rng.random() < self.timestamp:
            mechanisms["time"] = Root(noise=self.calendar)
            columns["time"] = Column("time", "timestamp")
            time_column = "time"
        return SCM(mechanisms, columns, time_column=time_column)

    def mechanism(
        self,
        sources: tuple[int, ...],
        dims: list[int],
        i: int,
        categorical: bool,
        rng: np.random.Generator,
    ) -> Mechanism:
        effects = tuple(self.effect(f"n{p}", dims[p], dims[i], categorical, rng) for p in sources)
        if categorical:
            return Softmax(effects, biases=tuple(rng.normal(0.0, 0.5, dims[i])))
        if not effects:
            return Root(dim=dims[i], noise=self.root_noise.draw(rng))
        op = self.ops.draw(rng) if len(effects) > 1 else "sum"
        return Combine(effects, op, noise=Normal(std=self.noise.draw(rng)))

    def effect(
        self, parent: str, d_in: int, d_out: int, block: bool, rng: np.random.Generator
    ) -> Effect:
        preserving = d_in == d_out and not block
        families = self.families if preserving else self.families.without(PRESERVING)
        return BUILDERS[families.draw(rng)](self, parent, d_in, d_out, rng)

    def column(self, i: int, dim: int, categorical: bool, rng: np.random.Generator) -> Column:
        node = f"n{i}"
        missing = self.missing.draw(rng) if rng.random() < self.missing_share else 0.0
        if categorical:
            categories = tuple(f"c{k}" for k in range(dim))
            return Column(node, "categorical", categories=categories, missing=missing)
        slot = int(rng.integers(dim))
        if rng.random() < self.binned:
            k = self.classes.draw(rng)
            probabilities = tuple(float(p) for p in rng.dirichlet(np.ones(k)))
            categories = tuple(f"c{j}" for j in range(k))
            return Column(
                node,
                "categorical",
                dims=slot,
                categories=categories,
                probabilities=probabilities,
                missing=missing,
            )
        return Column(node, dims=slot, marginal=self.marginals.draw(rng), missing=missing)
