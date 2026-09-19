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

FAMILIES = ("linear", "matrix", "mlp", "tree", "fourier", "quadratic")
ACTIVATIONS = tuple(name for name in TRANSFORM_NAMES if name != "identity")


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


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high or (self.log and self.low <= 0):
            raise ValueError("low must not exceed high, and a log range must be positive")

    def draw(self, rng: np.random.Generator) -> float | int:
        if self.log:
            value = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        else:
            value = float(rng.uniform(self.low, self.high))
        return int(np.clip(round(value), self.low, self.high)) if self.integer else value


@dataclass(frozen=True)
class TablePrior:
    nodes: Range = Range(3, 16, log=True, integer=True)
    layouts: Choices = Choices(
        (RandomCauchy(), RandomCauchy(2.0), BarabasiAlbert(2), Layered(3, 0.2))
    )
    width: Range = Range(1, 4, log=True, integer=True)
    categorical: float = 0.3
    classes: Range = Range(2, 8, integer=True)
    families: Choices = Choices(FAMILIES)
    ops: Choices = Choices(("sum", "product", "max", "logsumexp"), (6.0, 1.0, 1.0, 1.0))
    noise: Range = Range(0.01, 0.5, log=True)
    root_noise: Choices = Choices(
        (Normal(), Uniform(-1.7, 1.7), Mixture((Normal(-1.5, 0.5), Normal(1.5, 0.5))))
    )
    hidden: Range = Range(2, 16, log=True, integer=True)
    trees: Range = Range(1, 8, log=True, integer=True)
    depth: Range = Range(1, 4, integer=True)
    frequencies: int = 16
    columns: Range = Range(3, 12, integer=True)
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
        mechanisms: dict[str, Mechanism] = {}
        for i in range(n):
            name = f"n{i}"
            sources = tuple(f"n{p}" for p in parents[i])
            if categorical[i]:
                effects = tuple(
                    self.effect(source, dims[p], dims[i], rng, block=True)
                    for source, p in zip(sources, parents[i])
                )
                biases = tuple(rng.normal(0.0, 0.5, dims[i]))
                mechanisms[name] = Softmax(effects, biases=biases)
            elif not sources:
                mechanisms[name] = Root(dim=dims[i], noise=self.root_noise.draw(rng))
            else:
                effects = tuple(
                    self.effect(source, dims[p], dims[i], rng)
                    for source, p in zip(sources, parents[i])
                )
                op = self.ops.draw(rng) if len(effects) > 1 else "sum"
                mechanisms[name] = Combine(effects, op, noise=Normal(std=self.noise.draw(rng)))
        columns = self.columns_for(mechanisms, dims, categorical, rng)
        time_column = None
        if rng.random() < self.timestamp:
            mechanisms["time"] = Root(noise=self.calendar)
            columns["time"] = Column("time", "timestamp")
            time_column = "time"
        return SCM(mechanisms, columns, time_column=time_column)

    def effect(
        self, parent: str, d_in: int, d_out: int, rng: np.random.Generator, block: bool = False
    ) -> Effect:
        family = self.families.draw(rng)
        while family == "linear" and (block or d_in != d_out):
            family = self.families.draw(rng)
        scale = 1.0 / np.sqrt(d_in)
        if family == "linear":
            return LinearEffect(
                parent, float(rng.normal()), str(rng.choice(TRANSFORM_NAMES)), dim=d_in
            )
        if family == "matrix":
            return MatrixEffect(parent, rng.normal(0.0, scale, (d_in, d_out)))
        if family == "mlp":
            hidden = self.hidden.draw(rng)
            weights = (
                rng.normal(0.0, scale, (d_in, hidden)),
                rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, d_out)),
            )
            biases = (rng.normal(0.0, 0.5, hidden), np.zeros(d_out))
            return MLPEffect(
                parent, weights, biases, ("identity", str(rng.choice(ACTIVATIONS)), "identity")
            )
        if family == "tree":
            trees, depth = self.trees.draw(rng), self.depth.draw(rng)
            split_dims = rng.integers(0, d_in, (trees, depth))
            return TreeEffect(
                parent,
                split_dims,
                rng.normal(0.0, 1.0, (trees, depth)),
                rng.normal(0.0, 1.0, (trees, 2**depth, d_out)),
            )
        if family == "fourier":
            frequencies = rng.normal(0.0, 1.0, (d_in, self.frequencies)) * rng.uniform(0.5, 3.0)
            phases = rng.uniform(0.0, 2.0 * np.pi, self.frequencies)
            return FourierEffect(
                parent,
                frequencies,
                phases,
                rng.normal(0.0, 1.0 / np.sqrt(self.frequencies), (self.frequencies, d_out)),
            )
        return QuadraticEffect(
            parent, rng.normal(0.0, scale / (d_in + 1), (d_out, d_in + 1, d_in + 1))
        )

    def columns_for(self, mechanisms, dims, categorical, rng) -> dict[str, Column]:
        n = len(dims)
        feature_nodes = rng.permutation(n)[: rng.integers(1, n + 1)]
        columns: dict[str, Column] = {"id": Column(kind="key")}
        for c in range(self.columns.draw(rng)):
            i = int(rng.choice(feature_nodes))
            node = f"n{i}"
            missing = self.missing.draw(rng) if rng.random() < self.missing_share else 0.0
            if categorical[i]:
                categories = tuple(f"c{k}" for k in range(dims[i]))
                columns[f"col{c}"] = Column(
                    node, "categorical", categories=categories, missing=missing
                )
            elif rng.random() < self.binned:
                k = self.classes.draw(rng)
                probabilities = tuple(float(p) for p in rng.dirichlet(np.ones(k)))
                dim = int(rng.integers(dims[i]))
                columns[f"col{c}"] = Column(
                    node,
                    "categorical",
                    dims=dim,
                    categories=tuple(f"c{j}" for j in range(k)),
                    probabilities=probabilities,
                    missing=missing,
                )
            else:
                dim = int(rng.integers(dims[i]))
                columns[f"col{c}"] = Column(
                    node, dims=dim, marginal=self.marginals.draw(rng), missing=missing
                )
        return columns
