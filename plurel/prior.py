from dataclasses import dataclass, fields, replace

import numpy as np

from plurel.columns import DEFAULT_CALENDAR, Column
from plurel.distributions import (
    Calendar,
    Exponential,
    LogNormal,
    Mixture,
    Normal,
    Pareto,
    Uniform,
)
from plurel.layouts import (
    BarabasiAlbert,
    ErdosRenyi,
    Layered,
    RandomCauchy,
    RandomTree,
    ReverseRandomTree,
    WattsStrogatz,
)
from plurel.links import HSBMLink, Link, RandomLink, TreeLink
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
from plurel.schema import FK, Port, Schema
from plurel.scm import SCM

ACTIVATIONS = tuple(name for name in TRANSFORM_NAMES if name != "identity")


def _linear(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    return LinearEffect(parent, float(rng.normal()), str(rng.choice(TRANSFORM_NAMES)), dim=d_in)


def _matrix(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    return MatrixEffect(parent, rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, d_out)))


def _mlp(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    hidden = prior.mlp_hidden_width.draw(rng)
    weights = (
        rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, hidden)),
        rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, d_out)),
    )
    biases = (rng.normal(0.0, 0.5, hidden), np.zeros(d_out))
    return MLPEffect(
        parent, weights, biases, ("identity", str(rng.choice(ACTIVATIONS)), "identity")
    )


def _tree(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    trees, depth = prior.tree_count.draw(rng), prior.tree_depth.draw(rng)
    splits = rng.integers(0, d_in, (trees, depth))
    return TreeEffect(
        parent, splits, rng.normal(size=(trees, depth)), rng.normal(size=(trees, 2**depth, d_out))
    )


def _fourier(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Effect:
    frequencies = rng.normal(size=(d_in, prior.fourier_frequency_count)) * rng.uniform(0.5, 3.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, prior.fourier_frequency_count)
    weights = rng.normal(
        0.0, 1.0 / np.sqrt(prior.fourier_frequency_count), (prior.fourier_frequency_count, d_out)
    )
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

    def warp(self, rng: np.random.Generator) -> "Range":
        location, concentration = META_LOCATION.draw(rng), META_CONCENTRATION.draw(rng)
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


META_LOCATION = Range(0.0, 1.0)
META_CONCENTRATION = LogRange(0.1, 10_000.0)


@dataclass(frozen=True)
class TablePrior:
    """Random single-table SCM prior.

    Ranges and choices are warped once per table, then drawn per use.

    Attributes:
        node_count: Nodes in the table's DAG.
        node_layouts: DAG generator for the node graph.
        node_width: Latent dimensions of a numeric node.
        node_categorical_share: Probability that a node is categorical, a Softmax.
        node_class_count: Classes of a categorical node.
        effect_families: Effect family per edge; linear only when parent and node widths agree.
        combine_ops: Reduction over the effects of a node with several parents.
        combine_noise_std: Standard deviation of the Gaussian noise of a Combine node.
        root_noise: Exogenous distribution of a source node.
        mlp_hidden_width: Hidden width of an MLP effect.
        tree_count: Oblivious trees in a tree effect.
        tree_depth: Depth of each oblivious tree.
        fourier_frequency_count: Random Fourier features in a Fourier effect.
        column_count: Observed columns besides the key.
        column_marginals: Marginal a numeric column is rank-mapped onto; None keeps the latent.
        column_binned_share: Probability that a numeric column is binned into categories instead.
        column_bin_count: Categories of a binned column.
        column_missing_rate: Missing rate of a column that has missingness.
        column_missing_share: Probability that a column has missingness.
        time_probability: Probability that the table gets a calendar time column.
        time_calendar: Calendar the time column is drawn from.
    """

    node_count: Range = LogIntegersRange(3, 16)
    node_layouts: Choices = Choices(
        (RandomCauchy(), RandomCauchy(2.0), BarabasiAlbert(2), Layered(3, 0.2))
    )
    node_width: Range = LogIntegersRange(1, 4)
    node_categorical_share: float = 0.3
    node_class_count: Range = IntegersRange(2, 8)
    effect_families: Choices = Choices(FAMILIES)
    combine_ops: Choices = Choices(("sum", "product", "max", "logsumexp"), (6.0, 1.0, 1.0, 1.0))
    combine_noise_std: Range = LogRange(0.01, 0.5)
    root_noise: Choices = Choices(
        (Normal(), Uniform(-1.7, 1.7), Mixture((Normal(-1.5, 0.5), Normal(1.5, 0.5))))
    )
    mlp_hidden_width: Range = LogIntegersRange(2, 16)
    tree_count: Range = LogIntegersRange(1, 8)
    tree_depth: Range = IntegersRange(1, 4)
    fourier_frequency_count: int = 16
    column_count: Range = IntegersRange(3, 12)
    column_marginals: Choices = Choices(
        (None, Uniform(), LogNormal(), Pareto(2.0), Exponential()), (3.0, 1.0, 1.0, 1.0, 1.0)
    )
    column_binned_share: float = 0.2
    column_bin_count: Range = IntegersRange(2, 8)
    column_missing_rate: Range = Range(0.01, 0.1)
    column_missing_share: float = 0.3
    time_probability: float = 0.5
    time_calendar: Calendar = DEFAULT_CALENDAR

    def __post_init__(self) -> None:
        if set(self.effect_families.values) <= set(PRESERVING):
            raise ValueError("effect_families needs a family that can change width")

    def warp(self, rng: np.random.Generator) -> "TablePrior":
        knobs = {
            field.name: getattr(self, field.name).warp(rng)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Range | Choices)
        }
        return replace(self, **knobs)

    def realize(self, seed: Seed = None) -> SCM:
        rng = generator(seed)
        return self.warp(rng).build(rng)

    def build(self, rng: np.random.Generator) -> SCM:
        n = self.node_count.draw(rng)
        parents = self.node_layouts.draw(rng).sample(n, rng)
        categorical = rng.random(n) < self.node_categorical_share
        dims = [
            self.node_class_count.draw(rng) if categorical[i] else self.node_width.draw(rng)
            for i in range(n)
        ]
        mechanisms = {
            f"n{i}": self.mechanism(parents[i], dims, i, categorical[i], rng) for i in range(n)
        }
        columns = {"id": Column(kind="key")}
        feature_nodes = rng.permutation(n)[: rng.integers(1, n + 1)]
        for c in range(self.column_count.draw(rng)):
            i = int(rng.choice(feature_nodes))
            columns[f"col{c}"] = self.column(i, dims[i], categorical[i], rng)
        time_column = None
        if rng.random() < self.time_probability:
            mechanisms["time"] = Root(noise=self.time_calendar)
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
        op = self.combine_ops.draw(rng) if len(effects) > 1 else "sum"
        return Combine(effects, op, noise=Normal(std=self.combine_noise_std.draw(rng)))

    def effect(
        self, parent: str, d_in: int, d_out: int, block: bool, rng: np.random.Generator
    ) -> Effect:
        preserving = d_in == d_out and not block
        families = self.effect_families if preserving else self.effect_families.without(PRESERVING)
        return BUILDERS[families.draw(rng)](self, parent, d_in, d_out, rng)

    def column(self, i: int, dim: int, categorical: bool, rng: np.random.Generator) -> Column:
        node = f"n{i}"
        missing = (
            self.column_missing_rate.draw(rng) if rng.random() < self.column_missing_share else 0.0
        )
        if categorical:
            categories = tuple(f"c{k}" for k in range(dim))
            return Column(node, "categorical", categories=categories, missing=missing)
        slot = int(rng.integers(dim))
        if rng.random() < self.column_binned_share:
            k = self.column_bin_count.draw(rng)
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
        return Column(node, dims=slot, marginal=self.column_marginals.draw(rng), missing=missing)


def _consume(mechanism: Mechanism, effect: Effect) -> Mechanism:
    if isinstance(mechanism, Combine):
        return replace(mechanism, effects=mechanism.effects + (effect,))
    return Combine((effect,), noise=mechanism.noise)


@dataclass(frozen=True)
class SchemaPrior:
    """Random multi-table schema prior.

    A table's parents in the table graph are the tables it references. Each table is a fresh
    warp of `table_prior`. Gathered parent nodes become extra effects on existing child nodes;
    aggregated child nodes become new observed nodes of the parent, so the node graph across
    tables stays acyclic by construction.

    Attributes:
        table_count: Tables in the database.
        table_layouts: DAG generator for the table graph.
        table_prior: Prior of every table's SCM.
        entity_row_count: Rows of a table that other tables reference.
        activity_row_count: Rows of a table nothing references.
        link_level_count: Cluster levels of an HSBM link.
        link_cluster_count: Clusters per level on each side of an HSBM link.
        link_within: Affinity of matching clusters in an HSBM link.
        link_between: Affinity of mismatched clusters in an HSBM link.
        link_cluster_weights: Relative cluster sizes of an HSBM link; None for equal sizes.
        link_popularity: Per-parent popularity of an HSBM link; None for uniform.
        link_inactive_share: Share of parents an HSBM link leaves without children.
        link_random_share: Probability that a key uses a uniform link instead of an HSBM link.
        fk_nullable_share: Probability that a foreign key has null values.
        fk_nullable_rate: Null rate of a nullable foreign key.
        self_reference_probability: Probability that a table gets a self-referential tree key.
        self_reference_root_share: Share of roots in a self-referential tree.
        gather_count: Parent nodes gathered into the child per foreign key.
        aggregate_count: Child nodes aggregated into the parent per foreign key.
        aggregates: Aggregation of an aggregate port.
        time_follow_probability: Probability that a child's time follows its parent's time.
        time_delay: Mean delay in seconds of a child event after its parent event.
    """

    table_count: Range = LogIntegersRange(2, 8)
    table_layouts: Choices = Choices(
        (
            BarabasiAlbert(2),
            ReverseRandomTree(),
            RandomTree(),
            WattsStrogatz(2),
            Layered(3, 0.1),
            ErdosRenyi(0.4),
        )
    )
    table_prior: TablePrior = TablePrior()
    entity_row_count: Range = LogIntegersRange(500, 1000)
    activity_row_count: Range = LogIntegersRange(10_000, 30_000)
    link_level_count: Range = IntegersRange(1, 3)
    link_cluster_count: Range = IntegersRange(1, 3)
    link_within: Choices = Choices((0.9, Uniform(0.4, 0.95)))
    link_between: Choices = Choices((Uniform(0.001, 0.002), Pareto(1.0, 0.01)))
    link_cluster_weights: Choices = Choices((None, Pareto(1.5)))
    link_popularity: Choices = Choices((None, Pareto(2.5)))
    link_inactive_share: Range = Range(0.0, 0.7)
    link_random_share: float = 0.2
    fk_nullable_share: float = 0.3
    fk_nullable_rate: Range = Range(0.01, 0.3)
    self_reference_probability: float = 0.3
    self_reference_root_share: Range = Range(0.05, 0.5)
    gather_count: Range = IntegersRange(0, 3)
    aggregate_count: Range = IntegersRange(0, 2)
    aggregates: Choices = Choices(("count", "sum", "mean", "max", "min"))
    time_follow_probability: float = 0.7
    time_delay: Range = LogRange(3600.0, 90 * 24 * 3600.0)

    def __post_init__(self) -> None:
        clusters = self.link_cluster_count.high**self.link_level_count.high
        if min(self.entity_row_count.low, self.activity_row_count.low) < clusters:
            raise ValueError(f"row counts must allow {clusters} link clusters")

    def realize(self, seed: Seed = None) -> Schema:
        rng, _ = generator(seed).spawn(2)
        n = self.table_count.draw(rng)
        parents = self.table_layouts.draw(rng).sample(n, rng)
        priors = [self.table_prior.warp(rng) for _ in range(n)]
        tables = {f"t{i}": priors[i].build(rng) for i in range(n)}
        fkeys, following = [], set()
        for i, references in enumerate(parents):
            for p in references:
                fk = self.fkey(f"t{i}", f"t{p}", rng)
                fkeys.append(fk)
                self.gather(tables, fk, priors[i], rng)
                self.aggregate(tables, fk, rng)
                if not fk.nullable and fk.table not in following and self.follow(tables, fk, rng):
                    following.add(fk.table)
        for i in range(n):
            if rng.random() < self.self_reference_probability:
                fk = FK(
                    f"t{i}",
                    "parent_id",
                    f"t{i}",
                    TreeLink(self.self_reference_root_share.draw(rng)),
                )
                fkeys.append(fk)
                self.gather(tables, fk, priors[i], rng)
        return Schema(tables, tuple(fkeys))

    def rows(self, schema: Schema, seed: Seed = None) -> dict[str, int]:
        _, rng = generator(seed).spawn(2)
        referenced = {fk.parent for fk in schema.fkeys if fk.parent != fk.table}
        return {
            table: (self.entity_row_count if table in referenced else self.activity_row_count).draw(
                rng
            )
            for table in schema.tables
        }

    def link(self, rng: np.random.Generator) -> Link:
        if rng.random() < self.link_random_share:
            return RandomLink()
        levels = self.link_level_count.draw(rng)
        return HSBMLink(
            tuple(self.link_cluster_count.draw(rng) for _ in range(levels)),
            tuple(self.link_cluster_count.draw(rng) for _ in range(levels)),
            within=self.link_within.draw(rng),
            between=self.link_between.draw(rng),
            cluster_weights=self.link_cluster_weights.draw(rng),
            popularity=self.link_popularity.draw(rng),
            inactive=self.link_inactive_share.draw(rng),
        )

    def fkey(self, child: str, parent: str, rng: np.random.Generator) -> FK:
        nullable = self.fk_nullable_rate.draw(rng) if rng.random() < self.fk_nullable_share else 0.0
        return FK(child, f"{parent}_id", parent, self.link(rng), nullable=nullable)

    def gather(
        self, tables: dict[str, SCM], fk: FK, prior: TablePrior, rng: np.random.Generator
    ) -> None:
        child, parent = tables[fk.table], tables[fk.parent]
        sources = [
            name
            for name, m in parent.mechanisms.items()
            if not isinstance(m, Port) and name not in parent.timestamp_nodes
        ]
        consumers = [
            name
            for name, m in child.mechanisms.items()
            if not isinstance(m, Port) and name not in child.timestamp_nodes
        ]
        if fk.table == fk.parent:
            sources = [name for name in sources if not parent.mechanisms[name].parents]
            consumers = [name for name in consumers if child.mechanisms[name].parents]
        mechanisms = dict(child.mechanisms)
        count = min(self.gather_count.draw(rng), len(sources), len(consumers))
        for source in map(str, rng.choice(sources, count, replace=False)) if count else ():
            consumer = str(rng.choice(consumers))
            port = f"{fk.column}_{source}"
            mechanisms[port] = Port(
                fk.parent, source, via=fk.column, fill=0.0, dim=parent.mechanisms[source].dim
            )
            target = mechanisms[consumer]
            block = isinstance(target, Softmax)
            effect = prior.effect(port, parent.mechanisms[source].dim, target.dim, block, rng)
            mechanisms[consumer] = _consume(target, effect)
        tables[fk.table] = SCM(mechanisms, child.columns, time_column=child.time_column)

    def aggregate(self, tables: dict[str, SCM], fk: FK, rng: np.random.Generator) -> None:
        child, parent = tables[fk.table], tables[fk.parent]
        sources = [
            name
            for name, m in child.mechanisms.items()
            if m.dim == 1 and not isinstance(m, Port) and name not in child.timestamp_nodes
        ]
        mechanisms, columns = dict(parent.mechanisms), dict(parent.columns)
        count = min(self.aggregate_count.draw(rng), len(sources))
        for source in map(str, rng.choice(sources, count, replace=False)) if count else ():
            how = self.aggregates.draw(rng)
            name = f"{fk.table}_{source}_{how}"
            fill = None if how in ("count", "sum") else np.nan
            mechanisms[name] = Port(fk.table, source, via=fk.column, aggregate=how, fill=fill)
            columns[name] = Column(name)
        tables[fk.parent] = SCM(mechanisms, columns, time_column=parent.time_column)

    def follow(self, tables: dict[str, SCM], fk: FK, rng: np.random.Generator) -> bool:
        child, parent = tables[fk.table], tables[fk.parent]
        if child.time_column is None or parent.time_column is None:
            return False
        if rng.random() >= self.time_follow_probability:
            return False
        source = parent.columns[parent.time_column].node
        target = child.columns[child.time_column].node
        port = f"{fk.column}_{source}"
        mechanisms = dict(child.mechanisms)
        mechanisms[port] = Port(fk.parent, source, via=fk.column)
        delay = Exponential(self.time_delay.draw(rng))
        mechanisms[target] = Combine((LinearEffect(port),), noise=delay)
        tables[fk.table] = SCM(mechanisms, child.columns, time_column=child.time_column)
        return True
