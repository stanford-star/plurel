from collections.abc import Hashable, Mapping
from dataclasses import dataclass, fields, replace

import numpy as np

from plurel.columns import DEFAULT_CALENDAR, Column
from plurel.distributions import (
    AutoRegressive,
    Beta,
    Cycle,
    Exponential,
    Gumbel,
    LogNormal,
    Mixture,
    Normal,
    Pareto,
    Poisson,
    TimeSeries,
    Trend,
    Uniform,
)
from plurel.graph import (
    COMPLETE,
    TRANSFORM_NAMES,
    Edge,
    Foreign,
    FourierEdge,
    LinearEdge,
    LookupEdge,
    MatrixEdge,
    MLPEdge,
    NearestEdge,
    Node,
    QuadraticEdge,
    Summary,
    TreeEdge,
    nested_logits,
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
from plurel.random import Seed, generator
from plurel.schema import FK, SCM, Schema

ACTIVATIONS = tuple(name for name in TRANSFORM_NAMES if name != "identity")
WORKWEEK = (1.0,) * 5 + (0.3, 0.3)
LEISURE = (0.6,) * 4 + (0.8, 1.0, 1.0)
EVENING_HOURS = (0.2,) * 8 + (0.5,) * 9 + (1.0,) * 5 + (0.4,) * 2
CALENDARS = (
    DEFAULT_CALENDAR,
    replace(DEFAULT_CALENDAR, weekday_weights=WORKWEEK),
    replace(DEFAULT_CALENDAR, weekday_weights=LEISURE, hour_weights=EVENING_HOURS),
    replace(DEFAULT_CALENDAR, hour_weights=(1.0,) * 24),
)
ZERO_INFLATED = tuple(
    Mixture((Normal(0.0, 0.0), Exponential()), (share, 1.0 - share)) for share in (0.3, 0.7)
)
OUTLIERS = Mixture((Normal(), Normal(0.0, 8.0)), (0.97, 0.03))


def _linear(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    return LinearEdge(parent, float(rng.normal()), str(rng.choice(TRANSFORM_NAMES)), dim=d_in)


def _lookup(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    k = prior.node_class_count.draw(rng)
    probabilities = tuple(float(p) for p in rng.dirichlet(np.ones(k)))
    return LookupEdge(parent, tuple(float(v) for v in rng.normal(size=k)), probabilities)


def _matrix(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    return MatrixEdge(parent, rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, d_out)))


def _nearest(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    return NearestEdge(parent, rng.normal(size=(d_out, d_in)))


def _mlp(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    hidden = [prior.mlp_hidden_width.draw(rng) for _ in range(prior.mlp_layer_count.draw(rng))]
    widths, gain = [d_in, *hidden, d_out], prior.mlp_weight_scale.draw(rng)
    weights = tuple(rng.normal(0.0, gain / np.sqrt(a), (a, b)) for a, b in zip(widths, widths[1:]))
    biases = tuple(rng.normal(0.0, 0.5, b) for b in hidden) + (np.zeros(d_out),)
    activations = ("identity", *(str(rng.choice(ACTIVATIONS)) for _ in hidden), "identity")
    return MLPEdge(parent, weights, biases, activations)


def _tree(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    trees, depth = prior.tree_count.draw(rng), prior.tree_depth.draw(rng)
    splits = rng.integers(0, d_in, (trees, depth))
    return TreeEdge(
        parent, splits, rng.normal(size=(trees, depth)), rng.normal(size=(trees, 2**depth, d_out))
    )


def _fourier(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    frequencies = rng.normal(size=(d_in, prior.fourier_frequency_count)) * rng.uniform(0.5, 3.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, prior.fourier_frequency_count)
    weights = rng.normal(
        0.0, 1.0 / np.sqrt(prior.fourier_frequency_count), (prior.fourier_frequency_count, d_out)
    )
    return FourierEdge(parent, frequencies, phases, weights)


def _quadratic(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    scale = 1.0 / (np.sqrt(d_in) * (d_in + 1))
    return QuadraticEdge(parent, rng.normal(0.0, scale, (d_out, d_in + 1, d_in + 1)))


BUILDERS = {
    "linear": _linear,
    "lookup": _lookup,
    "matrix": _matrix,
    "nearest": _nearest,
    "mlp": _mlp,
    "tree": _tree,
    "fourier": _fourier,
    "quadratic": _quadratic,
}
FAMILIES = tuple(BUILDERS)
FITS = {
    "linear": lambda d_in, d_out, block: d_in == d_out and not block,
    "lookup": lambda d_in, d_out, block: d_in == d_out == 1 and not block,
    "nearest": lambda d_in, d_out, block: d_out >= 2,
}


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
        location, concentration = _open_unit(rng), META_CONCENTRATION.draw(rng)
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


def _open_unit(rng: np.random.Generator) -> float:
    return float(rng.integers(1, 2**53) / 2**53)


@dataclass(frozen=True)
class TablePrior:
    """Random single-table SCM prior.

    Ranges and choices are warped once per table, then drawn per use. A calendar time column
    is added only on request; a schema prior decides it by table role. Nodes standardize their
    signal, so noise, bias and crossing terms are at unit scale; a categorical node with a
    nested edge keeps its raw scores, so the mask holds.

    Attributes:
        node_count: Nodes in the table's DAG.
        node_layouts: DAG generator for the node graph.
        node_width: Latent dimensions of a numeric node; a concat node has its parents' widths
            together.
        node_categorical_share: Probability that a node is categorical, a one-hot node.
        node_class_count: Classes of a categorical node, and levels of a lookup edge.
        node_nested_share: Probability that a categorical node with categorical parents nests
            its classes in one of them: each parent class allows a drawn subset of the classes.
        node_ops: Reduction over the edges of a numeric node with several parents.
        node_noise_std: Standard deviation of the Gaussian noise of a numeric node, relative
            to its standardized signal.
        key_reader_share: Share of the table's nodes that read the edges crossing one key, drawn
            per key; at least one node reads.
        edge_families: Edge family per edge, among those fitting its widths: linear keeps a
            numeric node's width, lookup a width of one, nearest needs at least two outputs.
        root_noise: Exogenous distribution of a source node.
        root_series_share: Probability that a source node of a table with a time column is a
            time series over the rows, which a calendar keeps in time order.
        series_trend_alpha: Exponent of the trend of a time series.
        series_trend_scale: Rise of the trend over the rows.
        series_cycle_periods: Periods of the cycle over the rows.
        series_cycle_scale: Amplitude of the cycle.
        series_rho: Autocorrelation of the noise of a time series.
        series_noise_std: Standard deviation of that noise.
        mlp_hidden_width: Hidden width of an MLP edge.
        mlp_layer_count: Hidden layers of an MLP edge.
        mlp_weight_scale: Gain of the weights of an MLP edge over the 1/sqrt(fan-in) scale.
        tree_count: Oblivious trees in a tree edge.
        tree_depth: Depth of each oblivious tree.
        fourier_frequency_count: Random Fourier features in a Fourier edge.
        column_count: Observed columns besides the key.
        column_marginals: Marginal a numeric column is rank-mapped onto; None keeps the latent,
            a mixture with a point mass gives a zero-inflated column, one with a wide component
            gives outliers.
        column_binned_share: Probability that a numeric column is binned into categories instead.
        column_bin_count: Categories of a binned column.
        column_binning: Bin edges of a binned column: normal quantiles or the latent's own.
        column_missing_rate: Missing rate of a column that has missingness.
        column_missing_share: Probability that a column has missingness.
        column_missing_structured_share: Probability that such a column is missing where an
            indicator node says so, a two-class node reading a drawn node, instead of at random.
        time_calendars: Calendar the time column is drawn from: business hours over flat
            weekdays, a work week, evenings and weekends, or always on.
    """

    node_count: Range = LogIntegersRange(3, 16)
    node_layouts: Choices = Choices(
        (
            RandomCauchy(),
            RandomCauchy(2.0),
            BarabasiAlbert(2),
            Layered(3, 0.2),
            ErdosRenyi(0.3),
            ErdosRenyi(0.6),
            Layered(6, 0.1),
            RandomTree(),
            ReverseRandomTree(),
            WattsStrogatz(2),
        )
    )
    node_width: Range = LogIntegersRange(1, 4)
    node_categorical_share: float = 0.3
    node_class_count: Range = IntegersRange(2, 10)
    node_nested_share: float = 0.5
    node_ops: Choices = Choices(
        ("sum", "product", "max", "min", "logsumexp", "concat"), (6.0, 1.0, 1.0, 1.0, 1.0, 2.0)
    )
    node_noise_std: Range = LogRange(0.001, 0.5)
    key_reader_share: Range = Range(0.1, 1.0)
    edge_families: Choices = Choices(FAMILIES)
    root_noise: Choices = Choices(
        (
            Normal(),
            Uniform(-1.7, 1.7),
            Mixture((Normal(-1.5, 0.5), Normal(1.5, 0.5))),
            Beta(0.5, 0.5, -1.7, 1.7),
            Beta(2.0, 5.0, -1.7, 1.7),
            Exponential(),
            LogNormal(),
            Pareto(2.0),
            Poisson(0.5),
            Poisson(3.0),
        )
    )
    root_series_share: float = 0.3
    series_trend_alpha: Range = Range(0.0, 2.0)
    series_trend_scale: Range = Range(-1.5, 1.5)
    series_cycle_periods: Range = LogRange(1.0, 12.0)
    series_cycle_scale: Range = Range(0.0, 1.0)
    series_rho: Range = Range(0.0, 0.9)
    series_noise_std: Range = LogRange(0.1, 1.0)
    mlp_hidden_width: Range = LogIntegersRange(2, 16)
    mlp_layer_count: Range = IntegersRange(1, 3)
    mlp_weight_scale: Range = LogRange(0.1, 3.0)
    tree_count: Range = LogIntegersRange(1, 8)
    tree_depth: Range = IntegersRange(1, 4)
    fourier_frequency_count: int = 16
    column_count: Range = IntegersRange(3, 12)
    column_marginals: Choices = Choices(
        (
            None,
            Normal(),
            Uniform(),
            LogNormal(),
            Pareto(2.0),
            Exponential(),
            *ZERO_INFLATED,
            OUTLIERS,
        ),
        (3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5),
    )
    column_binned_share: float = 0.2
    column_bin_count: Range = IntegersRange(2, 10)
    column_binning: Choices = Choices(("normal", "empirical"))
    column_missing_rate: Range = Range(0.01, 0.1)
    column_missing_share: float = 0.3
    column_missing_structured_share: float = 0.5
    time_calendars: Choices = Choices(CALENDARS)

    def __post_init__(self) -> None:
        if set(self.edge_families.values) <= set(FITS):
            raise ValueError("edge_families needs a family that fits any widths")

    def warp(self, rng: np.random.Generator) -> "TablePrior":
        knobs = {
            field.name: getattr(self, field.name).warp(rng)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Range | Choices)
        }
        return replace(self, **knobs)

    def realize(self, seed: Seed = None, *, time: bool = False) -> SCM:
        rng = generator(seed)
        return self.warp(rng).build(rng, time=time)

    def build(self, rng: np.random.Generator, time: bool) -> SCM:
        n = self.node_count.draw(rng)
        parents = self.node_layouts.draw(rng).sample(n, rng)
        categorical = rng.random(n) < self.node_categorical_share
        ops = [
            self.node_ops.draw(rng) if len(parents[i]) > 1 and not categorical[i] else "sum"
            for i in range(n)
        ]
        dims: list[int] = []
        for i in range(n):
            if categorical[i]:
                dims.append(self.node_class_count.draw(rng))
            elif ops[i] == "concat":
                dims.append(sum(dims[p] for p in parents[i]))
            else:
                dims.append(self.node_width.draw(rng))
        nodes = {
            f"n{i}": self.node(parents[i], ops[i], dims, i, categorical, rng, time)
            for i in range(n)
        }
        columns = {"id": Column(kind="key")}
        feature_nodes = rng.permutation(n)[: rng.integers(1, n + 1)]
        for c in range(self.column_count.draw(rng)):
            i = int(rng.choice(feature_nodes))
            missing: float | str = 0.0
            if rng.random() < self.column_missing_share:
                missing = self.column_missing_rate.draw(rng)
                if rng.random() < self.column_missing_structured_share:
                    nodes[f"m{c}"] = self.indicator(missing, dims, rng)
                    missing = f"m{c}"
            columns[f"col{c}"] = self.column(i, dims[i], categorical[i], missing, rng)
        if time:
            nodes["time"] = Node(noise=self.time_calendars.draw(rng))
            columns["time"] = Column("time", "timestamp")
        return SCM(nodes, columns, time_column="time" if time else None)

    def node(
        self,
        sources: tuple[int, ...],
        op: str,
        dims: list[int],
        i: int,
        categorical: np.ndarray,
        rng: np.random.Generator,
        time: bool,
    ) -> Node:
        if categorical[i]:
            nested = -1
            if (parents := [p for p in sources if categorical[p]]) and (
                rng.random() < self.node_nested_share
            ):
                nested = int(rng.choice(parents))
            edges = tuple(
                self.nested(f"n{p}", dims[p], dims[i], rng)
                if p == nested
                else self.edge(f"n{p}", dims[p], dims[i], True, rng)
                for p in sources
            )
            bias = tuple(rng.normal(0.0, 0.5, dims[i]))
            return Node(edges, bias=bias, onehot=True, noise=Gumbel(), standardize=nested < 0)
        if not sources:
            series = time and rng.random() < self.root_series_share
            noise = self.series(rng) if series else self.root_noise.draw(rng)
            return Node(dim=dims[i], noise=noise, standardize=True)
        widths = [dims[p] if op == "concat" else dims[i] for p in sources]
        edges = tuple(self.edge(f"n{p}", dims[p], w, False, rng) for p, w in zip(sources, widths))
        noise = Normal(std=self.node_noise_std.draw(rng))
        return Node(edges, op, noise=noise, standardize=True)

    def edge(
        self, parent: str, d_in: int, d_out: int, block: bool, rng: np.random.Generator
    ) -> Edge:
        unfit = tuple(name for name, fits in FITS.items() if not fits(d_in, d_out, block))
        family = self.edge_families.without(unfit).draw(rng)
        return BUILDERS[family](self, parent, d_in, d_out, rng)

    def nested(self, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
        """A matrix edge from a categorical parent whose classes each allow a drawn, non-empty
        subset of the node's classes."""
        allowed = rng.random((d_in, d_out)) < 0.5
        allowed[np.arange(d_in), rng.integers(d_out, size=d_in)] = True
        subsets = tuple(tuple(int(k) for k in np.flatnonzero(row)) for row in allowed)
        probabilities = tuple(float(p) for p in rng.dirichlet(np.ones(d_out)))
        return MatrixEdge(parent, nested_logits(subsets, probabilities))

    def indicator(self, rate: float, dims: list[int], rng: np.random.Generator) -> Node:
        """A two-class node reading one drawn node, in its second class at about `rate`."""
        p = int(rng.integers(len(dims)))
        edge = self.edge(f"n{p}", dims[p], 2, True, rng)
        bias = (0.0, float(np.log(rate / (1.0 - rate))))
        return Node((edge,), bias=bias, onehot=True, noise=Gumbel(), standardize=True)

    def series(self, rng: np.random.Generator) -> TimeSeries:
        trend = Trend(self.series_trend_alpha.draw(rng), self.series_trend_scale.draw(rng))
        cycle = Cycle(
            self.series_cycle_periods.draw(rng),
            self.series_cycle_scale.draw(rng),
            float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        noise = AutoRegressive(self.series_rho.draw(rng), self.series_noise_std.draw(rng))
        return TimeSeries(trend, cycle, noise)

    def column(
        self, i: int, dim: int, categorical: bool, missing: float | str, rng: np.random.Generator
    ) -> Column:
        node = f"n{i}"
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
                binning=self.column_binning.draw(rng),
                missing=missing,
            )
        return Column(node, dims=slot, marginal=self.column_marginals.draw(rng), missing=missing)


def _wire(
    scm: SCM,
    prior: TablePrior,
    crossings: dict[tuple[str, str], tuple[Edge, ...]],
    table: str,
    tails: Mapping[str, tuple[Hashable, int]],
    count: int,
    rng: np.random.Generator,
    readers: list[str] | None = None,
) -> list[str]:
    """Give the tails of `count` drawn sources to a drawn share of the readers, every node of
    the table unless given, as crossing edges built by the table's prior; returns the readers."""
    readers = list(scm.nodes) if readers is None else readers
    count = min(count, len(tails))
    if not count or not readers:
        return []
    share = prior.key_reader_share.draw(rng)
    chosen = [node for node in readers if rng.random() < share] or [str(rng.choice(readers))]
    drawn = [tails[str(source)] for source in rng.choice(list(tails), count, replace=False)]
    for node in chosen:
        target = scm.nodes[node]
        edges = tuple(
            prior.edge(tail, d_in, target.dim, target.onehot, rng) for tail, d_in in drawn
        )
        crossings[table, node] = crossings.get((table, node), ()) + edges
    return chosen


def _downstream(scm: SCM, readers: list[str]) -> set[str]:
    """The readers and every node below them in the table's DAG."""
    below = set(readers)
    for node in scm.order:
        if any(parent in below for parent in scm.nodes[node].parents):
            below.add(node)
    return below


@dataclass(frozen=True)
class SchemaPrior:
    """Random multi-table schema prior.

    A table's parents in the table graph are the tables it references, some through two keys.
    Each table is a fresh warp of `table_prior`, built as a local plan first. The crossing
    edges are then decided on the built tables and given to the Schema: across each key, a
    drawn share of the child's nodes read drawn parent nodes; across a drawn subset of the keys
    between static tables, a drawn share of the parent's nodes read summaries of drawn child
    nodes. Summaries are wired first and keys never read a node downstream of one, so the node
    graph across tables stays acyclic by construction.

    Tables nothing references hold events and get a time column; referenced tables are static,
    so no key ever points into the future and cutting a database at any time leaves every key
    valid. Self-referential tree keys go only on static tables, and aggregates summarize only
    static children into their static parents: a parent row summarizing later events would
    leak the future.

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
        fk_duplicate_share: Probability that a table references a parent through a second
            foreign key as well.
        self_reference_probability: Probability that a static table gets a self-referential tree
            key.
        self_reference_root_share: Share of roots in a self-referential tree.
        gather_count: Parent nodes gathered into the child per foreign key.
        aggregate_share: Probability that a key between static tables feeds summaries of the
            child into the parent.
        aggregate_count: Child node summaries fed into the parent per such key.
        aggregates: Aggregation a summary edge draws from.
    """

    table_count: Range = LogIntegersRange(2, 20)
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
    link_level_count: Range = IntegersRange(1, 5)
    link_cluster_count: Range = IntegersRange(1, 3)
    link_within: Choices = Choices((0.9, Uniform(0.4, 0.95)))
    link_between: Choices = Choices((Uniform(0.001, 0.002), Pareto(1.0, 0.01)))
    link_cluster_weights: Choices = Choices((None, Pareto(1.5)))
    link_popularity: Choices = Choices((None, Pareto(2.5)))
    link_inactive_share: Range = Range(0.0, 0.7)
    link_random_share: float = 0.2
    fk_nullable_share: float = 0.3
    fk_nullable_rate: Range = Range(0.01, 0.3)
    fk_duplicate_share: float = 0.15
    self_reference_probability: float = 0.3
    self_reference_root_share: Range = Range(0.05, 0.5)
    gather_count: Range = IntegersRange(0, 3)
    aggregate_share: float = 0.5
    aggregate_count: Range = IntegersRange(1, 2)
    aggregates: Choices = Choices(("count", "sum", "mean", "max", "min"))

    def __post_init__(self) -> None:
        clusters = self.link_cluster_count.high**self.link_level_count.high
        if min(self.entity_row_count.low, self.activity_row_count.low) < clusters:
            raise ValueError(f"row counts must allow {clusters} link clusters")

    def realize(self, seed: Seed = None) -> Schema:
        rng, _ = generator(seed).spawn(2)
        layout = self.table_layouts.draw(rng).sample(self.table_count.draw(rng), rng)
        names = [f"t{i}" for i in range(len(layout))]
        references = {names[i]: [names[p] for p in parents] for i, parents in enumerate(layout)}
        static = {parent for parents in references.values() for parent in parents}
        priors = {name: self.table_prior.warp(rng) for name in names}
        tables = {name: priors[name].build(rng, time=name not in static) for name in names}
        fkeys = []
        for name in names:
            for parent in references[name]:
                fkeys.append(self.fkey(name, f"{parent}_id", parent, rng))
                if rng.random() < self.fk_duplicate_share:
                    fkeys.append(self.fkey(name, f"{parent}_id2", parent, rng))
        crossings: dict[tuple[str, str], tuple[Edge, ...]] = {}
        tainted: dict[str, set[str]] = {name: set() for name in names}
        for fk in fkeys:
            if fk.table in static and rng.random() < self.aggregate_share:
                child, parent = tables[fk.table], tables[fk.parent]
                tails = {n: self.summary(fk, n, node.dim, rng) for n, node in child.nodes.items()}
                count = self.aggregate_count.draw(rng)
                readers = _wire(parent, priors[fk.parent], crossings, fk.parent, tails, count, rng)
                tainted[fk.parent] |= _downstream(parent, readers)
        for fk in fkeys:
            child, parent = tables[fk.table], tables[fk.parent]
            plain = [n for n in parent.nodes if n not in tainted[fk.parent]]
            tails = {n: (Foreign(fk.column, n), parent.nodes[n].dim) for n in plain}
            count = self.gather_count.draw(rng)
            readers = [n for n in child.nodes if n not in child.timestamp_nodes]
            _wire(child, priors[fk.table], crossings, fk.table, tails, count, rng, readers)
        for name in names:
            if name in static and rng.random() < self.self_reference_probability:
                link = TreeLink(self.self_reference_root_share.draw(rng))
                fkeys.append(FK(name, "parent_id", name, link, fill=0.0))
                scm = tables[name]
                roots = [
                    n for n in scm.nodes if not scm.nodes[n].parents and n not in tainted[name]
                ]
                tails = {n: (Foreign("parent_id", n), scm.nodes[n].dim) for n in roots}
                readers = [n for n in scm.nodes if scm.nodes[n].parents]
                count = self.gather_count.draw(rng)
                _wire(scm, priors[name], crossings, name, tails, count, rng, readers)
        return Schema(tables, tuple(fkeys), crossings)

    def summary(
        self, fk: FK, source: str, dim: int, rng: np.random.Generator
    ) -> tuple[Summary, int]:
        """A drawn summary of `source` across `fk`, with the width it contributes."""
        how = self.aggregates.draw(rng)
        tail = Summary(fk.table, fk.column, source, how, fill=None if how in COMPLETE else 0.0)
        return tail, 1 if how == "count" else dim

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

    def fkey(self, child: str, column: str, parent: str, rng: np.random.Generator) -> FK:
        nullable = self.fk_nullable_rate.draw(rng) if rng.random() < self.fk_nullable_share else 0.0
        return FK(child, column, parent, self.link(rng), nullable=nullable, fill=0.0)
