from dataclasses import dataclass, fields, replace

import numpy as np

from plurel.columns import DEFAULT_CALENDAR, Column
from plurel.distributions import (
    Calendar,
    Exponential,
    Gumbel,
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
    Edge,
    FourierEdge,
    LinearEdge,
    MatrixEdge,
    MLPEdge,
    Node,
    QuadraticEdge,
    TreeEdge,
)
from plurel.random import Seed, generator
from plurel.schema import COMPLETE, FK, Foreign, Schema, Summary
from plurel.scm import SCM

ACTIVATIONS = tuple(name for name in TRANSFORM_NAMES if name != "identity")


def _linear(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    return LinearEdge(parent, float(rng.normal()), str(rng.choice(TRANSFORM_NAMES)), dim=d_in)


def _matrix(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    return MatrixEdge(parent, rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, d_out)))


def _mlp(prior, parent: str, d_in: int, d_out: int, rng: np.random.Generator) -> Edge:
    hidden = prior.mlp_hidden_width.draw(rng)
    weights = (
        rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_in, hidden)),
        rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, d_out)),
    )
    biases = (rng.normal(0.0, 0.5, hidden), np.zeros(d_out))
    return MLPEdge(parent, weights, biases, ("identity", str(rng.choice(ACTIVATIONS)), "identity"))


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
    is added only on request; a schema prior decides it by table role.

    Attributes:
        node_count: Nodes in the table's DAG.
        node_layouts: DAG generator for the node graph.
        node_width: Latent dimensions of a numeric node.
        node_categorical_share: Probability that a node is categorical, a one-hot node.
        node_class_count: Classes of a categorical node.
        edge_families: Edge family per edge; linear only when parent and node widths agree.
        node_ops: Reduction over the edges of a node with several parents.
        node_noise_std: Standard deviation of the Gaussian noise of a node.
        root_noise: Exogenous distribution of a source node.
        mlp_hidden_width: Hidden width of an MLP edge.
        tree_count: Oblivious trees in a tree edge.
        tree_depth: Depth of each oblivious tree.
        fourier_frequency_count: Random Fourier features in a Fourier edge.
        column_count: Observed columns besides the key.
        column_marginals: Marginal a numeric column is rank-mapped onto; None keeps the latent.
        column_binned_share: Probability that a numeric column is binned into categories instead.
        column_bin_count: Categories of a binned column.
        column_missing_rate: Missing rate of a column that has missingness.
        column_missing_share: Probability that a column has missingness.
        time_calendar: Calendar the time column is drawn from.
    """

    node_count: Range = LogIntegersRange(3, 16)
    node_layouts: Choices = Choices(
        (RandomCauchy(), RandomCauchy(2.0), BarabasiAlbert(2), Layered(3, 0.2))
    )
    node_width: Range = LogIntegersRange(1, 4)
    node_categorical_share: float = 0.3
    node_class_count: Range = IntegersRange(2, 8)
    edge_families: Choices = Choices(FAMILIES)
    node_ops: Choices = Choices(("sum", "product", "max", "logsumexp"), (6.0, 1.0, 1.0, 1.0))
    node_noise_std: Range = LogRange(0.01, 0.5)
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
    time_calendar: Calendar = DEFAULT_CALENDAR

    def __post_init__(self) -> None:
        if set(self.edge_families.values) <= set(PRESERVING):
            raise ValueError("edge_families needs a family that can change width")

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
        dims = [
            self.node_class_count.draw(rng) if categorical[i] else self.node_width.draw(rng)
            for i in range(n)
        ]
        nodes = {f"n{i}": self.node(parents[i], dims, i, categorical[i], rng) for i in range(n)}
        columns = {"id": Column(kind="key")}
        feature_nodes = rng.permutation(n)[: rng.integers(1, n + 1)]
        for c in range(self.column_count.draw(rng)):
            i = int(rng.choice(feature_nodes))
            columns[f"col{c}"] = self.column(i, dims[i], categorical[i], rng)
        if time:
            nodes["time"] = Node(noise=self.time_calendar)
            columns["time"] = Column("time", "timestamp")
        return SCM(nodes, columns, time_column="time" if time else None)

    def node(
        self,
        sources: tuple[int, ...],
        dims: list[int],
        i: int,
        categorical: bool,
        rng: np.random.Generator,
    ) -> Node:
        edges = tuple(self.edge(f"n{p}", dims[p], dims[i], categorical, rng) for p in sources)
        if categorical:
            return Node(
                edges, bias=tuple(rng.normal(0.0, 0.5, dims[i])), onehot=True, noise=Gumbel()
            )
        if not edges:
            return Node(dim=dims[i], noise=self.root_noise.draw(rng))
        op = self.node_ops.draw(rng) if len(edges) > 1 else "sum"
        return Node(edges, op, noise=Normal(std=self.node_noise_std.draw(rng)))

    def edge(
        self, parent: str, d_in: int, d_out: int, block: bool, rng: np.random.Generator
    ) -> Edge:
        preserving = d_in == d_out and not block
        families = self.edge_families if preserving else self.edge_families.without(PRESERVING)
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


def _consume(node: Node, edge: Edge) -> Node:
    return replace(node, edges=node.edges + (edge,))


def _plain(scm: SCM) -> list[str]:
    """Nodes with no summary among their ancestors, the ones a key may read."""
    tainted: set[str] = set()
    for name in scm.order:
        tails = scm.nodes[name].parents
        if any(isinstance(tail, Summary) or tail in tainted for tail in tails):
            tainted.add(name)
    return [name for name in scm.nodes if name not in tainted]


@dataclass(frozen=True)
class SchemaPrior:
    """Random multi-table schema prior.

    A table's parents in the table graph are the tables it references. Each table is a fresh
    warp of `table_prior`. Across each key, drawn parent nodes feed drawn child nodes and
    summaries of drawn child nodes feed drawn parent nodes, both as extra edges on existing
    nodes; summaries are wired first and keys never read a node downstream of one, so the node
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
        self_reference_probability: Probability that a static table gets a self-referential tree
            key.
        self_reference_root_share: Share of roots in a self-referential tree.
        gather_count: Parent nodes gathered into the child per foreign key.
        aggregate_count: Child node summaries fed into the parent per foreign key between static
            tables.
        aggregates: Aggregation a summary edge draws from.
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

    def __post_init__(self) -> None:
        clusters = self.link_cluster_count.high**self.link_level_count.high
        if min(self.entity_row_count.low, self.activity_row_count.low) < clusters:
            raise ValueError(f"row counts must allow {clusters} link clusters")

    def realize(self, seed: Seed = None) -> Schema:
        rng, _ = generator(seed).spawn(2)
        n = self.table_count.draw(rng)
        parents = self.table_layouts.draw(rng).sample(n, rng)
        referenced = {p for references in parents for p in references}
        priors = [self.table_prior.warp(rng) for _ in range(n)]
        tables = {f"t{i}": priors[i].build(rng, time=i not in referenced) for i in range(n)}
        fkeys = [
            self.fkey(f"t{i}", f"t{p}", rng)
            for i, references in enumerate(parents)
            for p in references
        ]
        for fk in fkeys:
            if tables[fk.table].time_column is None:
                self.aggregate(tables, fk, priors[int(fk.parent[1:])], rng)
        for fk in fkeys:
            self.gather(tables, fk, priors[int(fk.table[1:])], rng)
        for i in sorted(referenced):
            if rng.random() < self.self_reference_probability:
                fk = FK(
                    f"t{i}",
                    "parent_id",
                    f"t{i}",
                    TreeLink(self.self_reference_root_share.draw(rng)),
                    fill=0.0,
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
        return FK(child, f"{parent}_id", parent, self.link(rng), nullable=nullable, fill=0.0)

    def gather(
        self, tables: dict[str, SCM], fk: FK, prior: TablePrior, rng: np.random.Generator
    ) -> None:
        """Give drawn parent nodes to drawn child nodes as edges crossing the key."""
        child, parent = tables[fk.table], tables[fk.parent]
        sources = [name for name in _plain(parent) if name not in parent.timestamp_nodes]
        consumers = [name for name in _plain(child) if name not in child.timestamp_nodes]
        if fk.table == fk.parent:
            sources = [name for name in sources if not parent.nodes[name].parents]
            consumers = [name for name in consumers if child.nodes[name].parents]
        nodes = dict(child.nodes)
        count = min(self.gather_count.draw(rng), len(sources), len(consumers))
        for source in map(str, rng.choice(sources, count, replace=False)) if count else ():
            consumer = str(rng.choice(consumers))
            target = nodes[consumer]
            tail = Foreign(fk.column, source)
            edge = prior.edge(tail, parent.nodes[source].dim, target.dim, target.onehot, rng)
            nodes[consumer] = _consume(target, edge)
        tables[fk.table] = SCM(nodes, child.columns, time_column=child.time_column)

    def aggregate(
        self, tables: dict[str, SCM], fk: FK, prior: TablePrior, rng: np.random.Generator
    ) -> None:
        """Feed summaries of drawn child nodes into drawn parent nodes as edges crossing the key.

        Summaries are wired before any key is read, and keys read only nodes with no summary
        among their ancestors, so the node graph across tables stays acyclic.
        """
        child, parent = tables[fk.table], tables[fk.parent]
        sources = [name for name, node in child.nodes.items() if node.dim == 1]
        consumers = [name for name in parent.nodes if name not in parent.timestamp_nodes]
        nodes = dict(parent.nodes)
        count = min(self.aggregate_count.draw(rng), len(sources), len(consumers))
        for source in map(str, rng.choice(sources, count, replace=False)) if count else ():
            consumer, how = str(rng.choice(consumers)), self.aggregates.draw(rng)
            tail = Summary(fk.table, fk.column, source, how, fill=None if how in COMPLETE else 0.0)
            target = nodes[consumer]
            nodes[consumer] = _consume(target, prior.edge(tail, 1, target.dim, target.onehot, rng))
        tables[fk.parent] = SCM(nodes, parent.columns, time_column=parent.time_column)
