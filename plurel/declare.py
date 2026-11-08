"""What a table's entries say: its keys, its columns and the causes each column follows. Each
compiles to the nodes and columns of `plurel.graph` and `plurel.columns`. Any weight, effect, noise
or marginal may be a range or another drawable: one declaration is then a family of tables, and each
seed of `Table.realize` draws one."""

import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Literal

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.distributions import (
    CENTRAL,
    Affine,
    Bounded,
    Calendar,
    Distribution,
    Gamma,
    Gumbel,
    Laplace,
    Normal,
    StudentT,
    record_step,
    shape,
)
from plurel.graph import (
    TRANSFORMS,
    Edge,
    Foreign,
    Function,
    LevelEdge,
    LinearEdge,
    LookupEdge,
    MatrixEdge,
    Node,
    ReadEdge,
    Recorded,
    Summary,
)
from plurel.links import Link, RandomLink
from plurel.random import Range

Weight = float | Range
Levels = Mapping[object, Weight]  # an effect per level of a categorical cause
Effect = Weight | tuple[Weight, Function] | Levels | tuple[Weight, Levels]
# a cause names a column, a recorded value, a summary or a pair whose product it reads
Cause = str | Recorded | Summary | tuple[str, str]
Causes = Mapping[Cause, Effect]

CODE = ".code"  # a categorical column's hidden node holding its level's index
NOISE = 0.5  # the default standard deviation of the noise on a column's causes
BOUNDED_ROWS = 4096  # draws that place a scaled marginal on its range

STRONG, WEAK = Range(1.6, 3.0), Range(0.4, 1.4)
LOW, MEDIUM, HIGH = Range(0.25, 0.5), Range(0.5, 1.0), Range(1.0, 1.6)

HOW = {"sum": "sum", "max": "max", "min": "min", "product": "compound", "rules": "sum"}
UNITS = {
    "count": ("counts", "zero_inflated"),
    "amount": ("right", "zero_inflated", "symmetric", "bimodal"),
    "duration": ("right", "symmetric", "bimodal"),
    "ratio": ("right", "symmetric", "left"),
    "score": ("symmetric", "left", "bimodal", "right"),
    "percent": ("symmetric", "left", "right", "bimodal", "multimodal"),
    "signed": ("symmetric", "bimodal", "multimodal"),
}


def drawn(value: object, rng: np.random.Generator) -> object:
    """The value with every drawable in it drawn, through mappings and tuples."""
    if hasattr(value, "draw"):
        return value.draw(rng)
    if isinstance(value, Mapping):
        return {key: drawn(item, rng) for key, item in value.items()}
    return tuple(drawn(item, rng) for item in value) if isinstance(value, tuple) else value


def ranged(value: object) -> bool:
    """Whether drawing the value can give different values under different seeds."""
    if isinstance(value, Mapping):
        return any(ranged(item) for item in value.values())
    if hasattr(value, "draw"):
        return getattr(value, "family", True)
    return isinstance(value, tuple) and any(map(ranged, value))


@dataclass(frozen=True)
class Scaled:
    """A marginal on `[low, high]` whose shape is drawn per seed from those of `units`, or is the
    given `shape`; its central 99% spans the range and values are recorded to a fine step."""

    units: str
    low: float
    high: float
    shape: str | None = None

    def __post_init__(self) -> None:
        if self.units not in UNITS:
            raise ValueError(f"units must be one of {sorted(UNITS)}, not {self.units!r}")
        if self.shape is not None and self.shape not in UNITS[self.units]:
            raise ValueError(f"shape must be one of {UNITS[self.units]} for {self.units}")
        if not (np.isfinite(self.low) and np.isfinite(self.high) and self.low < self.high):
            raise ValueError("a scale is a finite low and a higher high")
        if self.units == "percent" and (self.low < 0.0 or self.high > 100.0):
            raise ValueError("a percent scale stays inside [0, 100]")
        if self.units == "signed" and not self.low < 0.0 < self.high:
            raise ValueError("a signed scale spans zero")
        if self.units not in ("signed", "score") and self.low < 0.0:
            raise ValueError(f"a {self.units} scale cannot go negative")

    def draw(self, rng: np.random.Generator) -> Bounded:
        kinds = UNITS[self.units]
        base = shape(self.shape or kinds[int(rng.integers(len(kinds)))], rng)
        ends = np.quantile(base.sample(BOUNDED_ROWS, rng), CENTRAL)
        scale = (self.high - self.low) / (ends[1] - ends[0] if ends[1] > ends[0] else 1.0)
        whole = float(self.low).is_integer() and float(self.high).is_integer()
        step = 1.0 if whole and self.units == "count" else record_step(self.low, self.high)
        return Bounded(base, self.low - scale * ends[0], scale, self.low, self.high, step)


@dataclass(frozen=True)
class Shares:
    """Level shares drawn per seed: around the given shares of a mapping, or from nothing but the
    levels of a sequence; a higher `concentration` holds them tighter, and no level falls below its
    authored share or a floor, whichever is lower."""

    levels: Sequence[object] | Mapping[object, float]
    concentration: float | Range = Range(0.2, 5.0)

    def draw(self, rng: np.random.Generator) -> dict[object, float]:
        given = isinstance(self.levels, Mapping)
        k = len(self.levels)
        center = np.asarray(tuple(self.levels.values())) if given else np.full(k, 1.0)
        shares = rng.dirichlet(drawn(self.concentration, rng) * center)
        shares = np.maximum(shares, np.minimum(center / center.sum(), min(0.04, 0.25 / k)))
        return dict(zip(self.levels, (shares / shares.sum()).tolist(), strict=True))


@dataclass(frozen=True)
class Noise:
    """Noise of a kind at a standard deviation (`gumbel`: at a scale), drawn per seed."""

    kind: str
    std: float | Range
    factor: float = 1.0

    @property
    def family(self) -> bool:
        return self.kind in ("student_t", "skewed") or ranged(self.std)

    def draw(self, rng: np.random.Generator) -> Distribution:
        std = drawn(self.std, rng) * self.factor
        if self.kind == "gaussian":
            return Normal(std=std)
        if self.kind == "gumbel":
            return Gumbel(scale=std)
        if self.kind == "laplace":
            return Laplace(scale=std / np.sqrt(2.0))
        if self.kind == "student_t":
            df = float(rng.uniform(2.5, 12.0))
            return StudentT(df, std * np.sqrt((df - 2.0) / df))
        if self.kind == "skewed":
            k, sign = float(rng.uniform(0.7, 6.0)), float(rng.choice([-1.0, 1.0]))
            return Affine(
                Gamma(k, 1.0), loc=-sign * std * np.sqrt(k), scale=sign * std / np.sqrt(k)
            )
        raise ValueError(f"unknown noise {self.kind!r}")


def _tail(
    cause: str | Recorded | Summary, keys: Collection[str]
) -> str | Recorded | Foreign | Summary:
    """The tail a cause reads: a local name, or `key.column` of the row the key points at."""
    if not isinstance(cause, str | Recorded):
        return cause
    head, dot, rest = (cause.column if isinstance(cause, Recorded) else cause).partition(".")
    if not (dot and head in keys):
        return cause
    return Foreign(head, replace(cause, column=rest) if isinstance(cause, Recorded) else rest)


def _finite(value: object) -> float:
    if not np.isfinite(value := float(value)):
        raise ValueError(f"weights and effects must be finite, not {value}")
    return value


def _levels(tail: str | Foreign | Summary, effects: Mapping[object, float]) -> Edge:
    if isinstance(tail, Summary | Recorded) or isinstance(getattr(tail, "node", ""), Recorded):
        raise ValueError(
            f"a summary is a number, as is a recorded value: give {tail!r} a weight, not an"
            " effect per level"
        )
    code = replace(tail, node=tail.node + CODE) if isinstance(tail, Foreign) else tail + CODE
    return LevelEdge(code, tuple((level, _finite(e)) for level, e in effects.items()))


def _checked(transform: Function) -> Function:
    if isinstance(transform, str) and transform not in TRANSFORMS:
        raise ValueError(f"transform must be a function or one of {tuple(TRANSFORMS)}")
    return transform


def _scaled_levels(effect: Effect) -> Effect:
    """A weight and the effects per level it scales, as those effects scaled."""
    if isinstance(effect, tuple) and isinstance(effect[1], Mapping):
        return {level: effect[0] * value for level, value in effect[1].items()}
    return effect


def _cause(cause: str | Recorded | Summary, effect: Effect, keys: Collection[str]) -> Edge:
    """The edge of one cause: a weight, a weight and a transform, one effect per level, or a weight
    and the effects it scales."""
    tail = _tail(cause, keys)
    effect = _scaled_levels(effect)
    if isinstance(effect, Mapping):
        return _levels(tail, effect)
    weight, transform = effect if isinstance(effect, tuple) else (effect, "identity")
    return LinearEdge(tail, _finite(weight), _checked(transform))


def _pair(
    name: str, pair: tuple[str, str], effect: Effect, how: str, keys: Collection[str]
) -> tuple[str, Node, Edge]:
    """The hidden product of two local causes and the edge reading it: a weight, optionally with a
    transform of both, or an effect per level of the first, which sets the slope of the second."""
    first, second = pair
    if any(not isinstance(cause, str) or _tail(cause, keys) != cause for cause in pair):
        raise ValueError(
            f"{pair} crosses a key or reads a record: a product reads latents of this table"
        )
    term = f"{name}.{'*'.join(pair)}"
    effect = _scaled_levels(effect)
    if isinstance(effect, Mapping):
        edges, weight = (_levels(first, effect), LinearEdge(second)), 1.0
    else:
        default = "step" if how == "rules" else "identity"
        weight, transform = effect if isinstance(effect, tuple) else (effect, default)
        edges = tuple(LinearEdge(cause, 1.0, _checked(transform)) for cause in pair)
    return term, Node(edges, op="product", noise=None), LinearEdge(term, _finite(weight))


def _noise(noise: float | Distribution | None) -> Distribution | None:
    if noise is None or isinstance(noise, Distribution):
        return noise
    if noise < 0.0:
        raise ValueError("noise is a standard deviation")
    return Normal(std=noise) if noise else None


def _node(
    name: str,
    causes: Causes,
    noise: float | Distribution | None,
    keys: Collection[str],
    how: str = "sum",
) -> tuple[Node, dict[str, Node]]:
    """The node summing a column's causes under `how`, plus noise, and the products it reads."""
    if how not in HOW:
        raise ValueError(f"how must be one of {tuple(HOW)}")
    if not causes:
        if noise is not None:
            raise ValueError("noise is relative to a column's causes, and this one has none")
        return Node(), {}
    if HOW[how] != "sum" and any(
        not isinstance(cause, tuple)
        and _tail(cause, keys) != cause
        or not isinstance(cause, tuple | str)
        for cause in causes
    ):
        raise ValueError(
            f"how={how!r} reads the latents of this table's own columns: a cause across a key or"
            " a recorded value adds"
        )
    edges, helpers = [], {}
    for cause, effect in causes.items():
        if isinstance(cause, tuple):
            term, helpers[term], edge = _pair(name, cause, effect, how, keys)
        else:
            edge = _cause(cause, effect, keys)
        edges.append(edge)
    noise = _noise(NOISE if noise is None else noise)
    return Node(tuple(edges), op=HOW[how], noise=noise), helpers


class Drawable:
    @property
    def family(self) -> bool:
        """Whether a seed must draw this before its table can be sampled."""
        return any(ranged(getattr(self, f.name)) for f in fields(self))

    def draw(self, rng: np.random.Generator) -> "Drawable":
        return replace(self, **{f.name: drawn(getattr(self, f.name), rng) for f in fields(self)})


class Declaration(Drawable):
    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        """The table entries this declaration compiles to; `keys` are the table's foreign keys."""
        raise NotImplementedError


def key_node(causes: Causes, noise: object, keys: Collection[str]) -> Node | None:
    """The node a key reads from its parent's columns, causes plus noise; noise alone without
    causes, and None without either."""
    if not causes:
        noise = _noise(noise)
        return None if noise is None else Node(noise=noise)
    node, helpers = _node("key", causes, noise, keys)
    if helpers:
        raise ValueError("a key reads its parent's columns one at a time: no products")
    return node


@dataclass(frozen=True)
class ByLevel:
    """Each level of the categorical `parent` puts a column on its own `(low, high)` range. The
    parent moves the column by `strength` at the ranges' standardized midpoints, logged when
    positive."""

    parent: str
    ranges: Mapping[object, tuple[float, float]]
    strength: float | Range = STRONG

    def __post_init__(self) -> None:
        if not all(np.isfinite(low) and np.isfinite(high) and low < high for low, high in self.ranges.values()):  # fmt: skip
            raise ValueError("each range is a finite low and a higher high")

    @property
    def effects(self) -> dict[object, float]:
        mids = np.array([(low + high) / 2.0 for low, high in self.ranges.values()])
        mids = np.log10(mids) if (mids > 0).all() else mids
        spread = float(mids.std()) or 1.0
        return dict(zip(self.ranges, ((mids - mids.mean()) / spread).tolist(), strict=True))

    @property
    def family(self) -> bool:
        return ranged(self.strength)

    def draw(self, rng: np.random.Generator) -> "ByLevel":
        return replace(self, strength=drawn(self.strength, rng))


@dataclass(frozen=True)
class Numeric(Declaration):
    """A number following `marginal`: the causes' weighted sum plus `noise`, its latent, orders the
    rows, and they take the marginal's values in that order. A cause is read as its latent, or as
    a `Recorded` cause as what its column records, in its units. Without a marginal the column is
    that sum, so `{Recorded("x"): 2.0}` with `noise=5.0` records `2 * x` plus noise in x's units. A
    share `zeros` of the rows record the marginal's floor, or 0, and `clip` (either side may be
    None) and `round` (a step) apply last. `how` combines the causes: their sum, largest, smallest,
    compounding product, or `rules`, a sum whose pairs are thresholds. A pair in `causes` is the
    product of two columns. `missing_by` puts the blanks on the `high` or `low` rows of a column."""

    marginal: Distribution | Scaled | ByLevel | None = None
    causes: Causes = field(default_factory=dict)
    noise: float | Range | Noise | Distribution | None = None
    missing: float = 0.0
    clip: tuple[float | None, float | None] | None = None
    round: float | None = None
    zeros: float = 0.0
    missing_by: tuple[str, str] | None = None
    how: str = "sum"
    doc: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.marginal, ByLevel):
            if self.clip is not None or self.round is not None or self.zeros:
                raise ValueError("ByLevel renders last: no clip, round or zeros with it")
            if self.marginal.parent in self.causes:
                raise ValueError(f"ByLevel sets the effect of {self.marginal.parent!r} itself")
        if isinstance(self.marginal, Scaled | Bounded):
            low, high = self.marginal.low, self.marginal.high
            if self.round is not None and self.round >= high - low:
                raise ValueError(f"round {self.round} is no step inside the range ({low}, {high})")
            if any(c is not None and not low <= c <= high for c in self.clip or ()):
                raise ValueError(f"clip {self.clip} must lie inside the range ({low}, {high})")

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        causes, marginal, by_level = self.causes, self.marginal, None
        if isinstance(marginal, ByLevel):
            causes = {**causes, marginal.parent: (marginal.strength, marginal.effects)}
            by_level = (marginal.parent, tuple((k, *span) for k, span in marginal.ranges.items()))
            marginal = None
        node, helpers = _node(name, causes, self.noise, keys, self.how)
        bounded, clip = isinstance(marginal, Bounded), self.clip
        if bounded:
            limits = self.clip or (None, None)
            clip = (
                marginal.low if limits[0] is None else limits[0],
                marginal.high if limits[1] is None else limits[1],
            )
        column = Column(
            node,
            marginal=marginal,
            missing=self.missing,
            clip=clip,
            round=self.round,
            zeros=(self.zeros, marginal.low if bounded else 0.0) if self.zeros else None,
            missing_by=self.missing_by,
            by_level=by_level,
        )
        return {**helpers, name: column}


@dataclass(frozen=True)
class Categorical(Declaration):
    """A level from `levels`: a sequence of equally common levels, a mapping of level to share, or
    `Shares`. Rows are ranked by the causes' sum plus `noise` and cut at the shares. Children read
    it through a mapping of level to effect or, when `ordered`, also through a weight, which
    follows the rank. With `scores` ({level: {column: weight}}) a row instead takes the level
    scoring highest, under Gumbel noise; realizing fits the scores to the shares."""

    levels: Sequence[object] | Mapping[object, float] | Shares
    causes: Causes = field(default_factory=dict)
    noise: float | Range | Noise | Distribution | None = None
    missing: float = 0.0
    ordered: bool = False
    scores: Mapping[object, Mapping[str, float | Range]] | None = None
    missing_by: tuple[str, str] | None = None
    how: str = "sum"
    doc: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.levels, str):
            raise ValueError("levels are a sequence or a mapping, not one string")
        if self.scores is not None:
            if self.causes or self.how != "sum" or self.ordered:
                raise ValueError("scores take the place of causes, how and an order")
            if not isinstance(self.levels, Mapping | Shares):
                raise ValueError("scores need the shares they are fit to")
            if not isinstance(self.noise, type(None) | Gumbel | Noise | Range | float | int):
                raise ValueError("scores take Gumbel noise")
            if isinstance(self.noise, Noise) and self.noise.kind != "gumbel":
                raise ValueError("scores take Gumbel noise")
            if isinstance(self.noise, float | int) and self.noise <= 0.0:
                raise ValueError("scores need noise above zero")

    @property
    def family(self) -> bool:
        return self.scores is not None or super().family

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        categories = tuple(self.levels)
        codes = tuple(float(code) for code in range(len(categories)))
        if self.scores is not None:
            node = self.scored(categories)
            read = MatrixEdge(name, np.asarray(codes)[:, None])
            column = Column(
                node,
                categories=categories,
                missing=self.missing,
                missing_by=self.missing_by,
                fit_shares=tuple(self.levels.values()),
                ordered=False,
            )
            helpers = {}
        else:
            shares = isinstance(self.levels, Mapping)
            node, helpers = _node(name, self.causes, self.noise, keys, self.how)
            column = Column(
                node,
                categories=categories,
                probabilities=tuple(self.levels.values()) if shares else None,
                missing=self.missing,
                missing_by=self.missing_by,
                ordered=self.ordered,
            )
            read = LookupEdge(name, codes, column.category_probabilities)
        return {**helpers, name: column, name + CODE: Node(read, noise=None)}

    def scored(self, categories: tuple[object, ...]) -> Node:
        if unknown := set(self.scores) - set(categories):
            raise ValueError(f"scores name unknown levels {sorted(map(str, unknown))}")
        weights: dict[str, np.ndarray] = {}
        for k, level in enumerate(categories):
            for feature, weight in self.scores.get(level, {}).items():
                weights.setdefault(feature, np.zeros((1, len(categories))))[0, k] = _finite(weight)
        edges = tuple(MatrixEdge(feature, matrix) for feature, matrix in weights.items())
        bias = (0.0,) * len(categories)
        noise = self.noise if isinstance(self.noise, Gumbel) else Gumbel(scale=self.noise or 1.0)
        return Node(edges, bias=bias, onehot=True, noise=noise)


@dataclass(frozen=True)
class Timestamp(Declaration):
    """A time in `[start, end)` on a calendar of business hours; causes move a row later. Without
    `start` and `end` the time is its causes' sum plus `noise` in epoch seconds, so it must read a
    time `Recorded` and say its noise: weight 1 and a positive noise, such as
    `Exponential(7 * 86400)`, place each row a week or so after that time, never before."""

    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    causes: Causes = field(default_factory=dict)
    noise: float | Range | Noise | Distribution | None = None
    missing: float = 0.0
    doc: str | None = None

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError("a timestamp takes both a start and an end, or neither")
        if self.start is None and (
            not any(isinstance(cause, Recorded) for cause in self.causes) or self.noise is None
        ):
            raise ValueError(
                "a timestamp without a start and an end is its causes plus its noise in seconds:"
                " read a time Recorded and give the noise, a delay such as Exponential(86400), or 0"
            )

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        calendar = None
        if self.start is not None:
            calendar = Calendar(pd.Timestamp(self.start), pd.Timestamp(self.end))
        node, helpers = _node(name, self.causes, self.noise, keys)
        return {**helpers, name: Column(node, "timestamp", marginal=calendar, missing=self.missing)}


@dataclass(frozen=True)
class Hidden(Declaration):
    """A cause no column records; other columns read it by its name. A root has unit scale."""

    causes: Causes = field(default_factory=dict)
    noise: float | Range | Noise | Distribution | None = None
    doc: str | None = None

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        node, helpers = _node(name, self.causes, self.noise, keys)
        return {**helpers, name: node}


@dataclass(frozen=True, eq=False)
class Derived:
    """A column computed from the recorded values of numeric or derived columns of its table, each
    named by a parameter of the function, or bound to one by `columns=`, which maps a parameter
    to a column such as `{"price": "Unit Price"}`; a blank input leaves a blank."""

    function: Callable[..., np.ndarray]
    doc: str | None = None
    columns: Mapping[str, str] = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        parameters = tuple(inspect.signature(self.function).parameters)
        if not parameters:
            raise ValueError("a derived column needs at least one input")
        if unknown := sorted(set(self.columns) - set(parameters)):
            raise ValueError(f"columns binds {unknown}, which the function takes no parameter of")
        object.__setattr__(self, "columns", dict(self.columns))

    @property
    def bindings(self) -> tuple[tuple[str, str], ...]:
        """Each parameter of the function and the column it reads."""
        parameters = inspect.signature(self.function).parameters
        return tuple((name, self.columns.get(name, name)) for name in parameters)

    @property
    def inputs(self) -> tuple[str, ...]:
        """The columns the function reads, in the order of its parameters."""
        return tuple(column for _, column in self.bindings)

    def compute(self, frame: Mapping[str, pd.Series]) -> np.ndarray:
        args = [frame[name].to_numpy(dtype=float) for name in self.inputs]
        with np.errstate(all="ignore"):
            computed = np.round(np.asarray(self.function(*args), dtype=float), 6)
        blank = np.isnan(args).any(0)
        if not np.isfinite(computed[~blank]).all():
            raise ValueError(f"{self.inputs} gave a non-finite value, such as a division by zero")
        return np.where(blank, np.nan, computed)


def _real_column(values: pd.Series) -> Column:
    if pd.api.types.is_datetime64_any_dtype(values):
        kind, categories = "timestamp", None
    elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        kind, categories = "numeric", None
    elif isinstance(values.dtype, pd.CategoricalDtype):
        kind, categories = "categorical", tuple(values.cat.categories)
    else:
        kind, categories = "categorical", tuple(sorted(values.dropna().unique(), key=str))
    return Column(Node(noise=None), kind, categories=categories, values=values, ordered=False)


@dataclass(frozen=True, eq=False)
class Real(Declaration):
    """A column of given values, kept as they are; other columns read it like any other."""

    values: pd.Series | np.ndarray | Sequence[object]
    doc: str | None = None

    family = False

    def draw(self, rng: np.random.Generator) -> "Real":
        return self

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        column = _real_column(pd.Series(self.values))
        if column.kind != "categorical":
            return {name: column}
        return {name: column, name + CODE: Node(ReadEdge(name), noise=None)}


def _unique(values: pd.Series) -> pd.Series:
    values = pd.Series(values).reset_index(drop=True)
    if values.isna().any() or not values.is_unique:
        raise ValueError("primary key values must be unique and present")
    return values


@dataclass(frozen=True, eq=False)
class PrimaryKey:
    """The table's key: row positions, or the given `values`."""

    values: pd.Series | np.ndarray | Sequence[object] | None = None
    kind = "key"
    missing = 0.0

    def __post_init__(self) -> None:
        if self.values is not None:
            object.__setattr__(self, "values", _unique(pd.Series(self.values)))

    def observe(
        self, name: str, latents: dict[str, np.ndarray], rng: np.random.Generator, n: int
    ) -> pd.Series:
        return pd.Series(np.arange(n)) if self.values is None else self.values


@dataclass(frozen=True)
class Counts(Drawable):
    """How many child rows point at each parent row: `mean` on average, each parent's share scaled
    by the exponential of its causes, read from the parent's columns, plus `noise`, 0.5 by default
    when there are causes; `noise=0` leaves Poisson counts given the causes. A row count
    given for the child table is split among the parents by those shares; without one each parent
    draws its own Poisson count, and the counts are the child table's size. `most` caps each
    parent's count, as a key unique per parent needs when the parents it points at are few, and
    `least` floors it, as every order having a line does: drawn counts are clipped to them, so
    `mean` is the Poisson mean before clipping, and a given total first gives every parent
    `least`."""

    mean: float | Range
    causes: Causes = field(default_factory=dict)
    noise: float | Range | Noise | Distribution | None = None
    most: int | None = None
    least: int = 0

    def __post_init__(self) -> None:
        if not ranged(self.mean) and not (np.isfinite(self.mean) and self.mean > 0.0):
            raise ValueError("mean is a positive count of children per parent")
        if isinstance(self.least, bool) or not isinstance(self.least, int) or self.least < 0:
            raise ValueError("least is a whole number of children per parent, at least 0")
        if self.most is not None and isinstance(self.most, int) and self.least > self.most:
            raise ValueError(f"least {self.least} is above most {self.most}")
        if self.most is not None and (
            isinstance(self.most, bool) or not isinstance(self.most, int) or self.most < 1
        ):
            raise ValueError("most is a whole number of children per parent, at least 1")


@dataclass(frozen=True, eq=False)
class ForeignKey(Drawable):
    """A key pointing at rows of `parent`, drawn by `link` or given as the parent's key `values`;
    a share `nullable` of the rows, drawn at random, point at nothing, and latents read across it
    take `fill` there. A `Recorded` cause across it takes the `Recorded` fill instead.

    A key may read its parent's columns instead of a structural link. `match` maps a column of
    this table to one of the parent that must hold the same level, or point at the same row when
    both are keys. A row no parent is left for, unmatched or past what a unique key allows,
    raises, or with `unmatched="null"` points at nothing; `nullable` drops matched rows at random
    on top of that. `prefer` weighs the eligible parents by the exponential of causes read from the
    parent's columns. `counts` sets how many rows point at each parent, and so the table's size.

    `unique=True` points no two rows at the same parent; `unique="student_id"`, or a tuple of
    columns, points no two rows that agree on those keys or categoricals at the same parent, so
    the pair is unique, as in a table of enrollments. A row blank in one of them is free, as SQL
    treats nulls; a group with more rows than parents to give them raises. Which options combine
    is in the README's "Foreign key options at a glance"."""

    parent: str
    link: Link = RandomLink()
    nullable: float = 0.0
    fill: float | None = None
    values: pd.Series | np.ndarray | Sequence[object] | None = None
    match: Mapping[str, str] = field(default_factory=dict)
    prefer: Causes = field(default_factory=dict)
    counts: Counts | None = None
    unique: bool | str | tuple[str, ...] = False
    unmatched: Literal["raise", "null"] = "raise"

    @property
    def together(self) -> tuple[str, ...]:
        """The columns a unique key is unique together with."""
        if isinstance(self.unique, bool):
            return ()
        return (self.unique,) if isinstance(self.unique, str) else tuple(self.unique)

    def __post_init__(self) -> None:
        if not 0.0 <= self.nullable < 1.0:
            raise ValueError("nullable must be in [0, 1)")
        reads = bool(self.match or self.prefer or self.counts)
        if reads and (self.link != RandomLink() or self.values is not None):
            raise ValueError("match, prefer and counts choose the parents: no link or values")
        if self.unique and self.link != RandomLink():
            raise ValueError("a unique key draws its parents without repeats: no link")
        if self.counts is not None and (self.match or self.prefer or self.nullable or self.unique):
            raise ValueError(
                "counts give every child its parent: no match, prefer, nullable or unique"
            )
        if self.unmatched not in ("raise", "null"):
            raise ValueError('unmatched is "raise" or "null"')
        if self.unmatched == "null" and not (self.match or self.unique) or self.unmatched == "null" and self.values is not None:  # fmt: skip
            raise ValueError('unmatched="null" is for keys that match or are unique and drawn')
        if self.fill is not None and not np.isfinite(self.fill):
            raise ValueError("fill must be finite")
        if self.values is not None:
            if self.nullable:
                raise ValueError("given key values hold their own blanks: no nullable")
            object.__setattr__(self, "values", pd.Series(self.values).reset_index(drop=True))
