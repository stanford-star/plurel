"""What a table's entries say: its keys, its columns and the causes each column follows. Each
compiles to the nodes and columns of `plurel.graph` and `plurel.columns`."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.distributions import Calendar, Distribution, Normal
from plurel.graph import (
    TRANSFORMS,
    Edge,
    Foreign,
    Function,
    LevelEdge,
    LinearEdge,
    LookupEdge,
    Node,
    ReadEdge,
    Summary,
)
from plurel.links import Link, RandomLink

Effect = float | tuple[float, Function] | Mapping[object, float]
Causes = Mapping[str | Summary, Effect]

CODE = ".code"  # a categorical column's hidden node holding its level's index
NOISE = 0.5  # the default standard deviation of the noise on a column's causes


def _cause(cause: str | Summary, effect: Effect, keys: Collection[str]) -> Edge:
    """The edge of one cause: a weight, a weight and a transform, or one effect per level."""
    tail = cause
    if isinstance(cause, str):
        head, dot, rest = cause.partition(".")
        tail = Foreign(head, rest) if dot and head in keys else cause
    if not isinstance(effect, Mapping):
        weight, transform = effect if isinstance(effect, tuple) else (effect, "identity")
        if isinstance(transform, str) and transform not in TRANSFORMS:
            raise ValueError(f"transform must be a function or one of {tuple(TRANSFORMS)}")
        return LinearEdge(tail, float(weight), transform)
    if isinstance(tail, Summary):
        raise ValueError("a summary is a number: give it a weight, not an effect per level")
    code = replace(tail, node=tail.node + CODE) if isinstance(tail, Foreign) else tail + CODE
    return LevelEdge(code, tuple(effect.items()))


def _node(
    causes: Causes, noise: float | None, keys: Collection[str], standardize: bool = True
) -> Node:
    if not causes:
        if noise is not None:
            raise ValueError("noise is relative to a column's causes, and this one has none")
        return Node()
    noise = NOISE if noise is None else noise
    if noise < 0.0:
        raise ValueError("noise is a standard deviation")
    edges = tuple(_cause(cause, effect, keys) for cause, effect in causes.items())
    return Node(edges, noise=Normal(std=noise) if noise else None, standardize=standardize)


class Declaration:
    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        """The table entries this declaration compiles to; `keys` are the table's foreign keys."""
        raise NotImplementedError


@dataclass(frozen=True)
class Numeric(Declaration):
    """A number following `marginal`: the causes' weighted sum is standardized, `noise` of that
    standard deviation is added and the rows take the marginal's values in that order. Without a
    marginal the column is that sum on the causes' unit scales plus the noise. `round` is a step
    and, with `clip`, applies last."""

    marginal: Distribution | None = None
    causes: Causes = field(default_factory=dict)
    noise: float | None = None
    missing: float = 0.0
    clip: tuple[float, float] | None = None
    round: float | None = None

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        node = _node(self.causes, self.noise, keys, standardize=self.marginal is not None)
        options = {"clip": self.clip, "round": self.round}
        return {name: Column(node, marginal=self.marginal, missing=self.missing, **options)}


@dataclass(frozen=True)
class Categorical(Declaration):
    """A level from `levels`, a sequence of equally common levels or a mapping of level to share.
    Rows are ranked by the causes' standardized sum plus `noise` and cut at the shares, so a
    positive weight pushes rows toward the later levels. Children read it through a weight, which
    follows that rank, or through a mapping of level to effect."""

    levels: Sequence[object] | Mapping[object, float]
    causes: Causes = field(default_factory=dict)
    noise: float | None = None
    missing: float = 0.0

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        if isinstance(self.levels, str):
            raise ValueError("levels are a sequence or a mapping, not one string")
        shares = isinstance(self.levels, Mapping)
        probabilities = tuple(self.levels.values()) if shares else None
        categories = tuple(self.levels)
        column = Column(
            _node(self.causes, self.noise, keys),
            categories=categories,
            probabilities=probabilities,
            missing=self.missing,
        )
        codes = tuple(float(code) for code in range(len(categories)))
        code = Node(LookupEdge(name, codes, column.category_probabilities), noise=None)
        return {name: column, name + CODE: code}


@dataclass(frozen=True)
class Timestamp(Declaration):
    """A time in `[start, end)` on a calendar of business hours; causes move a row later."""

    start: str | pd.Timestamp
    end: str | pd.Timestamp
    causes: Causes = field(default_factory=dict)
    noise: float | None = None
    missing: float = 0.0

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        calendar = Calendar(pd.Timestamp(self.start), pd.Timestamp(self.end))
        node = _node(self.causes, self.noise, keys)
        return {name: Column(node, "timestamp", marginal=calendar, missing=self.missing)}


@dataclass(frozen=True)
class Hidden(Declaration):
    """A cause no column records, on a unit scale; other columns read it by its name."""

    causes: Causes = field(default_factory=dict)
    noise: float | None = None

    def entries(self, name: str, keys: Collection[str]) -> dict[str, Node | Column]:
        return {name: _node(self.causes, self.noise, keys)}


def _real_column(values: pd.Series) -> Column:
    if pd.api.types.is_datetime64_any_dtype(values):
        kind, categories = "timestamp", None
    elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        kind, categories = "numeric", None
    elif isinstance(values.dtype, pd.CategoricalDtype):
        kind, categories = "categorical", tuple(values.cat.categories)
    else:
        kind, categories = "categorical", tuple(sorted(values.dropna().unique(), key=str))
    return Column(Node(noise=None), kind, categories=categories, values=values)


@dataclass(frozen=True, eq=False)
class Real(Declaration):
    """A column of given values, kept as they are; other columns read it like any other."""

    values: pd.Series | np.ndarray | Sequence[object]

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


@dataclass(frozen=True, eq=False)
class ForeignKey:
    """A key pointing at rows of `parent`, drawn by `link` or given as the parent's key `values`;
    a share `nullable` of the rows point at nothing, and causes read across it take `fill` there."""

    parent: str
    link: Link = RandomLink()
    nullable: float = 0.0
    fill: float | None = None
    values: pd.Series | np.ndarray | Sequence[object] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.nullable < 1.0:
            raise ValueError("nullable must be in [0, 1)")
        if self.fill is not None and not np.isfinite(self.fill):
            raise ValueError("fill must be finite")
        if self.values is not None:
            if self.nullable:
                raise ValueError("given key values hold their own blanks: no nullable")
            object.__setattr__(self, "values", pd.Series(self.values).reset_index(drop=True))
