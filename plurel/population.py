"""A population: the rows a table is sampled from, as the :class:`TableMetadata` of the table realized
from a plurel plan under a seed and how its rows are drawn. Every draw is a pure function of the
population's seed lanes and the draw's index, so a seed gives the same rows anywhere."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from plurel.columns import rank_map

ENTITY_ROWS_MAX = 256  # the most rows one entity of a grouped population holds
ENTITY_CODES = 26 * 100_000  # an entity's id is a letter and five digits


@dataclass
class TableMetadata:
    """The schema and semantics of a table. ``feature_names``/``feature_types`` are in feature order
    (types are ``"numeric"``/``"categorical"``); ``target`` is ``None`` until a label column is
    chosen; ``plan`` is a synthetic table's plurel SCM (``None`` on real ones)."""

    feature_names: list[str]
    feature_types: list[str]
    target: str | None = None
    domain: str | None = None
    plan: object | None = None
    # optional human-written schema text; column_descriptions is keyed by column name
    description: str | None = None
    column_descriptions: dict[str, str] | None = None
    # {rendered target value -> name} for a label column of bare codes; None when values are names
    target_value_names: dict[str, str] | None = None
    # {column -> values, low to high}: the generator's latent order
    category_order: dict[str, list[str]] | None = None
    # {"segment": codes} per row of a mixture table; must hold for the frame as shipped
    row_latents: dict[str, np.ndarray] | None = None
    # producer-owned flags of how a synthetic table was realized; tolerate None and missing keys
    realization: dict | None = None

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.feature_types):
            raise ValueError(
                f"{len(self.feature_names)} feature_names but {len(self.feature_types)} feature_types"
            )
        bad = {t for t in self.feature_types if t not in ("numeric", "categorical")}
        if bad:
            raise ValueError(f"feature_types must be 'numeric'/'categorical'; got {sorted(bad)}")


def levels(plan) -> dict[str, list[str]]:
    """String levels per categorical column; integer-coded columns are absent."""
    return {
        name: list(column.categories)
        for name, column in plan.columns.items()
        if column.kind == "categorical"
        and all(isinstance(level, str) for level in column.categories)
    }


def category_order(plan) -> dict[str, list[str]]:
    """``{categorical column -> levels, low to high}`` in latent order: authored names where
    present (index-aligned with the quantile bins), else the integer bin codes."""
    named = levels(plan)
    return {
        name: named.get(name) or [str(i) for i in range(len(column.categories))]
        for name, column in plan.columns.items()
        if column.kind == "categorical"
    }


def table_id(source: str, seed: int) -> str:
    """A realized table's id: its schema id (``random`` for a prior table) and its seed."""
    return f"{source}--s{seed}"


@dataclass(frozen=True)
class Grouping:
    """How a grouped population's rows share entities: each draw cuts its rows into entities whose sizes are
    log-normal around ``rows`` with log sd ``spread`` (the last cut short, so a draw holds the rows it is asked
    for), draws the plan once per entity, and holds the ``fixed`` nodes (a parent-closed set) at their entity's
    values in each of its rows, every other node drawn per row. A fixed numeric column records its entity's value,
    under per-row noise of sd ``measurement`` times the entities' spread when that is above 0 (replicate
    readings); ``mode`` names the kind (``replicates``: the label and all its causes are fixed; ``shared_causes``:
    some roots are), and ``column`` the ID column that names each row's entity, ``None`` for none."""

    mode: str
    fixed: tuple[str, ...]
    rows: float
    spread: float
    measurement: float
    column: str | None

    def sizes(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """The rows per entity of an ``n``-row draw, in order, summing to ``n``."""
        if n == 0:
            return np.zeros(0, dtype=int)
        drawn = np.exp(rng.normal(np.log(self.rows), self.spread, n))
        drawn = np.clip(np.round(drawn), 1, ENTITY_ROWS_MAX).astype(int)
        ends = np.cumsum(drawn)
        last = int(np.searchsorted(ends, n))  # the entity whose rows reach the n-th
        sizes = drawn[: last + 1]
        sizes[-1] -= ends[last] - n
        return sizes

    def draw(self, plan, n: int, rng: np.random.Generator) -> tuple:
        """``(frame, entities, names)``: ``n`` rows of ``plan``, each row's entity (contiguous, from 0) and each
        entity's id."""
        sizes_rng, entity_rng, row_rng, noise_rng, id_rng = rng.spawn(5)
        sizes = self.sizes(n, sizes_rng)
        entities = np.repeat(np.arange(len(sizes)), sizes)
        each, latents = plan.sample_with_latents(len(sizes), seed=entity_rng)
        held = {node: latents[node][entities] for node in self.fixed}
        frame = plan.sample_with_latents(n, seed=row_rng, held=held)[0]
        fixed = set(self.fixed)
        for name in plan.order:
            node = plan.nodes[name]
            if node.noise is None and node.parents and fixed.issuperset(node.parents):
                fixed.add(name)
        for name, column in plan.columns.items():
            if name not in fixed:
                continue
            if column.kind == "numeric" and self.measurement > 0:
                # a reading of the entity's value: noise on its latent, observed on the column's marginal as any
                # draw is; the fixed nodes downstream read the value itself
                latent = latents[name][:, 0]
                noise = self.measurement * latent.std() * noise_rng.standard_normal(n)
                noisy = latent[entities] + noise
                frame[name] = rank_map(noisy, column.marginal, noise_rng) if column.marginal else noisy  # fmt: skip
            else:  # the entity's value as its own draw observed it (an in-draw rank map would bias ties)
                frame[name] = each[name].iloc[entities].set_axis(frame.index)
        codes = id_rng.choice(ENTITY_CODES, size=len(sizes), replace=False)
        names = np.array([f"{chr(ord('A') + int(c) // 100_000)}{int(c) % 100_000:05d}" for c in codes], dtype=object)  # fmt: skip
        return frame, entities, names


def entity_cut(entities: np.ndarray, n: int) -> int:
    """The first row at or after row ``n`` that starts an entity (the row count if none does): rows before it share
    no entity with rows from it on. ``n`` itself for one entity per row."""
    entities = np.asarray(entities)
    if not len(entities):
        return 0
    starts = np.flatnonzero(np.r_[True, entities[1:] != entities[:-1]])
    later = starts[starts >= n]
    return int(later[0]) if len(later) else len(entities)


@dataclass(frozen=True)
class Population:
    """``rows`` gives draw after draw of the population's rows, each a pure function of ``lanes``
    (the generator's seed lane) and the draw. ``render`` maps a drawn frame and its draw index onto
    the table's own scales, or is ``None`` when the plan's columns are the table's; ``grouping`` makes
    rows share entities (:class:`Grouping`), ``None`` draws them independent."""

    metadata: TableMetadata
    lanes: tuple[int, ...]
    render: Callable | None
    grouping: Grouping | None = None

    @property
    def id(self) -> str:
        return self.metadata.realization["table_id"]

    @property
    def target(self) -> str:
        return self.metadata.target

    @property
    def plan(self):
        """The plurel SCM the table is realized from."""
        return self.metadata.plan

    @property
    def features(self) -> list[str]:
        return self.metadata.feature_names

    def rows(self, n: int, draw: int):
        """Draw ``draw`` of the rows, ``n`` of them, the features then the target; every draw is
        its own sample."""
        return self.entity_rows(n, draw)[0]

    def entity_rows(self, n: int, draw: int):
        """``(rows, entities)``: :meth:`rows` and each row's entity, its own row index unless the
        population is grouped, when an entity's rows are contiguous."""
        plan = self.plan
        rng = np.random.default_rng([*self.lanes, draw])
        if self.grouping is None:
            frame = plan.sample(n, seed=rng)
            return (self.render(frame, draw) if self.render else frame[[*self.features, self.target]]), np.arange(n)  # fmt: skip
        frame, entities, names = self.grouping.draw(plan, n, rng)
        column = self.grouping.column
        drawn = [name for name in self.features if name != column]
        frame = self.render(frame, draw) if self.render else frame[[*drawn, self.target]]
        if column is not None:
            frame.insert(0, column, names[entities])
        return frame, entities
