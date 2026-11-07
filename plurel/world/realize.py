"""A catalog's record realized under a seed as a :class:`Population`: :func:`realize_world` draws its
table, its grouped or joint transforms and the card a reader gets with its rows."""

import zlib
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from plurel import Categorical, Noise, Schema, Shares, Table
from plurel.columns import Column
from plurel.graph import LinearEdge, MatrixEdge, ReadEdge, bin_levels
from plurel.graph import Node as PlanNode
from plurel.population import Grouping, Population, TableMetadata, category_order, levels, table_id
from plurel.schema import CALIBRATION_ROWS
from plurel.world.record import (
    children,
    domain,
    features,
    is_numeric,
    level_names,
    parents,
    validate,
)

ENTITY_EFFECT = "entity.effect"  # the unrecorded cause per entity; a dotted name no column takes
ENTITY_WEIGHT = (0.4, 1.4)
JOINT_SEPARATOR = " / "  # between a joint label's parts, the target's first
JOINT_CLASSES = (4, 10)  # a joint target's class count
JOINT_FLOOR = 0.01  # the least share of a calibration draw every joint class must reach
# the share of a joint part's value the other columns explain (linearly): under it the part is mostly noise no
# column shows, over it a near-function of the columns shown, either way little of a second outcome to learn
JOINT_R2 = (0.3, 0.95)
# a numeric part's levels, split at its median: a third band met the floor on half as many schemas, and the
# schemas it met it on a median split meets it on too
BANDS = ("low", "high")

# The sentence a transformed table's description gains, so its text says what its rows and label are: {unit}
# names the entity, {column} its ID column, {target} the label, {name} a joint label, {part} its second part.
GROUPED_TEXT = {
    ("replicates", True): (
        "Each {unit} was recorded several times: the rows with the same {column} are repeated readings of one {unit}, and its {target} is the same in each.",
        "Rows repeat by {unit}: a {unit} ({column}) appears in several rows, each a fresh measurement of it, so its readings differ a little and its {target} not at all. The rows to predict come from {unit}s the context rows do not show.",
    ),
    ("replicates", False): (
        "Each {unit} was recorded several times, so the rows come in groups of repeated readings of one {unit} that share its {target}, though the table does not say which rows belong together.",
        "Several rows come from each {unit}: repeated measurements of it, with no column naming the {unit}. The rows to predict come from {unit}s the context rows do not show.",
    ),
    ("shared_causes", True): (
        "Rows come in groups: the rows with the same {column} were recorded on one {unit} and share some of its circumstances, while {target} is recorded row by row.",
        "A {unit} ({column}) appears in several rows, each recorded on its own occasion: what belongs to the {unit} carries over between them, what belongs to the occasion does not. The rows to predict come from {unit}s the context rows do not show.",
    ),
    ("shared_causes", False): (
        "Several rows come from each {unit} and share some of its circumstances, though the table does not say which rows belong together; {target} is recorded row by row.",
        "The rows were recorded on fewer {unit}s than there are rows, each {unit} appearing several times with no column naming it. The rows to predict come from {unit}s the context rows do not show.",
    ),
}  # fmt: skip
JOINT_TEXT = (
    "The outcome to predict, {name}, joins two: the {target} and the {part}, written with the {target} first and a slash between them.",
    "Each label of {name} pairs the row's {target} with its {part}, as '<{target}> / <{part}>'.",
)  # fmt: skip

# a shared-causes table's chance that a root is the entity's: an unrecorded one, a recorded one
HIDDEN_ROOT_SHARE, ROOT_SHARE = 0.7, 0.3

# Every realization coin comes from a lane of [WORLD_DOMAIN, crc32(name), seed, lane, ...], so the same
# (record, seed) gives the same table and the same row draws on any machine.
WORLD_DOMAIN = 0x57524C44
_LANE_WORLD, _LANE_TRANSFORM, _LANE_SAMPLE = 1, 2, 3
_EFFECT, _JOINT, _ENTITY, _TEXT = 0, 1, 2, 3  # the transform lane's own lanes

# a grouped table's draw, as ``realize_world`` takes it in ``grouping`` and its realization stamps it
GROUPING_KEYS = ("mode", "rows", "spread", "measurement", "column", "effect")
ENTITY_DOC = "the unit the row was recorded on; rows that share it come from one unit"


def _lane(table: Table, seed: int, which: int, *rest: int) -> np.random.Generator:
    return np.random.default_rng(
        [WORLD_DOMAIN, zlib.crc32(table.name.encode()), int(seed), which, *rest]
    )


def _louder(noise: object, factor: float) -> Noise:
    """The noise a target takes at ``factor``, whatever form it was authored in."""
    if isinstance(noise, Noise):
        return replace(noise, factor=factor)
    if noise is None:
        return Noise("gumbel", 1.0, factor)
    if isinstance(noise, float | int) or hasattr(noise, "draw"):
        return Noise("gaussian", noise, factor)
    raise ValueError(f"a target's noise of {noise!r} cannot be scaled")


def _varied(table: Table, target_noise: float, concentration: float | None) -> Table:
    """The record with its target's noise scaled and its authored shares redrawn per seed."""
    entries = dict(table.declared)
    for name, entry in entries.items():
        if concentration is not None and isinstance(entry, Categorical):
            if isinstance(entry.levels, Mapping):
                entry = replace(entry, levels=Shares(entry.levels, concentration))
        if name == table.target and target_noise != 1.0:
            entry = replace(entry, noise=_louder(entry.noise, target_noise))
        entries[name] = entry
    return Table(entries, **table.metadata)


def _with_entity_effect(plan: Table, rng: np.random.Generator) -> Table:
    """The plan with :data:`ENTITY_EFFECT`, a standard normal root, feeding the target (a scored one by a
    weight per level) and each other numeric column with probability one half, at the weak strength and a
    sign drawn per edge. Only nodes that sum their edges take it, so a max, min or compounding node keeps
    its authored form."""
    entries = {**plan.table, ENTITY_EFFECT: PlanNode()}

    def fed(column: Column) -> Column:
        weight = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(*ENTITY_WEIGHT))
        edges = (*column.node.edges, LinearEdge(ENTITY_EFFECT, weight))
        return replace(column, node=replace(column.node, edges=edges))

    for name, column in plan.columns.items():
        if name != plan.target and column.kind == "numeric":
            if rng.random() < 0.5 and column.node.op == "sum":
                entries[name] = fed(column)
    column = plan.columns[plan.target]
    node = column.node
    if node.onehot:
        weights = rng.normal(0.0, 1.0, (1, node.dim)) * float(rng.uniform(*ENTITY_WEIGHT))
        edges = (*node.edges, MatrixEdge(ENTITY_EFFECT, weights))
        entries[plan.target] = replace(column, node=replace(node, edges=edges))
    elif node.op == "sum":
        entries[plan.target] = fed(column)
    return plan.like(entries)


def part_levels(table: Table, leaf: str) -> list[str]:
    """The levels a joint target's part ``leaf`` takes: a categorical leaf's own, a numeric leaf's :data:`BANDS`."""
    entry = table.declared[leaf]
    return level_names(entry) if isinstance(entry, Categorical) else list(BANDS)


def joint_leaves(table: Table) -> list[str]:
    """The columns a joint target can join the record's target with: recorded leaves (a parent, no child, not
    derived) whose levels (:func:`part_levels`) times the target's are within :data:`JOINT_CLASSES`; a categorical
    one never blank, as its level is the label's, while a numeric one is cut into bands of its value, which no blank
    or recording artifact reaches. Both parts are effects of the features, so the joint label is too; an isolated
    root would add a part no column predicts."""
    target = table.declared[table.target]
    if not isinstance(target, Categorical):
        return []
    below, n = children(table), len(level_names(target))
    lo, hi = JOINT_CLASSES
    leaves = []
    for name in features(table):
        entry = table.declared[name]
        if parents(table, name) and not below[name] and hasattr(entry, "missing"):
            whole = not isinstance(entry, Categorical) or not entry.missing
            if whole and lo <= n * len(part_levels(table, name)) <= hi:
                leaves.append(name)
    return leaves


def _joint_part(table: Table, plan: Table, leaf: str, rng: np.random.Generator):
    """``(plan, column, r2)`` for the joint target's part ``leaf``, or ``None`` when it cannot be one. From one
    ``CALIBRATION_ROWS`` draw: a numeric leaf is cut into its bands at the draw's quantiles of its value, as a column
    of the plan reading the leaf's node, which moves no draw; the part is kept when
    the other columns explain a :data:`JOINT_R2` share of its value (``r2``: linearly, on the latents, the target
    aside) and every joint class holds :data:`JOINT_FLOOR` of the draw."""
    target = table.target
    frame, latents = plan.sample_with_latents(CALIBRATION_ROWS, seed=rng)
    value = latents[leaf][:, 0]
    column, named = leaf, part_levels(table, leaf)
    if is_numeric(table.declared[leaf]):
        column = f"{leaf}.band"  # a dotted name no column takes
        edges = tuple(float(e) for e in np.quantile(value, np.arange(1, len(named)) / len(named)))
        reading = PlanNode(ReadEdge(leaf), noise=None)
        band = Column(reading, "categorical", categories=tuple(named), binning=edges)
        plan = plan.like({**plan.table, column: band})
        parts = np.asarray(named)[bin_levels(value, edges)]
    else:
        parts = frame[leaf].astype(str).to_numpy()
    shown = sorted(set(plan.columns) - {target, leaf, column})
    x = np.column_stack([*(latents[n].reshape(len(value), -1) for n in shown), np.ones(len(value))])
    r2 = float(1.0 - np.var(value - x @ np.linalg.lstsq(x, value, rcond=None)[0]) / np.var(value))
    joined = pd.Series(list(zip(frame[target].astype(str), parts, strict=True))).value_counts()
    common = all(
        joined.get((t, p), 0) >= JOINT_FLOOR * CALIBRATION_ROWS
        for t in levels(plan)[target]
        for p in named
    )
    return (plan, column, r2) if common and JOINT_R2[0] <= r2 <= JOINT_R2[1] else None


def _part_text(table: Table, leaf: str) -> str:
    """How the transform text names a joint target's part: the leaf, or its band and how it is cut."""
    if not is_numeric(table.declared[leaf]):
        return leaf
    return f"{leaf} band ({' or '.join(BANDS)}, split at its median)"


def _entity_nodes(plan: Table, mode: str, labels: list[str], rng: np.random.Generator) -> tuple[str, ...]:  # fmt: skip
    """The nodes a grouped table holds per entity, in plan order: for ``replicates`` the ``labels`` and every
    ancestor of them; for ``shared_causes`` a draw of the roots (hidden ones more often, at least one root); with
    the entity effect either way."""
    if mode == "replicates":
        held, frontier = set(labels), list(labels)
        while frontier:
            for parent in plan.nodes[frontier.pop()].parents:
                if parent not in held:
                    held.add(parent)
                    frontier.append(parent)
    else:
        roots = [n for n, node in plan.nodes.items() if not node.parents and n != ENTITY_EFFECT]
        held = {name for name in roots if rng.random() < (ROOT_SHARE if name in plan.columns else HIDDEN_ROOT_SHARE)}  # fmt: skip
        if roots and not held:
            held = {roots[int(rng.integers(len(roots)))]}
    if ENTITY_EFFECT in plan.nodes:
        held.add(ENTITY_EFFECT)
    return tuple(name for name in plan.nodes if name in held)


def _column_docs(table: Table, plan: Table) -> dict[str, str]:
    """Per-column documentation: the authored doc plus each categorical column's levels as a plain list."""
    named, docs = levels(plan), {}
    for name in [*features(table), table.target]:
        parts = [table.declared[name].doc]
        if name in named:
            parts.append(f"categories: {', '.join(map(str, named[name]))}")
        docs[name] = "; ".join(parts)
    return docs


def realize_world(
    table: Table,
    seed: int,
    *,
    target_noise: float = 1.0,
    share_concentration: float | None = None,
    grouping: dict | None = None,
    joint: str | None = None,
) -> Population:
    """``table`` under ``seed`` as a :class:`Population`: every range drawn from the seed's lane, the target's noise
    scaled by ``target_noise`` and the authored shares redrawn under ``share_concentration``.

    Two transforms take the table past its record, their choices drawn from the seed's transform lane. ``grouping``
    (:data:`GROUPING_KEYS`: the mode, the median rows per entity and their log sd, the measurement noise, the ID
    column's name or ``None``, and whether an unrecorded cause per entity is added) makes rows share entities
    (:class:`Grouping`, the nodes held per entity drawn by :func:`_entity_nodes`). ``joint`` names a leaf of
    :func:`joint_leaves` whose level (a numeric leaf's band) the label joins to the target's
    (``"<target> / <part>"``, the target's levels slowest), the leaf no longer a column; a part :func:`_joint_part`
    refuses keeps the table's single target. A transformed table's description gains a sentence saying what its
    rows and label are, and a column it no longer has loses its note. The realization stamps the transforms as
    realized (``grouping``, ``joint``) and the sentences (``text``)."""
    validate(table)
    target = table.target
    root = int(_lane(table, seed, _LANE_WORLD).integers(2**63))
    plan = _varied(table, target_noise, share_concentration).drawn(root)
    if grouping is not None and grouping["effect"]:
        plan = _with_entity_effect(plan, _lane(table, seed, _LANE_TRANSFORM, _EFFECT))
    plan = Schema({"table": plan}).fitted(root).tables["table"]
    part = None
    if joint is not None:
        if joint not in joint_leaves(table):
            raise ValueError(f"{joint!r} is not one of the record's joint leaves")
        part = _joint_part(table, plan, joint, _lane(table, seed, _LANE_TRANSFORM, _JOINT))
        joint = joint if part else None
    name = f"{target}_and_{joint}" if joint else target
    shown = [c for c in features(table) if c != joint]
    types = ["numeric" if is_numeric(table.declared[c]) else "categorical" for c in shown]
    order, docs = category_order(plan), _column_docs(table, plan)
    realization = {
        "schema_id": table.name,
        "seed": int(seed),
        "table_id": table_id(table.name, seed),
        "target_noise": float(target_noise),
        "share_concentration": None if share_concentration is None else float(share_concentration),
    }
    texts, pick = [], _lane(table, seed, _LANE_TRANSFORM, _TEXT)
    render = None
    if part:
        plan, column, r2 = part
        named = part_levels(table, joint)
        rank = category_order(plan)
        order = {k: v for k, v in rank.items() if k not in (target, joint, column)}
        labels = [f"{t}{JOINT_SEPARATOR}{p}" for t in rank[target] for p in named]
        order[name] = labels
        for gone in (target, joint):  # columns the table no longer has
            docs.pop(gone, None)
        described = _part_text(table, joint)
        docs[name] = f"{table.declared[target].doc}, then the {described}: {table.declared[joint].doc}; categories: {', '.join(labels)}"  # fmt: skip
        realization["joint"] = {"part": joint, "separator": JOINT_SEPARATOR, "banded": column != joint,
                                "described": described, "r2": round(r2, 3),
                                "parts": [[target, rank[target]], [joint, named]]}  # fmt: skip
        texts.append(JOINT_TEXT[int(pick.integers(len(JOINT_TEXT)))].format(name=name, target=target, part=described))  # fmt: skip
        recorded = list(shown)

        def render(frame: pd.DataFrame, draw: int) -> pd.DataFrame:
            joined = frame[target].astype(str) + JOINT_SEPARATOR + frame[column].astype(str)
            return frame[recorded].assign(**{name: pd.Categorical(joined, categories=labels)})

    entity = None
    if grouping is not None:
        labelled = [target, joint] if joint else [target]
        fixed = _entity_nodes(plan, grouping["mode"], labelled, _lane(table, seed, _LANE_TRANSFORM, _ENTITY))  # fmt: skip
        entity = Grouping(mode=grouping["mode"], fixed=fixed, rows=grouping["rows"], spread=grouping["spread"],
                          measurement=grouping["measurement"], column=grouping["column"])  # fmt: skip
        if entity.column is not None:
            shown, types = [entity.column, *shown], ["categorical", *types]
            docs[entity.column] = ENTITY_DOC
        effect_on = sorted(n for n, node in plan.nodes.items() if ENTITY_EFFECT in node.parents)
        realization["grouping"] = {**{key: grouping[key] for key in GROUPING_KEYS}, "fixed": list(fixed),
                                   "effect_on": effect_on}  # fmt: skip
        unit = "unit" if entity.column in (None, "id") else entity.column.removesuffix("_id")
        phrasings = GROUPED_TEXT[entity.mode, entity.column is not None]
        texts.append(phrasings[int(pick.integers(len(phrasings)))].format(unit=unit, column=entity.column, target=name))  # fmt: skip
    description = (table.doc or "").strip() or None
    if texts:
        realization["text"] = " ".join(texts)
        description = f"{description} {realization['text']}" if description else realization["text"]
    metadata = TableMetadata(
        feature_names=shown,
        feature_types=types,
        target=name,
        domain=domain(table),
        plan=plan,
        description=description,
        column_descriptions=docs,
        category_order=order,
        realization=realization,
    )
    return Population(
        metadata=metadata,
        lanes=(WORLD_DOMAIN, zlib.crc32(table.name.encode()), int(seed), _LANE_SAMPLE),
        render=render,
        grouping=entity,
    )
