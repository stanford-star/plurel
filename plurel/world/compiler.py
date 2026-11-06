"""The compiler: a :class:`WorldSchema` realized under a seed as a :class:`Population`.
:func:`compile_schema` draws every magnitude inside the authored sign and strength class and
builds the plurel plan; :func:`_calibrate_argmax` fits an argmax target's class biases to its
authored shares; :func:`_fix_edges` pins every binned column's edges from one calibration draw;
:func:`realize_world` wraps the plan with the table's card and the renderer (:func:`_render`) that
puts each draw onto the authored scales, artifacts and blanks."""

from __future__ import annotations

import zlib
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd

from plurel import (
    SCM,
    Affine,
    Column,
    Gamma,
    Gumbel,
    Laplace,
    LinearEdge,
    LookupEdge,
    MatrixEdge,
    Normal,
    ReadEdge,
    StudentT,
)
from plurel import Node as PlanNode
from plurel.graph import REDUCTIONS
from plurel.population import Grouping, Population, TableMetadata, category_order, levels, table_id
from plurel.world.distributions import shape
from plurel.world.schema import (
    MAGNITUDES,
    NOISE_SCALES,
    TRANSFORM_OF_SHAPE,
    UNIT_FAMILIES,
    Edge,
    Node,
    WorldSchema,
)


def _min_level_share(k: int) -> float:
    return min(0.04, 0.25 / k)


def sample_bin_probs(
    scm_levels: dict[str, list[str]],
    rng: np.random.Generator,
    *,
    concentration: tuple[float, float],
) -> dict[str, list[float]]:
    """Per-column level probabilities for every categorical feature column with 2+ levels that
    authors none: a Dirichlet with a per-column concentration, floored at ``_min_level_share``.
    The target's base rate is always authored (``validate`` requires ``level_probs``)."""
    out: dict[str, list[float]] = {}
    for name, level_names in scm_levels.items():
        k = len(level_names)
        if k < 2:
            continue
        alpha = float(rng.uniform(*concentration))
        probs = np.clip(rng.dirichlet(np.full(k, alpha)), _min_level_share(k), None)
        out[name] = (probs / probs.sum()).tolist()
    return out


LEVEL_CONCENTRATION = (0.2, 5.0)


def _noise(kind: str, std: float, rng: np.random.Generator):
    """A unit-variance noise of the authored shape at ``std``: gaussian, heavier-tailed (laplace,
    student_t) or skewed (a centred gamma, either way round); only the parameters the shape uses
    are drawn."""
    if kind == "laplace":
        return Laplace(scale=std / np.sqrt(2.0))
    if kind == "student_t":
        df = float(rng.uniform(2.5, 12.0))  # > 2 so the variance exists
        return StudentT(df, std * np.sqrt((df - 2.0) / df))
    if kind == "skewed":
        k, sign = float(rng.uniform(0.7, 6.0)), float(rng.choice([-1.0, 1.0]))
        return Affine(Gamma(k, 1.0), loc=-sign * std * np.sqrt(k), scale=sign * std / np.sqrt(k))
    return Normal(std=std)


def _combine(kind: str, edges: tuple, noise):
    """The node a mechanism's edges combine under: their sum, their largest or smallest, or the
    compounding of their sum; rules are additive over threshold edges."""
    op = {"max": "max", "min": "min", "product": "compound"}.get(kind, "sum")
    return PlanNode(edges, op=op, noise=noise)


def _bind_lookups(nodes: dict, probabilities: dict) -> dict:
    """Every lookup edge without probabilities bound to its parent's, helper nodes included, so
    the mechanisms read the level the parent's column shows."""

    def bound(edge):
        if isinstance(edge, LookupEdge) and edge.probabilities is None and edge.parent in probabilities:  # fmt: skip
            return replace(edge, probabilities=tuple(float(v) for v in probabilities[edge.parent]))
        return edge

    return {name: replace(node, edges=tuple(bound(e) for e in node.edges)) for name, node in nodes.items()}  # fmt: skip


def vary_shares(authored: dict[str, list[float]], rng: np.random.Generator, *, concentration: float) -> dict[str, list[float]]:  # fmt: skip
    """The authored level shares redrawn for one realization: a Dirichlet centered on each authored
    vector with total concentration ``concentration`` (higher is tighter; at 100 a 12% class lands
    between about 6 and 20%), floored at ``_min_level_share`` so no level vanishes."""
    out = {}
    for name, probs in authored.items():
        p = np.asarray(probs, dtype=float)
        drawn = np.clip(rng.dirichlet(concentration * p), _min_level_share(len(p)), None)
        out[name] = (drawn / drawn.sum()).tolist()
    return out


def compile_schema(schema: WorldSchema, rng: np.random.Generator, *, target_noise: float = 1.0,
                   share_concentration: float | None = None, entity_effect: np.random.Generator | None = None):  # fmt: skip
    """``(plan, target)``: the schema realized as a PluRel SCM, every magnitude drawn from ``rng``
    inside the authored sign and strength class. ``target_noise`` scales the target's own noise
    (its latent's std, or an argmax target's score noise) after the draw: below 1 the target is
    closer to a function of its parents, above 1 further from one; the features are untouched.
    ``share_concentration`` redraws the authored level shares around their values
    (:func:`vary_shares`); ``None`` keeps them. Either way every binned column's edges are fixed
    once from a calibration draw (:func:`_fix_edges`), so a table's class shares follow its rows.
    ``entity_effect`` adds an unrecorded cause per entity drawn from it (:func:`_with_entity_effect`)
    before the calibration, so the shares hold with it."""
    schema.validate()

    order = schema.topo_order()
    # formula columns are computed on rendered values in _render, not as SCM nodes
    names = [n for n in order if n != schema.target and schema.nodes[n].formula is None
             and not schema.nodes[n].hidden] + [schema.target]  # fmt: skip
    target = schema.target

    def draw_weight(edge: Edge) -> float:
        lo, hi = MAGNITUDES[abs(edge.effect)]
        return float(np.sign(edge.effect)) * float(rng.uniform(lo, hi))

    # nominal parents get their bin probabilities bound into their lookup terms after sampling
    lookup_parents: set[str] = set()

    data_mechanisms: dict = {}
    for name in order:
        node = schema.nodes[name]
        if node.formula is not None:
            continue
        edges = schema.edges(name)
        if not edges and not node.class_scores:
            data_mechanisms[name] = PlanNode()
            continue
        if node.class_scores:
            # argmax target: plurel's one-hot node over per-class scores, each feature a matrix
            # edge carrying its authored sign at a drawn magnitude in the classes it drives, a
            # Gumbel per class so the draw is a softmax and class boundaries are soft
            classes = len(node.class_scores)
            weights: dict[str, np.ndarray] = {}
            for k, feats in enumerate(node.class_scores.values()):
                for feat, sign in feats:
                    weights.setdefault(feat, np.zeros((1, classes)))[0, k] += float(sign) * float(rng.uniform(1.0, 3.0))  # fmt: skip
            edges = tuple(MatrixEdge(feat, matrix) for feat, matrix in weights.items())
            noise = Gumbel(scale=float(rng.uniform(0.6, 1.8)) * target_noise)
            data_mechanisms[name] = PlanNode(edges, bias=(0.0,) * classes, onehot=True, noise=noise)
            continue
        data_effects = []
        for parent, edge in edges.items():
            if edge.level_effects is not None:
                scale = float(rng.uniform(0.7, 1.6))
                values = tuple(float(v) * scale for v in edge.level_effects.values())
                if edge.level_ranges is not None:
                    # hard-ranged edge: offsets are standardized log-midpoints of the envelopes,
                    # so latent distances mirror the real magnitude ratios
                    mids = np.array([(lo + hi) / 2 for lo, hi in edge.level_ranges.values()])
                    mids = np.log10(mids) if np.all(mids > 0) else mids
                    std = float(mids.std()) or 1.0
                    strength = float(rng.uniform(*MAGNITUDES[1.0]))
                    values = tuple((mids - float(mids.mean())) / std * strength)
                lookup_parents.add(parent)
                data_effects.append(LookupEdge(parent, values))
                if edge.level_slopes is not None:  # the regime: this level's extra slope of scaled
                    slopes = tuple(float(v) for v in edge.level_slopes.values())
                    regime = f"{name}.{parent}_{edge.scaled}"  # a hidden product node: the slope
                    data_mechanisms[regime] = PlanNode((LookupEdge(parent, slopes), LinearEdge(edge.scaled)), op="product", noise=None)  # fmt: skip
                    data_effects.append(LinearEdge(regime))
                continue
            weight = draw_weight(edge)
            transform = TRANSFORM_OF_SHAPE[edge.shape]
            data_effects.append(LinearEdge(parent, weight, transform))
        for a, b in node.interactions:  # under rules a pair is a conjunction of two thresholds
            weight = float(rng.uniform(0.8, 2.5))
            transform = "step" if node.mechanism == "rules" else "identity"
            term = f"{name}.{a}_{b}"  # a hidden product node: the interaction
            data_mechanisms[term] = PlanNode((LinearEdge(a, 1.0, transform), LinearEdge(b, 1.0, transform)), op="product", noise=None)  # fmt: skip
            data_effects.append(LinearEdge(term, weight))
        std = float(rng.uniform(*NOISE_SCALES[node.noise_scale])) * (target_noise if name == target else 1.0)  # fmt: skip
        noise = _noise(node.noise_dist, std, rng)
        data_mechanisms[name] = _combine(node.mechanism, tuple(data_effects), noise)

    # no pooled marginal for hard-ranged columns; their true marginal is the per-level mixture
    marginals: dict[str, str | None] = {}
    for name, node in schema.nodes.items():
        if node.kind == "numeric" and node.formula is None and not node.hidden:
            marginals[name] = (None if schema._range_edge(name) is not None
                               else node.dist or str(rng.choice(UNIT_FAMILIES[node.units])))  # fmt: skip

    levels = {name: list(vals) for name, vals in schema.levels.items()}
    authored_probs = {
        name: [float(p) for p in node.level_probs.values()]
        for name, node in schema.nodes.items()
        if node.level_probs is not None
    }
    drawn_levels = {k: v for k, v in levels.items() if k not in authored_probs}
    probabilities = sample_bin_probs(drawn_levels, rng, concentration=LEVEL_CONCENTRATION)
    probabilities.update(authored_probs)
    if share_concentration is not None:
        probabilities.update(vary_shares(authored_probs, rng, concentration=share_concentration))
    if entity_effect is not None:
        data_mechanisms = _with_entity_effect(data_mechanisms, schema, entity_effect)
    # the sampled bin probabilities bound into every lookup edge: shared realized boundaries
    data_mechanisms = _bind_lookups(data_mechanisms, probabilities)

    table = dict(data_mechanisms)
    for name in names:
        node, mechanism = schema.nodes[name], data_mechanisms[name]
        if node.kind == "numeric":
            kind = marginals[name]
            table[name] = Column(mechanism, marginal=None if kind is None else shape(kind, rng))
        elif node.class_scores:  # the mechanism emits the class
            table[name] = Column(mechanism, categories=tuple(levels[name]))
        else:
            table[name] = Column(
                mechanism, "categorical", categories=tuple(levels[name]),
                probabilities=tuple(probabilities[name]),
                binning="normal" if name in lookup_parents else "empirical",
            )  # fmt: skip

    # a node without a column (an authored hidden cause, a helper) is simulated, never observed
    plan = SCM(table)
    if schema.nodes[target].class_scores:
        plan = _calibrate_argmax(plan, target, probabilities[target], rng)
    return _fix_edges(plan, rng), target


# the unrecorded cause per entity; a dotted name no schema node or helper takes
ENTITY_EFFECT = "entity.effect"


def _with_entity_effect(mechanisms: dict, schema: WorldSchema, rng: np.random.Generator) -> dict:
    """``mechanisms`` with :data:`ENTITY_EFFECT`, a standard normal root, feeding the target (an argmax target's
    class scores by a weight per class) and each rendered numeric column with probability one half, at the weak
    strength class and a sign drawn per edge. Only nodes that sum their edges take it, so a max, min or compounding
    node keeps its authored form."""
    lo, hi = MAGNITUDES[0.5]
    out = {**mechanisms, ENTITY_EFFECT: PlanNode()}
    numeric = [n for n, node in schema.nodes.items() if node.kind == "numeric" and n in mechanisms and not node.hidden and n != schema.target]  # fmt: skip
    for name in numeric:
        if rng.random() < 0.5 and mechanisms[name].op == "sum":
            weight = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(lo, hi))
            out[name] = replace(mechanisms[name], edges=(*mechanisms[name].edges, LinearEdge(ENTITY_EFFECT, weight)))  # fmt: skip
    node = mechanisms[schema.target]
    if node.onehot:
        weights = rng.normal(0.0, 1.0, (1, node.dim)) * float(rng.uniform(lo, hi))
        out[schema.target] = replace(node, edges=(*node.edges, MatrixEdge(ENTITY_EFFECT, weights)))
    elif node.op == "sum":
        weight = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(lo, hi))
        out[schema.target] = replace(node, edges=(*node.edges, LinearEdge(ENTITY_EFFECT, weight)))
    return out


JOINT_SEPARATOR = " / "  # between a joint label's parts, the target's first
JOINT_CLASSES = (4, 10)  # a joint target's class count
JOINT_FLOOR = 0.01  # the least share of a calibration draw every joint class must reach
# the share of a joint part's value the other columns explain (linearly): under it the part is mostly noise no
# column shows, over it a near-function of the columns shown, either way little of a second outcome to learn
JOINT_R2 = (0.3, 0.95)
# a numeric part's levels, split at its median: a third band met the floor on half as many schemas, and the
# schemas it met it on a median split meets it on too
BANDS = ("low", "high")


def part_levels(schema: WorldSchema, leaf: str) -> list[str]:
    """The levels a joint target's part ``leaf`` takes: a categorical leaf's own, a numeric leaf's :data:`BANDS`."""
    return list(schema.levels[leaf]) if schema.nodes[leaf].kind == "categorical" else list(BANDS)


def joint_leaves(schema: WorldSchema) -> list[str]:
    """The columns a joint target can join the schema's target with: rendered leaves (a parent, no child, no
    formula) whose levels (:func:`part_levels`) times the target's are within :data:`JOINT_CLASSES`; a categorical
    one never blank, as its level is the label's, while a numeric one is cut into bands of its value, which no blank
    or recording artifact reaches. Both parts are effects of the features, so the joint label is too; an isolated
    root would add a part no column predicts."""
    children, n = schema.children(), len(schema.levels[schema.target])
    lo, hi = JOINT_CLASSES
    return [
        name for name, node in schema.nodes.items()
        if name != schema.target and not node.hidden and node.formula is None and node.parents and not children[name]
        and (node.kind == "numeric" or node.missing is None) and lo <= n * len(part_levels(schema, name)) <= hi
    ]  # fmt: skip


def _joint_part(schema: WorldSchema, plan, target: str, leaf: str, rng: np.random.Generator):
    """``(plan, column, r2)`` for the joint target's part ``leaf``, or ``None`` when it cannot be one. From one
    ``CALIBRATION_ROWS`` draw: a numeric leaf is cut into its bands at the draw's quantiles of its value, as a column
    of the plan reading the leaf's node, which moves no draw; the part is kept when
    the other columns explain a :data:`JOINT_R2` share of its value (``r2``: linearly, on the latents, the target
    aside) and every joint class holds :data:`JOINT_FLOOR` of the draw."""
    frame, latents = plan.sample_with_latents(CALIBRATION_ROWS, seed=rng)
    value = latents[leaf][:, 0]
    column, named = leaf, part_levels(schema, leaf)
    if schema.nodes[leaf].kind == "numeric":
        column = f"{leaf}.band"  # a dotted name no schema column takes
        edges = tuple(float(e) for e in np.quantile(value, np.arange(1, len(named)) / len(named)))
        reading = PlanNode(ReadEdge(leaf), noise=None)
        band = Column(reading, "categorical", categories=tuple(named), binning=edges)
        plan = SCM({**plan.table, column: band})
        parts = np.asarray(named)[np.digitize(value, edges)]
    else:
        parts = frame[leaf].astype(str).to_numpy()
    shown = sorted(set(plan.columns) - {target, leaf, column})
    x = np.column_stack([*(latents[n].reshape(len(value), -1) for n in shown), np.ones(len(value))])
    r2 = float(1.0 - np.var(value - x @ np.linalg.lstsq(x, value, rcond=None)[0]) / np.var(value))
    pairs = pd.Series(list(zip(frame[target].astype(str), parts, strict=True))).value_counts()
    common = all(pairs.get((t, p), 0) >= JOINT_FLOOR * CALIBRATION_ROWS for t in levels(plan)[target] for p in named)  # fmt: skip
    return (plan, column, r2) if common and JOINT_R2[0] <= r2 <= JOINT_R2[1] else None


def _part_text(schema: WorldSchema, leaf: str) -> str:
    """How the transform text names a joint target's part: the leaf, or its band and how it is cut."""
    if schema.nodes[leaf].kind == "categorical":
        return leaf
    return f"{leaf} band ({' or '.join(BANDS)}, split at its median)"


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


def _entity_nodes(plan, mode: str, labels: list[str], rng: np.random.Generator) -> tuple[str, ...]:
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
        roots = [
            name for name, node in plan.nodes.items() if not node.parents and name != ENTITY_EFFECT
        ]
        held = {name for name in roots if rng.random() < (ROOT_SHARE if name in plan.columns else HIDDEN_ROOT_SHARE)}  # fmt: skip
        if roots and not held:
            held = {roots[int(rng.integers(len(roots)))]}
    if ENTITY_EFFECT in plan.nodes:
        held.add(ENTITY_EFFECT)
    return tuple(name for name in plan.nodes if name in held)


CALIBRATION_ROWS = 4096
CALIBRATION_PASSES = 100
CALIBRATION_TOLERANCE = 0.01  # largest bias step, in noise units, at which the fit stops


def _fix_edges(plan, rng: np.random.Generator):
    """The plan with every empirically binned column cut at fixed edges: the quantiles of one
    ``CALIBRATION_ROWS`` draw of its latent at the column's cumulative shares. Later draws bin at
    the population's cuts, so a table's shares vary with its rows and a shift moves them; a per-draw quantile would meet the authored shares in every sample instead."""
    values = plan.sample_with_latents(CALIBRATION_ROWS, seed=rng)[1]
    table = dict(plan.table)
    for name, column in plan.columns.items():
        if column.kind == "categorical" and column.binning == "empirical":
            cuts = np.cumsum(column.category_probabilities)[:-1]
            edges = tuple(
                float(e) for e in np.quantile(np.asarray(values[name], dtype=float), cuts)
            )
            table[name] = replace(column, binning=edges)
    return SCM(table)


def _calibrate_argmax(plan, target: str, probabilities: list[float], rng: np.random.Generator):
    """The plan with per-class biases on ``target`` under which its class shares meet the
    authored ``probabilities``: a fixed point on one draw of the parents and the mechanism's own
    Gumbel noise, each pass shifting a class by the noise scale times the log of target over
    observed share (the prior-shift adjustment of a softmax)."""
    node = plan.nodes[target]
    noise_scale = node.noise.scale
    n = CALIBRATION_ROWS
    latents = plan.sample_with_latents(n, seed=rng)[1]
    scores = REDUCTIONS[node.op]([edge.apply(latents[edge.parent]) for edge in node.edges])
    noise = rng.gumbel(0.0, noise_scale, size=(n, node.dim))
    wanted = np.asarray(probabilities, dtype=float)
    biases = np.zeros(len(wanted))
    for _ in range(CALIBRATION_PASSES):
        codes = np.argmax(scores + biases + noise, axis=1)
        shares = np.bincount(codes, minlength=len(wanted)) / n
        step = noise_scale * np.log(wanted / np.maximum(shares, 1.0 / n))
        if float(np.abs(step).max()) < CALIBRATION_TOLERANCE * noise_scale:
            break
        biases += step
    calibrated = replace(node, bias=tuple(float(b - biases.mean()) for b in biases))
    return SCM({**plan.table, target: replace(plan.columns[target], node=calibrated)})


def _scaled_units(values: np.ndarray, lo: float, hi: float,
                  integer_valued: bool = False) -> np.ndarray:  # fmt: skip
    """Monotone affine of ``values`` onto the authored ``(lo, hi)``, rounded to ~1/500 of the span
    and clipped inside the envelope. Integer-valued draws round to integers only when ``lo`` and
    ``hi`` are both integers."""
    v = values.astype(np.float64)
    vlo, vhi = float(np.min(v)), float(np.max(v))
    unit = (v - vlo) / (vhi - vlo if vhi > vlo else 1.0)
    out = lo + unit * (hi - lo)
    if integer_valued and float(lo).is_integer() and float(hi).is_integer():
        return np.round(out).astype(np.float32)
    step = 10.0 ** np.floor(np.log10((hi - lo) / 500.0))
    # rounding must not leave the authored envelope (a floor rounded to 0 breaks formula divisions)
    return np.clip(np.round(out / step) * step, lo, hi).astype(np.float32)


# Every realization coin comes from a lane of [WORLD_DOMAIN, crc32(schema_id), seed, lane, ...], so
# the same (schema, seed) gives the same SCM and the same row draws on any machine.
WORLD_DOMAIN = 0x57524C44
_LANE_WORLD = 1
# a grouped or joint table's choices, apart so an untransformed table draws as before
_LANE_TRANSFORM = 2
_LANE_SAMPLE = 3
_LANE_ARTIFACT = 4
_LANE_BLANK = 5
_EFFECT, _JOINT, _ENTITY, _TEXT = 0, 1, 2, 3  # the transform lane's own lanes

# a grouped table's draw, as ``realize_world`` takes it in ``grouping`` and its realization stamps it
GROUPING_KEYS = ("mode", "rows", "spread", "measurement", "column", "effect")
ENTITY_DOC = "the unit the row was recorded on; rows that share it come from one unit"


def _lane(schema: WorldSchema, seed: int, which: int, *rest: int) -> np.random.Generator:
    return np.random.default_rng(
        [WORLD_DOMAIN, zlib.crc32(schema.schema_id.encode()), int(seed), which, *rest]
    )


def realize_world(schema: WorldSchema, seed: int, *, target_noise: float = 1.0,
                  share_concentration: float | None = None, grouping: dict | None = None,
                  joint: str | None = None) -> Population:  # fmt: skip
    """``schema`` under ``seed`` as a :class:`Population`: the magnitudes drawn from the
    seed's lane, the target's noise scaled by ``target_noise``, the level shares redrawn under
    ``share_concentration`` (:func:`compile_schema`); every draw of rows is rendered onto the
    authored scales (:func:`_render`) under lanes of the same seed.

    Two transforms take the table past its schema, their choices drawn from the seed's transform lane. ``grouping``
    (:data:`GROUPING_KEYS`: the mode, the median rows per entity and their log sd, the measurement noise, the ID
    column's name or ``None``, and whether an unrecorded cause per entity is added) makes rows share entities
    (:class:`Grouping`, the nodes held per entity drawn by :func:`_entity_nodes`). ``joint`` names a leaf of
    :func:`joint_leaves` whose level (a numeric leaf's band) the label joins to the target's
    (``"<target> / <part>"``, the target's levels slowest), the leaf no longer a column; a part :func:`_joint_part`
    refuses keeps the table's single target. A transformed table's description gains a sentence saying what its
    rows and label are (:data:`GROUPED_TEXT`, :data:`JOINT_TEXT`), and a column it no longer has loses its note.
    The realization stamps the transforms as realized (``grouping``, ``joint``) and the sentences (``text``)."""
    effect = grouping is not None and grouping["effect"]
    plan, target = compile_schema(schema, _lane(schema, seed, _LANE_WORLD), target_noise=target_noise,
                                  share_concentration=share_concentration,
                                  entity_effect=_lane(schema, seed, _LANE_TRANSFORM, _EFFECT) if effect else None)  # fmt: skip
    part = None
    if joint is not None:
        part = _joint_part(
            schema, plan, target, joint, _lane(schema, seed, _LANE_TRANSFORM, _JOINT)
        )
        joint = joint if part else None
    name = f"{target}_and_{joint}" if joint else target
    features = [c for c in schema.feature_names() if c != joint]
    types = [schema.nodes[c].kind for c in features]
    order, docs = category_order(plan), _column_docs(schema, plan) or {}
    realization = {
        "schema_id": schema.schema_id,
        "seed": int(seed),
        "table_id": table_id(schema.schema_id, seed),
        "target_noise": float(target_noise),
        "share_concentration": None
        if share_concentration is None
        else float(share_concentration),
    }  # fmt: skip
    texts, pick = [], _lane(schema, seed, _LANE_TRANSFORM, _TEXT)
    if part:
        plan, column, r2 = part
        named = part_levels(schema, joint)
        order = {k: v for k, v in category_order(plan).items() if k not in (target, joint, column)}
        order[name] = [
            f"{t}{JOINT_SEPARATOR}{p}" for t in category_order(plan)[target] for p in named
        ]
        for gone in (target, joint):  # columns the table no longer has
            docs.pop(gone, None)
        described = _part_text(schema, joint)
        docs[name] = f"{schema.nodes[target].desc}, then the {described}: {schema.nodes[joint].desc}; categories: {', '.join(order[name])}"  # fmt: skip
        realization["joint"] = {"part": joint, "separator": JOINT_SEPARATOR, "banded": column != joint,
                                "described": described, "r2": round(r2, 3),
                                "parts": [[target, category_order(plan)[target]], [joint, named]]}  # fmt: skip
        texts.append(JOINT_TEXT[int(pick.integers(len(JOINT_TEXT)))].format(name=name, target=target, part=described))  # fmt: skip
    entity = None
    if grouping is not None:
        labels = [target, joint] if joint else [target]
        fixed = _entity_nodes(
            plan, grouping["mode"], labels, _lane(schema, seed, _LANE_TRANSFORM, _ENTITY)
        )
        entity = Grouping(mode=grouping["mode"], fixed=fixed, rows=grouping["rows"], spread=grouping["spread"],
                          measurement=grouping["measurement"], column=grouping["column"])  # fmt: skip
        if entity.column is not None:
            features, types = [entity.column, *features], ["categorical", *types]
            docs[entity.column] = ENTITY_DOC
        fed = sorted(n for n, node in plan.nodes.items() if ENTITY_EFFECT in node.parents)
        realization["grouping"] = {**{key: grouping[key] for key in GROUPING_KEYS}, "fixed": list(fixed),
                                   "effect_on": fed}  # fmt: skip
        unit = "unit" if entity.column in (None, "id") else entity.column.removesuffix("_id")
        phrasings = GROUPED_TEXT[entity.mode, entity.column is not None]
        texts.append(phrasings[int(pick.integers(len(phrasings)))].format(unit=unit, column=entity.column, target=name))  # fmt: skip
    description = (schema.description or "").strip() or None
    if texts:
        realization["text"] = " ".join(texts)
        description = f"{description} {realization['text']}" if description else realization["text"]
    metadata = TableMetadata(
        feature_names=features,
        feature_types=types,
        target=name,
        domain=schema.domain,
        plan=plan,
        description=description,
        column_descriptions=docs or None,
        category_order=order,
        realization=realization,
    )
    render = partial(_render, schema, plan, target=target, seed=int(seed), joint=(joint, part[1], name) if part else None)  # fmt: skip
    return Population(
        metadata=metadata,
        lanes=(WORLD_DOMAIN, zlib.crc32(schema.schema_id.encode()), int(seed), _LANE_SAMPLE),
        render=render,
        grouping=entity,
    )


def _artifacts(node: Node, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The record's capture of a rendered column: a share at the floor, the limits, the quantum."""
    out = values.astype(np.float64)
    if node.zero_inflation is not None:
        out[rng.random(len(out)) < node.zero_inflation] = node.scale[0]
    if node.censor is not None:
        lo, hi = node.censor
        out = np.clip(out, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    if node.rounding is not None:
        out = np.clip(np.round(out / node.rounding) * node.rounding, *node.scale)
    return out.astype(np.float32)


def _blank(schema: WorldSchema, frame, rng: np.random.Generator):
    """The record's blanks: at random, or weighted toward the high or low rows of a column (the
    column itself for MNAR), with the authored share in expectation; before the derived columns,
    so a blank input leaves a blank result."""
    for name, node in schema.nodes.items():
        if node.missing is None:
            continue
        n = len(frame)
        if node.missing_by is None:
            mask = rng.random(n) < node.missing
        else:
            by, side = node.missing_by
            ranks = frame[by].rank(pct=True).fillna(0.5).to_numpy()
            weight = ranks if side == "high" else 1.0 - ranks
            mask = rng.random(n) < np.clip(2.0 * node.missing * weight, 0.0, 1.0)
        if node.kind == "numeric":
            frame.loc[mask, name] = np.nan
        else:
            frame[name] = frame[name].astype(object)
            frame.loc[mask, name] = None


def _render(schema: WorldSchema, plan, frame, draw: int, *, target: str, seed: int,
            joint: tuple[str, str, str] | None = None):  # fmt: skip
    """Draw ``draw``'s rows onto the table: numeric columns onto their authored scale (per column,
    or per parent level for hard-ranged columns) and through their recording artifacts, the
    record's blanks, then derived columns as exact arithmetic over the rendered values; authored
    column order, target last. The artifacts and blanks come from lanes of the seed and draw.
    ``joint`` ``(leaf, part, name)`` joins the target and the plan column ``part`` (the leaf, or
    its band) into the label ``name``, the leaf left out."""
    if len(frame) == 1:
        raise ValueError("a table renders onto its scales relative to its rows: draw at least two")
    artifacts = _lane(schema, seed, _LANE_ARTIFACT, draw)
    blanks = _lane(schema, seed, _LANE_BLANK, draw)
    named = levels(plan)
    for name, column in plan.columns.items():
        if name == target or column.kind != "numeric":
            continue
        col = frame[name].to_numpy()
        if not np.isfinite(col).all():
            continue
        integer_valued = bool(np.all(np.mod(col, 1) == 0))
        ranged = schema._range_edge(name)
        if ranged is None:
            node = schema.nodes[name]
            scaled = _scaled_units(col, *node.scale, integer_valued=integer_valued)
            frame[name] = _artifacts(node, scaled, artifacts)
            continue
        parent, redge = ranged
        realized = [str(v) for v in named[parent]]
        parent_col = frame[parent].astype(str).to_numpy()
        scaled = col.astype(np.float64)
        for idx, level in enumerate(schema.nodes[parent].levels):
            mask = parent_col == realized[idx]
            if mask.any():
                lo, hi = redge.level_ranges[level]
                scaled[mask] = _scaled_units(col[mask], lo, hi, integer_valued=integer_valued)
        frame[name] = scaled.astype(np.float32)
    _blank(schema, frame, blanks)
    for name in schema.feature_names():
        node = schema.nodes[name]
        if node.formula is not None:
            args = [frame[i].to_numpy(dtype=np.float64) for i in schema._formula_inputs(name)]
            derived = np.round(np.asarray(node.formula(*args), dtype=np.float64), 6)
            frame[name] = derived.astype(np.float32)
    if joint is None:
        return frame[[*schema.feature_names(), target]]
    leaf, part, label = joint
    joined = frame[target].astype(str) + JOINT_SEPARATOR + frame[part].astype(str)
    order = [f"{t}{JOINT_SEPARATOR}{p}" for t in named[target] for p in named[part]]
    features = [c for c in schema.feature_names() if c != leaf]
    return frame[features].assign(**{label: pd.Categorical(joined, categories=order)})


def _column_docs(schema: WorldSchema, plan) -> dict[str, str] | None:
    """Per-column documentation over the schema's nodes (derived columns included): the authored
    desc plus each categorical column's realized level names as a plain category list."""
    docs: dict[str, str] = {}
    named = levels(plan)
    for name in schema.nodes:
        if schema.nodes[name].hidden:
            continue
        parts = [schema.nodes[name].desc]
        level_names = named.get(name)
        if level_names is not None:
            # a plain list, never an ordered "a < b < c" scale
            parts.append(f"categories: {', '.join(map(str, level_names))}")
        docs[name] = "; ".join(parts)
    return docs or None


__all__ = [
    "ENTITY_EFFECT",
    "GROUPING_KEYS",
    "compile_schema",
    "joint_leaves",
    "part_levels",
    "realize_world",
    "sample_bin_probs",
    "vary_shares",
]
