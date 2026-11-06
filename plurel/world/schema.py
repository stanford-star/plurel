"""The authoring language of a world schema: :class:`WorldSchema`, a domain's causal DAG as one
authored literal, its :class:`Node` columns and :class:`Edge` terms, and ``validate``, which names
the first rule a schema breaks. A schema says what a table's columns are and how they depend on each
other; the magnitudes are drawn when it is realized."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

# The plurel transform a LinearEdge applies for each authored edge shape: an authored "linear"
# edge is the identity transform, its weight alone. Every shape but ``u`` is monotone, so an
# authored sign is the realized direction; a ``u`` edge bends back (both extremes push the same
# way) and has none.
TRANSFORM_OF_SHAPE: dict[str, str] = {
    "linear": "identity",
    "hinge": "relu",
    "saturating": "tanh",
    "threshold": "step",
    "u": "square",
}
# How a node's parents combine: a sum, the largest or smallest term, a compounding product, or
# threshold rules and their conjunctions.
MECHANISMS = ("additive", "max", "min", "product", "rules")
STRENGTHS = (0.5, 1.0)
# Strength class -> magnitude range. The ranges must not overlap, so strong always dominates weak.
MAGNITUDES: dict[float, tuple[float, float]] = {1.0: (1.6, 3.0), 0.5: (0.4, 1.4)}
# Noise std ranges, relative to unit-variance shapes and edge magnitudes of 0.4-3.0.
NOISE_SCALES: dict[str, tuple[float, float]] = {
    "low": (0.25, 0.5),
    "medium": (0.5, 1.0),
    "high": (1.0, 1.6),
}
NOISE_DISTS = ("gaussian", "laplace", "student_t", "skewed")

# Unit family -> allowed marginal shapes for a numeric node's dist.
UNIT_FAMILIES: dict[str, tuple[str, ...]] = {
    "count": ("counts", "zero_inflated"),
    "amount": ("right", "zero_inflated", "symmetric", "bimodal"),
    "duration": ("right", "symmetric", "bimodal"),
    "ratio": ("right", "symmetric", "left"),
    "score": ("symmetric", "left", "bimodal", "right"),
    "percent": ("symmetric", "left", "right", "bimodal", "multimodal"),
    "signed": ("symmetric", "bimodal", "multimodal"),
}


@dataclass(frozen=True)
class Edge:
    """One term of a child's structural equation: how one parent enters. ``effect`` is the authored
    sign and strength class (±1 strong, ±0.5 weak; magnitudes are drawn by the compiler); a nominal
    parent enters through ``level_effects`` instead, keyed in the parent's authored level order."""

    effect: float = 0.0  # ±1 strong, ±0.5 weak; unused with level_effects
    shape: str = "linear"  # "linear"|"hinge"|"saturating"|"threshold"
    level_effects: dict[str, float] | None = None  # nominal parent: per-level latent offset
    # per-level (lo, hi) the child renders into; needs level_effects, supersedes the child's scale
    level_ranges: dict[str, tuple[float, float]] | None = None
    # per-level extra slope of another parent of the child (``scaled``): the regime each level of
    # a nominal parent puts the child in, on a level_effects edge (E3)
    level_slopes: dict[str, float] | None = None
    scaled: str | None = None


@dataclass(frozen=True)
class Node:
    """One column of the schema: its properties plus its incoming edges. ``parents`` maps a parent
    to an :class:`Edge`, a bare ±1/±0.5 (``Edge(effect=...)``), or a ``{level: offset}`` dict
    (``Edge(level_effects=...)``); a node with no parents is a root."""

    kind: str  # "numeric" | "categorical"
    desc: str  # model-facing one-liner: what the column is, never a causal claim
    parents: dict[str, object] = field(default_factory=dict)
    units: str | None = None  # UNIT_FAMILIES key; REQUIRED numeric, forbidden categorical
    dist: str | None = None  # marginal shape; REQUIRED on numeric roots, from the family pool
    scale: tuple[float, float] | None = None  # rendered (lo, hi), REQUIRED on every numeric
    levels: list[str] | None = None  # REQUIRED categorical (unless class_scores derives them)
    ordinal: bool = False  # level ORDER is an intensity scale (enables scalar-sign edges out)
    # nominal ARGMAX outcome (target-only): {class: [(numeric feature, ±1), ...]}
    class_scores: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    interactions: list[tuple[str, str]] = field(default_factory=list)  # product terms, parent pairs
    noise_scale: str | None = None  # "low"|"medium"|"high"; REQUIRED on non-root additive nodes
    noise_dist: str | None = None  # "gaussian"|"laplace"|"student_t"|"skewed"; REQUIRED with it
    # derived leaf computed on rendered values; inputs are the callable's parameter names
    formula: object | None = None
    # authored base rates keyed by ``levels`` in order, summing to 1; None = drawn
    level_probs: dict[str, float] | None = None
    # a cause no column records: a numeric root, simulated and never rendered, that feeds two or
    # more nodes on the way to the target (the hidden confounder of the taxonomy)
    hidden: bool = False
    mechanism: str = "additive"  # one of MECHANISMS: how the parents combine
    # how the record captures a numeric value (D6): the quantum it is recorded to, the detection
    # or capping limits it is clipped to (None = that side open), the share recorded at the floor
    rounding: float | None = None
    censor: tuple[float | None, float | None] | None = None
    zero_inflation: float | None = None
    # what the record leaves blank (E2): the share of rows missing and, when the blanks follow a
    # column, that column and whether its high or low rows go missing (the column itself: MNAR)
    missing: float | None = None
    missing_by: tuple[str, str] | None = None


def _as_edge(spec: object) -> Edge:
    """Normalize an authored parent spec (number / dict / Edge) to an :class:`Edge`."""
    if isinstance(spec, Edge):
        return spec
    if isinstance(spec, dict):
        return Edge(level_effects=dict(spec))
    return Edge(effect=float(spec))  # type: ignore[arg-type]


@dataclass(frozen=True)
class WorldSchema:
    """A domain's causal DAG as one authored literal. ``schema_id`` is an opaque identity key;
    ``rationale`` is provenance and never model-facing. ``taxonomy`` (sector, division, group) and
    ``title`` (that group's title) say where a catalog places the schema: its loader sets them, never
    the author."""

    schema_id: str
    target: str
    description: str
    nodes: dict[str, Node]
    rationale: str = ""
    taxonomy: tuple[str, str, str] | None = field(default=None, compare=False)
    title: str | None = field(default=None, compare=False)

    def feature_names(self) -> list[str]:
        """The rendered feature columns: every node but the target and the hidden ones."""
        return [
            name for name, node in self.nodes.items() if name != self.target and not node.hidden
        ]

    def edges(self, name: str) -> dict[str, Edge]:
        """The node's incoming edges, shorthand-normalized."""
        return {parent: _as_edge(spec) for parent, spec in self.nodes[name].parents.items()}

    def parent_names(self, name: str) -> list[str]:
        """Every structural parent: edge parents, class-scoring features, and formula inputs,
        deduplicated in that order."""
        node = self.nodes[name]
        scored = [feat for feats in node.class_scores.values() for feat, _sign in feats]
        return list(dict.fromkeys([*node.parents, *scored, *self._formula_inputs(name)]))

    def _formula_inputs(self, name: str) -> list[str]:
        """A derived node's inputs: the formula callable's parameter names."""
        node = self.nodes[name]
        if node.formula is None:
            return []
        return list(inspect.signature(node.formula).parameters)

    def all_edges(self) -> list[tuple[str, str, Edge]]:
        """Every authored edge as ``(child, parent, edge)``, authored order."""
        return [
            (child, parent, edge)
            for child in self.nodes
            for parent, edge in self.edges(child).items()
        ]

    def children(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {name: [] for name in self.nodes}
        for child in self.nodes:
            for parent in self.parent_names(child):
                out[parent].append(child)
        return out

    def isolated_roots(self) -> list[str]:
        """Nodes with no parents and no children: structurally meaningless for the outcome."""
        children = self.children()
        return [
            name
            for name in self.nodes
            if not self.parent_names(name) and not children[name] and name != self.target
        ]

    def mediators(self) -> list[str]:
        """Non-root feature nodes on a directed path to the target; a non-root whose descendants
        never reach the target is not one."""
        ancestors: set[str] = set()
        frontier = list(self.parent_names(self.target))
        while frontier:
            name = frontier.pop()
            if name not in ancestors:
                ancestors.add(name)
                frontier.extend(self.parent_names(name))
        return [
            name
            for name in self.nodes
            if name != self.target and self.parent_names(name) and name in ancestors
        ]

    def _range_edge(self, name: str) -> tuple[str, Edge] | None:
        """The child's level_ranges edge as ``(parent, edge)``, if any (validate allows one)."""
        found = [(p, e) for p, e in self.edges(name).items() if e.level_ranges is not None]
        return found[0] if found else None

    def topo_order(self) -> list[str]:
        """Kahn topological order, authored order among ready nodes; raises on a cycle."""
        remaining = {name: set(self.parent_names(name)) for name in self.nodes}
        order: list[str] = []
        while remaining:
            ready = [name for name, parents in remaining.items() if not parents]
            if not ready:
                raise ValueError(f"{self.schema_id}: cycle among {sorted(remaining)}")
            for name in ready:
                order.append(name)
                del remaining[name]
            for parents in remaining.values():
                parents.difference_update(ready)
        return order

    @property
    def domain(self) -> str:
        """Model-facing subject line: the title in lower case when placed, else ``schema_id``."""
        return self.schema_id if self.title is None else self.title.lower()

    @property
    def columns(self) -> list[tuple[str, str]]:
        return [(name, node.kind) for name, node in self.nodes.items() if name != self.target and not node.hidden]  # fmt: skip

    @property
    def column_descriptions(self) -> dict[str, str]:
        return {name: node.desc for name, node in self.nodes.items() if not node.hidden}

    @property
    def levels(self) -> dict[str, list[str]]:
        out = {
            name: list(node.levels) for name, node in self.nodes.items() if node.levels is not None
        }
        target_node = self.nodes[self.target]
        if target_node.class_scores:
            out[self.target] = list(target_node.class_scores)
        return out

    def validate(self) -> None:
        """Structural + semantic sanity; raises ValueError naming the first violation."""

        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"{self.schema_id}: {msg}")

        check(bool(self.schema_id.strip()), "schema_id must be non-empty")
        check(self.target in self.nodes, f"target {self.target!r} is not a node")
        check(bool(self.description.strip()), "needs a table description")
        names = set(self.nodes)
        target_node = self.nodes[self.target]
        check(target_node.kind == "categorical", "the target must be categorical")
        check(
            bool(target_node.parents) or bool(target_node.class_scores),
            "the target needs parents (or class_scores)",
        )
        for name in self.nodes:  # before topo: an unknown reference is not a "cycle"
            for parent in self.parent_names(name):
                check(parent in self.nodes,
                      f"{name!r}: unknown parent or formula input {parent!r}")  # fmt: skip
        self.topo_order()  # raises on cycles
        check(not self.children()[self.target], "the target must have no children (label leakage)")

        for name, node in self.nodes.items():
            check(node.kind in ("numeric", "categorical"), f"{name!r}: bad kind {node.kind!r}")
            check(bool(node.desc.strip()), f"{name!r}: every node needs a desc")
            if node.hidden:
                check(node.kind == "numeric" and name != self.target and not node.parents
                      and node.formula is None,
                      f"{name!r}: a hidden node is a numeric root, never the target")  # fmt: skip
                check(node.units is None and node.dist is None and node.scale is None,
                      f"{name!r}: a hidden node is never rendered; author no units, dist or scale")  # fmt: skip
                check(len(self.children()[name]) >= 2,
                      f"{name!r}: a hidden node must feed two or more nodes, or it is noise")  # fmt: skip
                continue
            if node.missing is not None or node.missing_by is not None:
                check(node.missing is not None and 0 < node.missing < 1,
                      f"{name!r}: missing is a share in (0, 1)")  # fmt: skip
                check(name != self.target and node.formula is None,
                      f"{name!r}: the target and derived columns are never blank")  # fmt: skip
                if node.missing_by is not None:
                    by, side = node.missing_by
                    check(by in self.nodes and self.nodes[by].kind == "numeric"
                          and self.nodes[by].formula is None and side in ("high", "low"),
                          f"{name!r}: missing_by names a numeric column and high or low")  # fmt: skip

            if node.formula is not None:
                check(node.kind == "numeric", f"{name!r}: formula nodes must be numeric")
                check(name != self.target, f"{name!r}: the target cannot be a formula")
                check(not node.parents, f"{name!r}: formula inputs are the callable's "
                                        "parameter names; do not author parents")  # fmt: skip
                check(node.units is None and node.dist is None and node.scale is None,
                      f"{name!r}: a formula node's values are fully determined by its "
                      "inputs; do not author units/dist/scale")  # fmt: skip
                check(node.noise_scale is None and node.noise_dist is None,
                      f"{name!r}: formula nodes are deterministic; no noise")  # fmt: skip
                check(not self.children()[name],
                      f"{name!r}: formula nodes are leaves (no children)")  # fmt: skip
                inputs = self._formula_inputs(name)
                check(bool(inputs), f"{name!r}: a formula needs at least one input")
                for feat in inputs:
                    check(self.nodes[feat].kind == "numeric" and self.nodes[feat].formula is None
                          and not self.nodes[feat].hidden,
                          f"{name!r}: formula input {feat!r} must be a numeric, "
                          "non-derived node")  # fmt: skip
                continue

            if node.kind == "numeric":
                check(node.levels is None, f"{name!r}: numeric node with levels")
                check(node.units in UNIT_FAMILIES,
                      f"{name!r}: units must be one of {sorted(UNIT_FAMILIES)}, got {node.units!r}")  # fmt: skip
                allowed = UNIT_FAMILIES[node.units]
                if node.parents:
                    check(node.dist is None or node.dist in allowed,
                          f"{name!r}: dist {node.dist!r} outside units family {node.units!r}")  # fmt: skip
                else:
                    check(node.dist in allowed,
                          f"{name!r}: numeric root needs dist from its units family {allowed}")  # fmt: skip
                ranged = self._range_edge(name) is not None
                if ranged:
                    check(node.scale is None,
                          f"{name!r}: level_ranges supersede scale; author only the ranges")  # fmt: skip
                    check(node.dist is None,
                          f"{name!r}: a hard-ranged column has no pooled marginal: its "
                          "true marginal IS the per-level mixture; do not author dist")  # fmt: skip
                else:
                    check(node.scale is not None,
                          f"{name!r}: every numeric node needs an authored scale=(lo, hi): "
                          "rendered magnitudes are authored, never drawn")  # fmt: skip
                for lo, hi in (
                    [node.scale] if not ranged else self._range_edge(name)[1].level_ranges.values()
                ):
                    check(np.isfinite(lo) and np.isfinite(hi) and lo < hi,
                          f"{name!r}: scale/range must be a finite (lo, hi) with lo < hi")  # fmt: skip
                    if node.units == "percent":
                        check(lo >= 0.0 and hi <= 100.0,
                              f"{name!r}: a percent scale must stay inside [0, 100]")  # fmt: skip
                    if node.units == "signed":
                        check(lo < 0.0 < hi, f"{name!r}: a signed scale must span zero")
                    elif node.units != "score":  # only score may go negative
                        check(lo >= 0.0, f"{name!r}: a {node.units} scale cannot go negative")
                if (
                    node.rounding is not None
                    or node.censor is not None
                    or node.zero_inflation is not None
                ):
                    check(node.formula is None and not ranged,
                          f"{name!r}: artifacts apply to a numeric column with its own scale")  # fmt: skip
                    lo, hi = node.scale
                    if node.rounding is not None:
                        check(0 < node.rounding < hi - lo, f"{name!r}: rounding must be a quantum inside the scale")  # fmt: skip
                    if node.censor is not None:
                        c_lo, c_hi = node.censor
                        check(
                            c_lo is not None or c_hi is not None, f"{name!r}: censor needs a limit"
                        )
                        check((c_lo is None or lo <= c_lo < hi) and (c_hi is None or lo < c_hi <= hi)
                              and (c_lo is None or c_hi is None or c_lo < c_hi),
                              f"{name!r}: censor limits must lie inside the scale")  # fmt: skip
                    if node.zero_inflation is not None:
                        check(0 < node.zero_inflation < 1, f"{name!r}: zero_inflation is a share in (0, 1)")  # fmt: skip
            else:
                check(node.units is None and node.dist is None and node.scale is None,
                      f"{name!r}: categorical node with units/dist/scale")  # fmt: skip
                if node.class_scores:
                    check(node.levels is None,
                          f"{name!r}: class_scores derives levels; do not author both")  # fmt: skip
                else:
                    check(node.levels is not None and len(node.levels) >= 2,
                          f"{name!r}: categorical node needs 2+ levels")  # fmt: skip
                    check(len(set(node.levels)) == len(node.levels),
                          f"{name!r}: duplicate level names")  # fmt: skip
                    check(all(s.strip() for s in node.levels), f"{name!r}: empty level name")
                if node.level_probs is not None:
                    check(list(node.level_probs) == list(node.levels or node.class_scores),
                          f"{name!r}: level_probs keys must equal the levels (or classes), in order")  # fmt: skip
                    probs = list(node.level_probs.values())
                    check(all(p > 0 for p in probs) and abs(sum(probs) - 1.0) < 1e-6,
                          f"{name!r}: level_probs must be positive and sum to 1")  # fmt: skip
            if node.kind == "numeric":
                check(node.level_probs is None, f"{name!r}: level_probs on a numeric node")
            if name == self.target:
                check(node.level_probs is not None,
                      f"{name!r}: the target authors its base rate; write level_probs")  # fmt: skip
            if node.parents:
                check(node.noise_scale in NOISE_SCALES,
                      f"{name!r}: every non-root node needs an authored noise_scale "
                      f"(one of {sorted(NOISE_SCALES)}), got {node.noise_scale!r}")  # fmt: skip
                check(node.noise_dist in NOISE_DISTS,
                      f"{name!r}: every non-root node needs an authored noise_dist "
                      f"(one of {sorted(NOISE_DISTS)}), got {node.noise_dist!r}")  # fmt: skip
            else:
                check(node.noise_scale is None and node.noise_dist is None,
                      f"{name!r}: noise_scale/noise_dist on a node with no structural "
                      "equation (roots have no noise; argmax nodes use per-class score noise)")  # fmt: skip

            edges = self.edges(name)
            for parent, edge in edges.items():
                check(parent != name, f"{name!r}: self-edge")
                parent_node = self.nodes[parent]
                if edge.level_effects is not None:
                    check(parent_node.kind == "categorical",
                          f"{name!r}: level_effects on numeric parent {parent!r}")  # fmt: skip
                    check(not parent_node.ordinal,
                          f"{name!r}: level_effects on ORDINAL parent {parent!r} (use a sign)")  # fmt: skip
                    check(parent_node.levels is not None
                          and list(edge.level_effects) == list(parent_node.levels),
                          f"{name!r}: level_effects keys must equal {parent!r} levels, in order")  # fmt: skip
                else:
                    check(abs(edge.effect) in STRENGTHS,
                          f"{name!r}: edge from {parent!r} effect must be ±1 or ±0.5, "
                          f"got {edge.effect}")  # fmt: skip
                    check(edge.shape in TRANSFORM_OF_SHAPE,
                          f"{name!r}: bad shape {edge.shape!r} from {parent!r}")  # fmt: skip
                    if parent_node.kind == "categorical":
                        check(parent_node.ordinal,
                              f"{name!r}: scalar sign on categorical parent {parent!r} requires "
                              "ordinal=True (else author level_effects)")  # fmt: skip
                if edge.level_slopes is not None:
                    check(edge.level_effects is not None,
                          f"{name!r}: level_slopes ride on a level_effects edge")  # fmt: skip
                    check(list(edge.level_slopes) == list(parent_node.levels or []),
                          f"{name!r}: level_slopes keys must equal {parent!r} levels, in order")  # fmt: skip
                    scaled = edges.get(edge.scaled or "")
                    check(scaled is not None and edge.scaled != parent
                          and scaled.level_effects is None,
                          f"{name!r}: scaled must name another scalar parent of this node")  # fmt: skip
                if edge.level_ranges is not None:
                    check(node.kind == "numeric",
                          f"{name!r}: level_ranges on a categorical child")  # fmt: skip
                    check(edge.level_effects is not None,
                          f"{name!r}: level_ranges require level_effects on the same edge "
                          "(the envelope needs a causal direction)")  # fmt: skip
                    check(list(edge.level_ranges) == list(parent_node.levels or []),
                          f"{name!r}: level_ranges keys must equal {parent!r} levels, in order")  # fmt: skip
                    pairs = [(edge.level_effects[lv], (lo + hi) / 2)
                             for lv, (lo, hi) in edge.level_ranges.items()]  # fmt: skip
                    check(all(m1 <= m2 for (e1, m1), (e2, m2)
                              in zip(sorted(pairs), sorted(pairs)[1:], strict=False)),
                          f"{name!r}: level_ranges order disagrees with level_effects")  # fmt: skip
            check(sum(e.level_ranges is not None for e in edges.values()) <= 1,
                  f"{name!r}: at most one level_ranges edge per child")  # fmt: skip

            for a, b in node.interactions:
                check(a in edges and b in edges and a != b,
                      f"{name!r}: interaction pair ({a!r}, {b!r}) must be two distinct parents")  # fmt: skip
                check(name != self.target or not (self.nodes[a].hidden or self.nodes[b].hidden),
                      f"{name!r}: the target's interaction pair ({a!r}, {b!r}) must be two rendered columns")  # fmt: skip
            check(node.mechanism in MECHANISMS, f"{name!r}: bad mechanism {node.mechanism!r}")
            if node.mechanism != "additive":
                check(len(edges) >= 2 and not node.class_scores,
                      f"{name!r}: a {node.mechanism} mechanism combines two or more parents")  # fmt: skip
            if node.mechanism == "rules":
                check(all(e.shape == "threshold" for e in edges.values()
                          if e.level_effects is None),
                      f"{name!r}: a rules mechanism has threshold edges only")  # fmt: skip
            if node.class_scores:
                check(name == self.target, f"{name!r}: class_scores is target-only")
                check(not node.parents,
                      f"{name!r}: an argmax node's parents live in class_scores")  # fmt: skip
                check(len(node.class_scores) >= 3, f"{name!r}: a nominal target needs 3+ classes")
                check(any(feats for feats in node.class_scores.values()),
                      f"{name!r}: class_scores must give at least one class a driving feature")  # fmt: skip
                for cls, feats in node.class_scores.items():
                    for feat, sign in feats:
                        check(feat in names and self.nodes[feat].kind == "numeric",
                              f"{name!r}: class driver {feat!r} of {cls!r} must be a numeric node")  # fmt: skip
                        check(sign in (-1, 1), f"{name!r}: class driver sign {sign} for {feat!r}")
