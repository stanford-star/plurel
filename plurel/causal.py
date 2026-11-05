"""Reads of a plurel :class:`SCM` of one table: its draws (with interventions), the node a column
observes, the categorical levels and their order, and a one-hot target's class scores. A column
observes a node, several columns may observe one, and a node without a column is hidden; queries
take column names, interventions are on nodes."""

from __future__ import annotations

import numpy as np

from plurel.graph import REDUCTIONS
from plurel.random import Seed
from plurel.schema import SCM, Schema

TABLE = "table"


def sample(plan: SCM, n: int, *, seed: Seed = None, interventions: dict | None = None):
    """``(frame, latents)`` of ``n`` rows of the one-table schema, ``interventions`` on nodes."""
    forced = {TABLE: interventions} if interventions else None
    frames, latents = Schema({TABLE: plan}).sample_with_latents(
        {TABLE: n}, seed=seed, interventions=forced
    )
    return frames[TABLE], latents[TABLE]


def node_of(plan: SCM, name: str) -> str:
    """The node column ``name`` observes."""
    return plan.columns[name].node


def levels(plan: SCM) -> dict[str, list[str]]:
    """String levels per categorical column; integer-coded columns are absent."""
    return {
        name: list(column.categories)
        for name, column in plan.columns.items()
        if column.kind == "categorical"
        and all(isinstance(level, str) for level in column.categories)
    }


def category_order(plan: SCM) -> dict[str, list[str]]:
    """``{categorical column -> levels, low to high}`` in latent order: authored names where
    present (index-aligned with the quantile bins), else the integer bin codes."""
    named = levels(plan)
    return {
        name: named.get(name) or [str(i) for i in range(len(column.categories))]
        for name, column in plan.columns.items()
        if column.kind == "categorical"
    }


def class_scores(plan: SCM, target: str, n: int, seed: Seed = None) -> np.ndarray:
    """The noiseless ``(n, classes)`` score matrix of the one-hot node ``target`` observes on one
    draw of its parents: the exact read of which class each parent profile favors."""
    node = plan.nodes[node_of(plan, target)]
    if not node.onehot:
        raise ValueError(f"{target!r} does not observe a one-hot node")
    latents = sample(plan, n, seed=seed)[1]
    return REDUCTIONS[node.op]([edge.apply(latents[edge.parent]) for edge in node.edges])
