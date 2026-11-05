"""Marginal shape classes for authored numeric columns.

A shape class (symmetric, right, left, bimodal, multimodal, counts, zero_inflated) is what a
schema authors for a numeric column; :func:`shape` turns one into a plurel distribution, its
family and parameters drawn per call, so two realizations of the same class still differ in tail
weight, spread and texture."""

from __future__ import annotations

import numpy as np

from plurel import (
    Affine,
    Distribution,
    Exponential,
    Gamma,
    Gumbel,
    Laplace,
    LogNormal,
    Mixture,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    StudentT,
    Uniform,
)

SHAPE_KINDS = ("symmetric", "right", "left", "bimodal", "multimodal", "counts", "zero_inflated")


def _pick(rng: np.random.Generator, *families):
    return families[int(rng.integers(len(families)))]()


def shape(kind: str, rng: np.random.Generator) -> Distribution:
    """One plurel distribution of shape class ``kind``, its family and parameters drawn."""
    u = rng.uniform
    if kind == "symmetric":
        return _pick(
            rng,
            lambda: Normal(0.0, u(0.3, 4.0)),
            lambda: Uniform(-(w := u(0.5, 6.0)), w),
            lambda: Laplace(0.0, u(0.4, 2.5)),
            lambda: StudentT(u(2.2, 15.0), u(0.4, 1.8)),
        )
    if kind in ("right", "left"):
        tail = _pick(
            rng,
            lambda: LogNormal(0.0, u(0.4, 2.5)),
            lambda: Gamma(u(1.1, 5.0), u(0.4, 2.5)),
            lambda: Exponential(u(0.4, 2.5)),
            lambda: Pareto(u(1.2, 6.0)),
            lambda: Gumbel(0.0, u(0.4, 2.0)),
        )
        return Affine(tail, scale=-1.0) if kind == "left" else tail
    if kind in ("bimodal", "multimodal"):
        # unequal component weights and per-component spreads, jittered centers: real mixtures
        # are a dominant mode plus minorities, not evenly spaced equal humps
        modes = 2 if kind == "bimodal" else int(rng.integers(3, 6))
        span = u(2.5, 7.0)
        centers = np.linspace(-span / 2, span / 2, modes) + u(-0.15, 0.15, modes) * span
        weights = np.clip(rng.dirichlet(np.full(modes, u(0.8, 4.0))), 0.05, None)
        spreads = u(0.25, 1.2, modes)
        components = tuple(
            Normal(float(c), float(s)) for c, s in zip(centers, spreads, strict=True)
        )
        return Mixture(components, tuple(float(w) for w in weights / weights.sum()))
    if kind == "counts":
        # integer-valued measured quantities (events, items): right-skewed, tie-bearing,
        # non-negative, the dominant real numeric column shape
        return _pick(
            rng,
            lambda: Poisson(u(0.5, 80.0)),
            lambda: NegativeBinomial(u(0.8, 8.0), u(0.1, 0.8)),
        )
    if kind == "zero_inflated":
        # a point mass at zero plus a right tail (spend, claims, counts of rare events)
        zeros = float(u(0.15, 0.85))
        tail = _pick(
            rng,
            lambda: Exponential(u(0.4, 2.5)),
            lambda: LogNormal(0.0, u(0.4, 1.5)),
            lambda: Gamma(u(1.0, 4.0), u(0.4, 2.0)),
        )
        return Mixture((Normal(0.0, 0.0), tail), (zeros, 1.0 - zeros))
    raise ValueError(f"unknown shape {kind!r}; expected one of {SHAPE_KINDS}")
