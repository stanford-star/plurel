from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from plurel.distributions import Calendar, Distribution
from plurel.mechanisms import bin_levels

Kind = Literal["numeric", "categorical"]
Binning = Literal["normal", "empirical"] | tuple[float, ...]

DEFAULT_CALENDAR = Calendar(pd.Timestamp("1990-01-01"), pd.Timestamp("2025-01-01"))


def rank_map(latent: np.ndarray, marginal: Distribution, rng: np.random.Generator) -> np.ndarray:
    reference = np.sort(marginal.sample(len(latent), rng))
    return reference[np.searchsorted(np.sort(latent), latent)]


@dataclass(frozen=True)
class Column:
    node: str
    kind: Kind = "numeric"
    dims: int | tuple[int, ...] | None = None
    categories: tuple[object, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    binning: Binning = "normal"
    marginal: Distribution | None = None
    missing: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("numeric", "categorical"):
            raise ValueError(f"unknown column kind {self.kind!r}")
        if not 0.0 <= self.missing < 1.0:
            raise ValueError("missing must be in [0, 1)")
        if self.kind == "numeric":
            if self.categories is not None or self.probabilities is not None:
                raise ValueError("categories and probabilities belong to categorical columns")
            return
        if self.marginal is not None:
            raise ValueError("categorical columns have no marginal")
        if (
            self.categories is None
            or len(set(self.categories)) != len(self.categories)
            or len(self.categories) < 2
        ):
            raise ValueError("categorical columns need at least two distinct categories")
        probabilities = self.category_probabilities
        if len(probabilities) != len(self.categories):
            raise ValueError("one probability per category")
        if min(probabilities) <= 0 or not np.isclose(sum(probabilities), 1.0):
            raise ValueError("probabilities must be positive and sum to one")
        if isinstance(self.binning, tuple):
            if len(self.binning) != len(self.categories) - 1 or list(self.binning) != sorted(
                self.binning
            ):
                raise ValueError("edges are one fewer than the categories, ascending")
        elif self.binning not in ("normal", "empirical"):
            raise ValueError(f"unknown binning {self.binning!r}")

    @property
    def category_probabilities(self) -> tuple[float, ...]:
        if self.categories is None:
            return ()
        return self.probabilities or (1.0 / len(self.categories),) * len(self.categories)

    def observe(self, latent: np.ndarray, rng: np.random.Generator) -> pd.Series:
        latent = latent if self.dims is None else latent[:, self.dims]
        if self.kind == "categorical":
            observed = pd.Categorical.from_codes(self.codes(latent), list(self.categories))
        else:
            if latent.ndim == 2 and latent.shape[1] != 1:
                raise ValueError(f"numeric column needs one dimension of {self.node!r}")
            flat = latent.reshape(len(latent))
            observed = rank_map(flat, self.marginal, rng) if self.marginal else flat
            if isinstance(self.marginal, Calendar):
                observed = pd.to_datetime(observed, unit="s")
        series = pd.Series(observed)
        return series.mask(rng.random(len(series)) < self.missing) if self.missing else series

    def codes(self, latent: np.ndarray) -> np.ndarray:
        if latent.ndim == 2 and latent.shape[1] > 1:
            if latent.shape[1] != len(self.categories):
                raise ValueError(f"one-hot node {self.node!r} must have one column per category")
            return latent.argmax(1)
        flat = latent.reshape(len(latent))
        if isinstance(self.binning, tuple):
            return np.digitize(flat, self.binning)
        if self.binning == "empirical":
            cuts = np.cumsum(self.category_probabilities)[:-1]
            return np.digitize(flat, np.quantile(flat, cuts))
        return bin_levels(flat, self.category_probabilities)
