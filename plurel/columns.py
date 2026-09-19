from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from plurel.distributions import Calendar, Distribution
from plurel.mechanisms import bin_levels

Kind = Literal["numeric", "categorical", "timestamp"]
Binning = Literal["normal", "empirical"]

DEFAULT_CALENDAR = Calendar(pd.Timestamp("1990-01-01"), pd.Timestamp("2025-01-01"))


def rank_map(latent: np.ndarray, marginal: Distribution, rng: np.random.Generator) -> np.ndarray:
    ranks = np.argsort(np.argsort(latent, kind="stable"), kind="stable")
    return np.sort(marginal.sample(len(latent), rng))[ranks]


@dataclass(frozen=True)
class Column:
    node: str
    kind: Kind = "numeric"
    dims: int | tuple[int, ...] | None = None
    categories: tuple[object, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    binning: Binning = "normal"
    edges: tuple[float, ...] | None = None
    marginal: Distribution | None = None
    missing: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("numeric", "categorical", "timestamp"):
            raise ValueError(f"unknown column kind {self.kind!r}")
        if not 0.0 <= self.missing < 1.0:
            raise ValueError("missing must be in [0, 1)")
        if self.kind != "categorical":
            if (self.categories, self.probabilities, self.edges) != (None, None, None):
                raise ValueError(
                    "categories, probabilities and edges belong to categorical columns"
                )
            return
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
        if self.edges is not None and len(self.edges) != len(self.categories) - 1:
            raise ValueError("edges are one fewer than the categories")
        if self.binning not in ("normal", "empirical"):
            raise ValueError(f"unknown binning {self.binning!r}")
        if self.marginal is not None:
            raise ValueError("categorical columns have no marginal")

    @property
    def category_probabilities(self) -> tuple[float, ...]:
        if self.categories is None:
            return ()
        return self.probabilities or (1.0 / len(self.categories),) * len(self.categories)

    def observe(self, values: np.ndarray, rng: np.random.Generator) -> pd.Series:
        values = values if self.dims is None else values[:, self.dims]
        if self.kind == "categorical":
            observed = np.asarray(self.categories)[self.codes(values)]
        else:
            if values.ndim == 2 and values.shape[1] != 1:
                raise ValueError(f"{self.kind} column needs one dimension of {self.node!r}")
            latent = values.reshape(len(values))
            marginal = self.marginal or (DEFAULT_CALENDAR if self.kind == "timestamp" else None)
            observed = rank_map(latent, marginal, rng) if marginal else latent
            if self.kind == "timestamp":
                observed = pd.to_datetime(observed, unit="s")
        series = pd.Series(observed)
        return series.mask(rng.random(len(series)) < self.missing) if self.missing else series

    def codes(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 2 and values.shape[1] > 1:
            return values.argmax(1)
        latent = values.reshape(len(values))
        if self.edges is not None:
            return np.digitize(latent, self.edges)
        if self.binning == "empirical":
            cuts = np.cumsum(self.category_probabilities)[:-1]
            return np.digitize(latent, np.quantile(latent, cuts))
        return bin_levels(latent, self.category_probabilities)
