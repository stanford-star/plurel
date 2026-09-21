from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from plurel.distributions import Calendar, Distribution
from plurel.mechanisms import bin_levels

Kind = Literal["numeric", "categorical", "timestamp", "key"]
Binning = Literal["normal", "empirical"] | tuple[float, ...]

DEFAULT_CALENDAR = Calendar(pd.Timestamp("1990-01-01"), pd.Timestamp("2025-01-01"))


def rank_map(latent: np.ndarray, marginal: Distribution, rng: np.random.Generator) -> np.ndarray:
    known = ~np.isnan(latent)
    reference = np.sort(marginal.sample(int(known.sum()), rng))
    out = np.full(len(latent), np.nan)
    out[known] = reference[np.searchsorted(np.sort(latent[known]), latent[known])]
    return out


@dataclass(frozen=True)
class Column:
    node: str | None = None
    kind: Kind = "numeric"
    dims: int | tuple[int, ...] | None = None
    categories: tuple[object, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    binning: Binning = "normal"
    marginal: Distribution | None = None
    missing: float | str = 0.0
    after: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("numeric", "categorical", "timestamp", "key"):
            raise ValueError(f"unknown column kind {self.kind!r}")
        if self.kind == "key":
            options = (
                self.node,
                self.dims,
                self.categories,
                self.probabilities,
                self.marginal,
                self.after,
            )
            if (
                any(option is not None for option in options)
                or self.missing
                or self.binning != "normal"
            ):
                raise ValueError("a key column has no node and no options")
            return
        if self.node is None:
            raise ValueError("a column observes a node")
        if not isinstance(self.missing, str) and not 0.0 <= self.missing < 1.0:
            raise ValueError("missing must be a rate in [0, 1) or the name of a two-class node")
        if self.after is not None and self.kind != "timestamp":
            raise ValueError("only a timestamp column comes after another")
        if self.kind != "categorical":
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

    def observe(
        self, latents: dict[str, np.ndarray], rng: np.random.Generator, n: int
    ) -> pd.Series:
        if self.kind == "key":
            return pd.Series(np.arange(n))
        latent = latents[self.node] if self.dims is None else latents[self.node][:, self.dims]
        if self.kind == "categorical":
            observed = pd.Categorical.from_codes(self.codes(latent), list(self.categories))
        else:
            if latent.ndim == 2 and latent.shape[1] != 1:
                raise ValueError(f"{self.kind} column needs one dimension of {self.node!r}")
            flat = latent.reshape(len(latent))
            observed = rank_map(flat, self.marginal, rng) if self.marginal else flat
            if self.kind == "timestamp":
                observed = pd.to_datetime(observed, unit="s").as_unit("ns")
        series = pd.Series(observed)
        if isinstance(self.missing, str):
            indicator = latents[self.missing]
            if indicator.shape[1] != 2:
                raise ValueError(f"missingness node {self.missing!r} must have two classes")
            return series.mask(indicator[:, 1] == 1.0)
        return series.mask(rng.random(len(series)) < self.missing) if self.missing else series

    def codes(self, latent: np.ndarray) -> np.ndarray:
        if latent.ndim == 2 and latent.shape[1] > 1:
            if latent.shape[1] != len(self.categories):
                raise ValueError(f"one-hot node {self.node!r} must have one column per category")
            valid = (latent != 0).any(1) & ~np.isnan(latent).any(1)
            return np.where(valid, latent.argmax(1), -1)
        flat = latent.reshape(len(latent))
        if not len(flat):
            return np.empty(0, dtype=int)
        if isinstance(self.binning, tuple):
            codes = np.digitize(flat, self.binning)
        elif self.binning == "empirical":
            cuts = np.cumsum(self.category_probabilities)[:-1]
            codes = np.digitize(flat, np.nanquantile(flat, cuts))
        else:
            codes = bin_levels(flat, self.category_probabilities)
        return np.where(np.isnan(flat), -1, codes)
