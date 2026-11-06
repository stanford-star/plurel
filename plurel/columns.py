from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from plurel.distributions import Calendar, Distribution
from plurel.graph import Node, bin_levels, level_edges, standardize

Kind = Literal["numeric", "categorical", "timestamp"]

DEFAULT_CALENDAR = Calendar(pd.Timestamp("1990-01-01"), pd.Timestamp("2025-01-01"))


def rank_map(latent: np.ndarray, marginal: Distribution, rng: np.random.Generator) -> np.ndarray:
    """The marginal's values in the latent's order, ties broken at random, so a column holds
    exactly its draw of the marginal."""
    out = np.empty(len(latent))
    reference = np.sort(marginal.sample(len(latent), rng))
    out[np.lexsort((rng.random(len(latent)), latent))] = reference
    return out


@dataclass(frozen=True)
class Column:
    node: Node | None = None
    kind: Kind | None = None
    categories: tuple[object, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    binning: tuple[float, ...] | None = None
    marginal: Distribution | None = None
    missing: float | str = 0.0
    clip: tuple[float, float] | None = None
    round: float | None = None
    values: pd.Series | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.kind is None:
            kind = "categorical" if self.categories is not None else "numeric"
            object.__setattr__(self, "kind", kind)
        if self.kind not in ("numeric", "categorical", "timestamp"):
            raise ValueError(f"unknown column kind {self.kind!r}")
        if self.kind != "numeric" and (self.clip, self.round) != (None, None):
            raise ValueError("clip and round are for numeric columns")
        if self.clip is not None and self.clip[0] >= self.clip[1]:
            raise ValueError("clip is a low and a higher high")
        if self.round is not None and self.round <= 0.0:
            raise ValueError("round is a positive quantum")
        if not isinstance(self.node, Node):
            raise ValueError("a column holds its node")
        if self.values is not None:
            return
        if not isinstance(self.missing, str) and not 0.0 <= self.missing < 1.0:
            raise ValueError("missing must be a rate in [0, 1) or the name of a two-class node")
        if self.kind != "categorical":
            if (self.categories, self.probabilities, self.binning) != (None, None, None):
                raise ValueError(
                    "categories, probabilities and binning are for categorical columns"
                )
            if self.node.dim != 1:
                raise ValueError(f"a {self.kind} column needs a one-dimensional node")
            return
        if self.marginal is not None:
            raise ValueError("categorical columns have no marginal")
        if (
            self.categories is None
            or len(set(self.categories)) != len(self.categories)
            or len(self.categories) < 2
        ):
            raise ValueError("categorical columns need at least two distinct categories")
        if self.node.dim not in (1, len(self.categories)):
            raise ValueError("a categorical column needs one dimension or one per category")
        if self.node.dim > 1 and (self.probabilities is not None or self.binning is not None):
            raise ValueError("a one-hot node sets its own shares: no probabilities or binning")
        probabilities = self.category_probabilities
        if len(probabilities) != len(self.categories):
            raise ValueError("one probability per category")
        if min(probabilities) <= 0 or not np.isclose(sum(probabilities), 1.0):
            raise ValueError("probabilities must be positive and sum to one")
        if self.binning is not None and (
            len(self.binning) != len(self.categories) - 1
            or list(self.binning) != sorted(self.binning)
        ):
            raise ValueError("edges are one fewer than the categories, ascending")

    @property
    def category_probabilities(self) -> tuple[float, ...]:
        if self.categories is None:
            return ()
        return self.probabilities or (1.0 / len(self.categories),) * len(self.categories)

    @property
    def real_latent(self) -> np.ndarray:
        """Given values as a latent: a level's index, or a standardized rank with a blank at zero."""
        if self.kind == "categorical":
            codes = pd.Categorical(self.values, list(self.categories)).codes
            return codes.astype(float)[:, None]
        ranks = standardize(self.values.rank().to_numpy(dtype=float)[:, None])
        return np.nan_to_num(ranks)

    def observe(
        self, name: str, latents: dict[str, np.ndarray], rng: np.random.Generator, n: int
    ) -> pd.Series:
        if self.values is not None:
            return self.values
        latent = latents[name]
        if self.kind == "categorical":
            observed = pd.Categorical.from_codes(self.codes(latent), list(self.categories))
        else:
            flat = latent[:, 0]
            observed = rank_map(flat, self.marginal, rng) if self.marginal else flat
            if self.round is not None:
                observed = np.round(observed / self.round) * self.round
            if self.clip is not None:
                observed = np.clip(observed, *self.clip)
            if self.kind == "timestamp":
                observed = pd.to_datetime(observed, unit="s").as_unit("ns")
        series = pd.Series(observed)
        if isinstance(self.missing, str):
            return series.mask(latents[self.missing][:, 1] == 1.0)
        return series.mask(rng.random(len(series)) < self.missing) if self.missing else series

    def codes(self, latent: np.ndarray) -> np.ndarray:
        if latent.shape[1] > 1:
            return np.where((latent != 0).any(1), latent.argmax(1), -1)
        flat = latent[:, 0]
        edges = self.binning
        if edges is None:
            edges = level_edges(flat, self.category_probabilities)
        return bin_levels(flat, edges)
