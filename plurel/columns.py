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
    clip: tuple[float | None, float | None] | None = None
    round: float | None = None
    values: pd.Series | None = field(default=None, compare=False)
    ordered: bool = True
    missing_by: tuple[str, str] | None = None
    zeros: tuple[float, float] | None = None
    fit_shares: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind is None:
            kind = "categorical" if self.categories is not None else "numeric"
            object.__setattr__(self, "kind", kind)
        if self.kind not in ("numeric", "categorical", "timestamp"):
            raise ValueError(f"unknown column kind {self.kind!r}")
        if self.kind != "numeric" and (self.clip, self.round, self.zeros) != (None, None, None):
            raise ValueError("clip, round and zeros are for numeric columns")
        if self.clip is not None:
            low, high = self.clip
            if (low is None and high is None) or (None not in self.clip and not low < high):
                raise ValueError("clip is a low and a higher high; one may be None")
        if self.zeros is not None and not 0.0 < self.zeros[0] < 1.0:
            raise ValueError("zeros is a share of the rows in (0, 1)")
        if self.missing_by is not None:
            if self.missing_by[1] not in ("high", "low"):
                raise ValueError("missing_by names a column and high or low")
            if isinstance(self.missing, str) or not 0.0 < self.missing < 1.0:
                raise ValueError("missing_by needs a missing rate in (0, 1)")
        if self.round is not None and not 0.0 < self.round < np.inf:
            raise ValueError("round is a positive step: 1 for whole numbers, 0.01 for cents")
        if not isinstance(self.node, Node):
            raise ValueError("a column holds its node")
        if self.marginal is not None and not hasattr(self.marginal, "sample"):
            raise ValueError(f"marginal must be a distribution, not {self.marginal!r}")
        if self.values is not None:
            object.__setattr__(self, "values", self.values.reset_index(drop=True))
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
        if self.fit_shares is not None and (
            len(self.fit_shares) != len(self.categories)
            or min(self.fit_shares) <= 0
            or not np.isclose(sum(self.fit_shares), 1.0)
        ):
            raise ValueError("one positive share per level, summing to one")
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
        ranks = self.values.rank()
        ranks = ranks.fillna(ranks.mean()).fillna(0.0).to_numpy(dtype=float)
        return standardize(ranks[:, None])

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
            observed = self.shaped(flat, rng)
            if self.kind == "timestamp":
                observed = pd.to_datetime(observed, unit="s").as_unit("ns")
        series = pd.Series(observed)
        if isinstance(self.missing, str):
            return series.mask(latents[self.missing][:, 1] == 1.0)
        if self.missing_by is not None:
            by, side = self.missing_by
            ranks = pd.Series(latents[by][:, 0]).rank(pct=True).to_numpy()
            ranks = ranks if side == "high" else 1.0 - ranks
            rate = self.missing
            chance = 2.0 * rate * ranks if rate <= 0.5 else 1.0 - 2.0 * (1.0 - rate) * (1.0 - ranks)
            return series.mask(rng.random(len(series)) < chance)
        return series.mask(rng.random(len(series)) < self.missing) if self.missing else series

    def shaped(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A flat latent as the column's values: the marginal's in its order, then what the record
        does to them: a share at the floor, the step and the limits."""
        out = rank_map(latent, self.marginal, rng) if self.marginal else latent
        if self.zeros is not None:
            out = np.where(rng.random(len(out)) < self.zeros[0], self.zeros[1], out)
        if self.round is not None:
            out = np.round(np.round(out / self.round) * self.round, 10)
        return (out if self.clip is None else np.clip(out, *self.clip)) + 0.0  # no -0.0 on record

    def codes(self, latent: np.ndarray) -> np.ndarray:
        if latent.shape[1] > 1:
            return np.where((latent != 0).any(1), latent.argmax(1), -1)
        flat = latent[:, 0]
        edges = self.binning
        if edges is None:
            edges = level_edges(flat, self.category_probabilities)
        return bin_levels(flat, edges)
