"""
Synthetic time-series data
"""

import math

import numpy as np
import pandas as pd


def calendar_aware_timestamps(
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp,
    num_rows: int,
    candidate_factor: int = 5,
) -> pd.DatetimeIndex:
    """Sample sorted timestamps weighted by per-call weekday weights and a
    fixed business-hours envelope."""
    weekday_w = np.empty(7, dtype=np.float64)
    weekday_w[:5] = np.random.uniform(0.4, 1.0, size=5)
    weekday_w[5:] = np.random.uniform(0.1, 0.5, size=2)

    hour_w = np.full(24, 0.05, dtype=np.float64)
    hour_w[7:9] = 0.5
    hour_w[9:18] = 1.0
    hour_w[18:22] = 0.5
    hour_w[22:] = 0.1

    span_seconds = (max_ts - min_ts).total_seconds()
    n_cand = max(num_rows * candidate_factor, num_rows + 1)
    offsets_s = np.random.uniform(0.0, span_seconds, size=n_cand)
    ts = pd.DatetimeIndex(min_ts + pd.to_timedelta(offsets_s, unit="s"))
    weights = weekday_w[ts.weekday] * hour_w[ts.hour]
    weights = weights / weights.sum()
    sampled = np.random.choice(n_cand, size=num_rows, replace=False, p=weights)
    return ts[sampled].sort_values()


class Cycle:
    def __init__(self, min_value: float, max_value: float, frequency: int, scale: float):
        self.min_value = min_value
        self.max_value = max_value
        self.frequency = frequency
        self.scale = scale

    def get_value(self, row_idx: int) -> float:
        x = (row_idx / self.frequency) * math.pi
        val = self.scale * math.sin(x)
        return min(max(val, self.min_value), self.max_value)


class Trend:
    def __init__(
        self,
        num_points: int,
        min_value: float,
        max_value: float,
        alpha: float,
        scale: float,
        clip_max: bool = True,
    ):
        """For ``alpha >= 5`` the trend is boosted by ``exp((alpha - 5) * x)``
        and the ``max_value`` clip is released."""
        self.num_points = num_points
        self.min_value = min_value
        self.max_value = max_value
        self.alpha = alpha
        self.scale = scale
        self.clip_max = clip_max

    def get_value(self, row_idx: int) -> float:
        x = row_idx / self.num_points
        boost = math.exp(max(0.0, self.alpha - 5.0) * x)
        value = self.scale * math.pow(x, self.alpha) * boost + self.min_value
        if self.alpha < 5.0 and self.clip_max:
            return min(value, self.max_value)
        return value


class TSDataGenerator:
    def __init__(
        self,
        num_points: int,
        min_value: float,
        max_value: float,
        trend_alpha: float,
        trend_scale: float,
        cycle_frequency: float,
        cycle_scale: float,
        noise_scale: float = 0.05,
        ar_rho: float = 0.0,
    ):
        assert cycle_frequency <= num_points
        assert 0.0 <= ar_rho < 1.0
        self.num_points = num_points
        self.min_value = min_value
        self.max_value = max_value
        self.trend = Trend(
            num_points=num_points,
            min_value=min_value,
            max_value=max_value,
            alpha=trend_alpha,
            scale=trend_scale,
        )
        self.cycle = Cycle(
            min_value=min_value,
            max_value=max_value,
            frequency=cycle_frequency,
            scale=cycle_scale,
        )
        self.noise_scale = noise_scale
        self.ar_rho = ar_rho
        self._noise_state = 0.0

    def _get_noise_val(self):
        self._noise_state = self.ar_rho * self._noise_state + np.random.randn() * self.noise_scale
        return min(max(self._noise_state, self.min_value), self.max_value)

    def get_value(self, row_idx):
        trend_val = self.trend.get_value(row_idx=row_idx)
        cycle_val = self.cycle.get_value(row_idx=row_idx)
        noise_val = self._get_noise_val()
        return (trend_val + cycle_val + noise_val) / 3


class CategoricalTSDataGenerator:
    def __init__(self, ts_data_gens: list[TSDataGenerator]):
        self.ts_data_gens = ts_data_gens

    def get_value(self, row_idx):
        vals = [ts_data_gen.get_value(row_idx=row_idx) for ts_data_gen in self.ts_data_gens]
        vals = np.array(vals)
        exp_vals = np.exp(vals - np.max(vals))  # stable exponent
        probs = exp_vals / exp_vals.sum()
        category_idx = np.random.choice(len(vals), p=probs)
        return int(category_idx)


class UniformSourceGenerator:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def get_value(self, row_idx: int) -> float:
        return float(np.random.uniform(self.low, self.high))


class GaussianSourceGenerator:
    def __init__(self, mean: float, std: float, low: float, high: float):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

    def get_value(self, row_idx: int) -> float:
        return float(np.clip(np.random.normal(self.mean, self.std), self.low, self.high))


class BetaSourceGenerator:
    def __init__(self, alpha: float, beta: float, scale: float, offset: float):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.offset = offset

    def get_value(self, row_idx: int) -> float:
        return float(np.random.beta(self.alpha, self.beta) * self.scale + self.offset)


class MixedSourceGenerator:
    def __init__(self, generators: list):
        self.generators = generators

    def get_value(self, row_idx: int) -> float:
        gen = self.generators[np.random.randint(len(self.generators))]
        return gen.get_value(row_idx)


class IIDCategoricalGenerator:
    def __init__(self, num_categories: int):
        self.probs = np.random.dirichlet(np.ones(num_categories))
        self.num_categories = num_categories

    def get_value(self, row_idx: int) -> int:
        return int(np.random.choice(self.num_categories, p=self.probs))


class LogNormalSourceGenerator:
    def __init__(self, mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma

    def get_value(self, row_idx: int) -> float:
        return float(np.random.lognormal(mean=self.mean, sigma=self.sigma))


class ExponentialSourceGenerator:
    def __init__(self, scale: float):
        self.scale = scale

    def get_value(self, row_idx: int) -> float:
        return float(np.random.exponential(scale=self.scale))


class ParetoSourceGenerator:
    def __init__(self, alpha: float, scale: float):
        self.alpha = alpha
        self.scale = scale

    def get_value(self, row_idx: int) -> float:
        return float(np.random.pareto(a=self.alpha) * self.scale)


class PoissonSourceGenerator:
    def __init__(self, lam: float):
        self.lam = lam

    def get_value(self, row_idx: int) -> float:
        return float(np.random.poisson(lam=self.lam))
