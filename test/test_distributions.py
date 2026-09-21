import numpy as np
import pandas as pd
import pytest

from plurel.distributions import (
    DISTRIBUTIONS,
    AutoRegressive,
    Beta,
    Calendar,
    Cycle,
    Distribution,
    Exponential,
    Gumbel,
    LogNormal,
    Mixture,
    Normal,
    Pareto,
    Poisson,
    TimeSeries,
    Trend,
    Uniform,
)

START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2021-01-01")

EXAMPLES = {
    "normal": Normal(),
    "uniform": Uniform(),
    "beta": Beta(2.0, 3.0, low=-1.0, high=4.0),
    "lognormal": LogNormal(),
    "exponential": Exponential(),
    "pareto": Pareto(),
    "poisson": Poisson(3.0),
    "gumbel": Gumbel(),
    "mixture": Mixture((Normal(-3.0, 0.1), Normal(3.0, 0.1)), (0.25, 0.75)),
    "time_series": TimeSeries(noise=AutoRegressive(0.5, 0.1)),
    "calendar": Calendar(START, END),
}


def test_every_registered_distribution_has_an_example():
    assert set(EXAMPLES) == set(DISTRIBUTIONS)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_sample_shape_dtype_and_determinism(name):
    distribution = EXAMPLES[name]
    assert isinstance(distribution, Distribution)
    first = distribution.sample(50, np.random.default_rng(0))
    second = distribution.sample(50, np.random.default_rng(0))
    assert first.shape == (50,)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, second)


def test_mixture_follows_its_weights():
    values = EXAMPLES["mixture"].sample(20_000, np.random.default_rng(2))
    assert np.mean(values > 0) == pytest.approx(0.75, abs=0.02)


def test_autoregressive_is_persistent():
    values = AutoRegressive(0.9, 1.0).sample(20_000, np.random.default_rng(4))
    assert np.corrcoef(values[:-1], values[1:])[0, 1] == pytest.approx(0.9, abs=0.02)
    for rho, n in ((0.0, 50), (0.5, 7), (0.999, 3000), (1e-6, 1000), (0.9, 20_001)):
        expected = np.random.default_rng(4).normal(0.0, 2.0, n)
        for index in range(1, n):
            expected[index] += rho * expected[index - 1]
        actual = AutoRegressive(rho, 2.0).sample(n, np.random.default_rng(4))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_time_series_is_the_sum_of_its_parts():
    t = np.linspace(0.0, 1.0, 100)
    series = TimeSeries(Trend(2.0, 3.0), Cycle(2.0, 0.5), AutoRegressive(0.0, 0.0))
    expected = Trend(2.0, 3.0).values(t) + Cycle(2.0, 0.5).values(t)
    np.testing.assert_allclose(series.sample(100, np.random.default_rng(0)), expected)


def test_calendar_is_sorted_within_range_and_honors_zero_weights():
    weekday = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    hour = tuple(1.0 if 9 <= h < 17 else 0.0 for h in range(24))
    values = Calendar(START, END, weekday, hour).sample(500, np.random.default_rng(5))
    stamps = pd.to_datetime(values, unit="s")
    assert np.all(np.diff(values) >= 0)
    assert stamps.min() >= START and stamps.max() <= END
    assert set(stamps.weekday) <= {0, 1, 2, 3, 4}
    assert set(stamps.hour) <= set(range(9, 17))
    early = Calendar(pd.Timestamp("1965-01-01"), pd.Timestamp("1966-01-01"), weekday, hour)
    stamps = pd.to_datetime(early.sample(300, np.random.default_rng(6)), unit="s")
    assert set(stamps.weekday) <= {0, 1, 2, 3, 4} and set(stamps.hour) <= set(range(9, 17))
