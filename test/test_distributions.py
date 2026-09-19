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


def test_bounded_families_respect_their_bounds():
    rng = np.random.default_rng(1)
    beta = Beta(2.0, 3.0, low=-1.0, high=4.0).sample(2000, rng)
    uniform = Uniform(2.0, 5.0).sample(2000, rng)
    assert beta.min() >= -1.0 and beta.max() <= 4.0
    assert uniform.min() >= 2.0 and uniform.max() <= 5.0


def test_mixture_follows_its_weights():
    values = EXAMPLES["mixture"].sample(20_000, np.random.default_rng(2))
    assert np.mean(values > 0) == pytest.approx(0.75, abs=0.02)


def test_autoregressive_with_zero_rho_is_white_noise():
    rng = np.random.default_rng(3)
    values = AutoRegressive(0.0, 2.0).sample(5000, rng)
    assert values.std() == pytest.approx(2.0, rel=0.05)


def test_autoregressive_is_persistent():
    values = AutoRegressive(0.9, 1.0).sample(20_000, np.random.default_rng(4))
    assert np.corrcoef(values[:-1], values[1:])[0, 1] == pytest.approx(0.9, abs=0.02)


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


@pytest.mark.parametrize(
    "build",
    [
        lambda: Normal(std=0.0),
        lambda: Uniform(1.0, 1.0),
        lambda: Beta(0.0, 1.0),
        lambda: Mixture((Normal(),)),
        lambda: Mixture((Normal(), Normal()), (0.5, 0.6)),
        lambda: AutoRegressive(1.0),
        lambda: Cycle(0.0),
        lambda: Calendar(END, START),
        lambda: Calendar(START, END, (1.0,) * 6),
        lambda: Calendar(START, END, (0.0,) * 7),
    ],
)
def test_invalid_parameters_are_rejected(build):
    with pytest.raises(ValueError):
        build()
