import numpy as np
import pandas as pd
import pytest

from plurel.columns import DEFAULT_CALENDAR, Column, rank_map
from plurel.distributions import Calendar, Uniform

N = 2000
LEVELS = ("low", "mid", "high")
PROBABILITIES = (0.2, 0.5, 0.3)


@pytest.fixture
def latent():
    return np.random.default_rng(0).standard_normal((N, 1))


def test_column_validation():
    for kwargs in (
        {"kind": "text"},
        {"missing": 1.0},
        {"levels": LEVELS},
        {"kind": "categorical"},
        {"kind": "categorical", "levels": ("a", "a")},
        {"kind": "categorical", "levels": LEVELS, "probabilities": (0.5, 0.5)},
        {"kind": "categorical", "levels": LEVELS, "edges": (0.0,)},
        {"kind": "categorical", "levels": LEVELS, "binning": "kmeans"},
        {"kind": "categorical", "levels": LEVELS, "marginal": Uniform()},
    ):
        with pytest.raises(ValueError):
            Column("x", **kwargs)
    assert Column("x", "categorical", levels=LEVELS).level_probabilities == (1 / 3,) * 3


def test_numeric_columns_pass_through_or_rank_map_onto_the_marginal(latent):
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(Column("x").observe(latent, rng), latent.ravel())
    uniform = Column("x", marginal=Uniform(-1.0, 1.0)).observe(latent, np.random.default_rng(1))
    assert uniform.min() >= -1.0 and uniform.max() <= 1.0
    assert np.array_equal(np.argsort(uniform.to_numpy()), np.argsort(latent.ravel()))
    reference = np.sort(Uniform(-1.0, 1.0).sample(N, np.random.default_rng(1)))
    np.testing.assert_array_equal(np.sort(uniform.to_numpy()), reference)
    sparse = Column("x", missing=0.25).observe(latent, rng)
    assert abs(sparse.isna().mean() - 0.25) < 0.03
    with pytest.raises(ValueError):
        Column("h").observe(np.zeros((N, 3)), rng)


def test_categorical_columns_bin_a_latent_or_read_a_one_hot_block(latent):
    rng = np.random.default_rng(0)
    normal = Column("x", "categorical", levels=LEVELS, probabilities=PROBABILITIES)
    shares = normal.observe(latent, rng).value_counts(normalize=True)
    assert all(abs(shares[level] - p) < 0.04 for level, p in zip(LEVELS, PROBABILITIES))
    empirical = Column(
        "x", "categorical", levels=(0, 1, 2), probabilities=PROBABILITIES, binning="empirical"
    )
    counts = empirical.observe(latent, rng).value_counts()
    assert counts.index.dtype == np.int64 and abs(counts[0] - 0.2 * N) <= 1
    fixed = Column("x", "categorical", levels=LEVELS, edges=(-10.0, 10.0))
    assert set(fixed.observe(latent, rng)) == {"mid"}
    one_hot = np.eye(3)[np.arange(N) % 3]
    np.testing.assert_array_equal(
        Column("h", "categorical", levels=LEVELS).observe(one_hot, rng),
        np.asarray(LEVELS)[np.arange(N) % 3],
    )
    block = np.concatenate([np.zeros((N, 1)), one_hot], axis=1)
    sliced = Column("h", "categorical", levels=LEVELS, dims=(1, 2, 3)).observe(block, rng)
    np.testing.assert_array_equal(sliced, np.asarray(LEVELS)[np.arange(N) % 3])


def test_timestamp_columns_follow_the_calendar_in_latent_order(latent):
    rng = np.random.default_rng(0)
    stamps = Column("t", "timestamp").observe(latent, rng)
    assert stamps.dtype == "datetime64[ns]"
    assert stamps.min() >= DEFAULT_CALENDAR.start and stamps.max() <= DEFAULT_CALENDAR.end
    assert (np.diff(stamps.to_numpy()[np.argsort(latent.ravel())]) >= np.timedelta64(0, "ns")).all()
    calendar = Calendar(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))
    month = Column("t", "timestamp", marginal=calendar).observe(latent, rng)
    assert month.max() <= calendar.end
    order = rank_map(latent.ravel(), Uniform(), rng)
    assert np.array_equal(np.argsort(order), np.argsort(latent.ravel()))
