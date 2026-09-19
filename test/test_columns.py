import numpy as np
import pandas as pd
import pytest

from plurel.columns import DEFAULT_CALENDAR, Column, rank_map
from plurel.distributions import Calendar, Uniform

N = 2000
CATEGORIES = ("low", "mid", "high")
PROBABILITIES = (0.2, 0.5, 0.3)


@pytest.fixture
def latent():
    return np.random.default_rng(0).standard_normal((N, 1))


def test_column_validation():
    for kwargs in (
        {"kind": "timestamp"},
        {"missing": 1.0},
        {"categories": CATEGORIES},
        {"kind": "categorical"},
        {"kind": "categorical", "categories": ("a", "a")},
        {"kind": "categorical", "categories": CATEGORIES, "probabilities": (0.5, 0.5)},
        {"kind": "categorical", "categories": CATEGORIES, "binning": (0.0,)},
        {"kind": "categorical", "categories": CATEGORIES, "binning": (1.0, 0.0)},
        {"kind": "categorical", "categories": CATEGORIES, "binning": "kmeans"},
        {"kind": "categorical", "categories": CATEGORIES, "marginal": Uniform()},
    ):
        with pytest.raises(ValueError):
            Column("x", **kwargs)
    assert Column("x", "categorical", categories=CATEGORIES).category_probabilities == (1 / 3,) * 3


def test_numeric_columns_pass_through_or_rank_map_onto_the_marginal(latent):
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(Column("x").observe(latent, rng), latent.ravel())
    uniform = Column("x", marginal=Uniform(-1.0, 1.0)).observe(latent, np.random.default_rng(1))
    assert np.array_equal(np.argsort(uniform.to_numpy()), np.argsort(latent.ravel()))
    reference = np.sort(Uniform(-1.0, 1.0).sample(N, np.random.default_rng(1)))
    np.testing.assert_array_equal(np.sort(uniform.to_numpy()), reference)
    tied = rank_map(np.array([2.0, 0.0, 2.0, 1.0]), Uniform(), rng)
    assert tied[0] == tied[2] and tied[1] < tied[3] < tied[0]
    sparse = Column("x", missing=0.25).observe(latent, rng)
    assert abs(sparse.isna().mean() - 0.25) < 0.03
    with pytest.raises(ValueError):
        Column("h").observe(np.zeros((N, 3)), rng)


def test_categorical_columns_bin_a_latent_or_read_a_one_hot_block(latent):
    rng = np.random.default_rng(0)
    normal = Column("x", "categorical", categories=CATEGORIES, probabilities=PROBABILITIES)
    observed = normal.observe(latent, rng)
    assert observed.dtype == "category" and list(observed.cat.categories) == list(CATEGORIES)
    shares = observed.value_counts(normalize=True)
    assert all(abs(shares[c] - p) < 0.04 for c, p in zip(CATEGORIES, PROBABILITIES))
    empirical = Column(
        "x", "categorical", categories=(0, 1, 2), probabilities=PROBABILITIES, binning="empirical"
    )
    counts = empirical.observe(latent, rng).value_counts()
    assert counts.index.categories.dtype == np.int64 and abs(counts[0] - 0.2 * N) <= 1
    fixed = Column("x", "categorical", categories=CATEGORIES, binning=(-10.0, 10.0))
    assert set(fixed.observe(latent, rng)) == {"mid"}
    one_hot = np.eye(3)[np.arange(N) % 3]
    read = Column("h", "categorical", categories=CATEGORIES).observe(one_hot, rng)
    np.testing.assert_array_equal(read, np.asarray(CATEGORIES)[np.arange(N) % 3])
    block = np.concatenate([np.zeros((N, 1)), one_hot], axis=1)
    sliced = Column("h", "categorical", categories=CATEGORIES, dims=(1, 2, 3)).observe(block, rng)
    np.testing.assert_array_equal(sliced, read)
    with pytest.raises(ValueError):
        Column("h", "categorical", categories=CATEGORIES).observe(block, rng)
    masked = Column("h", "categorical", categories=(0, 1, 2), missing=0.5).observe(one_hot, rng)
    assert masked.dtype == "category" and masked.dropna().isin([0, 1, 2]).all()


def test_calendar_marginals_give_timestamps_in_latent_order(latent):
    rng = np.random.default_rng(0)
    stamps = Column("t", marginal=DEFAULT_CALENDAR).observe(latent, rng)
    assert stamps.dtype == "datetime64[ns]"
    assert stamps.min() >= DEFAULT_CALENDAR.start and stamps.max() <= DEFAULT_CALENDAR.end
    assert (np.diff(stamps.to_numpy()[np.argsort(latent.ravel())]) >= np.timedelta64(0, "ns")).all()
    calendar = Calendar(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))
    month = Column("t", marginal=calendar).observe(latent, rng)
    assert calendar.start <= month.min() and month.max() <= calendar.end
