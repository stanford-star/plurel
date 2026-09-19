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


def observe(column, latent, seed=0):
    return column.observe({column.node: latent}, np.random.default_rng(seed))


def test_column_validation():
    for kwargs in (
        {"kind": "timestamp"},
        {"missing": 1.0},
        {"missing": -0.1},
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
    np.testing.assert_array_equal(observe(Column("x"), latent), latent.ravel())
    uniform = observe(Column("x", marginal=Uniform(-1.0, 1.0)), latent, seed=1)
    assert np.array_equal(np.argsort(uniform.to_numpy()), np.argsort(latent.ravel()))
    reference = np.sort(Uniform(-1.0, 1.0).sample(N, np.random.default_rng(1)))
    np.testing.assert_array_equal(np.sort(uniform.to_numpy()), reference)
    tied = rank_map(np.array([2.0, 0.0, 2.0, 1.0]), Uniform(), np.random.default_rng(0))
    assert tied[0] == tied[2] and tied[1] < tied[3] < tied[0]
    with pytest.raises(ValueError):
        observe(Column("h"), np.zeros((N, 3)))


def test_categorical_columns_bin_a_latent_or_read_a_one_hot_node(latent):
    normal = Column("x", "categorical", categories=CATEGORIES, probabilities=PROBABILITIES)
    observed = observe(normal, latent)
    assert observed.dtype == "category" and list(observed.cat.categories) == list(CATEGORIES)
    shares = observed.value_counts(normalize=True)
    assert all(abs(shares[c] - p) < 0.04 for c, p in zip(CATEGORIES, PROBABILITIES))
    empirical = Column(
        "x", "categorical", categories=(0, 1, 2), probabilities=PROBABILITIES, binning="empirical"
    )
    counts = observe(empirical, latent).value_counts()
    assert counts.index.categories.dtype == np.int64 and abs(counts[0] - 0.2 * N) <= 1
    fixed = Column("x", "categorical", categories=CATEGORIES, binning=(-10.0, 10.0))
    assert set(observe(fixed, latent)) == {"mid"}
    one_hot = np.eye(3)[np.arange(N) % 3]
    read = observe(Column("h", "categorical", categories=CATEGORIES), one_hot)
    np.testing.assert_array_equal(read, np.asarray(CATEGORIES)[np.arange(N) % 3])
    block = np.concatenate([np.zeros((N, 1)), one_hot], axis=1)
    sliced = observe(Column("h", "categorical", categories=CATEGORIES, dims=(1, 2, 3)), block)
    np.testing.assert_array_equal(sliced, read)
    with pytest.raises(ValueError):
        observe(Column("h", "categorical", categories=CATEGORIES), block)


def test_calendar_marginals_give_timestamps_in_latent_order(latent):
    stamps = observe(Column("t", marginal=DEFAULT_CALENDAR), latent)
    assert stamps.dtype == "datetime64[ns]"
    assert stamps.min() >= DEFAULT_CALENDAR.start and stamps.max() <= DEFAULT_CALENDAR.end
    assert (np.diff(stamps.to_numpy()[np.argsort(latent.ravel())]) >= np.timedelta64(0, "ns")).all()
    calendar = Calendar(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))
    month = observe(Column("t", marginal=calendar), latent)
    assert calendar.start <= month.min() and month.max() <= calendar.end


def test_missingness_is_a_rate_or_an_indicator_node(latent):
    sparse = observe(Column("x", missing=0.25), latent)
    assert abs(sparse.isna().mean() - 0.25) < 0.03
    one_hot = np.eye(3)[np.arange(N) % 3]
    masked = observe(Column("h", "categorical", categories=(0, 1, 2), missing=0.5), one_hot)
    assert masked.dtype == "category" and masked.dropna().isin([0, 1, 2]).all()
    rng = np.random.default_rng(0)
    indicator = np.eye(2)[(latent.ravel() > 1.0).astype(int)]
    mnar = Column("x", missing="hidden").observe({"x": latent, "hidden": indicator}, rng)
    assert mnar.isna().to_numpy().tolist() == (latent.ravel() > 1.0).tolist()
    with pytest.raises(ValueError):
        Column("x", missing="hidden").observe({"x": latent, "hidden": np.eye(3)[[0] * N]}, rng)


def test_nan_latents_and_empty_one_hot_rows_observe_as_missing(latent):
    holed = latent.copy()
    holed[:5] = np.nan
    binned = observe(
        Column("x", "categorical", categories=CATEGORIES, probabilities=PROBABILITIES), holed
    )
    assert binned[:5].isna().all() and binned[5:].notna().all()
    for binning in ("empirical", (-0.5, 0.5)):
        column = Column("x", "categorical", categories=CATEGORIES, binning=binning)
        assert observe(column, holed)[:5].isna().all()
    mapped = observe(Column("x", marginal=Uniform()), holed)
    assert mapped[:5].isna().all() and mapped[5:].notna().all()
    one_hot = np.eye(3)[np.arange(N) % 3]
    one_hot[:5] = 0.0
    read = observe(Column("h", "categorical", categories=CATEGORIES), one_hot)
    assert read[:5].isna().all() and read[5:].notna().all()
