import numpy as np
import pandas as pd
import pytest

from plurel.columns import DEFAULT_CALENDAR, Column, rank_map
from plurel.distributions import Calendar, Poisson, Uniform
from plurel.graph import Node
from plurel.schema import SCM

N = 2000
CATEGORIES = ("low", "mid", "high")
PROBABILITIES = (0.2, 0.5, 0.3)
NODE = Node()


@pytest.fixture
def latent():
    return np.random.default_rng(0).standard_normal((N, 1))


def observe(column, latent, seed=0):
    return column.observe("x", {"x": latent}, np.random.default_rng(seed), len(latent))


def test_column_validation():
    for kwargs in (
        {"kind": "text"},
        {"missing": 1.0},
        {"missing": -0.1},
        {"kind": "numeric", "categories": CATEGORIES},
        {"binning": "empirical"},
        {"kind": "categorical"},
        {"kind": "categorical", "categories": ("a", "a")},
        {"kind": "categorical", "categories": CATEGORIES, "probabilities": (0.5, 0.5)},
        {"kind": "categorical", "categories": CATEGORIES, "binning": (0.0,)},
        {"kind": "categorical", "categories": CATEGORIES, "binning": (1.0, 0.0)},
        {"kind": "categorical", "categories": CATEGORIES, "binning": "kmeans"},
        {"kind": "categorical", "categories": CATEGORIES, "marginal": Uniform()},
    ):
        with pytest.raises(ValueError):
            Column(NODE, **kwargs)
    assert Column(NODE, "categorical", categories=CATEGORIES).category_probabilities == (1 / 3,) * 3


def test_numeric_columns_pass_through_or_rank_map_onto_the_marginal(latent):
    np.testing.assert_array_equal(observe(Column(NODE), latent), latent.ravel())
    uniform = observe(Column(NODE, marginal=Uniform(-1.0, 1.0)), latent, seed=1)
    assert np.array_equal(np.argsort(uniform.to_numpy()), np.argsort(latent.ravel()))
    reference = np.sort(Uniform(-1.0, 1.0).sample(N, np.random.default_rng(1)))
    np.testing.assert_array_equal(np.sort(uniform.to_numpy()), reference)
    assert rank_map(np.arange(4.0), Poisson(3.0), np.random.default_rng(0)).dtype == float
    discrete = rank_map(np.repeat([0.0, 1.0], [900, 100]), Uniform(), np.random.default_rng(0))
    assert np.unique(discrete).size == 1000 and abs(discrete[:900].mean() - 0.45) < 0.05
    assert discrete[900:].min() > discrete[:900].max()
    for kind in ("numeric", "timestamp"):
        with pytest.raises(ValueError, match="one-dimensional"):
            Column(Node(dim=3), kind)


def test_categorical_columns_bin_a_latent_or_read_a_one_hot_node(latent):
    normal = Column(NODE, "categorical", categories=CATEGORIES, probabilities=PROBABILITIES)
    observed = observe(normal, latent)
    assert observed.dtype == "category" and list(observed.cat.categories) == list(CATEGORIES)
    shares = observed.value_counts(normalize=True)
    assert all(abs(shares[c] - p) < 0.04 for c, p in zip(CATEGORIES, PROBABILITIES))
    coded = Column(NODE, "categorical", categories=(0, 1, 2), probabilities=PROBABILITIES)
    counts = observe(coded, latent).value_counts()
    assert counts.index.categories.dtype == np.int64 and abs(counts[0] - 0.2 * N) <= 1
    far = observe(normal, 100.0 + 50.0 * latent).value_counts(normalize=True)
    assert all(abs(far[c] - p) < 0.01 for c, p in zip(CATEGORIES, PROBABILITIES))
    ties = observe(coded, np.repeat([0.0, 1.0], [int(0.9 * N), N - int(0.9 * N)])[:, None])
    assert ties.nunique() > 1
    fixed = Column(NODE, "categorical", categories=CATEGORIES, binning=(-10.0, 10.0))
    assert set(observe(fixed, latent)) == {"mid"}
    one_hot = np.eye(3)[np.arange(N) % 3]
    read = observe(Column(NODE, "categorical", categories=CATEGORIES), one_hot)
    np.testing.assert_array_equal(read, np.asarray(CATEGORIES)[np.arange(N) % 3])
    with pytest.raises(ValueError, match="one per category"):
        Column(Node(dim=4), categories=CATEGORIES)
    for options in ({"probabilities": PROBABILITIES}, {"binning": "empirical"}):
        with pytest.raises(ValueError, match="sets its own shares"):
            Column(Node(dim=3), categories=CATEGORIES, **options)


def test_calendar_marginals_give_timestamps_in_latent_order(latent):
    stamps = observe(Column(NODE, "timestamp", marginal=DEFAULT_CALENDAR), latent)
    assert stamps.dtype == "datetime64[ns]"
    assert stamps.min() >= DEFAULT_CALENDAR.start and stamps.max() <= DEFAULT_CALENDAR.end
    assert (np.diff(stamps.to_numpy()[np.argsort(latent.ravel())]) >= np.timedelta64(0, "ns")).all()
    calendar = Calendar(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))
    month = observe(Column(NODE, "timestamp", marginal=calendar), latent)
    assert calendar.start <= month.min() and month.max() <= calendar.end
    seconds = Calendar(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")).sample(
        N, np.random.default_rng(0)
    )
    plain = observe(Column(NODE, "timestamp"), seconds[:, None])
    assert plain.dtype == "datetime64[ns]" and plain.is_monotonic_increasing
    assert plain.iloc[0] == pd.Timestamp(seconds[0], unit="s")


def test_missingness_is_a_rate_or_an_indicator_node(latent):
    sparse = observe(Column(NODE, missing=0.25), latent)
    assert abs(sparse.isna().mean() - 0.25) < 0.03
    one_hot = np.eye(3)[np.arange(N) % 3]
    masked = observe(Column(NODE, "categorical", categories=(0, 1, 2), missing=0.5), one_hot)
    assert masked.dtype == "category" and masked.dropna().isin([0, 1, 2]).all()
    rng = np.random.default_rng(0)
    indicator = np.eye(2)[(latent.ravel() > 1.0).astype(int)]
    mnar = Column(NODE, missing="hidden").observe("x", {"x": latent, "hidden": indicator}, rng, N)
    assert mnar.isna().to_numpy().tolist() == (latent.ravel() > 1.0).tolist()


def test_key_columns_are_row_identities():
    key = Column(kind="key")
    np.testing.assert_array_equal(key.observe("id", {}, np.random.default_rng(0), 5), np.arange(5))
    for kwargs in ({"node": NODE}, {"missing": 0.1}, {"marginal": Uniform()}):
        with pytest.raises(ValueError):
            Column(kind="key", **kwargs)
    assert (
        Column(NODE).kind == "numeric" and Column(NODE, categories=CATEGORIES).kind == "categorical"
    )
    for bad in ((), ("x",)):
        with pytest.raises(ValueError, match="holds its node"):
            Column(*bad)
    for table in ({}, {"three": Node(dim=3)}):
        with pytest.raises(ValueError, match="two-class"):
            SCM({**table, "x": Column(NODE, missing="three")})


def test_empty_one_hot_rows_observe_as_missing():
    one_hot = np.eye(3)[np.arange(N) % 3]
    one_hot[:5] = 0.0
    read = observe(Column(NODE, "categorical", categories=CATEGORIES), one_hot)
    assert read[:5].isna().all() and read[5:].notna().all()
