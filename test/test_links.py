import numpy as np
import pytest

import plurel.links
from plurel.distributions import Normal, Pareto
from plurel.links import LINKS, ForestLink, HSBMLink, Link, RandomLink, clusters

EXAMPLES = {
    "random": RandomLink(),
    "hsbm": HSBMLink((2, 3), (2, 2), attractiveness=Pareto(2.5), inactive=0.3),
    "forest": ForestLink(roots=0.2),
}


def test_every_registered_link_meets_the_contract():
    assert set(EXAMPLES) == set(LINKS)
    for name, link in EXAMPLES.items():
        assert isinstance(link, Link)
        n_parent = 1000 if name == "forest" else 400
        parents = link.sample(1000, n_parent, np.random.default_rng(0))
        np.testing.assert_array_equal(
            parents, link.sample(1000, n_parent, np.random.default_rng(0))
        )
        assert parents.shape == (1000,) and parents.dtype == np.int64
        assert parents.min() >= -1 and parents.max() < n_parent


def test_hsbm_links_within_blocks_with_skewed_and_inactive_parents():
    rng = np.random.default_rng(0)
    parents = HSBMLink((2,), (2,)).sample(4000, 400, rng)
    same_block = clusters(400, (2,))[parents, 0] == clusters(4000, (2,))[:, 0]
    assert same_block.mean() > 0.95
    skewed = HSBMLink(attractiveness=Pareto(1.5), inactive=0.5).sample(4000, 400, rng)
    degrees = np.bincount(skewed, minlength=400)
    assert (degrees == 0).sum() >= 200 and degrees.max() > 5 * degrees[degrees > 0].mean()
    with pytest.raises(ValueError):
        HSBMLink((2,), (2, 2))


def test_cluster_shares_set_unequal_sizes():
    equal = [[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 1], [1, 1]]
    np.testing.assert_array_equal(clusters(7, (2, 2)), equal)
    skewed = [[0, 0], [0, 1], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1]]
    np.testing.assert_array_equal(clusters(7, (2, 2), shares=(1, 1, 1, 4)), skewed)
    for shares in ((1, 1, 1), (1, 0, 1, 1)):
        with pytest.raises(ValueError):
            clusters(7, (2, 2), shares=shares)
    link = HSBMLink((4,), (4,), between=(1e-6, 2e-6), cluster_weights=Pareto(1.0))
    parents = link.sample(4000, 400, np.random.default_rng(0))
    draws = np.random.default_rng(0)
    parent_labels = clusters(400, (4,), link.shares((4,), draws))[:, 0]
    child_labels = clusters(4000, (4,), link.shares((4,), draws))[:, 0]
    sizes = np.bincount(parent_labels, minlength=4)
    assert sizes.min() >= 1 and sizes.max() > 3 * sizes.min()
    assert (parent_labels[parents] == child_labels).mean() > 0.95
    assert np.bincount(clusters(3, (2, 2))[:, 0], minlength=2).tolist() == [2, 1]


def test_forest_links_point_to_earlier_rows_or_nowhere():
    parents = ForestLink(roots=0.2).sample(1000, 1000, np.random.default_rng(0))
    assert parents[0] == -1 and abs((parents == -1).mean() - 0.2) < 0.05
    linked = np.flatnonzero(parents >= 0)
    assert (parents[linked] < linked).all()
    with pytest.raises(ValueError):
        ForestLink().sample(10, 20, np.random.default_rng(0))


def test_links_handle_empty_and_tiny_tables_and_reject_bad_parameters():
    rng = np.random.default_rng(0)
    for name, link in EXAMPLES.items():
        empty = link.sample(0, 0, rng)
        assert empty.shape == (0,) and empty.dtype == np.int64
        if name != "forest":
            assert not link.sample(50, 1, rng).any()
            with pytest.raises(ValueError):
                link.sample(50, 0, rng)
        with pytest.raises(ValueError):
            link.sample(-1, 10, rng)
    assert (ForestLink(roots=1.0).sample(20, 20, rng) == -1).all()
    for kwargs in (
        {"parent_hierarchy": ()},
        {"parent_hierarchy": (0,), "child_hierarchy": (1,)},
        {"within": 0.0},
        {"between": (0.0, 0.1)},
        {"between": (0.2, 0.1)},
        {"inactive": 1.0},
    ):
        with pytest.raises(ValueError):
            HSBMLink(**kwargs)
    with pytest.raises(ValueError):
        HSBMLink((2,), (2,), cluster_weights=Normal()).sample(100, 100, rng)
    with pytest.raises(ValueError):
        ForestLink(roots=0.0)


def test_hsbm_draws_do_not_depend_on_chunking(monkeypatch):
    link = HSBMLink((2, 2), (3, 2), attractiveness=Pareto(2.0), inactive=0.2)
    whole = link.sample(500, 300, np.random.default_rng(3))
    monkeypatch.setattr(plurel.links, "CHUNK_BYTES", 8 * 300 * 7)
    np.testing.assert_array_equal(link.sample(500, 300, np.random.default_rng(3)), whole)
