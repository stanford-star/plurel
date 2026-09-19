import numpy as np
import pytest

from plurel.distributions import Pareto
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
    expected = [[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]]
    np.testing.assert_array_equal(clusters(7, (2, 2)), expected)
    with pytest.raises(ValueError):
        HSBMLink((2,), (2, 2))


def test_forest_links_point_to_earlier_rows_or_nowhere():
    parents = ForestLink(roots=0.2).sample(1000, 1000, np.random.default_rng(0))
    assert parents[0] == -1 and abs((parents == -1).mean() - 0.2) < 0.05
    linked = np.flatnonzero(parents >= 0)
    assert (parents[linked] < linked).all()
    with pytest.raises(ValueError):
        ForestLink().sample(10, 20, np.random.default_rng(0))
