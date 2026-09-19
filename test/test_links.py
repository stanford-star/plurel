import numpy as np
import pytest

import plurel.links
from plurel.distributions import Pareto, Uniform
from plurel.links import LINKS, HSBMLink, Link, RandomLink, TreeLink, clusters

EXAMPLES = {
    "random": RandomLink(),
    "hsbm": HSBMLink((2, 3), (2, 2), attractiveness=Pareto(2.5), inactive=0.3),
    "tree": TreeLink(roots=0.2),
}


class Weights:
    def __init__(self, *values):
        self.values = values

    def sample(self, n, rng):
        return np.resize(np.asarray(self.values, dtype=float), n)


def test_every_registered_link_meets_the_contract():
    assert set(EXAMPLES) == set(LINKS)
    for name, link in EXAMPLES.items():
        assert isinstance(link, Link)
        n_parent = 1000 if name == "tree" else 400
        parents = link.sample(1000, n_parent, np.random.default_rng(0))
        np.testing.assert_array_equal(
            parents, link.sample(1000, n_parent, np.random.default_rng(0))
        )
        assert parents.shape == (1000,) and parents.dtype == np.int64
        assert parents.min() >= -1 and parents.max() < n_parent
        empty = link.sample(0, 0, np.random.default_rng(0))
        assert empty.shape == (0,) and empty.dtype == np.int64


def test_hsbm_links_within_blocks_with_skewed_and_inactive_parents():
    rng = np.random.default_rng(0)
    parents = HSBMLink((2,), (2,)).sample(4000, 400, rng)
    same_block = clusters(400, (2,))[parents, 0] == clusters(4000, (2,))[:, 0]
    assert same_block.mean() > 0.95
    skewed = HSBMLink(attractiveness=Pareto(1.5), inactive=0.5).sample(4000, 400, rng)
    degrees = np.bincount(skewed, minlength=400)
    assert (degrees == 0).sum() >= 200 and degrees.max() > 5 * degrees[degrees > 0].mean()
    excluded = HSBMLink(attractiveness=Weights(0.0, 1.0)).sample(2000, 10, rng)
    assert not (excluded % 2 == 0).any()


def test_cluster_shares_set_unequal_sizes():
    equal = [[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 1], [1, 1]]
    np.testing.assert_array_equal(clusters(7, (2, 2)), equal)
    skewed = [[0, 0], [0, 1], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1]]
    np.testing.assert_array_equal(clusters(7, (2, 2), shares=(1, 1, 1, 4)), skewed)
    link = HSBMLink((4,), (4,), between=Uniform(1e-6, 2e-6), cluster_weights=Pareto(1.0))
    parents = link.sample(4000, 400, np.random.default_rng(0))
    draws = np.random.default_rng(0)
    parent_labels = link.labels(400, (4,), draws)[:, 0]
    child_labels = link.labels(4000, (4,), draws)[:, 0]
    sizes = np.bincount(parent_labels, minlength=4)
    assert sizes.min() >= 1 and sizes.max() > 3 * sizes.min()
    assert (parent_labels[parents] == child_labels).mean() > 0.95


def test_tree_links_point_to_earlier_rows_or_nowhere():
    parents = TreeLink(roots=0.2).sample(1000, 1000, np.random.default_rng(0))
    assert parents[0] == -1 and abs((parents == -1).mean() - 0.2) < 0.05
    linked = np.flatnonzero(parents >= 0)
    assert (parents[linked] < linked).all()
    assert (TreeLink(roots=1.0).sample(20, 20, np.random.default_rng(0)) == -1).all()
    with pytest.raises(ValueError):
        TreeLink().sample(10, 20, np.random.default_rng(0))
    with pytest.raises(ValueError):
        TreeLink(roots=0.0)


def test_links_reject_unexpected_sizes_shares_and_parameters():
    rng = np.random.default_rng(0)
    for link in (RandomLink(), HSBMLink()):
        assert not link.sample(50, 1, rng).any()
        with pytest.raises(ValueError):
            link.sample(50, 0, rng)
        with pytest.raises(ValueError):
            link.sample(-1, 10, rng)
    for kwargs in (
        {"parent_clusters": ()},
        {"parent_clusters": (0,), "child_clusters": (1,)},
        {"parent_clusters": (2,), "child_clusters": (2, 2)},
        {"within": 0.0},
        {"inactive": 1.0},
    ):
        with pytest.raises(ValueError):
            HSBMLink(**kwargs)
    for shares in ((1, 1, 1), (1, 0, 1, 1), (1, np.nan, 1, 1)):
        with pytest.raises(ValueError):
            clusters(7, (2, 2), shares=shares)
    for link, n_child, n_parent in (
        (HSBMLink((4,), (4,)), 100, 3),
        (HSBMLink((2,), (2,), within=Uniform(-1.0, 1.0)), 100, 10),
        (HSBMLink((2,), (2,), between=Uniform(-1.0, 1.0)), 100, 10),
        (HSBMLink((1,), (4,)), 3, 100),
        (HSBMLink((2,), (2,), inactive=0.9), 100, 3),
        (HSBMLink(cluster_weights=Weights(np.nan)), 100, 10),
        (HSBMLink(attractiveness=Weights(np.inf)), 100, 10),
        (HSBMLink(attractiveness=Weights(0.0)), 100, 10),
        (HSBMLink(attractiveness=Weights(-1.0, 1.0)), 100, 10),
    ):
        with pytest.raises(ValueError):
            link.sample(n_child, n_parent, rng)


def test_cluster_bounds_always_cover_every_row():
    rng = np.random.default_rng(0)
    for _ in range(200):
        shares = rng.pareto(0.3, 16) + 1e-9
        labels = clusters(1000, (4, 4), shares)
        base = labels[:, 0] * 4 + labels[:, 1]
        assert (np.diff(base) >= 0).all() and np.bincount(base, minlength=16).min() >= 1


def test_hsbm_raises_when_exclusions_leave_no_linkable_parent():
    link = HSBMLink(attractiveness=Weights(1.0, 0.0), inactive=0.5)
    outcomes = set()
    for seed in range(10):
        try:
            parents = link.sample(100, 2, np.random.default_rng(seed))
        except ValueError:
            outcomes.add("raised")
        else:
            outcomes.add("linked")
            assert not parents.any()
    assert outcomes == {"raised", "linked"}


def test_hsbm_draws_do_not_depend_on_chunking(monkeypatch):
    link = HSBMLink((2, 2), (3, 2), attractiveness=Pareto(2.0), inactive=0.2)
    whole = link.sample(500, 300, np.random.default_rng(3))
    monkeypatch.setattr(plurel.links, "CHUNK_BYTES", 8 * 300 * 7)
    np.testing.assert_array_equal(link.sample(500, 300, np.random.default_rng(3)), whole)


def test_hsbm_matches_the_closed_form_distribution():
    link = HSBMLink(
        (2, 3),
        (2, 2),
        within=Uniform(0.4, 0.9),
        between=Pareto(1.0, 0.01),
        attractiveness=Pareto(2.0),
    )
    n_child, n_parent = 30_000, 6
    parents = link.sample(n_child, n_parent, np.random.default_rng(7))
    rng = np.random.default_rng(7)
    parent_labels = link.labels(n_parent, link.parent_clusters, rng)
    child_labels = link.labels(n_child, link.child_clusters, rng)
    levels = [
        np.log(link.affinity(p, c, rng)) for p, c in zip(link.parent_clusters, link.child_clusters)
    ]
    log_p = link.log_weights(n_parent, rng)[:, None] + sum(
        level[parent_labels[:, i][:, None], child_labels[:, i][None, :]]
        for i, level in enumerate(levels)
    )
    expected = np.exp(log_p) / np.exp(log_p).sum(axis=0)
    base = child_labels[:, 0] * 2 + child_labels[:, 1]
    for block in np.unique(base):
        rows = np.flatnonzero(base == block)
        observed = np.bincount(parents[rows], minlength=n_parent) / len(rows)
        np.testing.assert_allclose(observed, expected[:, rows[0]], atol=0.02)


def test_links_hold_their_invariants_under_random_configurations():
    rng = np.random.default_rng(11)
    for _ in range(60):
        depth = int(rng.integers(1, 4))
        link = HSBMLink(
            tuple(int(c) for c in rng.integers(1, 4, depth)),
            tuple(int(c) for c in rng.integers(1, 4, depth)),
            within=float(rng.uniform(0.3, 1.0)),
            cluster_weights=Pareto(1.5) if rng.random() < 0.5 else None,
            attractiveness=Pareto(2.0) if rng.random() < 0.5 else None,
            inactive=float(rng.uniform(0.0, 0.6)),
        )
        n_parent = int(rng.integers(np.prod(link.parent_clusters), 200))
        n_child = 0 if rng.random() < 0.1 else int(rng.integers(np.prod(link.child_clusters), 500))
        seed = int(rng.integers(1 << 30))
        parents = link.sample(n_child, n_parent, np.random.default_rng(seed))
        assert parents.shape == (n_child,) and parents.dtype == np.int64
        np.testing.assert_array_equal(
            parents, link.sample(n_child, n_parent, np.random.default_rng(seed))
        )
        if n_child == 0:
            continue
        assert parents.min() >= 0 and parents.max() < n_parent
        draws = np.random.default_rng(seed)
        link.labels(n_parent, link.parent_clusters, draws)
        link.labels(n_child, link.child_clusters, draws)
        for p, c in zip(link.parent_clusters, link.child_clusters):
            link.affinity(p, c, draws)
        assert np.isfinite(link.log_weights(n_parent, draws)[parents]).all()


def test_affinities_can_vary_per_block_and_per_pair():
    rng = np.random.default_rng(0)
    fixed = HSBMLink().affinity(3, 3, rng)
    assert (np.diag(fixed) == 0.9).all() and (fixed[~np.eye(3, dtype=bool)] < 0.9).all()
    varied = HSBMLink(within=Uniform(0.2, 0.9), between=Pareto(1.0, 0.01)).affinity(3, 3, rng)
    diagonal = np.diag(varied)
    assert (0.2 <= diagonal).all() and (diagonal <= 0.9).all() and len(set(diagonal)) == 3
    off = varied[~np.eye(3, dtype=bool)]
    assert off.min() > 0 and off.max() > 10 * off.min()
