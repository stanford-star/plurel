import numpy as np
import pytest

from plurel.layouts import (
    LAYOUTS,
    BarabasiAlbert,
    ErdosRenyi,
    Layered,
    Layout,
    RandomCauchy,
    RandomTree,
    ReverseRandomTree,
    WattsStrogatz,
)

EXAMPLES = {
    "erdos_renyi": ErdosRenyi(0.3),
    "barabasi_albert": BarabasiAlbert(2, sink_dropout=0.5),
    "random_tree": RandomTree(),
    "reverse_random_tree": ReverseRandomTree(),
    "watts_strogatz": WattsStrogatz(4, rewire=0.3),
    "random_cauchy": RandomCauchy(),
    "layered": Layered(4, dropout=0.2),
}


def connected(parents):
    label = list(range(len(parents)))
    for child, own in enumerate(parents):
        for parent in own:
            old, new = label[parent], label[child]
            label = [new if value == old else value for value in label]
    return len(set(label)) <= 1


def edges(parents):
    return sum(len(own) for own in parents)


def test_every_registered_layout_meets_the_contract():
    assert set(EXAMPLES) == set(LAYOUTS)
    for layout in EXAMPLES.values():
        assert isinstance(layout, Layout)
        for n in (0, 1, 2, 3, 12, 40):
            parents = layout.sample(n, np.random.default_rng(0))
            assert parents == layout.sample(n, np.random.default_rng(0))
            assert len(parents) == n
            assert all(0 <= parent < child for child, own in enumerate(parents) for parent in own)
            assert all(len(set(own)) == len(own) for own in parents)
            assert connected(parents)
        with pytest.raises(ValueError):
            layout.sample(-1, np.random.default_rng(0))


def test_trees_have_one_edge_less_than_nodes_and_point_away_from_or_towards_the_root():
    rng = np.random.default_rng(1)
    forward = RandomTree().sample(50, rng)
    assert edges(forward) == 49 and all(len(own) <= 1 for own in forward)
    backward = ReverseRandomTree().sample(50, rng)
    assert edges(backward) == 49
    children = np.zeros(50, dtype=int)
    for own in backward:
        for parent in own:
            children[parent] += 1
    assert (children <= 1).all() and children.sum() == 49


def test_density_hubs_rings_and_layers():
    rng = np.random.default_rng(2)
    dense = ErdosRenyi(0.9).sample(40, rng)
    sparse = ErdosRenyi(0.1).sample(40, rng)
    assert edges(dense) > 3 * edges(sparse)
    hubs = BarabasiAlbert(2, sink_dropout=0.0).sample(200, rng)
    degree = np.zeros(200, dtype=int)
    for child, own in enumerate(hubs):
        degree[child] += len(own)
        for parent in own:
            degree[parent] += 1
    assert degree.max() > 5 * degree.mean()
    ring = WattsStrogatz(4, rewire=0.0).sample(30, rng)
    assert edges(ring) == 60
    layered = Layered(3, dropout=0.5).sample(30, rng)
    roots = [child for child, own in enumerate(layered) if not own]
    assert 1 <= len(roots) < 30 and max(roots) < 30
    assert edges(layered) >= 29
    for kwargs, layout in (
        ({"p": 1.5}, ErdosRenyi),
        ({"m": 0}, BarabasiAlbert),
        ({"k": 3}, WattsStrogatz),
        ({"depth": 0}, Layered),
        ({"dropout": 1.0}, Layered),
    ):
        with pytest.raises(ValueError):
            layout(**kwargs)
