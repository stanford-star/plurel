import numpy as np
import pytest

from plurel.distributions import Mixture, Normal
from plurel.mechanisms import (
    MECHANISMS,
    REDUCTIONS,
    Combine,
    LinearEffect,
    LookupEffect,
    Mechanism,
    Root,
    bin_levels,
)

N = 200
PROBABILITIES = (0.2, 0.5, 0.3)
TERMS = (
    LinearEffect("x", 1.5, "tanh"),
    LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES),
    LinearEffect("y", 0.5, "exp"),
)
EXAMPLES = {
    "root": Root(dim=3, noise=Mixture((Normal(-2.0), Normal(2.0)))),
    "combine": Combine(TERMS, noise=Normal(std=0.5)),
}


@pytest.fixture
def parents():
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal((N, 1)) for name in ("x", "y", "s")}


def test_every_registered_mechanism_meets_the_contract(parents):
    assert set(EXAMPLES) == set(MECHANISMS)
    for mechanism in EXAMPLES.values():
        assert isinstance(mechanism, Mechanism)
        noise = mechanism.sample_noise(N, np.random.default_rng(1))
        again = mechanism.sample_noise(N, np.random.default_rng(1))
        np.testing.assert_array_equal(noise, again)
        assert mechanism.evaluate(parents, noise).shape == (N, mechanism.dim)


def test_every_reduction_reduces_the_effects_terms(parents):
    terms = np.stack([effect.evaluate(parents) for effect in TERMS])
    zeros = np.zeros((N, 1))
    for op, reduce in REDUCTIONS.items():
        np.testing.assert_allclose(
            Combine(TERMS, op, noise=Normal(std=0.0)).evaluate(parents, zeros), reduce(terms)
        )


def test_lookup_effects_share_the_level_binning(parents):
    levels = bin_levels(parents["s"], PROBABILITIES)
    lookup = LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES).evaluate(parents)
    np.testing.assert_array_equal(lookup, np.asarray([10.0, 20.0, 30.0])[levels])
    assert set(np.unique(levels)) == {0, 1, 2}


def test_interactions_are_product_nodes(parents):
    zeros = np.zeros((N, 1))
    product = Combine((LinearEffect("x", 1.0), LinearEffect("y", 1.0)), op="product")
    interaction = product.evaluate(parents, zeros)
    np.testing.assert_allclose(interaction, parents["x"] * parents["y"])
    target = Combine((LinearEffect("x", 2.0), LinearEffect("h", -0.5)))
    values = target.evaluate({**parents, "h": interaction}, zeros)
    np.testing.assert_allclose(values, 2.0 * parents["x"] - 0.5 * parents["x"] * parents["y"])
    assert target.parents == ("x", "h")
