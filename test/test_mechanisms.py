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
def values():
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal((N, 1)) for name in ("x", "y", "s")}


def test_every_registered_mechanism_meets_the_contract(values):
    assert set(EXAMPLES) == set(MECHANISMS)
    for mechanism in EXAMPLES.values():
        assert isinstance(mechanism, Mechanism)
        exogenous = mechanism.sample_noise(N, np.random.default_rng(1))
        again = mechanism.sample_noise(N, np.random.default_rng(1))
        np.testing.assert_array_equal(exogenous, again)
        assert mechanism.evaluate(values, exogenous).shape == (N, mechanism.dim)


def test_every_reduction_reduces_the_transformed_parents(values):
    terms = np.stack([effect.apply(values[effect.parent]) for effect in TERMS])
    rng = np.random.default_rng(0)
    for op, reduce in REDUCTIONS.items():
        mechanism = Combine(TERMS, op, noise=None)
        exogenous = mechanism.sample_noise(N, rng)
        assert exogenous.shape == (N, 1) and not exogenous.any()
        np.testing.assert_allclose(mechanism.evaluate(values, exogenous), reduce(terms))


def test_lookup_effects_share_the_level_binning(values):
    levels = bin_levels(values["s"], PROBABILITIES)
    lookup = LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES).apply(values["s"])
    np.testing.assert_array_equal(lookup, np.asarray([10.0, 20.0, 30.0])[levels])
    assert set(np.unique(levels)) == {0, 1, 2}


def test_interactions_are_product_nodes(values):
    zeros = np.zeros((N, 1))
    product = Combine((LinearEffect("x"), LinearEffect("y")), op="product")
    interaction = product.evaluate(values, zeros)
    np.testing.assert_allclose(interaction, values["x"] * values["y"])
    target = Combine((LinearEffect("x", 2.0), LinearEffect("h", -0.5)))
    out = target.evaluate({**values, "h": interaction}, zeros)
    np.testing.assert_allclose(out, 2.0 * values["x"] - 0.5 * values["x"] * values["y"])
    assert target.parents == ("x", "h")
