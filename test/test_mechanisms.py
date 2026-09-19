import numpy as np
import pytest

from plurel.distributions import Mixture, Normal
from plurel.mechanisms import (
    MECHANISMS,
    REDUCTIONS,
    Combine,
    LinearEffect,
    LookupEffect,
    LookupScaleEffect,
    Mechanism,
    ProductEffect,
    Root,
    TransformedProductEffect,
    bin_levels,
)

N = 200
PROBABILITIES = (0.2, 0.5, 0.3)
TERMS = (
    LinearEffect("x", 1.5, "tanh"),
    LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES),
    ProductEffect(("x", "y"), 0.5),
    LookupScaleEffect("s", "y", (2.0, -3.0, 1.0), PROBABILITIES),
    TransformedProductEffect(("x", "y"), 0.25, "step"),
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
