import numpy as np
import pytest

from plurel.distributions import Beta, Mixture, Normal
from plurel.mechanisms import (
    MECHANISMS,
    Additive,
    Aggregate,
    Effect,
    LookupEffect,
    LookupScaleEffect,
    Mechanism,
    Multiplicative,
    Noise,
    ProductEffect,
    Root,
    TransformedProductEffect,
    bin_levels,
)

N = 200
PROBABILITIES = (0.2, 0.5, 0.3)
TERMS = (
    Effect("x", 1.5, "tanh"),
    LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES),
    ProductEffect(("x", "y"), 0.5),
    LookupScaleEffect("s", "y", (2.0, -3.0, 1.0), PROBABILITIES),
    TransformedProductEffect(("x", "y"), 0.25, "step"),
)
EXAMPLES = {
    "root": Root(Mixture((Normal(-2.0), Normal(2.0))), dim=3),
    "additive": Additive(TERMS, Noise(0.5, scale_effects=(Effect("y", 0.3),))),
    "aggregate": Aggregate(TERMS, Noise(0.0), op="min"),
    "multiplicative": Multiplicative(TERMS, Noise(0.0), span=1.0),
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


def test_additive_is_the_sum_of_its_contributions(parents):
    mechanism = Additive(TERMS, Noise(0.0))
    contributions = mechanism.contributions(parents)
    assert set(contributions) == {"x", "s", ("x", "y"), ("s", "y")}
    expected = sum(term.evaluate(parents) for term in TERMS)
    np.testing.assert_allclose(mechanism.evaluate(parents, np.zeros((N, 1))), expected)
    np.testing.assert_allclose(sum(contributions.values()), expected)


def test_aggregate_and_multiplicative_reduce_the_same_terms(parents):
    terms = Additive(TERMS, Noise(0.0)).terms(parents, N)
    zeros = np.zeros((N, 1))
    np.testing.assert_allclose(
        Aggregate(TERMS, Noise(0.0), op="max").evaluate(parents, zeros), terms.max(0)
    )
    clipped = np.expm1(np.clip(terms.sum(0), -1.0, 1.0))
    np.testing.assert_allclose(EXAMPLES["multiplicative"].evaluate(parents, zeros), clipped)


def test_lookup_effects_share_the_level_binning(parents):
    levels = bin_levels(parents["s"], PROBABILITIES)
    lookup = LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES).evaluate(parents)
    np.testing.assert_array_equal(lookup, np.asarray([10.0, 20.0, 30.0])[levels])
    assert set(np.unique(levels)) == {0, 1, 2}


def test_noise_is_heteroscedastic_and_takes_any_distribution(parents):
    rng = np.random.default_rng(2)
    noise = Noise(0.5, Beta(2.0, 2.0, low=-1.0, high=1.0), (Effect("y", 1.0),))
    draw = noise.sample(N, rng)
    assert draw.min() >= -1.0 and draw.max() <= 1.0
    scaled = noise.apply(parents, np.ones((N, 1)))
    np.testing.assert_allclose(scaled, 0.5 * np.exp(np.clip(parents["y"], -3.0, 3.0)))


def test_structure_is_exposed_for_the_oracle():
    mechanism = EXAMPLES["additive"]
    assert mechanism.parents == ("x", "s", "y")
    assert mechanism.mean_parents == ("x", "s", "y")
    assert mechanism.noise_parents == ("y",)
    assert mechanism.interaction_pairs == (("x", "y"), ("s", "y"))
    assert Root().parents == () and Root().interaction_pairs == ()
