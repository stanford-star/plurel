import numpy as np
import pytest

from plurel.distributions import Beta, Mixture, Normal
from plurel.mechanisms import (
    MECHANISMS,
    REDUCTIONS,
    Combine,
    Linear,
    Lookup,
    LookupScale,
    Mechanism,
    Noise,
    Product,
    Root,
    TransformedProduct,
    bin_levels,
)

N = 200
PROBABILITIES = (0.2, 0.5, 0.3)
TERMS = (
    Linear("x", 1.5, "tanh"),
    Lookup("s", (10.0, 20.0, 30.0), PROBABILITIES),
    Product(("x", "y"), 0.5),
    LookupScale("s", "y", (2.0, -3.0, 1.0), PROBABILITIES),
    TransformedProduct(("x", "y"), 0.25, "step"),
)
EXAMPLES = {
    "root": Root(Mixture((Normal(-2.0), Normal(2.0))), dim=3),
    "combine": Combine(TERMS, Noise(0.5, scale_effects=(Linear("y", 0.3),))),
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


def test_combine_sums_its_contributions(parents):
    mechanism = Combine(TERMS, Noise(0.0))
    contributions = mechanism.contributions(parents)
    assert set(contributions) == {"x", "s", ("x", "y"), ("s", "y")}
    expected = sum(term.evaluate(parents) for term in TERMS)
    np.testing.assert_allclose(mechanism.evaluate(parents, np.zeros((N, 1))), expected)
    np.testing.assert_allclose(sum(contributions.values()), expected)


def test_every_reduction_reduces_the_same_terms(parents):
    terms = Combine(TERMS, Noise(0.0)).terms(parents, N)
    zeros = np.zeros((N, 1))
    for op, reduce in REDUCTIONS.items():
        np.testing.assert_allclose(
            Combine(TERMS, Noise(0.0), op).evaluate(parents, zeros), reduce(terms)
        )


def test_lookup_effects_share_the_level_binning(parents):
    levels = bin_levels(parents["s"], PROBABILITIES)
    lookup = Lookup("s", (10.0, 20.0, 30.0), PROBABILITIES).evaluate(parents)
    np.testing.assert_array_equal(lookup, np.asarray([10.0, 20.0, 30.0])[levels])
    assert set(np.unique(levels)) == {0, 1, 2}


def test_noise_is_heteroscedastic_and_takes_any_distribution(parents):
    rng = np.random.default_rng(2)
    noise = Noise(0.5, Beta(2.0, 2.0, low=-1.0, high=1.0), (Linear("y", 1.0),))
    draw = noise.sample(N, rng)
    assert draw.min() >= -1.0 and draw.max() <= 1.0
    scaled = noise.apply(parents, np.ones((N, 1)))
    np.testing.assert_allclose(scaled, 0.5 * np.exp(np.clip(parents["y"], -3.0, 3.0)))


def test_structure_is_exposed_for_the_oracle():
    mechanism = EXAMPLES["combine"]
    assert mechanism.parents == ("x", "s", "y")
    assert mechanism.mean_parents == ("x", "s", "y")
    assert mechanism.noise_parents == ("y",)
    assert mechanism.interaction_pairs == (("x", "y"), ("s", "y"))
    assert Root().parents == () and Root().interaction_pairs == ()
