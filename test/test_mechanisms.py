import numpy as np
import pytest

from plurel.distributions import Mixture, Normal
from plurel.mechanisms import (
    MECHANISMS,
    REDUCTIONS,
    ArgmaxScores,
    Combine,
    Embed,
    LinearEffect,
    LookupEffect,
    Mechanism,
    Nearest,
    NestedLevels,
    Root,
    Softmax,
    bin_levels,
)

N = 200
PROBABILITIES = (0.2, 0.5, 0.3)
TERMS = (
    LinearEffect("x", 1.5, "tanh"),
    LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES),
    LinearEffect("y", 0.5, "exp"),
)
CENTERS = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
SCORES = ((LinearEffect("x", 1.0),), (LinearEffect("x", -1.0),), (LinearEffect("y", 2.0),))
EXAMPLES = {
    "root": Root(dim=3, noise=Mixture((Normal(-2.0), Normal(2.0)))),
    "combine": Combine(TERMS, noise=Normal(std=0.5)),
    "argmax_scores": ArgmaxScores(SCORES, biases=(0.0, 0.0, -1.0)),
    "nested_levels": NestedLevels("s", PROBABILITIES, ((0, 1), (2,), (3, 4)), (0.2,) * 5),
    "nearest": Nearest("block", CENTERS),
    "softmax": Softmax("block", 3, scale=5.0),
    "embed": Embed("code", CENTERS),
}


@pytest.fixture
def parents():
    rng = np.random.default_rng(0)
    parents = {name: rng.standard_normal((N, 1)) for name in ("x", "y", "s")}
    parents["block"] = rng.standard_normal((N, 3))
    parents["code"] = rng.integers(0, 3, (N, 1)).astype(float)
    return parents


def test_every_registered_mechanism_meets_the_contract(parents):
    assert set(EXAMPLES) == set(MECHANISMS)
    for mechanism in EXAMPLES.values():
        assert isinstance(mechanism, Mechanism)
        exogenous = mechanism.sample_noise(N, np.random.default_rng(1))
        again = mechanism.sample_noise(N, np.random.default_rng(1))
        np.testing.assert_array_equal(exogenous, again)
        assert mechanism.evaluate(parents, exogenous).shape == (N, mechanism.dim)


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


def test_argmax_scores_claims_regions_by_identity(parents):
    mechanism = EXAMPLES["argmax_scores"]
    codes = mechanism.evaluate(parents, np.zeros((N, 3))).ravel()
    np.testing.assert_array_equal(codes, mechanism.score_matrix(parents, N).argmax(1))
    assert set(codes) == {0, 1, 2}


def test_nested_levels_stay_within_the_parent_subset(parents):
    rng = np.random.default_rng(3)
    country = EXAMPLES["nested_levels"]
    city = NestedLevels("country", None, ((0,), (1,), (2,), (3,), (4, 5)), (1.0 / 6,) * 6)
    countries = country.evaluate(parents, country.sample_noise(N, rng)).ravel().astype(int)
    cities = city.evaluate({"country": countries[:, None]}, city.sample_noise(N, rng)).ravel()
    levels = bin_levels(parents["s"], PROBABILITIES).ravel()
    assert all(c in country.allowed[level] for c, level in zip(countries, levels))
    assert all(c in city.allowed[parent] for c, parent in zip(cities, countries))


def test_nearest_then_embed_recovers_the_closest_center(parents):
    codes = Nearest("block", CENTERS).evaluate(parents, np.zeros((N, 0)))
    embedded = Embed("code", CENTERS).evaluate({"code": codes}, np.zeros((N, 0)))
    distances = ((parents["block"][:, None] - CENTERS[None]) ** 2).sum(-1)
    np.testing.assert_array_equal(embedded, CENTERS[distances.argmin(1)])


def test_softmax_follows_its_block_and_biases(parents):
    sharp = Softmax("block", 3, scale=50.0).evaluate(parents, np.zeros((N, 3))).ravel()
    np.testing.assert_array_equal(sharp, parents["block"].argmax(1))
    biased = Softmax("block", 3, scale=0.0, biases=(0.0, 0.0, 5.0))
    assert set(biased.evaluate(parents, np.zeros((N, 3))).ravel()) == {2}
