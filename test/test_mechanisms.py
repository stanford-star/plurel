import numpy as np
import pytest

from plurel.distributions import Mixture, Normal
from plurel.mechanisms import (
    EFFECTS,
    MECHANISMS,
    REDUCTIONS,
    Combine,
    LinearEffect,
    LookupEffect,
    MatrixEffect,
    Mechanism,
    NearestEffect,
    Root,
    Softmax,
    bin_levels,
    nested_logits,
)

N = 200
PROBABILITIES = (0.2, 0.5, 0.3)
CENTERS = np.eye(3)
TERMS = (
    LinearEffect("x", 1.5, "tanh"),
    LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES),
    LinearEffect("y", 0.5, "exp"),
)
SCORES = (
    MatrixEffect("x", np.array([[1.0, -1.0, 0.0]])),
    MatrixEffect("y", np.array([[0.0, 0.0, 2.0]])),
)
EFFECT_EXAMPLES = {
    "linear": TERMS[0],
    "lookup": TERMS[1],
    "matrix": SCORES[0],
    "nearest": NearestEffect("h", CENTERS),
}
EXAMPLES = {
    "root": Root(dim=3, noise=Mixture((Normal(-2.0), Normal(2.0)))),
    "combine": Combine(TERMS, noise=Normal(std=0.5)),
    "softmax": Softmax(SCORES),
}
REFERENCE = {
    "sum": lambda t: t.sum(0),
    "product": lambda t: t.prod(0),
    "max": lambda t: t.max(0),
    "min": lambda t: t.min(0),
    "logsumexp": lambda t: np.log(np.exp(t).sum(0)),
    "concat": lambda t: np.concatenate(t, axis=1),
}


@pytest.fixture
def values():
    rng = np.random.default_rng(0)
    scalars = {name: rng.standard_normal((N, 1)) for name in ("x", "y", "s")}
    return scalars | {"h": rng.standard_normal((N, 3))}


def test_every_registered_mechanism_meets_the_contract(values):
    assert set(EXAMPLES) == set(MECHANISMS)
    for mechanism in EXAMPLES.values():
        assert isinstance(mechanism, Mechanism)
        exogenous = mechanism.sample_noise(N, np.random.default_rng(1))
        again = mechanism.sample_noise(N, np.random.default_rng(1))
        np.testing.assert_array_equal(exogenous, again)
        assert mechanism.evaluate(values, exogenous).shape == (N, mechanism.dim)


def test_every_registered_effect_declares_its_width(values):
    assert set(EFFECT_EXAMPLES) == set(EFFECTS)
    for effect in EFFECT_EXAMPLES.values():
        x = values[effect.parent]
        assert effect.apply(x).shape == (N, effect.dim or x.shape[1])


def test_every_reduction_reduces_the_transformed_parents(values):
    assert set(REFERENCE) == set(REDUCTIONS)
    terms = np.stack([effect.apply(values[effect.parent]) for effect in TERMS])
    for op, reference in REFERENCE.items():
        mechanism = Combine(TERMS, op, noise=None)
        exogenous = mechanism.sample_noise(N, np.random.default_rng(0))
        assert exogenous.shape == (N, mechanism.dim) and not exogenous.any()
        np.testing.assert_allclose(mechanism.evaluate(values, exogenous), reference(terms))


def test_block_effects_set_the_width_and_broadcast(values):
    block = MatrixEffect("h", np.ones((3, 2)))
    mixed = Combine((block, LinearEffect("x")), noise=None)
    assert mixed.dim == 2
    expected = np.repeat(values["h"].sum(1, keepdims=True) + values["x"], 2, axis=1)
    np.testing.assert_allclose(mixed.evaluate(values, np.zeros((N, 2))), expected)
    assert Combine((block, LinearEffect("x")), op="concat").dim == 3
    assert Combine((LinearEffect("h"),), dim=3).dim == 3


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


def test_softmax_is_a_gumbel_argmax_over_the_combined_scores(values):
    softmax = EXAMPLES["softmax"]
    assert softmax.dim == 3 and softmax.parents == ("x", "y")
    one_hot = softmax.evaluate(values, np.zeros((N, 3)))
    scores = np.concatenate([values["x"], -values["x"], 2.0 * values["y"]], axis=1)
    assert (one_hot.sum(1) == 1).all()
    np.testing.assert_array_equal(one_hot.argmax(1), scores.argmax(1))
    marginal = Softmax(biases=tuple(np.log(PROBABILITIES)), dim=3)
    draws = marginal.evaluate({}, marginal.sample_noise(20_000, np.random.default_rng(0)))
    np.testing.assert_allclose(draws.mean(0), PROBABILITIES, atol=0.02)


def test_nested_levels_are_a_softmax_over_masked_logits():
    allowed = ((0, 1), (2,), (3, 4))
    rng = np.random.default_rng(0)
    country = Softmax(dim=3)
    city = Softmax((MatrixEffect("country", nested_logits(allowed, (0.2,) * 5)),))
    countries = country.evaluate({}, country.sample_noise(N, rng))
    cities = city.evaluate({"country": countries}, city.sample_noise(N, rng))
    assert city.dim == 5
    for code, subset in enumerate(allowed):
        assert set(cities[countries.argmax(1) == code].argmax(1)) <= set(subset)


def test_nearest_effect_one_hot_encodes_the_closest_center(values):
    one_hot = NearestEffect("h", CENTERS).apply(values["h"])
    assert (one_hot.sum(1) == 1).all()
    np.testing.assert_array_equal(one_hot.argmax(1), values["h"].argmax(1))
    table = np.arange(9.0).reshape(3, 3)
    np.testing.assert_array_equal(MatrixEffect("c", table).apply(one_hot), table[one_hot.argmax(1)])
