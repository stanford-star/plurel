import numpy as np
import pytest

from plurel.distributions import Gumbel, Mixture, Normal
from plurel.mechanisms import (
    EFFECTS,
    MECHANISMS,
    REDUCTIONS,
    Combine,
    FourierEffect,
    LinearEffect,
    LookupEffect,
    MatrixEffect,
    Mechanism,
    MLPEffect,
    NearestEffect,
    QuadraticEffect,
    TreeEffect,
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
PARAMS = np.random.default_rng(0)
MLP = MLPEffect("h", (PARAMS.standard_normal((3, 4)), PARAMS.standard_normal((4, 2))))
TREE = TreeEffect(
    "h", np.array([[0, 2], [1, 1]]), np.zeros((2, 2)), PARAMS.standard_normal((2, 4, 2))
)
FOURIER = FourierEffect(
    "h",
    PARAMS.standard_normal((3, 5)),
    PARAMS.uniform(0.0, 2.0 * np.pi, 5),
    PARAMS.standard_normal((5, 2)),
)
QUADRATIC = QuadraticEffect("h", PARAMS.standard_normal((2, 4, 4)))
EFFECT_EXAMPLES = {
    "linear": TERMS[0],
    "lookup": TERMS[1],
    "matrix": SCORES[0],
    "nearest": NearestEffect("h", CENTERS),
    "mlp": MLP,
    "tree": TREE,
    "fourier": FOURIER,
    "quadratic": QUADRATIC,
}
EXAMPLES = {
    "root": Combine(dim=3, noise=Mixture((Normal(-2.0), Normal(2.0)))),
    "combine": Combine(TERMS, noise=Normal(std=0.5)),
    "onehot": Combine(SCORES, onehot=True, noise=Gumbel()),
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
def latents():
    rng = np.random.default_rng(0)
    scalars = {name: rng.standard_normal((N, 1)) for name in ("x", "y", "s")}
    return scalars | {"h": rng.standard_normal((N, 3))}


def test_every_registered_mechanism_meets_the_contract(latents):
    assert set(MECHANISMS) == {"combine"}
    for mechanism in EXAMPLES.values():
        assert isinstance(mechanism, Mechanism | MECHANISMS["combine"])
        exogenous = mechanism.sample_noise(N, np.random.default_rng(1))
        again = mechanism.sample_noise(N, np.random.default_rng(1))
        np.testing.assert_array_equal(exogenous, again)
        assert mechanism.evaluate(latents, exogenous).shape == (N, mechanism.dim)


def test_one_node_type_covers_roots_combines_and_one_hot_nodes(latents):
    root = Combine()
    assert root.dim == 1 and root.parents == () and isinstance(root.noise, Normal)
    exogenous = root.sample_noise(N, np.random.default_rng(0))
    np.testing.assert_array_equal(root.evaluate(latents, exogenous), exogenous)
    assert Combine(dim=4).sample_noise(N, np.random.default_rng(0)).shape == (N, 4)
    assert Combine(bias=(0.0, 1.0, 2.0)).dim == 3
    classes = Combine(bias=(0.0, 0.0, 5.0), onehot=True, noise=Gumbel())
    onehot = classes.evaluate(latents, classes.sample_noise(N, np.random.default_rng(0)))
    assert onehot.shape == (N, 3) and (onehot.sum(1) == 1).all() and onehot[:, 2].mean() > 0.9
    scores = Combine(SCORES, bias=(0.0, 0.0, 0.0), onehot=True, noise=None)
    assert scores.dim == 3 and scores.evaluate(latents, np.zeros((N, 3))).sum() == N
    for bad in (
        lambda: Combine(TERMS, dim=2),
        lambda: Combine(bias=(0.0, 1.0), dim=3),
        lambda: Combine(TERMS, bias=(0.0, 1.0)),
        lambda: Combine(onehot=True),
        lambda: Combine(TERMS, op="median"),
    ):
        with pytest.raises(ValueError):
            bad()


def test_every_registered_effect_declares_its_width(latents):
    assert set(EFFECT_EXAMPLES) == set(EFFECTS)
    for effect in EFFECT_EXAMPLES.values():
        x = latents[effect.parent]
        assert effect.apply(x).shape == (N, effect.dim)


def test_every_reduction_reduces_the_transformed_parents(latents):
    assert set(REFERENCE) == set(REDUCTIONS)
    terms = np.stack([effect.apply(latents[effect.parent]) for effect in TERMS])
    for op, reference in REFERENCE.items():
        mechanism = Combine(TERMS, op, noise=None)
        exogenous = mechanism.sample_noise(N, np.random.default_rng(0))
        assert exogenous.shape == (N, mechanism.dim) and not exogenous.any()
        np.testing.assert_allclose(mechanism.evaluate(latents, exogenous), reference(terms))


def test_block_effects_set_the_width_and_broadcast(latents):
    block = MatrixEffect("h", np.ones((3, 2)))
    mixed = Combine((block, LinearEffect("x")), noise=None)
    assert mixed.dim == 2
    expected = np.repeat(latents["h"].sum(1, keepdims=True) + latents["x"], 2, axis=1)
    np.testing.assert_allclose(mixed.evaluate(latents, np.zeros((N, 2))), expected)
    assert Combine((block, LinearEffect("x")), op="concat").dim == 3
    assert Combine((LinearEffect("h", dim=3),)).dim == 3
    node = Combine((MLP, TREE), op="logsumexp")
    assert node.dim == 2
    assert node.evaluate(latents, node.sample_noise(N, np.random.default_rng(0))).shape == (N, 2)


def test_lookup_effects_share_the_level_binning(latents):
    levels = bin_levels(latents["s"], PROBABILITIES)
    lookup = LookupEffect("s", (10.0, 20.0, 30.0), PROBABILITIES).apply(latents["s"])
    np.testing.assert_array_equal(lookup, np.asarray([10.0, 20.0, 30.0])[levels])
    assert set(np.unique(levels)) == {0, 1, 2}


def test_interactions_are_product_nodes(latents):
    zeros = np.zeros((N, 1))
    product = Combine((LinearEffect("x"), LinearEffect("y")), op="product")
    interaction = product.evaluate(latents, zeros)
    np.testing.assert_allclose(interaction, latents["x"] * latents["y"])
    target = Combine((LinearEffect("x", 2.0), LinearEffect("h", -0.5)))
    out = target.evaluate({**latents, "h": interaction}, zeros)
    np.testing.assert_allclose(out, 2.0 * latents["x"] - 0.5 * latents["x"] * latents["y"])
    assert target.parents == ("x", "h")


def test_one_hot_node_is_a_gumbel_argmax_over_the_combined_scores(latents):
    softmax = EXAMPLES["onehot"]
    assert softmax.dim == 3 and softmax.parents == ("x", "y")
    one_hot = softmax.evaluate(latents, np.zeros((N, 3)))
    scores = np.concatenate([latents["x"], -latents["x"], 2.0 * latents["y"]], axis=1)
    assert (one_hot.sum(1) == 1).all()
    np.testing.assert_array_equal(one_hot.argmax(1), scores.argmax(1))
    marginal = Combine(bias=tuple(np.log(PROBABILITIES)), onehot=True, noise=Gumbel())
    draws = marginal.evaluate({}, marginal.sample_noise(20_000, np.random.default_rng(0)))
    np.testing.assert_allclose(draws.mean(0), PROBABILITIES, atol=0.02)


def test_nested_levels_are_a_softmax_over_masked_logits():
    allowed = ((0, 1), (2,), (3, 4))
    rng = np.random.default_rng(0)
    country = Combine(bias=(0.0,) * 3, onehot=True, noise=Gumbel())
    city = Combine(
        (MatrixEffect("country", nested_logits(allowed, (0.2,) * 5)),), onehot=True, noise=Gumbel()
    )
    countries = country.evaluate({}, country.sample_noise(N, rng))
    cities = city.evaluate({"country": countries}, city.sample_noise(N, rng))
    assert city.dim == 5
    for code, subset in enumerate(allowed):
        assert set(cities[countries.argmax(1) == code].argmax(1)) <= set(subset)


def test_nearest_effect_one_hot_encodes_the_closest_center(latents):
    one_hot = NearestEffect("h", CENTERS).apply(latents["h"])
    assert (one_hot.sum(1) == 1).all()
    np.testing.assert_array_equal(one_hot.argmax(1), latents["h"].argmax(1))
    table = np.arange(9.0).reshape(3, 3)
    np.testing.assert_array_equal(MatrixEffect("c", table).apply(one_hot), table[one_hot.argmax(1)])


def test_mlp_effect_places_activations_between_layers(latents):
    h = latents["h"]
    w1, w2 = MLP.weights
    np.testing.assert_allclose(MLP.apply(h), h @ w1 @ w2)
    hidden = MLPEffect("h", (w1, w2), activations=("identity", "tanh", "identity"))
    np.testing.assert_allclose(hidden.apply(h), np.tanh(h @ w1) @ w2)
    first = MLPEffect("h", (w1,), activations=("tanh", "identity"))
    np.testing.assert_allclose(first.apply(h), np.tanh(h) @ w1)
    with pytest.raises(ValueError):
        MLPEffect("h", (w1,), activations=("tanh",))


def test_tree_effect_averages_oblivious_tree_leaves(latents):
    h = latents["h"]
    expected = np.zeros((N, 2))
    for tree, (dims, points) in enumerate(zip(TREE.split_dims, TREE.split_points)):
        index = sum((h[:, d] > p).astype(int) << k for k, (d, p) in enumerate(zip(dims, points)))
        expected += TREE.leaves[tree, index]
    np.testing.assert_allclose(TREE.apply(h), expected / 2)


def test_fourier_and_quadratic_effects_match_their_formulas(latents):
    h = latents["h"]
    features = np.cos(h @ FOURIER.frequencies + FOURIER.phases)
    np.testing.assert_allclose(FOURIER.apply(h), features @ FOURIER.weights)
    ones = np.concatenate([h, np.ones((N, 1))], axis=1)
    expected = np.stack([np.einsum("ni,ij,nj->n", ones, a, ones) for a in QUADRATIC.tensor], axis=1)
    np.testing.assert_allclose(QUADRATIC.apply(h), expected)
