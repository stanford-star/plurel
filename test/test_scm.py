import numpy as np
import pytest

from plurel.distributions import Normal
from plurel.mechanisms import Combine, LinearEffect, MatrixEffect, NearestEffect, Root, Softmax
from plurel.scm import SCM

N = 300
TABLE = np.arange(6.0).reshape(3, 2)
MECHANISMS = {
    "y": Combine((LinearEffect("x", 2.0), LinearEffect("xz", -0.5)), noise=Normal(std=0.1)),
    "xz": Combine((LinearEffect("x"), LinearEffect("z")), op="product", noise=None),
    "x": Root(),
    "z": Root(),
    "h": Root(dim=3),
    "segment": Softmax((MatrixEffect("h", np.eye(3)),)),
    "cluster": Combine((NearestEffect("h", np.eye(3)),), noise=None),
    "embedding": Combine((MatrixEffect("segment", TABLE),), noise=None),
}


@pytest.fixture
def scm():
    return SCM(MECHANISMS)


def test_simulate_evaluates_every_node_in_topological_order(scm):
    values = scm.simulate(N, seed=0)
    assert set(values) == set(MECHANISMS)
    assert scm.order.index("x") < scm.order.index("xz") < scm.order.index("y")
    for name, mechanism in MECHANISMS.items():
        assert values[name].shape == (N, mechanism.dim)
    np.testing.assert_allclose(values["xz"], values["x"] * values["z"])
    np.testing.assert_array_equal(values["embedding"], TABLE[values["segment"].argmax(1)])
    again = scm.simulate(N, seed=0)
    assert all(np.array_equal(values[name], again[name]) for name in MECHANISMS)
    assert not np.array_equal(values["x"], scm.simulate(N, seed=1)["x"])


def test_interventions_replace_a_node_and_keep_common_random_numbers(scm):
    factual = scm.simulate(N, seed=0)
    counterfactual = scm.simulate(N, seed=0, interventions={"x": 0.0})
    assert not counterfactual["x"].any()
    for name in ("z", "h", "segment"):
        np.testing.assert_array_equal(counterfactual[name], factual[name])
    residuals = [v["y"] - 2.0 * v["x"] + 0.5 * v["x"] * v["z"] for v in (factual, counterfactual)]
    np.testing.assert_allclose(*residuals)
    forced = scm.simulate(N, seed=0, interventions={"segment": np.eye(3)[1]})
    np.testing.assert_array_equal(forced["embedding"], np.tile(TABLE[1], (N, 1)))
    per_row = scm.simulate(N, seed=0, interventions={"x": np.arange(N)})
    np.testing.assert_array_equal(per_row["x"], np.arange(N)[:, None])
    with pytest.raises(ValueError):
        scm.simulate(N, seed=0, interventions={"missing": 1.0})
    with pytest.raises(ValueError):
        scm.simulate(N, seed=0, interventions={"h": np.ones((N, 2))})


def test_construction_rejects_unknown_parents_and_cycles():
    with pytest.raises(ValueError):
        SCM({"y": Combine((LinearEffect("x"),))})
    with pytest.raises(ValueError):
        SCM({"a": Combine((LinearEffect("b"),)), "b": Combine((LinearEffect("a"),))})


def test_simulate_rejects_a_node_that_breaks_its_declared_width():
    scm = SCM({"h": Root(dim=3), "y": Combine((LinearEffect("h"),))})
    with pytest.raises(ValueError, match="declared"):
        scm.simulate(N, seed=0)
