import numpy as np
import pandas as pd
import pytest

from plurel.columns import DEFAULT_CALENDAR, Column
from plurel.distributions import Exponential, Gumbel, Normal, Uniform
from plurel.mechanisms import Combine, LinearEffect, MatrixEffect, NearestEffect
from plurel.scm import SCM

N = 300
TABLE = np.arange(6.0).reshape(3, 2)
MECHANISMS = {
    "y": Combine((LinearEffect("x", 2.0), LinearEffect("xz", -0.5)), noise=Normal(std=0.1)),
    "xz": Combine((LinearEffect("x"), LinearEffect("z")), op="product", noise=None),
    "x": Combine(),
    "z": Combine(),
    "h": Combine(dim=3),
    "segment": Combine((MatrixEffect("h", np.eye(3)),), onehot=True, noise=Gumbel()),
    "cluster": Combine((NearestEffect("h", np.eye(3)),), noise=None),
    "embedding": Combine((MatrixEffect("segment", TABLE),), noise=None),
    "hidden": Combine(
        (MatrixEffect("y", np.array([[0.0, 2.0]])),), bias=(0.0, -1.0), onehot=True, noise=Gumbel()
    ),
}
COLUMNS = {name: Column(name) for name in ("y", "xz", "x", "z")}


@pytest.fixture
def scm():
    return SCM(MECHANISMS, COLUMNS)


def test_simulate_evaluates_every_node_in_topological_order(scm):
    latents = scm.simulate(N, seed=0)
    assert set(latents) == set(MECHANISMS)
    assert scm.order.index("x") < scm.order.index("xz") < scm.order.index("y")
    for name, mechanism in MECHANISMS.items():
        assert latents[name].shape == (N, mechanism.dim)
    np.testing.assert_allclose(latents["xz"], latents["x"] * latents["z"])
    np.testing.assert_array_equal(latents["embedding"], TABLE[latents["segment"].argmax(1)])
    again = scm.simulate(N, seed=0)
    assert all(np.array_equal(latents[name], again[name]) for name in MECHANISMS)
    assert not np.array_equal(latents["x"], scm.simulate(N, seed=1)["x"])


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
        SCM({"y": Combine((LinearEffect("x"),))}, {})
    with pytest.raises(ValueError):
        SCM({"a": Combine((LinearEffect("b"),)), "b": Combine((LinearEffect("a"),))}, {})


def test_simulate_rejects_a_node_that_breaks_its_declared_width():
    scm = SCM({"h": Combine(dim=3), "y": Combine((LinearEffect("h"),))}, {})
    with pytest.raises(ValueError, match="declared"):
        scm.simulate(N, seed=0)
    overflow = SCM({"x": Combine(), "y": Combine((LinearEffect("x", np.inf),))}, {})
    with pytest.raises(ValueError, match="non-finite"):
        overflow.simulate(N, seed=0)


def test_sample_observes_columns_from_one_draw(scm):
    frame, latents = scm.sample_with_latents(N, seed=0)
    assert list(frame) == ["y", "xz", "x", "z"] and len(frame) == N
    np.testing.assert_array_equal(frame["x"], latents["x"].ravel())
    assert all(np.array_equal(latents[k], v) for k, v in scm.simulate(N, seed=0).items())
    columns = {
        "amount": Column("y", marginal=Uniform(), missing=0.1),
        "segment": Column("segment", "categorical", categories=("a", "b", "c")),
        "when": Column("z", "timestamp", marginal=DEFAULT_CALENDAR),
    }
    typed = SCM(MECHANISMS, columns)
    frame = typed.sample(N, seed=0)
    assert list(frame) == list(columns) and frame["when"].dtype == "datetime64[ns]"
    assert set(frame["segment"]) == {"a", "b", "c"} and 0.05 < frame["amount"].isna().mean() < 0.15
    same = typed.sample(N, seed=0, interventions={"x": 0.0})
    pd.testing.assert_series_equal(same["when"], frame["when"])
    mnar = SCM(MECHANISMS, {"y": Column("y", missing="hidden")}).sample_with_latents(N, seed=0)
    frame, latents = mnar
    assert frame["y"].isna().to_numpy().tolist() == (latents["hidden"][:, 1] == 1.0).tolist()
    assert latents["y"][frame["y"].isna()].mean() > latents["y"][frame["y"].notna()].mean()
    with pytest.raises(ValueError):
        SCM(MECHANISMS, {"c": Column("missing")})
    with pytest.raises(ValueError):
        SCM(MECHANISMS, {"c": Column("y", missing="nobody")})


def test_declared_time_order_is_enforced_on_the_observed_table():
    def table(delay, marginal=None):
        return SCM(
            {
                "placed": Combine(noise=DEFAULT_CALENDAR),
                "shipped": Combine((LinearEffect("placed"),), noise=delay),
            },
            {
                "placed": Column("placed", "timestamp", marginal=marginal),
                "shipped": Column("shipped", "timestamp", marginal=marginal, after="placed"),
            },
            time_column="placed",
        )

    frame = table(Exponential(3600.0)).sample(N, seed=0)
    assert (frame["shipped"] >= frame["placed"]).all()
    assert table(Exponential(3600.0)).timestamp_nodes == {"placed", "shipped"}
    with pytest.raises(ValueError, match="precedes"):
        table(Normal(std=3600.0)).sample(N, seed=0)
    with pytest.raises(ValueError, match="precedes"):
        table(Exponential(3600.0), marginal=DEFAULT_CALENDAR).sample(N, seed=0)
    for after in ("nothing", "t"):
        with pytest.raises(ValueError, match="another timestamp"):
            SCM({"t": Combine()}, {"t": Column("t", "timestamp", after=after)})
