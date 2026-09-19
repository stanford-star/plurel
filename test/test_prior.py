import numpy as np
import pytest

from plurel.mechanisms import EFFECTS, Root, Softmax
from plurel.prior import FAMILIES, Choices, Range, TablePrior


def test_choices_and_ranges_draw_within_their_declarations():
    rng = np.random.default_rng(0)
    weighted = Choices(("a", "b", "c"), (0.0, 1.0, 3.0))
    draws = [weighted.draw(rng) for _ in range(2000)]
    assert "a" not in draws and 0.6 < draws.count("c") / 2000 < 0.9
    assert Choices((7,)).draw(rng) == 7
    log_ints = [Range(1, 32, log=True, integer=True).draw(rng) for _ in range(2000)]
    assert min(log_ints) == 1 and max(log_ints) == 32 and np.median(log_ints) < 8
    floats = [Range(-1.0, 1.0).draw(rng) for _ in range(2000)]
    assert -1.0 <= min(floats) and max(floats) < 1.0 and abs(np.mean(floats)) < 0.1
    for bad in (
        lambda: Choices(()),
        lambda: Choices((1, 2), (1.0,)),
        lambda: Choices((1, 2), (0.0, 0.0)),
        lambda: Range(2.0, 1.0),
        lambda: Range(0.0, 1.0, log=True),
    ):
        with pytest.raises(ValueError):
            bad()


def test_table_prior_realizes_valid_diverse_tables():
    prior = TablePrior()
    families, kinds = set(), set()
    for seed in range(40):
        scm = prior.realize(seed)
        assert prior.realize(seed).order == scm.order
        frame = scm.sample(200, seed=seed)
        assert frame.equals(scm.sample(200, seed=seed)) and len(frame) == 200
        assert scm.pkey_column == "id" and 4 <= len(frame.columns) <= 14
        assert 3 <= sum(name.startswith("n") for name in scm.mechanisms) <= 16
        for mechanism in scm.mechanisms.values():
            assert 1 <= mechanism.dim <= 8
            families.update(type(effect) for effect in getattr(mechanism, "effects", ()))
        kinds.update(column.kind for column in scm.columns.values())
    assert families == {EFFECTS[name] for name in FAMILIES} - {EFFECTS["lookup"]} | {
        EFFECTS["linear"]
    }
    assert kinds == {"key", "numeric", "categorical", "timestamp"}


def test_table_prior_knobs_are_respected():
    plain = TablePrior(categorical=0.0, timestamp=0.0, binned=0.0, missing_share=0.0)
    for seed in range(10):
        scm = plain.realize(seed)
        assert not any(isinstance(m, Softmax) for m in scm.mechanisms.values())
        assert scm.time_column is None and all(
            c.kind != "categorical" for c in scm.columns.values()
        )
        assert all(c.missing == 0.0 for c in scm.columns.values())
        assert not scm.sample(50, seed=seed).isna().any().any()
    single = TablePrior(nodes=Range(1, 1, integer=True), columns=Range(1, 1, integer=True))
    scm = single.realize(0)
    assert len(scm.mechanisms) <= 2 and isinstance(scm.mechanisms["n0"], Root | Softmax)
    assert scm.sample(5, seed=0).shape[0] == 5
