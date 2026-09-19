import numpy as np
import pytest

from plurel.io import create_database
from plurel.links import TreeLink
from plurel.mechanisms import EFFECTS, Root, Softmax
from plurel.prior import (
    FAMILIES,
    Choices,
    IntegersRange,
    LogIntegersRange,
    LogRange,
    Range,
    SchemaPrior,
    TablePrior,
)


def test_choices_and_ranges_draw_within_their_declarations():
    rng = np.random.default_rng(0)
    weighted = Choices(("a", "b", "c"), (0.0, 1.0, 3.0))
    draws = [weighted.draw(rng) for _ in range(2000)]
    assert "a" not in draws and 0.6 < draws.count("c") / 2000 < 0.9
    assert Choices((7,)).draw(rng) == 7
    log_ints = [LogIntegersRange(1, 32).draw(rng) for _ in range(4000)]
    assert min(log_ints) == 1 and max(log_ints) == 32 and np.median(log_ints) < 8
    counts = np.bincount([IntegersRange(2, 4).draw(rng) for _ in range(6000)], minlength=5)[2:]
    assert (abs(counts / 6000 - 1 / 3) < 0.03).all()
    floats = [Range(-1.0, 1.0).draw(rng) for _ in range(2000)]
    assert -1.0 <= min(floats) and max(floats) < 1.0 and abs(np.mean(floats)) < 0.1
    logs = [LogRange(0.01, 100.0).draw(rng) for _ in range(2000)]
    assert 0.01 <= min(logs) and max(logs) < 100.0 and 0.5 < np.median(logs) < 2.0
    for bad in (
        lambda: Choices(()),
        lambda: Choices((1, 2), (1.0,)),
        lambda: Choices((1, 2), (0.0, 0.0)),
        lambda: Range(2.0, 1.0),
        lambda: LogRange(0.0, 1.0),
        lambda: LogIntegersRange(0, 4),
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
    assert families == {EFFECTS[name] for name in FAMILIES}
    assert kinds == {"key", "numeric", "categorical", "timestamp"}


def test_table_prior_knobs_are_respected():
    plain = TablePrior(
        node_categorical_share=0.0,
        time_probability=0.0,
        column_binned_share=0.0,
        column_missing_share=0.0,
    )
    for seed in range(10):
        scm = plain.realize(seed)
        assert not any(isinstance(m, Softmax) for m in scm.mechanisms.values())
        assert scm.time_column is None
        assert all(c.kind != "categorical" for c in scm.columns.values())
        assert all(c.missing == 0.0 for c in scm.columns.values())
        assert not scm.sample(50, seed=seed).isna().any().any()
    with pytest.raises(ValueError, match="change width"):
        TablePrior(effect_families=Choices(("linear",)))
    single = TablePrior(node_count=IntegersRange(1, 1), column_count=IntegersRange(1, 1))
    scm = single.realize(0)
    assert len(scm.mechanisms) <= 2 and isinstance(scm.mechanisms["n0"], Root | Softmax)
    assert scm.sample(5, seed=0).shape[0] == 5


def test_warping_gives_each_realization_its_own_style():
    rng = np.random.default_rng(0)
    tight = Range(0.0, 1.0, shape=(1000.0, 1000.0))
    assert all(abs(tight.draw(rng) - 0.5) < 0.1 for _ in range(100))
    skewed = LogIntegersRange(1, 32, shape=(0.5, 20.0))
    assert np.median([skewed.draw(rng) for _ in range(500)]) == 1
    warped = IntegersRange(2, 8).warp(rng)
    assert warped.shape is not None and all(2 <= warped.draw(rng) <= 8 for _ in range(200))
    choices = Choices(("a", "b", "c"), (1.0, 1.0, 0.0)).warp(rng)
    assert choices.values == ("a", "b", "c") and choices.weights[2] == 0.0
    prior = TablePrior().warp(rng)
    assert prior.node_count.shape is not None and prior.effect_families.weights is not None
    assert prior.node_categorical_share == TablePrior().node_categorical_share
    meta = [
        np.mean([m.dim for m in TablePrior().realize(seed).mechanisms.values()])
        for seed in range(60)
    ]
    flat = [
        np.mean(
            [m.dim for m in TablePrior().build(np.random.default_rng(seed)).mechanisms.values()]
        )
        for seed in range(60)
    ]
    assert np.var(meta) > 1.5 * np.var(flat)


SMALL = dict(entity_row_count=IntegersRange(60, 120), activity_row_count=IntegersRange(300, 600))


def test_schema_prior_realizes_databases_that_influence_each_other_both_ways():
    prior = SchemaPrior(**SMALL)
    seen = set()
    for seed in range(25):
        schema = prior.realize(seed)
        assert prior.realize(seed).order == schema.order
        rows = prior.rows(schema, seed)
        referenced = {fk.parent for fk in schema.fkeys if fk.parent != fk.table}
        for table, count in rows.items():
            low, high = (60, 120) if table in referenced else (300, 600)
            assert low <= count <= high
        frames, latents = schema.sample_with_latents(rows, seed=seed)
        assert all(len(frames[t]) == n for t, n in rows.items())
        for (table, name), (port, fk) in schema.ports.items():
            seen.add("aggregate" if port.aggregate else "gather")
            if isinstance(fk.link, TreeLink):
                seen.add("self")
            if name.endswith("_time"):
                seen.add("follow")
                assert port.fill is None and fk.nullable == 0.0
                child_time = frames[fk.table][schema.tables[fk.table].time_column]
                parent_time = frames[fk.parent][schema.tables[fk.parent].time_column]
                keys = frames[fk.table][fk.column].to_numpy(dtype=int)
                assert (child_time.to_numpy() >= parent_time.to_numpy()[keys]).all()
            if port.aggregate in ("mean", "max", "min"):
                keys = frames[fk.table][fk.column].dropna().astype(int)
                childless = ~np.isin(np.arange(rows[table]), keys)
                assert frames[table][name].isna().to_numpy().tolist() == childless.tolist()
        create_database(schema, frames)
    assert seen == {"gather", "aggregate", "self", "follow"}


def test_schema_prior_knobs_switch_cross_table_structure_off():
    quiet = SchemaPrior(
        **SMALL,
        gather_count=IntegersRange(0, 0),
        aggregate_count=IntegersRange(0, 0),
        self_reference_probability=0.0,
        time_follow_probability=0.0,
        fk_nullable_share=0.0,
    )
    for seed in range(8):
        schema = quiet.realize(seed)
        assert not schema.ports and all(fk.nullable == 0.0 for fk in schema.fkeys)
        assert all(fk.table != fk.parent for fk in schema.fkeys)
        frames = schema.sample(quiet.rows(schema, seed), seed=seed)
        assert all(frames[fk.table][fk.column].notna().all() for fk in schema.fkeys)


def test_schema_prior_edge_cases():
    single = SchemaPrior(**SMALL, table_count=IntegersRange(1, 1), self_reference_probability=1.0)
    for seed in range(6):
        schema = single.realize(seed)
        assert len(schema.tables) == 1 and all(fk.table == fk.parent for fk in schema.fkeys)
        schema.sample(single.rows(schema, seed), seed=seed)
    tiny = SchemaPrior(
        **SMALL,
        table_count=IntegersRange(1, 1),
        table_prior=TablePrior(node_count=IntegersRange(1, 2), time_probability=0.0),
        self_reference_probability=1.0,
        gather_count=IntegersRange(3, 3),
    )
    for seed in range(12):
        schema = tiny.realize(seed)
        for (table, name), (port, fk) in schema.ports.items():
            assert name != port.node and fk.table == fk.parent
        schema.sample(tiny.rows(schema, seed), seed=seed)
    with pytest.raises(ValueError, match="clusters"):
        SchemaPrior(entity_row_count=IntegersRange(5, 10), activity_row_count=IntegersRange(5, 10))
