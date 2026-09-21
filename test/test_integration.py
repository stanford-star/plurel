import numpy as np
import pandas as pd
import pytest

from plurel import (
    DEFAULT_CALENDAR,
    SCM,
    Column,
    Exponential,
    LinearEdge,
    LogNormal,
    MatrixEdge,
    Node,
    Normal,
    Pareto,
    Uniform,
    create_database,
    read_database,
    write_database,
)
from plurel.distributions import Gumbel
from plurel.links import HSBMLink, RandomLink, TreeLink
from plurel.schema import COMPLETE, FK, Foreign, Schema, Summary


def random_schema(rng):
    names = [f"t{i}" for i in range(int(rng.integers(1, 4)))]
    mechanisms = {t: {} for t in names}
    columns = {t: {f"{t}_id": Column(kind="key")} for t in names}
    fkeys = []
    for t in names:
        roots = [f"r{j}" for j in range(int(rng.integers(1, 4)))]
        for j, root in enumerate(roots):
            mechanisms[t][root] = Node(noise=rng.choice([Normal(), Uniform(), LogNormal()]))
            marginal = rng.choice([None, Uniform(), LogNormal(), Pareto(2.0)])
            columns[t][f"c{j}"] = Column(
                root, marginal=marginal, missing=float(rng.choice([0.0, 0.2]))
            )
        k = int(rng.integers(2, 5))
        mechanisms[t]["seg"] = Node(bias=tuple(rng.normal(size=k)), onehot=True, noise=Gumbel())
        columns[t]["seg"] = Column(
            "seg", "categorical", categories=tuple(f"s{i}" for i in range(k))
        )
        mechanisms[t]["h"] = Node(dim=int(rng.integers(1, 4)))
        if rng.random() < 0.5:
            mechanisms[t]["stamp"] = Node(noise=DEFAULT_CALENDAR)
            mechanisms[t]["later"] = Node((LinearEdge("stamp"),), noise=Exponential(3600.0))
            columns[t]["stamp"] = Column("stamp", "timestamp")
            columns[t]["later"] = Column("later", "timestamp", after="stamp")
        terms = tuple(
            LinearEdge(root, float(rng.normal()), str(rng.choice(["identity", "tanh", "square"])))
            for root in roots
        )
        mechanisms[t]["y"] = Node(terms, noise=Normal(std=0.3))
        columns[t]["y"] = Column("y", marginal=rng.choice([None, Uniform(-1.0, 1.0)]))
        binning = str(rng.choice(["normal", "empirical"]))
        columns[t]["bin"] = Column(
            "y",
            "categorical",
            categories=("lo", "mid", "hi"),
            probabilities=(0.2, 0.3, 0.5),
            binning=binning,
        )
    for i, t in enumerate(names[1:], start=1):
        parent = names[int(rng.integers(0, i))]
        link = rng.choice(
            [
                RandomLink(),
                HSBMLink((2,), (2,), cluster_weights=Pareto(1.5)),
                HSBMLink((1, 2), (2, 1), popularity=Pareto(2.0), inactive=0.2),
            ]
        )
        key = f"{parent}_fk"
        fkeys.append(FK(t, key, parent, link, nullable=float(rng.choice([0.0, 0.2])), fill=0.0))
        observed = bool(rng.random() < 0.5)
        width = mechanisms[parent]["seg"].dim
        mechanisms[t]["g_r0"] = Node((LinearEdge(Foreign(key, "r0")),), noise=None)
        mechanisms[t]["g_seg"] = Node((LinearEdge(Foreign(key, "seg"), dim=width),), noise=None)
        columns[t]["g_seg"] = Column(
            "g_seg", "categorical", categories=tuple(f"p{i}" for i in range(width))
        )
        if observed:
            columns[t]["g_r0"] = Column("g_r0", marginal=Uniform())
        else:
            edges = (LinearEdge("g_r0", 2.0), MatrixEdge("g_seg", np.ones((width, 1))))
            mechanisms[t]["z"] = Node(edges, noise=Normal(std=0.1))
            columns[t]["z"] = Column("z")
        how = str(rng.choice(["count", "sum", "mean", "max"]))
        summary = Summary(t, key, "y", how, fill=None if how in COMPLETE else 0.0)
        mechanisms[parent][f"a_{t}"] = Node((LinearEdge(summary),), noise=None)
        mechanisms[parent][f"w_{t}"] = Node((LinearEdge(f"a_{t}", 0.5),), noise=Normal())
        columns[parent][f"a_{t}"] = Column(f"a_{t}")
        columns[parent][f"w_{t}"] = Column(f"w_{t}")
    if rng.random() < 0.4:
        t = names[0]
        fkeys.append(FK(t, "boss", t, TreeLink(roots=0.3), fill=0.0))
        mechanisms[t]["boss_r0"] = Node((LinearEdge(Foreign("boss", "r0")),), noise=None)
        mechanisms[t]["reports"] = Node(
            (LinearEdge(Summary(t, "boss", "r0", "count")),), noise=None
        )
        columns[t]["reports"] = Column("reports")
    tables = {
        t: SCM(mechanisms[t], columns[t], time_column="stamp" if "stamp" in columns[t] else None)
        for t in names
    }
    return Schema(tables, tuple(fkeys))


@pytest.mark.parametrize("seed", range(30))
def test_random_schemas_sample_wrap_and_round_trip(seed, tmp_path):
    rng = np.random.default_rng(seed)
    schema = random_schema(rng)
    rows = {t: int(rng.choice([0, 2, 7, 40, 150])) for t in schema.tables}
    for fk in schema.fkeys:
        rows[fk.parent] = max(rows[fk.parent], 4)
    frames, latents = schema.sample_with_latents(rows, seed=seed)
    again, _ = schema.sample_with_latents(rows, seed=seed)
    for t, scm in schema.tables.items():
        pd.testing.assert_frame_equal(frames[t], again[t])
        assert len(frames[t]) == rows[t]
        np.testing.assert_array_equal(frames[t][scm.pkey_column], np.arange(rows[t]))
        for name, latent in latents[t].items():
            assert latent.shape == (rows[t], scm.nodes[name].dim)
        for name, column in scm.columns.items():
            if column.kind == "categorical":
                assert frames[t][name].dropna().isin(column.categories).all()
            if column.kind == "timestamp":
                assert frames[t][name].dtype == "datetime64[ns]"
    for fk in schema.fkeys:
        keys = frames[fk.table][fk.column]
        linked = keys.dropna().astype(int)
        assert keys.dtype == "Int64" and linked.between(0, rows[fk.parent] - 1).all()
        if isinstance(fk.link, TreeLink):
            assert (linked.to_numpy() < linked.index.to_numpy()).all()
        elif fk.nullable == 0:
            assert keys.notna().all()
    for fk in schema.fkeys:
        if fk.table != fk.parent:
            orphan = frames[fk.table][fk.column].isna().to_numpy()
            assert frames[fk.table]["g_seg"].isna().to_numpy()[orphan].all()
    db = create_database(schema, frames)
    timed = [t for t, scm in schema.tables.items() if scm.time_column and rows[t]]
    if timed:
        path = write_database(
            db,
            tmp_path / "db",
            name="fuzz",
            val_timestamp=db.min_timestamp,
            test_timestamp=db.max_timestamp,
        )
        back = read_database(path)
        for t, table in db.table_dict.items():
            pd.testing.assert_frame_equal(back.table_dict[t].df, table.df, check_categorical=False)
    first = next(iter(schema.tables))
    assert set(schema.sample(rows, seed=seed, interventions={first: {"r0": 0.0}})) == set(rows)
