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
from plurel.graph import COMPLETE, Foreign, Summary
from plurel.links import HSBMLink, RandomLink, TreeLink
from plurel.schema import FK, Schema


def random_schema(rng):
    names = [f"t{i}" for i in range(int(rng.integers(1, 4)))]
    tables = {t: {f"{t}_id": Column(kind="key")} for t in names}
    for t in names:
        roots = [f"c{j}" for j in range(int(rng.integers(1, 4)))]
        for root in roots:
            tables[t][root] = Column(
                Node(noise=rng.choice([Normal(), Uniform(), LogNormal()])),
                marginal=rng.choice([None, Uniform(), LogNormal(), Pareto(2.0)]),
                missing=float(rng.choice([0.0, 0.2])),
            )
        k = int(rng.integers(2, 5))
        tables[t]["seg"] = Column(
            Node(bias=tuple(rng.normal(size=k)), onehot=True, noise=Gumbel()),
            categories=tuple(f"s{i}" for i in range(k)),
        )
        tables[t]["h"] = Node(dim=int(rng.integers(1, 4)))
        if rng.random() < 0.5:
            tables[t]["stamp"] = Column(Node(noise=DEFAULT_CALENDAR), "timestamp")
            tables[t]["later"] = Column(
                Node((LinearEdge("stamp"),), noise=Exponential(3600.0)), "timestamp"
            )
        terms = tuple(
            LinearEdge(root, float(rng.normal()), str(rng.choice(["identity", "tanh", "square"])))
            for root in roots
        )
        tables[t]["y"] = Column(
            Node(terms, noise=Normal(std=0.3)), marginal=rng.choice([None, Uniform(-1.0, 1.0)])
        )
        tables[t]["bin"] = Column(
            Node(LinearEdge("y"), noise=None),
            categories=("lo", "mid", "hi"),
            probabilities=(0.2, 0.3, 0.5),
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
        tables[t][key] = FK(parent, link, nullable=float(rng.choice([0.0, 0.2])), fill=0.0)
        observed = bool(rng.random() < 0.5)
        width = tables[parent]["seg"].node.dim
        gathered = Node(LinearEdge(Foreign(key, "c0")), noise=None)
        tables[t]["g_seg"] = Column(
            Node(LinearEdge(Foreign(key, "seg"), dim=width), noise=None),
            categories=tuple(f"p{i}" for i in range(width)),
        )
        if observed:
            tables[t]["g_c0"] = Column(gathered, marginal=Uniform())
        else:
            tables[t]["g_c0"] = gathered
            edges = (LinearEdge("g_c0", 2.0), MatrixEdge("g_seg", np.ones((width, 1))))
            tables[t]["z"] = Column(Node(edges, noise=Normal(std=0.1)))
        if "stamp" not in tables[t]:
            how = str(rng.choice(["count", "sum", "mean", "max"]))
            summary = Summary(t, key, "y", how, fill=None if how in COMPLETE else 0.0)
            tables[parent][f"a_{t}"] = Column(Node(LinearEdge(summary), noise=None))
            tables[parent][f"w_{t}"] = Column(Node((LinearEdge(f"a_{t}", 0.5),), noise=Normal()))
    if rng.random() < 0.4:
        t = names[0]
        tables[t]["boss"] = FK(t, TreeLink(roots=0.3), fill=0.0)
        tables[t]["boss_c0"] = Node(LinearEdge(Foreign("boss", "c0")), noise=None)
        if "stamp" not in tables[t]:
            tables[t]["reports"] = Column(
                Node(LinearEdge(Summary(t, "boss", "c0", "count")), noise=None)
            )
    return Schema(
        {
            t: SCM(table, time_column="stamp" if "stamp" in table else None)
            for t, table in tables.items()
        }
    )


@pytest.mark.parametrize("seed", range(30))
def test_random_schemas_sample_wrap_and_round_trip(seed, tmp_path):
    rng = np.random.default_rng(seed)
    schema = random_schema(rng)
    rows = {t: int(rng.choice([0, 2, 7, 40, 150])) for t in schema.tables}
    for fk in schema.keys.values():
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
    for (table, column), fk in schema.keys.items():
        keys = frames[table][column]
        linked = keys.dropna().astype(int)
        assert keys.dtype == "Int64" and linked.between(0, rows[fk.parent] - 1).all()
        if isinstance(fk.link, TreeLink):
            assert (linked.to_numpy() < linked.index.to_numpy()).all()
        elif fk.nullable == 0:
            assert keys.notna().all()
    for (table, column), fk in schema.keys.items():
        if table != fk.parent:
            orphan = frames[table][column].isna().to_numpy()
            assert frames[table]["g_seg"].isna().to_numpy()[orphan].all()
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
