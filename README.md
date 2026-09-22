<div align="center">
  <h1>PluRel</h1>
  <p>
Synthetic Data unlocks Scaling Laws for Relational Foundation Models
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=github)](https://star-project.stanford.edu/plurel)
[![arXiv](https://img.shields.io/badge/arXiv-2602.04029-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2602.04029)
[![PyPI](https://img.shields.io/pypi/v/plurel.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/plurel/)

<img src="docs/static/images/scaling_law.png" alt="Scaling Law Plot"/>
</div>
<br>

## Latest Updates

- [07/2026] Released **v1.1.0** on [PyPI](https://pypi.org/project/plurel/) with the latest features and performance improvements.
- [04/2026] PluRel is accepted to **ICML 2026!**

## Overview

PluRel synthesizes relational databases from structural causal models: every table is a small
causal graph, foreign keys carry causal influence between tables, and a prior draws thousands of
different databases in the [relbench](https://github.com/stanford-star/relbench) format.

## Installation

PluRel needs Python 3.12 or newer.

```bash
pip install plurel
```

## Step 1: one table, by hand

A table is a small causal graph. Each `Node` holds a block of latent values, one row per table
row and `dim` columns; its value is a reduction over the `Edge`s from its parents plus bias and
noise. A node without edges is a root whose value is its noise. A node with `onehot=True`
emits the one-hot argmax of its scores, and with Gumbel noise that samples a class from their
softmax. A node with `standardize=True` standardizes its signal before the noise is added, so
the noise is relative to a unit-scale signal. A `Column` observes one node, or one slot of it:
numeric, rank-mapped onto a marginal distribution if one is given; categorical, from a one-hot
node or by binning a numeric one; a timestamp; or the key. The nodes and columns together are
the `SCM` of the table.

```python
import numpy as np
from plurel import SCM, Column, Gumbel, LinearEdge, LogNormal, MatrixEdge, Node, Normal, Schema, Uniform

customers = SCM(
    nodes={
        "age": Node(noise=Uniform(-1.7, 1.7)),
        "segment": Node(
            (MatrixEdge("age", np.array([[1.0, -1.0, 0.3]])),),
            bias=(0.0, 0.0, -0.5), onehot=True, noise=Gumbel(), standardize=True,
        ),
        "spend": Node((LinearEdge("age", 0.8, "tanh"),), noise=Normal(std=0.3), standardize=True),
    },
    columns={
        "id": Column(kind="key"),
        "age": Column("age", marginal=Uniform(18.0, 80.0)),
        "segment": Column("segment", "categorical", categories=("bronze", "silver", "gold")),
        "spend": Column("spend", marginal=LogNormal(3.0, 0.5), missing=0.05),
    },
)
frame = Schema({"customers": customers}).sample({"customers": 500}, seed=0)["customers"]
```

An `SCM` is a plan; a `Schema` executes it, here for a single table. Sampling is deterministic
in the seed, so the same plan can be sampled again at other row counts or with other noise.
The edge families are `LinearEdge` (a weight and a transform), `MatrixEdge`, `MLPEdge`,
`TreeEdge` (oblivious trees), `FourierEdge`, `QuadraticEdge`, `LookupEdge` (levels of a binned
scalar) and `NearestEdge` (one-hot of the closest center); every one is a frozen dataclass
whose parameters you can read and set.

## Step 2: one table, from the prior

`TablePrior` draws such plans. Every knob is a `Range` or a `Choices`, warped once per table so
that each table has its own style, then drawn per use. Its docstring lists every knob.

```python
from plurel import IntegersRange, Range, TablePrior

prior = TablePrior(node_count=IntegersRange(4, 8), column_missing_share=Range(0.0, 0.0))
events = prior.realize(seed=0, time=True)
frame = Schema({"events": events}).sample({"events": 1000}, seed=0)["events"]
```

With `time=True` the table gets a calendar time column and its rows come out in time order.
The realized plan is inspectable: `events.nodes` are the nodes with their edges, `events.columns`
the columns, `events.order` the topological order.

## Step 3: a database, by hand

A `Schema` links tables through foreign keys and through *crossings*, the edges that read
across a key. A `Foreign(key, node)` tail reads the node of the row the key points at; a
`Summary(table, key, node, how)` tail aggregates, per row, the rows that point back at it
(`count`, `sum`, `mean`, `max` or `min`). The Schema draws the key links, evaluates every node
of every table in one topological order across tables, and observes the columns.

```python
import pandas as pd
from plurel import FK, Calendar, Foreign, HSBMLink, Pareto

orders = SCM(
    nodes={
        "time": Node(noise=Calendar(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"))),
        "amount": Node(noise=Normal(std=0.5), standardize=True),
    },
    columns={
        "id": Column(kind="key"),
        "amount": Column("amount", marginal=LogNormal()),
        "time": Column("time", "timestamp"),
    },
    time_column="time",
)
schema = Schema(
    tables={"customers": customers, "orders": orders},
    fkeys=(
        FK("orders", "customer_id", "customers", HSBMLink(popularity=Pareto(2.5)), nullable=0.02, fill=0.0),
    ),
    crossings={("orders", "amount"): (LinearEdge(Foreign("customer_id", "spend"), 1.0),)},
)
frames = schema.sample({"customers": 500, "orders": 5000}, seed=0)
```

An order's amount follows the spend of its customer; an order without a customer reads `fill`
instead. The link model decides which customers get many orders, here with a heavy-tailed
popularity. The same schema samples under an intervention, and returns its latents:

```python
frames, latents = schema.sample_with_latents(
    {"customers": 500, "orders": 5000}, seed=0, interventions={"customers": {"age": 1.0}}
)
```

## Step 4: a database, from the prior

`SchemaPrior` draws whole schemas: the table graph, a fresh `TablePrior` warp per table, the
keys, and the crossings. Then the database is written in the
[relbench](https://github.com/stanford-star/relbench) format.

```python
from plurel import SchemaPrior, create_database, split_timestamps, write_database

prior = SchemaPrior()
schema = prior.realize(seed=0)
frames = schema.sample(prior.rows(schema, seed=0), seed=0)
db = create_database(schema, frames)
val_timestamp, test_timestamp = split_timestamps(db)
write_database(
    db, "~/.cache/relbench/plurel-0", name="plurel-0",
    val_timestamp=val_timestamp, test_timestamp=test_timestamp,
)
```

`create_database` orders event tables by time and turns keys into row positions, the relbench
contract, and `write_database` saves parquet files and the manifest that `plurel.read_database`
or relbench load back. The prior draws every mechanism the library has: all eight edge
families, all six reductions, roots that are normal, uniform, bimodal, skewed, heavy-tailed,
counts or time series, twelve transforms, columns that are binned, rank-mapped, zero-inflated
or outlier-laden, missingness at random or driven by other nodes, four calendars,
self-referential trees, keys that are nullable or doubled, and summaries of children into
parents. Its docstring lists every knob.

Three rules hold in every generated database. Tables nothing references are event tables and
get a time column; referenced tables are static, so no key points into the future and cutting a
database at any time leaves every key valid. Summaries run only from static tables into static
tables, so no row summarizes later events. Latents are always finite; missing values enter only
when columns are observed.

## Step 5: a corpus

The generator script runs one process per database:

```bash
python scripts/synthetic_gen.py --seed_offset 0 --num_dbs 1000 --num_proc 16
```

Each database lands in `~/.cache/relbench/plurel-<seed>` (`--out_dir` and `--db_prefix`
change that) with validation and test timestamps leaving `--val_share` and `--test_share` of
the timestamped rows after them. A default database has 2 to 20 tables and takes about a second
or less.

## Development

Set up the environment with [pixi](https://pixi.sh/latest/installation/):

```bash
pixi install
pixi run pytest
pixi run ruff check .
pixi run ruff format .
pixi run pre-commit install
```

The v1 generator, the paper-exact code, remains at the
[`v1.1.0`](https://github.com/stanford-star/plurel/tree/v1.1.0) and
[`v1.0.0`](https://github.com/stanford-star/plurel/tree/v1.0.0) tags.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{kothapalli2026plurel,
    title={{PluRel:} Synthetic Data unlocks Scaling Laws for Relational Foundation Models},
    author={Vignesh Kothapalli and Rishabh Ranjan and Valter Hudovernik and Vijay Prakash Dwivedi and Johannes Hoffart and Carlos Guestrin and Jure Leskovec},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026}
}
```
