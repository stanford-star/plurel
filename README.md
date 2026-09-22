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

## Getting Started

### Build a table

A table is a causal graph. A `Node` holds a block of latent values; its value is a reduction
over the `Edge`s from its parents, plus bias and noise. A `Column` observes a node as a numeric,
categorical, timestamp or key column. Together they form the table's `SCM`.

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

A `Schema` executes the plan, here for one table. Sampling is deterministic in the seed.

### Draw a table from the prior

`TablePrior` draws such plans; every knob is a `Range` or a `Choices`, and its docstring lists
them all. With `time=True` the table gets a time column.

```python
from plurel import IntegersRange, Range, TablePrior

prior = TablePrior(node_count=IntegersRange(4, 8), column_missing_share=Range(0.0, 0.0))
events = prior.realize(seed=0, time=True)
frame = Schema({"events": events}).sample({"events": 1000}, seed=0)["events"]
```

### Link tables into a database

A `Schema` links tables through foreign keys and through *crossings*, edges that read across a
key: `Foreign(key, node)` reads the row the key points at, `Summary(table, key, node, how)`
aggregates the rows pointing back.

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

An order's amount follows the spend of its customer; an order without one reads `fill`. The
same schema samples under interventions and returns its latents:

```python
frames, latents = schema.sample_with_latents(
    {"customers": 500, "orders": 5000}, seed=0, interventions={"customers": {"age": 1.0}}
)
```

### Draw a database from the prior

`SchemaPrior` draws the table graph, one table plan per table, the keys and the crossings.
`create_database` orders event tables by time and `write_database` saves the
[relbench](https://github.com/stanford-star/relbench) layout that `read_database` loads back.

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

Every generated database keeps its keys valid under any time cut: only unreferenced tables hold
events, and summaries run only between static tables.

### Generate a corpus

One process per database:

```bash
python scripts/synthetic_gen.py --seed_offset 0 --num_dbs 1000 --num_proc 16
```

Databases land in `~/.cache/relbench/plurel-<seed>`; see `--help` for the output directory,
name prefix and validation and test shares.

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
