<div align="center">
  <h1>PluRel</h1>
  <p>
Synthetic Data unlocks Scaling Laws for Relational Foundation Models
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=github)](https://star-project.stanford.edu/plurel)
[![arXiv](https://img.shields.io/badge/arXiv-2602.04029-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2602.04029)
[![Model: rt-plurel](https://img.shields.io/badge/%F0%9F%A4%97%20Model-rt--plurel-yellow?style=flat)](https://huggingface.co/stanford-star/rt-plurel)
[![Dataset: plurel](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-plurel-yellow?style=flat)](https://huggingface.co/datasets/stanford-star/plurel)
[![PyPI](https://img.shields.io/pypi/v/plurel.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/plurel/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat)](LICENSE)

<img src="docs/static/images/scaling_law.png" alt="Scaling Law Plot"/>
</div>
<br>

Models pretrained on PluRel data lead public relational benchmarks (at the time of submission):  
🥇 **RT-PluRel**, a [Relational Transformer](https://github.com/stanford-star/relational-transformer) pretrained on PluRel synthetic data, is #1 on the combined system + model leaderboard on [RelArena-α](https://star-project.stanford.edu/relarena-alpha) ([repro](https://github.com/stanford-star/relational-transformer/tree/main/examples/finetune)).  
🥇 **RT-J**, warm-started from RT-PluRel, is the #1 in-context model on the [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard) ([repro](https://github.com/stanford-star/relational-transformer/tree/main/examples/icl)).

## Latest Updates

- [10/2026] 🥇 **RT-PluRel** takes the top spot on [RelArena-α](https://star-project.stanford.edu/relarena-alpha). Checkpoints are on [Hugging Face](https://huggingface.co/stanford-star/rt-plurel).
- [07/2026] Released **v1.1.0** on [PyPI](https://pypi.org/project/plurel/) with the latest features and performance improvements.
- [04/2026] PluRel is accepted to **ICML 2026!**

## Overview

PluRel synthesizes relational databases from structural causal models: every table is a small
causal graph, foreign keys carry causal influence between tables, and a prior draws thousands of
different databases in the [relbench](https://github.com/stanford-star/relbench) format. It is the
reference implementation for the [PluRel paper](https://arxiv.org/abs/2602.04029). People and agents
can also [write semi-synthetic tables](#write-your-own-tables) in a schema language that plurel
realizes as the same causal graphs.

> [!NOTE]
> Pretraining models on PluRel data (preprocessing, checkpoints, inference with [Relational Transformer](https://github.com/stanford-star/relational-transformer)) is covered in [`examples/relational_transformer/`](examples/relational_transformer/).

## Framework Design

<img src="docs/static/images/plurel_animated.gif" alt="PluRel Logo"/>

## Installation

PluRel needs Python 3.12 or newer. This README describes PluRel 2, which is not on PyPI yet, so
install it from this repository (`pip install plurel` gives 1.1.0, the
[earlier API](https://github.com/stanford-star/plurel/tree/v1.1.0)):

```bash
pip install git+https://github.com/stanford-star/plurel.git
```

## Getting Started

### Build a table

A `Table` is a dict of its columns. Each column says what it holds and what causes it:

```python
from plurel import Categorical, LogNormal, Numeric, PrimaryKey, Table, Uniform

customers = Table({
    "id": PrimaryKey(),
    "age": Numeric(Uniform(18, 80), round=1),
    "segment": Categorical({"bronze": 0.6, "silver": 0.3, "gold": 0.1}, causes={"age": 1.0}),
    "spend": Numeric(LogNormal(3, 0.5), causes={"age": 0.8, "segment": {"gold": 1.5}}, missing=0.05),
})
frame = customers.sample(500, seed=0)  # a DataFrame; hidden causes are not in it
frame.to_csv("customers.csv", index=False)
```

| Column | Holds |
|---|---|
| `PrimaryKey()` | Row positions, or the given `values` |
| `Numeric(marginal, causes, noise, missing, clip, round)` | Numbers following `marginal`, any distribution such as `Normal(50, 10)`, `Uniform(18, 80)`, `LogNormal(3, 0.5)`, `Poisson(4)` or `Exponential()` |
| `Categorical(levels, causes, noise, missing)` | Levels, as a sequence of equally common levels or a dict of level to share |
| `Timestamp(start, end, causes, noise, missing)` | Times in the span |
| `Hidden(causes, noise)` | A cause that other columns read but no column records |
| `Real(values)` | Given values, kept as they are |

**Causes** map a column's name to its effect:

| Effect | Means |
|---|---|
| `0.8` | A weight: higher in the cause means higher here, and a negative weight means lower |
| `(0.8, "tanh")` | A weight and a transform: `tanh`, `square` (U-shaped), `step`, `relu`, `abs`, `sin`, `exp`, `log`, or any function |
| `{"gold": 1.5}` | An effect per level of a categorical cause; levels left out have none |

A column adds up its causes' effects and puts them on a unit scale, then adds `noise` of that
standard deviation, 0.5 by default, so a smaller `noise` follows the causes more closely. The rows
then take the `marginal`'s values in that order, which means a column holds its marginal whatever
its causes. A categorical column cuts the same order at its shares, so a positive weight pushes rows
toward its later levels. A `Numeric` without a marginal is the weighted sum itself, plus the noise,
as floats.

`round` is a step: `round=1` gives whole numbers and `round=0.01` cents. `clip=(lo, hi)` caps the
values. `Timestamp`'s `end` is exclusive.

Sampling is deterministic in the seed.

### Link tables into a database

A `ForeignKey` points at another table, and `"key.column"` reads a column of the row it points at.
A `Schema` joins the tables, and `time_column` names the column that orders a table's events.

```python
from plurel import ForeignKey, HSBMLink, Pareto, Schema, Timestamp

orders = Table(
    {
        "id": PrimaryKey(),
        "customer_id": ForeignKey("customers", HSBMLink(popularity=Pareto(2.5))),
        "time": Timestamp("2020-01-01", "2024-01-01"),
        "amount": Numeric(LogNormal(), causes={"customer_id.spend": 1.0, "customer_id.segment": {"gold": 0.5}}),
    },
    time_column="time",
)
schema = Schema({"customers": customers, "orders": orders})
frames = schema.sample({"customers": 500, "orders": 5000}, seed=0)  # a dict of DataFrames by table
```

`link` says how rows pick their parent: `RandomLink()` by default, `HSBMLink(popularity=Pareto(2.5))`
for a few very popular parents, or `TreeLink(roots=0.05)` on a key to the same table, whose roots
have no parent. With `nullable=0.02` a share of the rows point at nothing, and a number read across
the key takes `fill`, which such a key must then set: `ForeignKey("customers", nullable=0.02, fill=0.0)`.

`Summary` reads the rows pointing back. It takes the table and its key, and counts them unless told
to aggregate a column of theirs (`Summary("orders", "customer_id", "amount", "sum")`; also `mean`,
`max` and `min`, which need a `fill` for rows that nothing points at). With `noise=0` and no marginal
the count is exact, and a `Hidden` count lets rows with more reports rank higher:

```python
from plurel import Hidden, Summary, TreeLink

teams = Table({"id": PrimaryKey(), "size": Numeric(causes={Summary("staff", "team_id"): 1.0}, noise=0)})
staff = Table({
    "id": PrimaryKey(),
    "team_id": ForeignKey("teams"),
    "boss_id": ForeignKey("staff", TreeLink(roots=0.05), fill=0.0),
    "reports": Hidden(causes={Summary("staff", "boss_id"): 1.0}),
    "grade": Categorical((1, 2, 3), causes={"reports": 1.0}),
    "pay": Numeric(LogNormal(11, 0.3), causes={"grade": 1.0, "team_id.size": 0.3}),
})
frames = Schema({"teams": teams, "staff": staff}).sample({"teams": 8, "staff": 200}, seed=0)
```

A cause is read within a row's own draw, so a column cannot read itself across a key, and a summary
cannot read a table with events, which would be reading the future.

### Start from real rows

`Table.from_frame` keeps a real table's rows exactly as they are. Generated tables point at its real
keys and read its columns as causes, and `extra` adds generated columns to the real rows themselves.
A table of real rows needs no row count.

```python
import pandas as pd

real = pd.DataFrame({
    "customer_id": ["C1", "C2", "C3", "C4"],
    "city": ["Boston", "Lima", "Boston", "Oslo"],
    "age": [34, 51, 29, 62],
})
people = Table.from_frame(
    real,
    primary_key="customer_id",
    extra={"churned": Categorical({"no": 0.8, "yes": 0.2}, causes={"age": -1.0})},
)
purchases = Table({
    "id": PrimaryKey(),
    "customer_id": ForeignKey("people"),
    "amount": Numeric(LogNormal(), causes={"customer_id.city": {"Boston": 1.0}}),
})
frames = Schema({"people": people, "purchases": purchases}).sample({"purchases": 200}, seed=0)
```

Real child tables work the same way: `foreign_keys={"customer_id": "people"}` keeps their key values
and `time_column` names their events. Real levels have no order, so read them by level.

### Save a database

`create_database` orders event tables by time and gives keys as row positions, the
[relbench](https://github.com/stanford-star/relbench) form, which `write_database` saves and
`read_database` loads back.

```python
from plurel import create_database, split_timestamps, write_database

db = create_database(schema, schema.sample({"customers": 500, "orders": 5000}, seed=0))
val_timestamp, test_timestamp = split_timestamps(db)
write_database(
    db, "~/.cache/relbench/shop", name="shop",
    val_timestamp=val_timestamp, test_timestamp=test_timestamp,
)
```

### Draw random tables and databases

`TablePrior` and `SchemaPrior` draw whole tables and databases. Every knob is a `Range` or a
`Choices`, and their docstrings list them all. By default a database has 2 to 20 tables
(`table_count`), 500 to 1,000 rows per entity table (`entity_row_count`) and 10,000 to 30,000 per
event table (`activity_row_count`).

```python
from plurel import IntegersRange, LogIntegersRange, SchemaPrior, TablePrior

prior = TablePrior(column_count=IntegersRange(8, 8))
events = prior.realize(seed=0, time=True)
frame = events.sample(1000, seed=0)

prior = SchemaPrior(table_count=IntegersRange(3, 5), entity_row_count=LogIntegersRange(1000, 2000))
drawn = prior.realize(seed=0)
frames = drawn.sample(prior.rows(drawn, seed=0), seed=0)
db = create_database(drawn, frames)
```

Every generated database keeps its keys valid under any time cut: only unreferenced tables hold
events, and summaries run only between static tables.

### Go lower

The columns above compile to the causal graph in `plurel.graph` and `plurel.columns`, which the
priors build directly: a `Node` is a reduction over `Edge`s (`LinearEdge`, `MatrixEdge`, `MLPEdge`,
`TreeEdge` and others) plus bias and noise, and a `Column` observes one. A `Table` accepts them next
to the columns above, for mechanisms the causes can't state.

### Generate a corpus

One process per database:

```bash
python scripts/synthetic_gen.py --seed_offset 0 --num_dbs 1000 --num_proc 16
```

Databases land in `~/.cache/relbench/plurel-<seed>`; see `--help` for the output directory,
name prefix and validation and test shares.

## Write your own tables

To author many tables from one description, a `WorldSchema` says what a table means: what each column is, on what scale, and how it depends on
the others. A schema authors signs and strength classes; a seed draws the magnitudes, so one schema
gives many tables. `validate()` names the first rule a schema breaks, which makes the language easy
for an agent to write against.

```python
from plurel.world import Edge, Node, WorldSchema, realize_world

schema = WorldSchema(
    schema_id="clinic_referral_outcome",
    target="outcome",
    description="Each row is one referral a clinic received; outcome records how it ended.",
    nodes={
        "age": Node(kind="numeric", units="duration", dist="right", scale=(0, 90), desc="the patient's age"),
        "wait_days": Node(
            kind="numeric", units="duration", scale=(0, 60), rounding=1,
            parents={"age": Edge(effect=+0.5)}, noise_scale="low", noise_dist="gaussian",
            desc="days waited",
        ),
        "outcome": Node(
            kind="categorical", levels=["declined", "accepted"], level_probs={"declined": 0.7, "accepted": 0.3},
            parents={"wait_days": Edge(effect=-1, shape="saturating")},
            noise_scale="medium", noise_dist="gaussian", desc="how it ended",
        ),
    },
)
population = realize_world(schema, seed=0)
rows = population.rows(n=1000, draw=0)
```

Nodes can also declare hidden causes, nominal parents, interactions, non-additive mechanisms, derived
columns and what the record rounds, caps or leaves blank; `compile_schema` returns the plain `SCM`.
`realize_random(seed, columns=...)` in `plurel.random_scm` gives a table of the same shape from the
prior, with no text.

A catalog of schemas is a package whose file layout is its taxonomy (`load_catalog`), declared by one
plan per sector (`Taxonomy`). It can live in any repository. `plurel.world.authoring.realism.check`
is the gate on a new schema, `plurel.world.authoring.dedupe` lists near-duplicates, and `brief` and
`lint` support writing a schema's world knowledge. [`examples/world/`](examples/world/) shows them
together.

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
