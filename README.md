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

- [11/2026] 🚧 PluRel v2 is under active development toward a **v2-alpha** release.
- [10/2026] 🥇 **RT-PluRel** takes the top spot on [RelArena-α](https://star-project.stanford.edu/relarena-alpha). Checkpoints are on [Hugging Face](https://huggingface.co/stanford-star/rt-plurel).
- [07/2026] Released **v1.1.0** on [PyPI](https://pypi.org/project/plurel/) with the latest features and performance improvements.
- [04/2026] PluRel is accepted to **ICML 2026!**

## Overview

PluRel generates synthetic relational databases from structural causal models. Each table is a
small causal graph, and foreign keys pass causal effects from one table to another. A random prior
can draw thousands of different databases in the [relbench](https://github.com/stanford-star/relbench)
format. The [PluRel paper](https://arxiv.org/abs/2602.04029) pretrains models on these databases.

You can also write your own tables and databases, start from real rows, and draw as many datasets
from them as you need.

<img src="docs/static/images/plurel_animated.gif" alt="PluRel"/>

> [!NOTE]
> Pretraining models on PluRel data (preprocessing, checkpoints, inference with
> [Relational Transformer](https://github.com/stanford-star/relational-transformer)) is covered in
> [`examples/relational_transformer/`](examples/relational_transformer/).

## Installation

PluRel needs Python 3.12 or newer.

> [!IMPORTANT]
> PluRel v2 is under active development toward a **v2-alpha** release. It is not on PyPI yet, and
> its API may still change. Install it from `main`, or pin a commit to reproduce a run.
> `pip install plurel` gives version 1.1.0, which has the
> [earlier API](https://github.com/stanford-star/plurel/tree/v1.1.0).

```bash
pip install "git+https://github.com/stanford-star/plurel.git"
```

## Quickstart

Generate a random database, as the paper does. The seed picks the database's tables, keys and
causal mechanisms. `prior.rows` picks how many rows each table gets:

<!-- standalone -->
```python
from plurel import SchemaPrior, create_database, split_timestamps, write_database

prior = SchemaPrior()
database = prior.realize(seed=0)                              # tables, keys and mechanisms
frames = database.sample(prior.rows(database, seed=0), seed=0)  # a dict of DataFrames
db = create_database(database, frames)                        # the relbench form
write_database(db, "~/.cache/relbench/quickstart", name="quickstart", overwrite=True,
               **dict(zip(("val_timestamp", "test_timestamp"), split_timestamps(db))))
```

To generate many databases in parallel, run the script below. Each database is saved to
`~/.cache/relbench/plurel-<seed>`:

```bash
python scripts/synthetic_gen.py --seed_offset 0 --num_dbs 1000 --num_proc 16
```

You can also write a table yourself. Each column says what it holds and what causes it:

<!-- standalone -->
```python
from plurel import Categorical, LogNormal, Numeric, PrimaryKey, Table, Uniform

customers = Table({
    "id": PrimaryKey(),
    "age": Numeric(Uniform(18, 80), round=1),
    "segment": Categorical({"bronze": 0.6, "silver": 0.3, "gold": 0.1}, causes={"age": 1.0}),
    "spend": Numeric(LogNormal(3, 0.5), causes={"age": 0.8, "segment": {"gold": 1.5}}),
})
print(customers.explain())
frame = customers.sample(500, seed=0)
```

And you can write a whole database and save it for relbench:

<!-- standalone -->
```python
from plurel import ForeignKey, Numeric, PrimaryKey, Schema, Summary, Table, Uniform, create_database, write_database

schema = Schema({
    "teams": Table({"id": PrimaryKey(), "size": Numeric(causes={Summary("staff", "team_id"): 1.0}, noise=0)}),
    "staff": Table({"id": PrimaryKey(), "team_id": ForeignKey("teams"), "pay": Numeric(Uniform(40, 120))}),
})
frames = schema.sample({"teams": 5, "staff": 60}, seed=0)
write_database(create_database(schema, frames), "~/.cache/relbench/teams", name="teams", overwrite=True)
```

See [`examples/`](examples/) for more. It has short [concept scripts](examples/concepts/) that each
explain one idea, and [domain recipes](examples/generation/) for retail orders, clinic visits and
student enrollments.

## Build a table

A `Table` is a dict of columns:

| Column | Holds |
|---|---|
| `PrimaryKey()` | Row positions, or the given `values` |
| `Numeric(marginal, causes, noise, missing, clip, round, zeros)` | Numbers following `marginal`, any distribution such as `Normal(50, 10)`, `Uniform(18, 80)`, `LogNormal(3, 0.5)` or `Poisson(4)` |
| `Categorical(levels, causes, noise, missing, ordered, scores)` | String levels, as a sequence of equally common levels or a dict of level to share |
| `Timestamp(start, end, causes, noise, missing)` | Times between `start` and `end`. Without them, the sum of the causes, in epoch seconds |
| `Hidden(causes, noise)` | A cause that other columns read but no column records |
| `Derived(function)` | A number computed from other columns, which the function's parameters name |

`causes` maps each cause to its effect on the column:

| Cause and effect | Means |
|---|---|
| `"age": 0.8` | A weight. Higher age means a higher value here, and a negative weight means a lower one |
| `"age": (0.8, "tanh")` | A weight and a transform: `tanh`, `square`, `step`, `relu`, `abs`, `sin`, `exp`, `log` or any function |
| `"segment": {"gold": 1.5}` | An effect per level of a categorical cause |
| `("age", "tenure"): 0.8` | A weight on the product of two causes |
| `Recorded("price"): 2.0` | A weight on what `price` records, in its units |

Each column adds up the effects of its causes, plus some `noise` (0.5 by default).

- With a marginal, the rows are then sorted by that sum and given the marginal's values in that
  order. So the column follows its marginal exactly.
- A `Numeric` without a marginal records the sum itself.

Either way, a weight says how closely a column follows its causes. It does not say how many units
the column moves. With one cause and the default noise, weights of 0.25, 0.5, 1 and 2 give rank
correlations of about 0.4, 0.7, 0.9 and 0.97. To work in real units, read a cause with `Recorded`,
or use `Derived`. [`02_latent_and_recorded.py`](examples/concepts/02_latent_and_recorded.py) shows
both.

`sample_with_latents(rows, seed=...)` also returns the hidden values behind every column. With
`held={"x": values}` it fixes the hidden values of `x`. This is an intervention, and the columns
that read `x` change to match.

More options for a column:

- `round=1` gives whole numbers, and `round=0.01` gives cents.
- `clip=(lo, hi)` keeps values between `lo` and `hi`.
- `zeros=0.2` sets 20% of rows to 0.
- `missing=0.1` blanks 10% of rows. Columns that read a blanked column still see its value.

[`11_missing_and_unmatched.py`](examples/concepts/11_missing_and_unmatched.py) covers blanks and
fills.

## Link tables into a database

A `ForeignKey` points at a row of another table. A cause written as `"key.column"` reads a column
of that row. `time_column` names the column that orders a table's events. A table with a key into
a table of events must have its own `time_column`. That way, no key points at a row that a cut in
time removes:

```python
from plurel import Categorical, ForeignKey, LogNormal, Numeric, PrimaryKey, Schema, Table, Timestamp, Uniform

customers = Table({
    "id": PrimaryKey(),
    "segment": Categorical({"bronze": 0.6, "silver": 0.3, "gold": 0.1}),
    "spend": Numeric(LogNormal(3, 0.5), causes={"segment": {"gold": 1.5}}),
})
orders = Table(
    {
        "id": PrimaryKey(),
        "customer_id": ForeignKey("customers"),
        "time": Timestamp("2020-01-01", "2024-01-01"),
        "amount": Numeric(LogNormal(), causes={"customer_id.spend": 1.0}),
    },
    time_column="time",
)
schema = Schema({"customers": customers, "orders": orders})
frames = schema.sample({"customers": 500, "orders": 5000}, seed=0)
```

By default, each row picks its parent at random. A key's options change that:

| Option | Says |
|---|---|
| `link` | How rows pick parents: `RandomLink()`, `HSBMLink(popularity=Pareto(2.5))` for a few popular parents, or `TreeLink(roots=0.05)` for a hierarchy within one table |
| `match={"region": "region"}` | Child and parent columns that must agree |
| `prefer={Recorded("beds"): 0.05}` | Parents with more beds are picked more often |
| `counts=Counts(4.0, causes=..., least=1, most=8)` | How many rows point at each parent, which sets the child table's size |
| `unique=True` or `unique="student_id"` | No parent repeated, across the table or per group |
| `unmatched="null"` | A row with no parent left gets an empty key, instead of an error |
| `nullable=0.02, fill=0.0` | 2% of rows get an empty key at random. Columns reading through an empty key see `fill` |

`Summary("staff", "team_id")` reads the rows that point back at a row. It gives their count, or,
with a column, their `sum`, `mean`, `max` or `min`. Every summary except a count or a sum needs a
`fill` for rows that nothing points at. A summary of a `Recorded` column skips blanks.

When a key has `counts`, its table needs no row count of its own:

```python
from plurel import Counts

school = Schema({
    "students": Table({"id": PrimaryKey(), "year": Numeric(Uniform(1, 4), round=1)}),
    "courses": Table({"id": PrimaryKey(), "level": Categorical(("intro", "advanced"))}),
    "enrollments": Table({
        "student_id": ForeignKey("students", counts=Counts(3.0, causes={"year": 0.5}, most=5)),
        "level": Categorical(("intro", "advanced"), causes={"student_id.year": 1.0}),
        "course_id": ForeignKey("courses", match={"level": "level"}, unique="student_id", unmatched="null", fill=0.0),
    }),
})
frames = school.sample({"students": 200, "courses": 30}, seed=0)
```

[`10_relational_schema.py`](examples/concepts/10_relational_schema.py) and
[`15_edge_cases.py`](examples/concepts/15_edge_cases.py) show which options work together, and how
counts and summaries behave at the edges.
[`12_temporal_data.py`](examples/concepts/12_temporal_data.py) covers the order of events.

## Start from real rows

`Table.from_frame` keeps the rows of a real table exactly. `extra` adds generated columns to it.
Generated tables can point at its real keys and read its columns:

```python
import pandas as pd

real = pd.DataFrame({"customer_id": ["C1", "C2", "C3", "C4"], "age": [34, 51, 29, 62]})
people = Table.from_frame(
    real,
    primary_key="customer_id",
    extra={"churned": Categorical({"no": 0.8, "yes": 0.2}, causes={"age": -1.0})},
    target="churned",
)
purchases = Table({"id": PrimaryKey(), "customer_id": ForeignKey("people"), "amount": Numeric(LogNormal())})
frames = Schema({"people": people, "purchases": purchases}).sample({"purchases": 200}, seed=0)
assert frames["people"][["customer_id", "age"]].equals(real)
```

This adds to the real rows. It does not learn to draw new rows like them. See
[`08_semi_synthetic_table.py`](examples/concepts/08_semi_synthetic_table.py) and
[`09_semi_synthetic_relations.py`](examples/concepts/09_semi_synthetic_relations.py).

## Author many tables from one description

Any weight, effect, noise or marginal can be a range. The table then describes a family of
tables, and generating data takes two steps, each with its own seed:

1. `realize(seed)` picks one table from the family. It draws a value for every range, then fits
   the categorical columns to a trial draw. This seed is the mechanism seed.
2. `sample(rows, seed=...)` draws a dataset from that table. This seed is the data seed. Sampling
   never changes the table.

```python
from plurel import MEDIUM, STRONG, Range, Scaled

family = Table({
    "wait_days": Numeric(Scaled("duration", 0, 60), round=1),
    "outcome": Categorical({"declined": 0.7, "accepted": 0.3}, causes={"wait_days": -STRONG}),
    "cost": Numeric(Scaled("amount", 0, 5000), causes={"wait_days": Range(0.2, 1.0)}, noise=MEDIUM),
})
generator = family.realize(seed=1)                  # one mechanism
first, second = generator.sample(1000, seed=10), generator.sample(1000, seed=11)
others = [family.realize(seed).sample(1000, seed=10) for seed in range(3)]  # three mechanisms
```

| Drawable | Draws |
|---|---|
| `Range(low, high)`, `WEAK`, `STRONG`, `LOW`, `MEDIUM`, `HIGH` | A number for a weight or a noise level. `WEAK` and `STRONG` are weights, and `LOW`, `MEDIUM` and `HIGH` are noise levels |
| `Scaled(units, low, high, shape=None)` | A marginal on the range, shaped for `count`, `amount`, `duration`, `ratio`, `score`, `percent` or `signed` values |
| `Shares(levels)` | A categorical column's shares |
| `Noise(kind, std)` | Noise: `gaussian`, `laplace`, `student_t`, `skewed` or `gumbel` |
| `ByLevel(column, {level: (low, high)})` | A marginal with its own range per level of a categorical column |

`Schema.realize` does the same for a database. A table with no ranges and no categorical `scores`
can be sampled directly. A table with `scores` must be realized first, to fit them.
[`03_realization.py`](examples/concepts/03_realization.py) through
[`07_calibration_sizes.py`](examples/concepts/07_calibration_sizes.py) explain what realizing does
and when it matters.

## Random tables and databases

`TablePrior` and `SchemaPrior` draw whole tables and databases at random. Every setting is a
`Range` or a `Choices`, and their docstrings list them all. By default, a database has 2 to 20
tables. Tables that others point at have 500 to 1,000 rows, and event tables have 10,000 to
30,000:

```python
from plurel import IntegersRange, LogIntegersRange, SchemaPrior, TablePrior

table = TablePrior(column_count=IntegersRange(8, 8)).realize(seed=0, time=True)
frame = table.sample(1000, seed=0)

prior = SchemaPrior(table_count=IntegersRange(3, 5), entity_row_count=LogIntegersRange(1000, 2000))
database = prior.realize(seed=0)
frames = database.sample(prior.rows(database, seed=0), seed=0)
```

Each drawn table has its own causal mechanism. The figure shows eight tables from `TablePrior`.
For each one, two numeric values that feed into a categorical column vary over a grid, and the
table's other inputs stay at typical values. Each cell shows the class the column takes most often there. The figure
comes from [`decision_boundaries.py`](examples/figures/decision_boundaries.py):

<img src="docs/static/images/decision_boundaries.png" alt="Decision boundaries of eight tables drawn by TablePrior"/>

Every generated database keeps its keys valid when cut at any point in time.
[`examples/analysis/`](examples/analysis/) and [`examples/reference/`](examples/reference/) measure
how diverse the prior's databases are, how hard their columns are to predict, and how one compares
with a real database.

## Save and inspect

- `create_database` puts the frames in relbench form. It sorts event tables by time and turns keys
  into row positions.
- `write_database` saves the database, and `read_database` loads it back. A database with events
  needs split timestamps, which `split_timestamps(db)` gives. A database without events needs
  none.
- `describe()` gives a table or schema as plain data, ready for JSON, including every drawn weight
  and fitted cut. `explain()` gives the same as readable text.

```python
from plurel import create_database, read_database, split_timestamps, write_database

db = create_database(schema, schema.sample({"customers": 500, "orders": 5000}, seed=0))
val, test = split_timestamps(db)
path = write_database(db, "~/.cache/relbench/shop", name="shop", val_timestamp=val, test_timestamp=test, overwrite=True)
loaded = read_database(path)
print(schema.explain())
```

[`13_static_export.py`](examples/concepts/13_static_export.py) shows a database without events.

## Reproducibility

The same seeds and row counts give the same frames, byte for byte, on the same platform and type
of processor. On a different platform or processor, the last digits of some numbers can differ.
[COMPATIBILITY.md](COMPATIBILITY.md) says exactly what repeats, and [CHANGELOG.md](CHANGELOG.md)
lists the changes that alter draws.

Each draw is one whole dataset, so `sample(200, seed=0).head(100)` is not the same as
`sample(100, seed=0)`. To build a larger dataset, draw several parts, each with its own seed
([`04_sampling.py`](examples/concepts/04_sampling.py)).

## Latency and memory

Making one random database from the default prior takes about a third of a second and a few
hundred megabytes of memory. That covers drawing it, sampling its rows, saving them to parquet and
reading them back. So a corpus of thousands of databases fits on a laptop. The numbers below come
from 100 tables and 100 databases, measured with `plurel` at commit `287fef8` on an Apple M4 and
on GitHub's `ubuntu-latest` runner:

| | Median | 90th percentile | Largest |
|---|---:|---:|---:|
| database: rows | 29,054 | 102,220 | 211,318 |
| database: seconds, M4 / Linux | 0.29 / 0.44 | 0.98 / 1.4 | 4.1 / 4.4 |
| database: peak memory, M4 / Linux | 264 / 290 MiB | 422 / 393 MiB | 776 / 702 MiB |
| table: seconds, M4 / Linux | 0.13 / 0.14 | 0.31 / 0.35 | 0.59 / 0.64 |

[benchmarks/](benchmarks/) has the harness and the full reports.

## Development

Set up the environment with [pixi](https://pixi.sh/latest/installation/):

```bash
pixi install
pixi run pre-commit install
pixi run test        # the suite
pixi run lint        # ruff check and format check
pixi run typecheck   # pyright over the public API
pixi run examples    # every README block, and each quickstart on its own
pixi run recipes     # every script in examples/, small, each in its own process
pixi run bench       # generation time and memory, see benchmarks/README.md
pixi run fingerprints  # rewrite test/fingerprints.json, for a change that moves draws on purpose
```

CI runs these on every pull request to `main` and `v2`, on Linux and macOS. The v1 generator, the
paper-exact code, remains at the [`v1.1.0`](https://github.com/stanford-star/plurel/tree/v1.1.0)
and [`v1.0.0`](https://github.com/stanford-star/plurel/tree/v1.0.0) tags.

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
