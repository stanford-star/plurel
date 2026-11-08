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
install it from this repository's `v2` branch (`pip install plurel` gives 1.1.0, the
[earlier API](https://github.com/stanford-star/plurel/tree/v1.1.0)):

```bash
pip install "git+https://github.com/stanford-star/plurel.git@v2"
```

`v2` moves; for a training run you want to reproduce, pin a commit instead, as in
`pip install "git+https://github.com/stanford-star/plurel.git@<commit>"`.

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
| `Numeric(marginal, causes, noise, missing, clip, round, zeros, missing_by, how)` | Numbers following `marginal`, any distribution such as `Normal(50, 10)`, `Uniform(18, 80)`, `LogNormal(3, 0.5)`, `Poisson(4)` or `Exponential()` |
| `Categorical(levels, causes, noise, missing, ordered, scores, missing_by, how)` | Levels, as a sequence of equally common levels or a dict of level to share |
| `Timestamp(start, end, causes, noise, missing)` | Times in the span; without one, the causes' sum in epoch seconds |
| `Hidden(causes, noise)` | A cause that other columns read but no column records |
| `Derived(function)` | A number computed from recorded numbers, named by the function's parameters |
| `Real(values)` | Given values, kept as they are |

**Causes** map a column's name to its effect:

| Cause and effect | Means |
|---|---|
| `"age": 0.8` | A weight: higher in the cause means higher here, and a negative weight means lower |
| `"age": (0.8, "tanh")` | A weight and a transform: `tanh`, `square` (U-shaped), `step`, `relu`, `abs`, `sin`, `exp`, `log`, or any function |
| `"segment": {"gold": 1.5}` | An effect per level of a categorical cause; levels left out have none |
| `"segment": 0.8` | A weight on a categorical cause, which needs `ordered=True` on it: its levels are then a scale |
| `("age", "tenure"): 0.8` | A weight on the product of two causes |
| `("segment", "age"): {"gold": 1.5}` | A slope of the second cause per level of the first |
| `Recorded("price"): 2.0` | A weight on what `price` records, in its units, instead of on its latent |

A column is its causes' effects added up plus `noise`, a standard deviation of 0.5 by default, and
a root is a unit normal. So `y = 2 * x + noise` is what `causes={"x": 2.0}` says: a larger weight or
a smaller `noise` follows the causes more closely. `how` combines the effects another way: `max`,
`min`, or `product`, their compounding. The rows then take the `marginal`'s values in that order,
which means a column holds its marginal whatever its causes. A categorical column cuts the same
order at its shares, so a positive weight pushes rows toward its later levels. Causes are read as
those sums, not in the units of their marginals, unless read `Recorded`; a `Numeric` without a
marginal is the sum itself, as floats. [Latent and recorded values](#latent-and-recorded-values)
works through both.

A weight's size is how closely a column follows a cause. With one unit-normal cause and the default
noise, the rank correlation is:

| Weight | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| Rank correlation | 0.41 | 0.68 | 0.88 | 0.97 |

Each further cause adds to the sum and so weakens the others. An effect per level moves that level's
rows by the effect on the same scale, so `{"Boston": 1.0}` shifts Boston's rows by twice the default
noise.

`round` is a step: `round=1` gives whole numbers and `round=0.01` cents. `clip=(lo, hi)` caps the
values, and either side may be `None`. `zeros=0.2` records a share of the rows at 0, or at the floor
of a `Scaled` marginal. `missing=0.1, missing_by=("age", "high")` puts the blanks on the high rows of
a number. Blanks are only in the record: columns that read a blanked column still see its value.
`Timestamp`'s `end` is exclusive.

Sampling is deterministic: the same seed and row counts give the same frames, byte for byte. A
draw is one dataset, not rows streamed from a population. Each draw holds its marginals exactly, by
rank, and a categorical that depends on a key cuts its shares in each draw, so a row's values depend
on the rows drawn with it: `sample(200, seed=0).head(100)` is not `sample(100, seed=0)`. Links,
counts and summaries depend on the sizes of the tables they join too. To build a larger dataset from
parts, draw each part as its own database under its own seed, and add the row counts of the parts
before it to each part's key columns, primary and foreign, before joining them.

### Latent and recorded values

Every column has a latent value, the sum of its causes' effects plus noise, and a recorded value,
what its frame holds. A marginal maps the one to the other by rank: the row with the highest latent
takes the highest of the marginal's values. A cause named bare reads the latent:

```python
import numpy as np

table = Table({
    "x": Numeric(Uniform(100, 200)),
    "y": Numeric(causes={"x": 2.0}, noise=0),          # no marginal: y is the sum itself
    "z": Numeric(Uniform(0, 1), causes={"x": 2.0}),   # a marginal: z holds Uniform(0, 1) by rank
})
frame, latents = table.sample_with_latents(1000, seed=0)
assert np.allclose(frame.y, 2 * latents["x"][:, 0])   # x is about 100 to 200, y about -5 to 6
```

`y` is twice the latent `x`, a unit normal, not twice the recorded `x`. A weight therefore says how
closely a column follows a cause, not how many units it moves per unit of the cause.

`held` sets a node's latents, which is how to intervene. Holding `x` one higher shifts `y` by
exactly 2, but leaves the recorded `x` and `z` as they were: every row moved alike, so their ranks
and hence their recorded values did not change. An intervention is guaranteed to show in a column
without a marginal and in the order of rows; a recorded column with a marginal changes only where
the intervention reorders its rows.

```python
held, _ = table.sample_with_latents(1000, seed=0, held={"x": latents["x"] + 1.0})
assert np.allclose(held.y - frame.y, 2.0) and held.x.equals(frame.x) and held.z.equals(frame.z)
```

A key's `fill` and a `Summary` of a bare name are in latent units too: `fill=0.0` reads as the
latent mean, and `Summary("staff", "team_id", "pay", "sum")` sums the latent `pay`. Only a count is
in rows.

`Recorded` reads what a column records instead, in its units: a number, a `Derived` value, or a
time in epoch seconds. It sees the record after everything the column does to it: its marginal,
`zeros`, `round`, `clip`, ranges by level and blanks. A blank, or a key pointing at nothing, reads
the `fill` given to `Recorded`, and raises without one. Weights and `noise` are then in those units,
so this `y` is twice the recorded `x` plus noise of five:

```python
from plurel import Exponential, Recorded, Timestamp

table = Table({
    "x": Numeric(Uniform(100, 200), round=1, missing=0.1),
    "y": Numeric(causes={Recorded("x", fill=0.0): 2.0}, noise=5.0),  # about 200 to 400, about 0 where x is blank
})
```

A `Summary` of a `Recorded` column aggregates the recorded values of the rows pointing back,
skipping blanks as pandas does, so `Summary("staff", "team_id", Recorded("pay"), "sum")` with
`noise=0` is the payroll `groupby("team_id").pay.sum()` gives; a count counts the recorded rows. A
`Timestamp` without a span is its causes' sum plus `noise` in epoch seconds, so it must read a
time `Recorded` and give its `noise`; with weight 1 on that time and a positive noise it always
comes after it. Times must stay within pandas' range, 1677 to 2262, and a time read `Recorded`
must have no time zone:

```python
orders = Table({
    "placed": Timestamp("2020-01-01", "2024-01-01"),
    "shipped": Timestamp(causes={Recorded("placed"): 1.0}, noise=Exponential(2 * 86400)),  # two days later on average
}, time_column="placed")
```

A column with `Recorded` causes has its latent in their units too, so a column naming it bare
reads that unscaled latent. Records and latents form one graph, so a cycle through either raises.

When latents tie, as with `noise=0` over a discrete or repeated cause, the tied rows take the same
level, so a categorical column can miss its shares rather than split equal rows at random. Give it
some `noise` when the shares matter. Requested shares are met exactly by each draw of a table that
is not realized; a realized table pins its cuts on one calibration draw, so its draws vary around
them; a column that depends on a key cuts each draw.

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

A key can instead choose parents by their columns. `match` maps a column of the child to one of the
parent that must agree: a categorical holding the same level, or a key pointing at the same row.
`prefer` weighs the eligible parents by the exponential of causes read from the parent's columns,
with no noise, so a weight of 1 on a `Recorded` number sends nearly every row to the largest.
A row that no parent matches points at nothing if the key is `nullable`, and raises otherwise:

```python
from plurel import Counts

clinics = Table({"id": PrimaryKey(), "region": Categorical(("north", "south")), "beds": Numeric(Uniform(10, 100))})
patients = Table({
    "region": Categorical(("north", "south")),
    "clinic_id": ForeignKey("clinics", match={"region": "region"}, prefer={Recorded("beds"): 0.05}),
})
```

`counts` sets how many rows point at each parent instead, and so the child table's size:
`ForeignKey("customers", counts=Counts(5.0, causes={"tenure": 0.8}))` gives each customer about 5
orders, more with longer tenure, and some none. `noise` (0.5 by default with causes) spreads the
counts beyond Poisson; `noise=0` keeps them Poisson given the causes. Each customer then draws its own count, so
`sample` may leave the orders table out of its rows. A row count given for it is instead split
among the customers by the same shares. A table takes at most one such key, and a key cannot match
on, or count from, a column that reads across it.

`unique` keeps a key from repeating a parent: `unique=True` across the table, or
`unique="student_id"` among the rows sharing a student, so each (student, course) pair of an
enrollments table appears once. Parents are drawn without repeats, by `prefer` if given; a row
blank in a grouping column is free, as SQL treats nulls, and given key values are checked. A
student with more enrollments than there are courses raises, or, on a `nullable` key, leaves the
extra rows pointing at nothing; `Counts(most=...)` caps how many rows each parent gets:

```python
enrollments = Table({
    "student_id": ForeignKey("students", counts=Counts(4.0, most=6)),  # with 6 or more courses
    "course_id": ForeignKey("courses", unique="student_id"),
})
```

A position within a group, such as an order's line number, is no key: compute it afterwards with
`frame.groupby("order_id").cumcount() + 1`.

`Summary` reads the rows pointing back. It takes the table and its key, and counts them unless told
to aggregate a column of theirs, as its sum (`Summary("staff", "boss_id", "pay", "mean",
fill=0.0)`; also `sum`, `max` and `min`; all but `sum` need a `fill` for rows that nothing points
at, or, for a `Recorded` column, whose rows pointing back are all blank). With `noise=0` and no marginal a count is exact, and a `Hidden` count lets rows with more
reports rank higher:

```python
from plurel import Hidden, Summary, TreeLink

teams = Table({"id": PrimaryKey(), "size": Numeric(causes={Summary("staff", "team_id"): 1.0}, noise=0)})
staff = Table({
    "id": PrimaryKey(),
    "team_id": ForeignKey("teams"),
    "boss_id": ForeignKey("staff", TreeLink(roots=0.05), fill=0.0),
    "reports": Hidden(causes={Summary("staff", "boss_id"): 1.0}),
    "grade": Categorical((1, 2, 3), causes={"reports": 1.0}, ordered=True),
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

This is augmentation, not learned synthesis. The real table keeps its own row count and keys in every
draw, `extra` and the generated tables are authored mechanisms that read it, and those tables take
any row count. plurel does not fit a model of the real rows or draw more rows like them. As a
cause, a real number is read as its standardized rank, with a blank at the middle rank, and a real
level by an effect per level, so generated columns follow the real values' order, not their units.

### Save a database

`create_database` orders event tables by time and gives keys as row positions, the
[relbench](https://github.com/stanford-star/relbench) form, which `write_database` saves and
`read_database` loads back; it writes a `db` directory and a `manifest.yaml` under the given path.
It refuses a path that already holds a dataset unless given `overwrite=True`, and a path holding
anything else. It writes into a hidden sibling directory and renames it into place, so the path
holds a complete dataset or none, and a failed or interrupted write keeps the previous one. A
symlinked path is written where it points. Of writers racing for
one path, each publishes a complete dataset or raises `FileExistsError`.

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
event table (`activity_row_count`). To scale a database's rows, scale both ranges: twice the default
is `entity_row_count=LogIntegersRange(1000, 2000), activity_row_count=LogIntegersRange(20000, 60000)`,
and `prior.rows(drawn, seed=...)` then draws the counts. `TablePrior`'s `column_count` counts the
columns besides the key.

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

## Author many tables from one description

Any weight, effect, noise or marginal above may be drawn from a range. The table is then a family: `realize(seed)` draws one
table of it, and one description gives many tables, each reproducible from its seed. `Schema.realize`
does the same for a whole database, so an agent can author either.

```python
from plurel import LOW, MEDIUM, STRONG, WEAK, Categorical, Noise, Numeric, Scaled, Table

referrals = Table(
    {
        "age": Numeric(Scaled("duration", 0, 90, shape="right"), doc="the patient's age"),
        "wait_days": Numeric(
            Scaled("duration", 0, 60), round=1, causes={"age": WEAK}, noise=LOW, doc="days waited"
        ),
        "outcome": Categorical(
            {"declined": 0.7, "accepted": 0.3},
            causes={"wait_days": (-STRONG, "tanh")},
            noise=Noise("gaussian", MEDIUM),
            doc="how it ended",
        ),
    },
    name="clinic_referral_outcome",
    doc="Each row is one referral a clinic received; outcome records how it ended.",
    target="outcome",
)
rows = referrals.realize(seed=0).sample(1000, seed=0)
```

`STRONG` and `WEAK` are strengths and `LOW`, `MEDIUM` and `HIGH` noise levels, each a `Range` a seed
draws from; `-WEAK` is a negative strength.

| Drawable | Draws |
|---|---|
| `Range(low, high)` | A number, wherever a weight, an effect or a noise goes |
| `Scaled(units, low, high, shape=None)` | A marginal on the range, of the given shape or one the `units` allow: `count`, `amount`, `duration`, `ratio`, `score`, `percent`, `signed` |
| `Shares(levels, concentration)` | A categorical column's shares, around a dict of them or from its levels alone |
| `Noise(kind, std)` | Noise: `gaussian`, `laplace`, `student_t`, `skewed` or `gumbel` |
| `ByLevel(column, {level: (low, high)})` | A marginal with its own range per level of a categorical column |

A database is the same: `schema.realize(seed=0).sample({"clinics": 40, "visits": 5000}, seed=0)` gives
a dict of DataFrames.

`Scaled` places the central 99% of a shape on `[low, high]`. Without a `shape` it draws one per seed
from those its units allow, so name the `shape` when a column's form matters:

| Units | Shapes |
|---|---|
| `count` | `counts`, `zero_inflated` |
| `amount` | `right`, `zero_inflated`, `symmetric`, `bimodal` |
| `duration` | `right`, `symmetric`, `bimodal` |
| `ratio` | `right`, `symmetric`, `left` |
| `score` | `symmetric`, `left`, `bimodal`, `right` |
| `percent` | `symmetric`, `left`, `right`, `bimodal`, `multimodal` |
| `signed` | `symmetric`, `bimodal`, `multimodal` |

`right` is skewed by a degree each seed draws, from mild to a long tail, so its median can sit near
the middle of the range. For values that are mostly small with a long tail in every seed, give the
marginal itself, such as `Numeric(LogNormal(4, 1), clip=(20, 2000))`.

`Categorical(shares, scores={level: {column: weight}})` takes the level that scores highest instead
of cutting a rank. Realizing fits these scores and the level cuts to one draw of the rows, so a
realized table's shares follow its rows and it can be sampled at any size. A column that depends on a
key keeps its shares exactly in every draw instead, since rows sharing a parent are not independent.

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
