# Generation benchmark

How long PluRel takes, and how much memory it uses, to draw a random table or database and save it.

## What is measured

Five workloads, each drawn once from seed 1, so every run draws the same structure. Seed 1 is the
first whose workloads all sample at every size; at seed 0 the 128-table database's smallest tables
were too small for their links below 500k rows:

| Workload | Tables | How the rows are split |
|---|---:|---|
| random table, 20 columns (`table`) | 1 | all in the one table |
| random database, 2 tables (`database-2`) | 2 | in the shares the prior drew for that database |
| random database, 8 tables (`database-8`) | 8 | the same |
| random database, 32 tables (`database-32`) | 32 | the same |
| random database, 128 tables (`database-128`) | 128 | the same |

The table comes from `TablePrior`, the databases from `SchemaPrior` with the table count fixed.
Each runs at 100k, 250k, 500k and 1M rows. That is the number of rows drawn in total: a database
splits it across its tables in the same shares at every size, rounded to add up exactly. Below
100k rows, the 128 tables would be too small for the links between them.

Each run is a fresh process with no warm-up, timed in the phases the README's random-database example
and `scripts/synthetic_gen.py` run:

| Phase | What it times |
|---|---|
| draw | drawing the table or database from its prior: `prior.realize(0)` |
| sample | `sample(rows, seed=0)` |
| database | `create_database`: time order and positional keys |
| write | `write_database` to parquet |
| read | `read_database` back |

Each workload and size runs three times; the report gives the median and the range. Peak memory is
the process's, so it includes the interpreter and imports, about 195 MiB.

## Running it

```bash
pixi run bench          # every workload and size, three runs each, to results/latest.json
pixi run bench-report   # REPORT.md from results/latest.json, without figures
pixi run bench-smoke    # 100k rows once, as CI runs it
pixi exec --spec python=3.12 --spec matplotlib python benchmarks/report.py \
    benchmarks/results/latest.json --out benchmarks/REPORT.md   # with figures
```

The report reads only the results JSON, so its figures need matplotlib alone, outside the
project's environment. They are saved beside the results, named after them:
`results/<results>-time.png`, `-memory.png` and `-phases.png`. On Linux, the
[`measure`](../.github/workflows/measure.yml) workflow runs the benchmark on GitHub's
`ubuntu-latest` runner, outside pull-request CI: push a commit to a branch named `measure/...` and
download its artifact.

A result holds for the machine, packages and commit its report names; compare two only on the same
machine.
