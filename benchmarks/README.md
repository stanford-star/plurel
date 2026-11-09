# Generation benchmark

How long PluRel takes, and how much memory it uses, to generate random tables and databases the
way a corpus is generated.

## What is measured

A sample of draws from the default priors, one per seed, so the results describe what the priors
generate rather than a few hand-picked structures:

| Kind | Draw `seed` |
|---|---|
| random table | `TablePrior().realize(seed)`, at a row count drawn as `SchemaPrior` draws an event table's: 10,000 to 30,000 |
| random database | `SchemaPrior().realize(seed)`, at the row counts `prior.rows(database, seed)` draws: 2 to 20 tables, 500 to 1,000 rows per referenced table and 10,000 to 30,000 per event table |

By default seeds 0 to 99 of each. Every draw samples with data seed 0, in a fresh process with no
warm-up, timed in the phases the README's random-database example and `scripts/synthetic_gen.py`
run:

| Phase | What it times |
|---|---|
| draw | drawing the table or database from its prior |
| sample | `sample(rows, seed=0)` |
| database | `create_database`: time order and positional keys |
| write | `write_database` to parquet |
| read | `read_database` back |

Each record keeps the draw's tables, keys, rows and cells, the time of every phase, its peak memory
(the process's, so including the interpreter and imports, about 195 MiB) and its size on disk. The
report gives, per kind, the median, the 90th percentile and the extremes of each, the median share
of each phase, the slowest draws and every failure.

## Running it

```bash
pixi run bench          # 100 tables and 100 databases, to results/latest.json
pixi run bench-report   # REPORT.md from results/latest.json, without figures
pixi run bench-smoke    # 3 of each, as CI runs it
pixi exec --spec python=3.12 --spec matplotlib python benchmarks/report.py \
    benchmarks/results/latest.json --out benchmarks/REPORT.md   # with figures
```

`--count` sets the draws of each kind and `--kinds` picks one. The report reads only the results
JSON, so its figures need matplotlib alone, outside the project's environment: each draw's time
against its rows and its peak memory against its cells, saved beside the results as
`results/<results>-time.png` and `-memory.png`. On Linux, the
[`measure`](../.github/workflows/measure.yml) workflow runs the benchmark on GitHub's
`ubuntu-latest` runner, outside pull-request CI: push a commit to a branch named `measure/...` and
download its artifact.

A result holds for the machine, packages and commit its report names; compare two only on the same
machine, and on the same prior: a change to a prior changes what is drawn.
