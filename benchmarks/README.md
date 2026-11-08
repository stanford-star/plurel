# Generation benchmarks

How long PluRel takes and how much memory it uses to generate, from declaring a schema to loading
the saved database back. Results are saved as JSON in [`results/`](results/) and
[`REPORT.md`](REPORT.md) is generated from them; the expensive runs stay out of pull-request CI,
which only runs the smallest size once.

```bash
pixi run bench          # every workload at 1k, 10k, 100k and 1M rows, seeds 0-2, each run twice
pixi run bench-report   # REPORT.md from results/latest.json, without plots
pixi run bench-smoke    # 1k rows, seed 0, once, as CI runs it
pixi exec --spec python=3.12 --spec matplotlib python benchmarks/report.py \
    benchmarks/results/latest.json --out benchmarks/REPORT.md   # with time.png and memory.png
python benchmarks/run.py --workloads wide retail --sizes 1000 10000 --seeds 0 --repeats 1 --out /tmp/a.json
```

The report reads only the JSON, never PluRel, so plots need matplotlib alone, not in the project's
environment, whose locked versions the measurements depend on.

On Linux, the [`measure`](../.github/workflows/measure.yml) workflow runs the full benchmark on
GitHub's `ubuntu-latest` runner, outside pull-request CI: push a commit to a branch named
`measure/...`, then download its artifact, which holds the results, their report and the
fingerprinted draws for [COMPATIBILITY.md](../COMPATIBILITY.md).

## What is measured

Each run is one workload at one size under one seed, in a fresh process, so its peak memory
(`ru_maxrss`) is its own and includes the interpreter and imports, about 190 MiB. There is no
warm-up: every run is cold, and imports are not timed. Each seed is run twice, so the report can
tell how much the time moves with the draw, across seeds, from timing noise, across repeats of one
seed. Its phases are timed apart and summed:

| Phase | What it times |
|---|---|
| construct | declaring the tables and checking the schema; for a prior workload, drawing it from `SchemaPrior` |
| realize | `realize(seed)`: drawing ranges and fitting level cuts and scores to a calibration draw |
| sample | `sample(rows, seed=...)`: links, latents, records and frames |
| database | `create_database`: time order and positional keys |
| write | `write_database` to parquet, with a split only when a table holds events |
| read | `read_database` back |

Each record keeps the rows drawn, split into generated and retained real rows, the cells, the
tables, the size on disk and the peak memory. A run that fails, runs out of memory or times out is
kept with its configuration and message, and the report lists it.

## Workloads

| Name | Shape |
|---|---|
| `narrow` | one table of five columns and a short chain of causes |
| `wide` | one table of 50 numbers, each caused by the three before it, and a categorical |
| `columns-10`, `columns-200` | one table of 10 or 200 numbers, each caused by the three before it; 200 columns stop at 100k rows, 20M cells |
| `dense` | one table of 40 numbers, each caused by every one before it: 780 causes |
| `tables-2`, `tables-8`, `tables-32` | a chain of 2, 8 or 32 tables of five columns, each keyed to the one before and caused by it, the rows split evenly, none below 256 |
| `prior-0`, `prior-1`, `prior-2` | `SchemaPrior` databases under three seeds, scaled so the largest table takes the size; no table goes below 256 rows, which a link's clusters need |
| `prior-table` | one table of 20 columns drawn by `TablePrior`, the same table at every size |
| `mixed` | hand-authored customers and orders around a product table drawn by `TablePrior`; orders' amount is caused across both keys |
| `keyed` | students counted per student with `least` and `most`, matched to real courses by level, preferred by recorded size and unique per student |
| `retail`, `visits`, `enrollments` | the recipes in [`examples/generation`](../examples/generation/), their fixtures scaled to about the size; retail and enrollments grow their retained tables with the size |

The size is a target: the report gives the rows actually drawn. A recipe's retained tables count
as retained rows, not generated ones.

## Reading the results

A result holds for the machine, packages and commit its environment block names; compare two only
on the same machine. To compare a change, run the same workloads before and after it and keep both
files.

## First results: Apple M4, 8 November 2026

[`REPORT.md`](REPORT.md) holds the measurements, from
[`results/2026-11-08-apple-m4.json`](results/2026-11-08-apple-m4.json): commit `e0c49b7` with no
local changes to `plurel/` or `examples/`, 108 runs, 9 workloads × 4 sizes × 3 repeats. What follows
reads those numbers; it is interpretation, labeled as such, and holds for this machine only.

- **Every run finished.** No failure, timeout or running out of memory up to the largest run:
  `prior-2` at its 1M size draws 6.0M rows and 65M cells in 35 s, at a peak of 2.9 GiB of this
  machine's 16 GiB. Larger sizes were not tried, so the supported range measured here ends there.
- **Sampling dominates at scale.** At the 1M size, sampling is 80 to 95% of the time for the
  single tables, the prior databases and the keyed schema. For retail and enrollments it is 43 to
  46%, because realizing fits their cuts to a draw as large as their retained tables, which grow
  with the size here. At that size sampling makes 1.6 to 10M generated cells a second, slowest for
  the prior databases, likely because their nodes carry many nonlinear edges each; time grows about
  linearly in rows beyond 100k.
- **Memory grows about 40 to 130 bytes per cell** over the interpreter's 195 MiB: least for the
  wide table and the largest prior database, most for smaller multi-table schemas with many keys.
- **Small draws are bound by fixed costs.** At 1k rows a run takes about 0.06 s, two thirds of it
  reading the saved database back through relbench. A prior database adds 0.4 to 0.6 s to realize
  at any size, since calibration always draws 4096 rows per table.
- **Bottlenecks to look at, each a separate change:** calibrating over every retained row (a probe
  with 1M real customers spent 7 s realizing), since fixed by calibrating on a random 4096 of a real
  table's rows: retail's realize at the 1M size fell from 1.7 s to 0.05 s and its total from 3.9 s
  to 1.9 s; the per-cell cost of the prior's edges; and the fixed 4096-row calibration for small
  draws.

Saving is cheap throughout: under 1 s for every run, and 395 MiB on disk for the 51M-cell wide
table.
