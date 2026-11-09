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
    benchmarks/results/latest.json --out benchmarks/REPORT.md   # with its figures, from figures.py
python benchmarks/run.py --workloads wide retail --sizes 1000 10000 --seeds 0 --repeats 1 --out /tmp/a.json
```

The report reads only the JSON, never PluRel, so plots need matplotlib alone, not in the project's
environment, whose locked versions the measurements depend on. Its figures, drawn by
[`figures.py`](figures.py), are saved beside the results they come from, named after them:
`results/<results>-time.png`, `-memory.png` and `-phases.png`. They name each workload by what it
is; the table below gives the short names the results and the command line use.

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

## Results, 8 November 2026

Two machines at commit `7241cbc`, with no local changes to `plurel/` or `examples/`: an Apple M4
with 16 GiB ([REPORT.md](REPORT.md), with plots, from
[`results/2026-11-08-apple-m4-7241cbc.json`](results/2026-11-08-apple-m4-7241cbc.json)) and
GitHub's `ubuntu-latest` runner, 4 cores of an AMD EPYC 9V74 with 16 GiB
([REPORT-linux.md](REPORT-linux.md), from
[`results/2026-11-08-github-ubuntu-7241cbc.json`](results/2026-11-08-github-ubuntu-7241cbc.json),
made by the `measure` workflow). Each ran 402 runs, 17 workloads at 4 sizes, 3 seeds, twice each,
and skipped `columns-200` at 1M. The main README's
[latency and memory](../README.md#latency-and-memory) table reads the 1M sizes. What follows
interprets the numbers, as such, for these machines only.

- **Every run finished** on both: no failure, timeout or running out of memory. The largest,
  `prior-2` at 1M, draws 6.0M rows and 65M cells in 35 s on either machine, at a peak of 2.6 GiB
  on the M4 and 3.3 GiB on the runner. Larger sizes were not tried, so the measured range ends there.
- **Sampling dominates at scale**, and grows about linearly in rows beyond 100k. At 1M the M4
  samples hand-authored tables at 7 to 16M generated cells a second, keyed schemas with counts,
  matching or uniqueness at 3 to 4M, and random priors at 1.6 to 4.9M. Profiling `prior-2` puts
  the priors' time in their edges' arithmetic: Fourier cosines about 20%, quadratic products 13%,
  Gumbel noise for categorical scores 9%, MLPs 8%, with little Python overhead. The runner takes
  1.0 to 1.7 times as long, least for the prior databases and keyed schemas.
- **Variability is small:** a seed's reruns differ by a few percent, and seeds of one workload by
  up to about a third, the most for retail, the narrow table and the largest prior, where the
  draw's own sizes and fixed costs move the total.
- **Memory grows about 35 to 130 bytes per cell** over the interpreter's 195 MiB, least for wide
  tables and long chains of tables. The runner peaks lower for most workloads, likely its
  allocator, and higher for the largest prior.
- **Small draws are bound by fixed costs.** At 1k rows a run takes about 0.05 s, most of it
  reading the database back through relbench. A random prior adds about 0.45 s to realize at any
  size, since calibration draws 4,096 rows per table; fewer would make its fitted cuts noisier.

The earlier run at commit `e0c49b7`, three repeats of one seed each, stays in
[`results/2026-11-08-apple-m4.json`](results/2026-11-08-apple-m4.json) as a baseline. Since then
realizing calibrates on at most 4,096 rows of a large real table (retail's realize at 1M fell
from 1.7 s to 0.04 s), and the rank map sorts a latent without ties once instead of by two keys.
