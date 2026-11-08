# Compatibility

What PluRel v2 promises about its API and its draws, and what has been checked. API changes and
migrations are listed in [CHANGELOG.md](CHANGELOG.md).

## Public surface

The public API is what `plurel.__all__` exports, with the methods and attributes of those objects
that the [README](README.md) documents:

- declarations: `Table`, `Schema`, `Numeric`, `Categorical`, `Timestamp`, `Hidden`, `Derived`,
  `Real`, `PrimaryKey`, `ForeignKey`, `Counts`, `Recorded`, `Summary`, `ByLevel`, `Scaled`, `Shares`,
  `Noise`, `Range` and the named strengths `WEAK`, `STRONG`, `LOW`, `MEDIUM`, `HIGH`;
- distributions and links: `Normal`, `Uniform`, `LogNormal`, `Exponential`, `Gamma` and the others
  exported, `RandomLink`, `HSBMLink`, `TreeLink`, and the `Distribution` and `Link` protocols as
  extension points;
- random schemas: `SchemaPrior`, `TablePrior` and the `Choices` ranges;
- databases: `create_database`, `write_database`, `read_database`, `split_timestamps`;
- inspection: `describe()` and `explain()` on tables and schemas.

Everything else is internal and may change without notice: the modules `plurel.graph`,
`plurel.columns`, `plurel.draw`, `plurel.fit` and `plurel.links` beyond the link classes, the
compiled `Node` and `Column` (accepted by `Table` as a lower-level escape hatch, not a stable
format), and the layout of `describe()` beyond being plain JSON-ready data.

## The lifecycle

1. Declare a `Table` or `Schema`. Ranges make it a family.
2. `realize(mechanism_seed)` makes a generator: every range drawn, level cuts and scores fit to a
   calibration draw. Declarations with neither parameter ranges nor categorical scores sample
   directly.
3. `sample(rows, seed=data_seed)` returns frames. Sampling never changes the generator.

## What a draw repeats

| Promise | Holds when | Checked by |
|---|---|---|
| **Exact**: byte-identical frames | the same PluRel commit, dependency versions and platform (operating system and processor); the same declarations, table names and keys; the same mechanism seed, data seed and row counts | `test/test_fingerprints.py` against sha256 digests stored per platform, on every CI run |
| **Exact**: a generator rebuilt from its seed | as above, without the data seed | the same test: a realized generator equals one realized again, and its draws repeat |
| **Distributional**: the same mechanism, statistically | across platforms or dependency versions: the random streams are the same, but floating-point results of functions such as `exp`, `log` and matrix products may differ, and reach values that no marginal re-ranks | measured, not promised: the matrix below says which workloads still matched exactly |
| **Not promised**: the same draws across PluRel commits | a commit may change how draws are made; it says so in its PR and updates the fingerprints | `pixi run fingerprints` rewrites this platform's; a failing test prints every platform's new digests in its CI log, ready to paste |

A table's draws follow its name, else its key in the schema, and every column, key and node draws
from its own named stream, so adding a column moves no other column's draw. Row counts are part of
a draw: rows are not stable across draw sizes, and shards are independent draws under their own
seeds (see [the README](README.md#build-a-table)). The API itself is compatible within v2 except
for the breaking changes the changelog lists, each with its migration.

## Measured across platforms

The same pixi lock on macOS (osx-arm64) and Linux (linux-64) drew the same bytes for one of the
seven fingerprinted workloads, the school schema, whose numbers pass through marginals and linear
causes; the other six, which use log-normal marginals, `exp`/`log` transforms or the prior's
matrix and tree edges, differed. So exact repeatability is a within-platform promise, and the
digests are stored per platform.

How far they differ was measured at commit `7241cbc`: every fingerprinted draw saved on the Apple
M4 and on a GitHub `ubuntu-latest` runner (AMD EPYC 9V74), with `test/fingerprints.py --save`, then
set side by side with `scripts/compare_draws.py`:

| Draws | Cells that differ | Largest difference | Categorical cells changed | Marginal and correlation shift |
|---|---:|---:|---:|---:|
| the four prior databases, two seeds each | 1.7 to 5.0% | under 3e-14 standard deviations | none | under 1e-15 |
| shop and teams, two seeds each | 0.1 to 0.6% | under 6e-14 standard deviations | none | none |
| school, two seeds | none | none | none | none |

Every table drew as many rows on both, every categorical and every key took the same value, and
the rank order of every number held: the differences are in the last bits of numbers and change
nothing a model could see. Within this measurement the distributional promise holds and is close
to exact. It is one pair of machines; another processor may differ more.

One CI run on Linux has failed the fingerprints for six prior draws and passed on rerun without a
change ([#155](https://github.com/stanford-star/plurel/pull/155)): GitHub's Linux runners are not
all one processor, and the run above matched the stored `linux-x86_64` digests for every draw it
could check (12 of 14; the teams draws change in the parquet round trip), so exact bytes may
depend on the processor and not only on the platform name. CI now logs the processor beside the
suite, so the next such failure names it.

## Tested environments

| Environment | Python and dependencies | Fingerprints | Suite |
|---|---|---|---|
| macOS 26.1, Apple M4 (osx-arm64), locally | pixi lock: Python 3.12.0, numpy 2.5.1, pandas 3.0.5, pyarrow 25.0.0, scipy 1.17.0, relbench 3.0.1 | match the `darwin-arm64` digests, which it wrote | passes |
| GitHub Actions `macos-latest` (osx-arm64) | the same pixi lock | match the `darwin-arm64` digests written locally: exact across these two Apple silicon machines | passes |
| GitHub Actions `ubuntu-latest` (linux-64) | the same pixi lock | match its own `linux-x86_64` digests; 12 of 14 differ from `darwin-arm64` | passes |
| GitHub Actions `ubuntu-latest`, wheel installed clean | Python 3.12 and 3.13, latest released dependencies | not checked: the smoke test and the recipes run, which promise no bytes | smoke and recipes pass |

No other platform, Python version or dependency set is claimed. Windows is untested. On a platform
with no stored digests the fingerprint test skips locally, printing its digests, and fails on CI.
