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
   calibration draw. A declaration without ranges is a generator already and samples directly.
3. `sample(rows, seed=data_seed)` returns frames. Sampling never changes the generator.

## What a draw repeats

| Promise | Holds when | Checked by |
|---|---|---|
| **Exact**: byte-identical frames | the same PluRel commit, dependency versions and platform; the same declarations, table names and keys; the same mechanism seed, data seed and row counts | `test/test_fingerprints.py` against stored sha256 digests, on every CI run |
| **Exact**: a generator rebuilt from its seed | as above, without the data seed | the same test: a realized generator equals one realized again, and its draws repeat |
| **Distributional**: the same mechanism, statistically | across platforms or dependency versions: floating point and library internals may differ, so draws may differ in their last bits or their tie-breaking while following the same mechanism | not checked as exact; see the matrix below for where exact also held |
| **Not promised**: the same draws across PluRel commits | a commit may change how draws are made; it says so in its PR and updates the fingerprints | `pixi run fingerprints` rewrites them, so the diff shows which draws moved |

A table's draws follow its name, else its key in the schema, and every column, key and node draws
from its own named stream, so adding a column moves no other column's draw. Row counts are part of
a draw: rows are not stable across draw sizes, and shards are independent draws under their own
seeds (see [the README](README.md#build-a-table)). The API itself is compatible within v2 except
for the breaking changes the changelog lists, each with its migration.

## Tested environments

| Environment | Python and dependencies | Fingerprints | Suite |
|---|---|---|---|
| macOS 26.1, Apple M4 (osx-arm64), locally | pixi lock: Python 3.12.0, numpy 2.5.1, pandas 3.0.5, pyarrow 25.0.0, scipy 1.17.0, relbench 3.0.1 | match | passes |
| GitHub Actions `macos-latest` (osx-arm64) | the same pixi lock | see this PR's CI | passes |
| GitHub Actions `ubuntu-latest` (linux-64) | the same pixi lock | see this PR's CI | passes |
| GitHub Actions `ubuntu-latest`, wheel installed clean | Python 3.12 and 3.13, latest released dependencies | not checked: the smoke test and the recipes run, which promise no bytes | smoke and recipes pass |

No other platform, Python version or dependency set is claimed. Windows is untested.
