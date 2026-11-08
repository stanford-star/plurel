# Generation concepts

Each script teaches one part of generating data, prints what happened, and checks its claims
with assertions. Run them in order or start with the concept you need. Each is independent:
there is no shared notebook state, private data, network access, or model-training dependency.
The fixtures are invented, and the mechanisms illustrate API behavior.

From a checkout with the v2 environment installed:

```bash
pixi run python examples/concepts/01_synthetic_table.py
pixi run python examples/concepts/05_calibration.py
pixi run recipes  # all concept examples and domain recipes, each in a fresh process
```

With v2 installed in another environment, use that environment's `python` in place of
`pixi run python`. Run without `-O`, which disables the assertions. All file-writing examples
use temporary directories; no output needs to be cleaned up.

## The lifecycle

**Declare → realize when needed → sample → inspect and check → export.**

| Stage | What it does | What comes back |
|---|---|---|
| Declare | Specify columns, relationships and authored mechanisms. Ranges make parameter choices unresolved; scored categoricals also require fitting. | A `Table` or `Schema` |
| Realize | Choose concrete parameters with a mechanism seed, then calibrate applicable categorical cuts or score biases. | A generator, still a `Table` or `Schema` |
| Sample | Produce a dataset with a data seed and row counts. Sampling does not change the generator. | A DataFrame or a dict of DataFrames |
| Inspect and check | Read `describe()`/`explain()`, examine frames, and check the relationships you intended. | Descriptions and checks, not a new generator |
| Export | Convert frames to a relbench database, then save and reload it. Keys become row positions; event tables are sorted by time. | A database and an on-disk dataset |

A fixed declaration without scored categoricals can sample directly. Calling `realize()` on
fixed categorical declarations can still change their behavior by pinning cuts. Calibration is
a step inside realization, not a prerequisite for every numeric table and not an operation that
learns a general-purpose synthesizer from an input DataFrame.

Use different mechanism seeds for different generators, and different data seeds for datasets
from one generator. `rows` passed to `realize` controls a calibration draw; `rows` passed to
`sample` controls the emitted dataset. Both draws must be feasible for the schema. See the
[compatibility contract](../../COMPATIBILITY.md) for the environment and naming requirements
behind exact repeatability. No example promises stable row prefixes across dataset sizes.

## Reading order

| Script | Question it answers |
|---|---|
| [01_synthetic_table.py](01_synthetic_table.py) | How do I generate a table entirely from authored declarations? |
| [02_latent_and_recorded.py](02_latent_and_recorded.py) | Does a coefficient act on a latent or on an emitted value? When do I use `Recorded` or `Derived`? |
| [03_realization.py](03_realization.py) | What changes with a mechanism seed, and what stays fixed when I sample? |
| [04_sampling.py](04_sampling.py) | What repeats, why does row count matter, and why are independent draws not streaming chunks? |
| [05_calibration.py](05_calibration.py) | Why do categorical shares behave differently before and after calibration? |
| [06_score_calibration.py](06_score_calibration.py) | How are score biases fit, and why does calibrating on retained rows preserve all source rows? |
| [07_calibration_sizes.py](07_calibration_sizes.py) | What if default calibration sizes cannot satisfy a relationship constraint? |
| [08_semi_synthetic_table.py](08_semi_synthetic_table.py) | How do I preserve source columns while adding an exact formula and a noisy target? |
| [09_semi_synthetic_relations.py](09_semi_synthetic_relations.py) | How do retained foreign keys and newly generated child rows compose? |
| [10_relational_schema.py](10_relational_schema.py) | How do bounded counts, inferred versus explicit totals, and recorded summaries interact? |
| [11_missing_and_unmatched.py](11_missing_and_unmatched.py) | How do output blanks, `Recorded` fills, missing matches and randomly missing keys differ? |
| [12_temporal_data.py](12_temporal_data.py) | How do I guarantee event order and save a temporal database? |
| [13_static_export.py](13_static_export.py) | How do I save a static database, and what happens to external identifiers? |
| [14_random_priors.py](14_random_priors.py) | How is drawing a random structure different from authoring a family? |
| [15_edge_cases.py](15_edge_cases.py) | How do I handle unresolved parameters, cycles, empty or one-row draws, ties, fixed real-row counts and impossible totals? |

## Synthetic and semi-synthetic generation

| Workflow | Source of rows | What the API learns or chooses |
|---|---|---|
| Authored synthetic | All rows are generated. | You choose the schema and mechanisms; realization resolves parameter ranges and calibration. |
| Random prior | All rows are generated. | A prior also chooses structural features such as tables, columns and dependencies. Anonymous structures do not establish domain realism. |
| Semi-synthetic | Supplied rows and values remain; extra columns and related tables are generated. | You author the extra mechanisms. `from_frame` does not learn how to create additional source-like rows. |

Calibration uses a temporary draw. Current v2 uses up to a bounded subset of a large retained
table during calibration, while subsequent sampling retains the full input. Ordinary
categoricals on retained rows and categoricals that depend on keys continue to cut each draw;
score-based categoricals fit biases. The examples show these differences without importing
internal implementation classes. `describe()` is useful for observing the fitted values, but
its detailed layout is not a stable serialization format for reconstructing a generator.

Once these pieces are familiar, the [domain recipes](../generation/) combine them into retail
orders, clinic visits and student enrollments. Structural checks and learnable authored targets
demonstrate that the chosen mechanisms work; empirical realism and training benefit require
separate evaluation.
