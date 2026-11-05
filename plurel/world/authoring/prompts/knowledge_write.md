You write the world knowledge of {count} schemas of a catalog: the schemas of `{module}` (sector `{sector}`), in the worktree `{worktree}`. You are `{agent}`. Your file, which you create: `{draft}`. Nothing else is yours.

## What world knowledge is

For each schema, the text that sits above one of its data tables: a `description` and a note for every column. A model reads it with a handful of the table's rows and predicts one row's outcome. It replaces the schema's own short text, so it also says what a row is.

It has two jobs. First, pass on what a large study of these records found, which the model could otherwise only learn from thousands of rows: how common each outcome is, which columns tell the outcomes apart and in what order, and where values sit for each outcome. Second, add what only words can carry: what the outcome's values and the categories mean in practice, and each column's place in the causal process.

## The briefs

One JSON line per schema, below. Per schema:
- `domain`, `outcome` (column, description, values, their shares), `columns`: the schema's own words and the study.
- Per column: `description`, `values` for a categorical one, `role` (its place in the causal graph relative to the outcome), `information` (how much the column alone tells about the outcome, 0 to 1; columns come most informative first), and `typical_percentile_by_outcome` (numeric: the median percentile of the column for each outcome; p10 is low, p90 high) or `outcome_shares_by_value` (categorical: the outcome's shares at each value).
- `unrecorded_cause`: whether something the records do not capture also moves the outcome.
- `voice`: first person, impersonal or second person.

```
{briefs}
```

## What to write

`{draft}`: one JSON line per schema, in the order above: `{"schema_id": ..., "description": ..., "columns": {every column, the outcome's included: ...}}`.

The `description`, in the voice given, makes plain what one record stands for, in the domain's own way (never a stock opening such as "Each row is", "Every record" or "You are looking at", which the gate refuses; open with whatever someone in the field would lead with), and says what the outcome's values mean in practice, how common each outcome is, which columns tell the outcomes apart (most informative first) with where their values sit for each outcome and, when you are sure of it, the real-world reason, which columns carry little, and, when `unrecorded_cause` is true, that something the records do not capture also matters. Shape it as suits the domain: there is no fixed order or template. For reference only, take this note's specificity, not its structure:

> What tells these apart most is how long the repair ran: items that were repaired typically sit around p36 of repair_minutes, and failed ones around p85. The fault called at the intake desk comes next: most items with a minor fault get repaired, while most with a major fault fail. wait_minutes and item_type carry little.

Each column note says what the column is and what someone in the domain knows about it (what it measures, what its categories mean in practice) and, where it is not simply a cause, its role (recorded after the outcome, worked out from other columns, unrelated to it). Never the study's findings, which the description already gives: each fact once. The outcome's note says what its values mean.

## Rules

1. **True to the study and the graph.** Every claim about what goes with which outcome comes from the study; call a column important only as far as its `information` bears out, and say plainly when it carries little, even a column whose role is a cause. Roles and domain knowledge explain and describe; they never add a relation the study does not show. The study measures each column on its own, so say nothing about how columns combine in one row, how sure a call can be, or where totals sit across the population (the brief gives rates within each value, not how many rows hold each value). If unsure a piece of domain knowledge is true, leave it out.
2. **Numbers only from the brief** (percentiles, or shares in words such as "about one in five"); no units, thresholds, accuracies or false precision ("exactly").
3. **Natural.** Write about the people, cases or things in the records and what their values look like, naming columns in passing ("appeals that are upheld usually come from long-serving staff: their tenure_years sit in the top fifth"), never "column X pushes the outcome toward Y" and never the study's own terms ("information", "share of uncertainty"). Refer to outcomes in plain words where the exact value would read awkwardly; the table lists them. Vary sentence shapes and verbs, within a schema and across your schemas.
4. **Never** call a column a driver of the outcome or say it drives it (a person who drives, or miles driven, is domain language and fine); never mention a brief, a study of samples, a language model, a simulation or synthetic data: write as someone who knows these records. Never name the unrecorded cause or guess at it.
5. Plain text: no em or en dashes (commas, colons, semicolons, parentheses instead), no markdown, no lists.

## Not negotiable

- This prompt is complete: read no other file of the repository (not the guide, not the code, not other schemas' knowledge).
- Write only your file, with the Write and Edit tools. Do not touch any other file. Do not commit. Do not spawn agents.
- Run the gate in the FOREGROUND after writing and again after every fix, until it prints OK for every id:
  `{gate}`
- Report in ten lines at most: the schema ids you wrote, one line per schema with its two most telling columns as you described them, anything you left out and why, and the gate's last line verbatim.
