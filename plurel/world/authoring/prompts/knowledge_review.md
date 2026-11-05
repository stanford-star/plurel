Read-only review of the world knowledge written for {count} schemas of a catalog: the schemas of `{module}` (sector `{sector}`). Worktree `{worktree}`. Draft `{draft}`. Do not edit any file and do not run any program (read with the Read tool, cat and grep only). Report findings; the caller fixes.

A model will trust this text, so a false or overreaching sentence teaches it something wrong. The mechanical gate already passed (every column noted, no dashes, no numbers outside the brief, no mention of synthetic data); you judge what it cannot. Be strict: when in doubt, report.

## The briefs the text was written from

One JSON line per schema. `information` is how much a column alone tells about the outcome (0 to 1); `typical_percentile_by_outcome` the column's median percentile per outcome; `outcome_shares_by_value` the outcome's shares at each value of a categorical column; `role` the column's place in the causal graph; `unrecorded_cause` whether something the records do not capture also moves the outcome.

```
{briefs}
```

## For every schema judge

A. **Findings.** Every claim that a column goes with an outcome is in the study: no reversed direction, no misquoted percentile or share, no column called important beyond its `information`, no column with near-zero `information` described as mattering.
B. **Beyond the brief.** No relation, threshold or number the brief does not give; nothing about how columns combine in one row, how sure a call can be, timing ("an early sign"), or where totals sit across the population (the brief gives rates within each value, not how many rows hold it); no false precision ("exactly").
C. **Domain knowledge.** True of the real domain, and implying no relation to the outcome that the study does not show. Doubtful is a finding.
D. **Unrecorded cause.** Mentioned only when `unrecorded_cause` is true, never named, guessed at or illustrated.
E. **Form.** Makes plain what one record stands for, without a stock opening ("Each row is", "You are looking at"), and how common each outcome is; column notes say what a column is, never the study's findings (each fact once); natural prose about the records, never "column X pushes toward Y" or the study's own terms ("information", "share of uncertainty"); no sentence frame repeated across the file's schemas; no column called a driver of the outcome; no mention of briefs, studies of samples, language models or synthetic data.

## Output

One finding per line, most severe first, each in the form
`<schema_id>: <letter>: "<offending words>": <what is wrong>; fix: <the fix in one sentence>`
so the lines can go straight into the fix step's findings file. Then one line: schemas read, schemas with no finding. Max 40 lines.
