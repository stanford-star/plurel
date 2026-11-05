You fix named defects in the world knowledge drafted for the schemas of `{module}` (sector `{sector}`). Worktree `{worktree}`. Draft `{draft}`. Nothing else is yours.

Findings to resolve, from the review (each names a schema id):
{findings}

## The briefs the text was written from

```
{briefs}
```

## Rules the fix must keep

1. **True to the study and the graph.** Every claim about what goes with which outcome comes from the brief's study; a column is important only as far as its `information` bears out. Nothing about how columns combine in one row, how sure a call can be, timing, or where totals sit across the population. Domain knowledge explains and describes; it never adds a relation the study does not show.
2. **Numbers only from the brief**; no units, thresholds, accuracies or false precision.
3. **Natural.** About the people, cases or things in the records, naming columns in passing; never "column X pushes toward Y" or the study's own terms.
4. **Never** a column called a driver of the outcome (a person who drives is fine), the unrecorded cause's name, or a mention of briefs, studies of samples, language models or synthetic data.
5. Plain text: no em or en dashes, no markdown, no lists; no stock opening ("Each row is", "Every record", "You are looking at"). Column notes say what a column is; the description holds the findings; each fact once.

## Not negotiable

- This prompt is complete: read no other file of the repository (not the guide, not the code, not other schemas' knowledge).
- Edit only the lines of the schemas named in the findings, in your draft, with the Read, Edit and Write tools; keep every other line and the order unchanged. Do not touch any other file. Do not commit. Do not spawn agents.
- Change only what a finding names, unless a fix forces a neighbouring sentence to change.
- Run the gate in the FOREGROUND after your edits, until it prints OK for every id:
  `{gate}`
- Report in ten lines at most: per finding what you changed (schema id, old words, new words), anything you could not do and why, and the gate's last line verbatim.
