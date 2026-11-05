# Adding schemas to a catalog: the agent's guide

Written for an agent that has to add schemas to a catalog and get them accepted. A catalog is a three level tree of sector, division and group, declared by one plan per sector (`<plans>/<sector>.yml`); a group is a directory of batch modules, one per authoring order. Part one (steps 1 to 5) adds schemas; part two (steps 6 to 10) writes each schema's world knowledge.

Every command below is `python -m plurel.world.authoring --workspace <module>:<name> <step>`, written `$AUTHOR <step>`, run from the worktree root. A `Workspace` says where the catalog lives; `examples/world/` has a working one.

## Design decisions

- **The tables are the catalog.** What a model trained on these tables learns about the world is the set of schemas and their shares. Fix a thin area by authoring more schemas there, not by reweighting at build time.
- **A schema is a real record.** Each row is a record an organisation keeps about one entity, with that record's own fields, units and codes; the SCM mirrors the real process (base rates, marginals, dependencies), sourced in the rationale. Numbers are approximations of named aggregates ("about"), never a table copied to the decimal. Real names may be level values; a level list that reproduces a published table row for row is a copy. Public datasets' column sets, codings and targets are never reproduced.
- **The target is what the process decides or records.** A measurement cut into bands is a column, not a target.
- **One record per subject across the catalog.** Two schemas about the same record and target are a duplicate, resolved in the schemas (delete or tell apart), never in an allowlist.
- **Validation and hygiene are kept apart.** Validation reads the mechanism and blocks: two to ten classes, realizable shares, `level_probs` on wide roots, every class and no constant column in realized rows. Hygiene is the authoring discipline for a new file (the "Each row is" opener, no identifier features, no binned targets, the copy guards, a sourced rationale): the gate enforces it on new work and the census reports it over a catalog.
- **Tables show the world's messiness where the process has it.** The six phenomena (blanks with a reason, censoring or recording quanta, zero inflation, a hidden cause, a non-additive mechanism, an interaction) are mechanisms the author adds when the real record has them; no count is required. Rare outcomes are rare, and shares are never uniform.
- **The file layout is the taxonomy.** A schema's placement derives from its path `<catalog>/<sector>/<division>/<group>/<batch>.py` and is never authored inline. A node is added only when no existing node holds the records without stretching its title.
- **Authoring is a pipeline of small, checkable steps**, one command each: plan, orders (one self-contained prompt per batch; the agent reads one example file and nothing else, so column vocabulary does not spread), gate, dedupe, review (the judgement rules; off by more than 2x is a finding), fix, sweep.
- **Agents work on their own file and report verbatim.** An agent edits only its module, never commits, and ends its report with the commands' last lines as printed. A declined record (the real process keeps no such record) is not a defect.

## Step 1. Add batches to the plan

Growth means adding batches to a group of `<plans>/<sector>.yml`; each batch is one order and one module:

```yaml
divisions:
  repair_and_reuse:
    title: Repair and reuse
    groups:
      repair_cafes:
        title: Repair cafes
        batches:
          item_visits:
          - an item brought to a volunteer repair cafe, how the visit ended
```

- Batch slugs are lower snake case, unique within the group, named for the records, never a code.
- Records name the row and the target the module will hold.
- A new sector needs a `scope` no other plan covers. Titles are plain words with the first capitalised.
- Anchors are aggregate statistics; naming a dataset is a defect.

`$AUTHOR plan --sector <sector>` prints the tree and the batch modules missing on disk.

## Step 2. Render the orders

```bash
$AUTHOR orders --sector <sector> --out <scratch>/orders --per-file 6 --worktree <worktree> --branch <branch> --only-missing
```

One `agentNN.md` per missing batch module plus `agents.md`. Each prompt is complete: the agent needs the prompt, the example file it names, and its own file. Do not add reading. `--run` sets the command prefix agents run Python with, such as a cluster's job launcher.

## Step 3. Run the authors

- One agent per prompt, files disjoint, in the background.
- Each report must end with the commands' last lines verbatim. Accept a file only when the gate prints OK for every id and "every schema in the file validates", and dedupe prints "no candidate pairs".
- A declined record is not a defect: the agent says why and picks another from the plan.

## Step 4. Review

`$AUTHOR review --sector <sector> --out <scratch>/review --worktree <worktree>` renders one read-only reviewer prompt per file with its duplicate candidates. Reviewers report and do not edit. Then write each file's findings to a text file, one per line naming the schema id, and render its fix prompt:

```bash
$AUTHOR fix --file <module> --findings <scratch>/findings/<name>.txt --out <scratch>/fix/<name>.md --worktree <worktree>
```

A duplicate candidate is resolved in the schemas: delete the one whose organisation does not keep the record, or tell the two apart in their ids, descriptions and target class names until dedupe no longer pairs them.

## Step 5. Sweep

`$AUTHOR sweep --sector <sector>` validates every schema of the sector with realization, lists duplicate candidates across the catalog, counts the hygiene findings and prints the census. Exit 0 is the acceptance condition. `$AUTHOR census --out <csv>` tabulates a whole catalog.

## Failure modes, and what to do

| symptom | cause | do |
|---|---|---|
| gate: "rationale names no statistical source" | publisher not named | name the office, agency, regulator or association |
| gate: "target ... cut into bins" | band, quintile, decile or bucket target | keep the number as a column, predict the event or decision |
| gate: "quotes 'X percent' to the decimal" | copied figure | round and write "about" |
| gate: "names a public dataset" | microdata or open data cited as the record | cite the aggregate publication, make the row the organisation's own record |
| gate: "a class share under 0.2%" | rare event on unconditioned rows | condition the rows (a queue, a filing, an inspection) and say so in the description |
| gate: "target has N classes" | a code list used as the target | merge the rarest codes into their real parent category, or predict the decision the record holds |
| the target is a function of its columns | no noise the process really has | add it, add a hidden cause, or predict a less deterministic decision. Child latents are not standardized: a `u` edge or a raw product `interactions` term on a strong child carries variance in the hundreds against target noise of about 1, so halve the edges or move the product off the target |
| no column moves the target | `zero_inflation` on the drivers zeroes rendered cells at random after the target was computed; or a `threshold` edge under `max` or `min` put most rows on a plateau | make the point mass part of the marginal (`dist="zero_inflated"`) and keep artifacts light; use hinges or an additive tally |
| validation: constant feature column | a count on a tiny integer scale rounded every row to one value | author the column's `dist` and `rounding=1` explicitly |
| dedupe: candidate pairs | the same record elsewhere, or two records described alike | delete from the sector that does not keep the record, or tell the two apart |
| validate() error | a schema model constraint | percent scale within 0 to 100; duration, amount and count take no `left` dist; a rules mechanism needs threshold edges; an argmax target needs `level_probs` and numeric drivers; formula nodes are leaves |

## Part two: world knowledge

Every schema can carry world knowledge: a `description` and a note per column, stored in `<knowledge>/<sector>.json` by schema id. Run part two for a new sector, and with `--only-missing` after part one adds schemas.

- **Two jobs, nothing else.** The text passes on what a large sample of the schema's population shows, and adds what only words carry: what the outcome and the categories mean in practice and each column's place in the causal process.
- **The study, not the authored graph, says what matters.** `study` pools 8192 rows over four realizations of the population: each outcome's share, how much each column alone tells about it and where its values sit per outcome. The authored graph can disagree with what the population shows, so a column's importance comes from the study; the graph gives only roles and whether an unrecorded cause matters.
- **Only what the brief supports.** The study measures each column on its own: no text about how columns combine in one row, how sure a call is, timing, or where totals sit across the population. The gate cannot see this kind of claim, so the review is not optional.
- **Natural, no template.** Written like someone who knows the records, naming columns in passing.
- **Findings once.** The description holds the findings; each column note says what the column is and what the domain knows about it.

## Step 6. Render the knowledge orders

`$AUTHOR knowledge-orders --sector <sector> --out <scratch>/knowledge --worktree <worktree> --only-missing` renders one `knowledgeNN.md` per catalog file with schemas to write, plus `knowledge_agents.md` and `knowledge_orders.json`, which the later steps read. Each prompt is complete, with the briefs inline.

## Step 7. Run the writers

One agent per prompt, drafts disjoint (`<scratch>/knowledge/knowledgeNN.jsonl`). Each writer runs `$AUTHOR knowledge-gate <draft> <schema ids>` until it prints OK for every id. The gate lints each entry against a freshly computed brief: every column and the outcome noted, no empty note, no em or en dash, no number outside the brief, no mention of synthetic data, the unrecorded cause never named. Accept a draft only when its report ends with "every schema in the draft passes".

## Step 8. Review

`$AUTHOR knowledge-review --out <scratch>/knowledge --worktree <worktree>` renders one read-only reviewer prompt per written draft, with the briefs inline. Reviewers judge what the gate cannot: findings true to the study, nothing beyond the brief, domain knowledge true and implying no unmeasured relation, the unrecorded cause unnamed, the form.

## Step 9. Fix

Write each draft's findings to a text file, one per line naming the schema id, and run `$AUTHOR knowledge-fix --out <scratch>/knowledge --agent knowledgeNN --findings <file> --worktree <worktree>`. The fix agent edits only the named schemas' lines and ends with the same gate.

## Step 10. Merge and sweep

`$AUTHOR knowledge-merge --out <scratch>/knowledge` writes every passing draft line into `<knowledge>/<sector>.json` and lists what it left out. `$AUTHOR knowledge-sweep --sector <sector> --workers 4` checks that every schema of the sector has world knowledge that passes the lint.

### Failure modes of world knowledge

| symptom | cause | do |
|---|---|---|
| gate: "columns must be exactly" | a column or the outcome's note left out | note every column, the outcome included |
| gate: "number not in the brief" | a unit, threshold, accuracy or a figure worked out from the brief | say it in words the brief supports |
| gate: "banned wording" | synthetic or simulated in the prose (a column's own name is exempt) | describe the records as someone who keeps them |
| gate: "stock opening" | "Each row is", "Every record", "You are looking at" | open with what someone in the field would lead with |
| gate: "names the unrecorded cause" | the hidden node's name or a paraphrase of it | say only that something the records do not capture also matters |
| review: a reading rule or confidence ("sure when X and Y agree") | the study measures each column on its own | drop it; state where each column's values sit |
| review: where totals sit | a population mix read off rates within each value | keep to the rate within each value |
| review: timing ("an early sign of") | a same-period association told as a forecast | state where the values sit for each outcome |
| review: a cause called important with near zero information | the role read as importance | say it carries little; the role goes in the column note |
