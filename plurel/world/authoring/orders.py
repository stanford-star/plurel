"""Authoring orders: a sector plan becomes one assignment per batch module and one self-contained prompt per
agent. The prompt carries every rule the agent needs (the table model, the realism rules the gate and the
review apply, the exclusions, the batch's records); the agent reads it, one example file, and nothing else."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plurel.world.authoring.workspace import Workspace
from plurel.world.plans import Plan


@dataclass(frozen=True, kw_only=True)
class Assignment:
    agent: str
    catalog_dir: Path
    sector: str
    division: str
    group: str
    batch: str

    @property
    def path(self) -> Path:
        return self.catalog_dir / self.sector / self.division / self.group / f"{self.batch}.py"


def assignments(
    plan: Plan, workspace: Workspace, *, only_missing: bool, root: Path
) -> list[Assignment]:
    """One agent per batch module of the plan, in plan order; ``only_missing`` skips modules already on disk."""
    out = []
    for division, group, batch in plan.batches():
        a = Assignment(
            agent=f"agent{len(out) + 1:02d}",
            catalog_dir=workspace.catalog_dir,
            sector=plan.sector,
            division=division,
            group=group,
            batch=batch,
        )
        if only_missing and (root / a.path).exists():
            continue
        out.append(a)
    return out


SCHEMA_MODEL = """The table model (`from plurel import ...`; `plurel.world.validate` is the first gate):
- `Table({column: entry}, name=, doc=, target=, sources=)`. `doc` opens "Each row is" and names the record and who keeps it. `sources` says where every number comes from. Any number may be a range, which a seed draws: `STRONG` and `WEAK` for strengths (`-WEAK` for a negative one), `LOW`, `MEDIUM`, `HIGH` for noise, `Range(low, high)` otherwise.
- `Numeric(Scaled(units, low, high, shape=), causes=, noise=, doc=)`, units count|amount|duration|ratio|score|percent|signed; a root names its `shape` from the units' family (count: counts, zero_inflated; amount: right, zero_inflated, symmetric, bimodal; duration: right, symmetric, bimodal; ratio: right, symmetric, left; score: symmetric, left, bimodal, right; percent: symmetric, left, right, bimodal, multimodal; signed: symmetric, bimodal, multimodal). A column with causes authors `noise=Noise(gaussian|laplace|student_t|skewed, LOW|MEDIUM|HIGH)`. Recording: `round=`, `clip=(low, high)` (a side may be None), `zeros=` (a share at the floor), `missing=` (at random), with `missing_by=(column, "high"|"low")` (by another column, or by itself).
- `Categorical({level: share}, causes=, noise=, doc=)`; shares on every categorical root and the target, summing to 1. `ordered=True` only for an intensity scale.
- Causes, `{column: effect}`: for a number or an ordered level, a weight or `(weight, "relu"|"tanh"|"step"|"square")`; for an unordered level, `{level: effect}` or `(Range(0.7, 1.6), {level: effect})`. `(a, b): weight` multiplies two causes, `(level_column, number): {level: slope}` sets a slope per level, and `Numeric(ByLevel(column, {level: (low, high)}))` puts each level on its own range. `how=` combines causes: max|min|product|rules (rules: every weight has `step`). `Derived(lambda a, b: ..., doc=)` computes a leaf from recorded numbers. A nominal target may use `Categorical({level: share}, scores={level: {number: Range(1, 3) or -Range(1, 3)}}, noise=Noise("gumbel", Range(0.6, 1.8)))`, three or more levels and no causes. `Hidden(doc=)` is a cause no column records, feeding two or more columns on the way to the target.
- `doc` says what the column is, never a causal claim. No em or en dash anywhere. Names lower snake case, unique, naming the record and the target."""

REALISM_RULES = """Realism rules (validation and hygiene are the gate on your file; the review reads for the judgement rules):
- Validation, read off the mechanism, blocks merge: the target has two to ten classes, shares not uniform and no class under 0.2 percent (condition the rows on a queue, a filing or an inspection when the real rate is lower, and say so in the description); categorical roots with four or more levels carry shares; realized rows show every class and no constant column.
- Hygiene, the authoring rules for a new file: description opens "Each row is"; no identifier rendered as a numeric feature; the target is not a measurement cut into bins (no *_band, quintile, decile or bucket target: keep the number as a column and predict the event or decision the record holds); no share copied to the decimal and no percentage quoted to a decimal without "about"; no dataset, microdata or public use file named as the record; the rationale carries two or more quantitative facts and names a statistical source (an office, agency, regulator, association, registry report or survey by name).
- Phenomena, the record's real messiness, where the process has it and only there: blanks with their reason (MAR by another column, MNAR by the value), censoring or recording quanta at real limits, zero inflation, a hidden cause, a non-additive mechanism, an interaction. The census reports their shares; no count is required.
- Review: every base rate, share and marginal is a realistic approximation of the real process, anchored to a named aggregate statistic with "about" wording, never a public table copied row for row; the record is one an organisation keeps about the row entity, with that record's own fields, units and codes, never a public dataset's column set, coding or target; real instrument scales, no invented index; declared selection of the rows; real names may be level values (a force, a carrier, a region), but a level list carrying a published table's shares row for row is a copy, not an approximation; one record per subject across the catalog (the dedupe command lists candidate pairs; a pair is resolved before the file is accepted)."""


def realism_rules(workspace: Workspace) -> str:
    """The realism rules with the catalog's own."""
    return f"{REALISM_RULES}\n{workspace.extra_rules}" if workspace.extra_rules else REALISM_RULES


PROMPT = """You author {per_file} new world schemas for a catalog, sector `{sector}` ({sector_title}), in the worktree `{worktree}` (branch `{branch}`). You are `{agent}`. Your file, which you create: `{path}`. Its module docstring opens "{group_title}: {batch_title}." and lists the records the file holds.

The sector's scope, which every schema stays inside: {scope}

Your batch's records and targets, from the sector plan (author these; decline one only when the real process keeps no such record, and say so):
{records}

Row entities the plan names for this sector: {row_entities}
Anchors the plan names (aggregate statistics only, never a dataset): {anchors}
This sector leaves out: {exclusions}{subjects}

Read first: this prompt in full, then one example file for style and imports: `{example}`. Do not read other catalog files; do not imitate their column vocabulary.

Rules that are not negotiable:
- Edit only your file, with the Write and Edit tools. Do not touch any other file. Do not commit.
- The only commands you run are these, in the FOREGROUND, once your schemas are written and again after fixes:
{commands}
  The gate must print OK for every id and "every schema in the file validates"; dedupe must print "no candidate pairs" or you resolve each pair (drop or recut yours); any other command must pass. If the gate prints a Python error from the table or validate, read it, fix and rerun.
- Anything you need to write outside your file goes in `{scratch}/{agent}/`.

{schema_model}

{realism_rules}

Your {per_file} schemas together span at least three row entities, at least two target kinds (binary, graded ordinal ladder, nominal with scores) and a range of how much the columns tell about the target; each carries the phenomena its record really has; rare outcomes are rare (down to 0.2 percent) and never uniform.

Report in twelve lines at most: the {per_file} schema ids with row entity, target kind and class count, column count, rarest class share and phenomena; any record you declined and why; the last line of every command verbatim. Say plainly if anything failed.
"""


def render(
    assignment: Assignment,
    *,
    workspace: Workspace,
    plan: Plan,
    per_file: int,
    worktree: str,
    branch: str,
    run: str,
    scratch: str,
) -> str:
    division = plan.divisions[assignment.division]
    group = division.groups[assignment.group]
    records = group.batches[assignment.batch]
    subjects = ""
    if workspace.subjects:
        subjects = (
            f"\nSubjects no schema may be about under any wording: {'; '.join(workspace.subjects)}"
        )
    return PROMPT.format(
        per_file=per_file,
        sector=plan.sector,
        sector_title=plan.title,
        scope=plan.scope,
        worktree=worktree,
        branch=branch,
        agent=assignment.agent,
        path=assignment.path,
        group_title=group.title,
        batch_title=assignment.batch.replace("_", " ").capitalize(),
        records="\n".join(f"- {r}" for r in records),
        row_entities=", ".join(plan.row_entities) or "as the records imply",
        anchors="; ".join(plan.anchors) or "the sources the group's organisations publish",
        exclusions="; ".join(plan.exclusions) or "nothing named",
        subjects=subjects,
        example=workspace.example_file,
        commands=workspace.check_commands(run, assignment.path, "<your schema ids>"),
        scratch=scratch,
        schema_model=SCHEMA_MODEL,
        realism_rules=realism_rules(workspace),
    )


def write_orders(
    *,
    workspace: Workspace,
    plan: Plan,
    out: Path,
    root: Path,
    only_missing: bool,
    per_file: int,
    worktree: str,
    branch: str,
    run: str,
) -> list[Assignment]:
    """One prompt file per agent plus an index; returns the assignments."""
    out.mkdir(parents=True, exist_ok=True)
    todo = assignments(plan, workspace, only_missing=only_missing, root=root)
    for a in todo:
        (out / f"{a.agent}.md").write_text(
            render(
                a,
                workspace=workspace,
                plan=plan,
                per_file=per_file,
                worktree=worktree,
                branch=branch,
                run=run,
                scratch=str(out.resolve()),
            )
        )
    (out / "agents.md").write_text("\n".join(f"- {a.agent}: `{a.path}`" for a in todo) + "\n")
    return todo
