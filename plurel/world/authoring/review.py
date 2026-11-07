"""Review and fix orders: one read-only reviewer prompt per catalog file, carrying the review rules, the file's schema
ids and its candidate duplicate pairs, and one fix prompt per file with the findings to resolve. The reviewer
reports; the fix agent edits its file and nothing else."""

from __future__ import annotations

from pathlib import Path

from plurel.world.authoring.orders import realism_rules
from plurel.world.authoring.workspace import Workspace

PROMPT = """Read-only realism review of one catalog file. Worktree `{worktree}`. File `{path}` with schemas: {ids}.
Do not edit any file and do not run any program (read with the Read tool or cat and grep only). Report findings; the caller fixes.

{realism_rules}

For every schema judge:
A. Numbers: base rates, shares and marginals against what you know of the real process; a figure off by more than 2x is a finding (state your reference figure and its source); a share or percentage copied to the decimal from a published table is a finding.
B. Record: the row is a record an organisation keeps, with its own fields; a public dataset's column set, coding or target (census and survey public use files, well-known public tables) is a finding; the target is an event or decision, not a band of a printed measurement.
C. Levels: real names as level values are fine; a level list that reproduces a published table row for row (its units with its shares to the decimal) is a finding.
D. Duplicates: these candidate pairs from `python -m {module} dedupe --file {path}` need a verdict (same record and same or equivalent target, or not):
{pairs}
E. Style: description opens "Each row is"; desc says what a column is with no causal claim; no em or en dash; module docstring lists the file's schemas.

Output: numbered findings, most severe first, each with file:line, schema id, the problem in one sentence with your reference figure or source, and the fix in one sentence. Then one line: schemas read, schemas with no finding. Max 40 lines.
"""


def render(
    *,
    workspace: Workspace,
    path: Path,
    ids: list[str],
    pairs: list[tuple[str, str, float]],
    worktree: str,
) -> str:
    return PROMPT.format(
        worktree=worktree,
        path=path,
        ids=", ".join(ids),
        realism_rules=realism_rules(workspace),
        module=workspace.module,
        pairs="\n".join(f"- {a} vs {b} ({score:.2f})" for a, b, score in pairs) or "- none",
    )


FIX_PROMPT = """You fix named defects in one catalog file. Worktree `{worktree}`. File `{path}`. Nothing else is yours.

Findings to resolve, from the review (each names a schema id):
{items}

{realism_rules}

Rules that are not negotiable:
- Edit only your file, with the Read, Edit and Write tools. Do not touch any other file. Do not commit. Do not spawn agents.
- The only commands you run are these, in the FOREGROUND, after your edits and again after fixes:
{commands}
  The gate must print OK for every id and "every schema in the file validates"; dedupe must print "no candidate pairs"; any other command must pass.{extra}
- dedupe imports the whole catalog, and other fix agents edit other files in this worktree at the same time. If a command fails with an import error naming a schema that is not in your file, wait a minute and retry once; if it fails again, report the gate's lines, name the blocking file, and stop. A candidate pair that names one of your ids is yours to resolve even when the other id is in another file. The operator then runs `sweep --sector` once for the whole batch; it must pass before the batch is committed.
- Delete a schema: remove its whole `NAME = Table(...)` block, any constant only it used and its line in the module docstring. Never leave a dangling name.
- Recut a schema: keep the id unless the finding says otherwise; make the row a record the named organisation keeps, with that record's own fields, units and codes; drop any public dataset's variable list, weights, top codes and wordings; keep the aggregate statistics as anchors in the rationale.
- Numbers: approximations with "about" wording anchored to a named aggregate publisher, rounded to the point or half point; level probabilities sum to 1 within 1e-6.
- A duplicate candidate is resolved in the schemas: delete the one whose organisation does not keep the record, or tell two genuinely different records apart in their ids, descriptions and target class names until dedupe no longer pairs them. There is no list of cleared pairs.
- Style: desc says what a column is, never a causal claim; no em or en dash anywhere; ids lower snake case, unique, specific; the module docstring matches the file after your edits.

Report in twelve lines at most: per finding what you changed (schema id, old value, new value), the last line of every command verbatim, and anything you could not do and why.
"""


def render_fix(
    *,
    workspace: Workspace,
    path: Path,
    items: list[str],
    worktree: str,
    run: str,
    scratch: str,
) -> str:
    extra = workspace.extra_fix_rules.format(run=run, module=workspace.module, path=path, scratch=scratch)  # fmt: skip
    return FIX_PROMPT.format(
        worktree=worktree,
        path=path,
        items="\n".join(f"- {item}" for item in items),
        realism_rules=realism_rules(workspace),
        commands=workspace.check_commands(run, path, "<the schema ids you changed>"),
        extra=f"\n{extra}" if extra else "",
    )
