"""World knowledge orders: the second half of the authoring protocol (``GUIDE.md``, part two). Each catalog file
of a sector becomes one writer order whose prompt carries the briefs of the file's schemas
(:func:`plurel.world.knowledge.brief`) and every rule the writer needs; each written draft gets one
read-only reviewer prompt, and one fix prompt from its findings. The prompts' text is the stage files in
``prompts/`` (``knowledge_write.md``, ``knowledge_review.md``, ``knowledge_fix.md``). :func:`gate` lints a
draft, :func:`merge` moves a draft's clean lines into the catalog's world knowledge, :func:`sweep` checks a
sector's."""

from __future__ import annotations

import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from plurel.world.authoring.workspace import Workspace, written_knowledge
from plurel.world.knowledge import brief, lint
from plurel.world.schema import WorldSchema

PROMPTS = Path(__file__).parent / "prompts"
INDEX = "knowledge_orders.json"  # the orders of one --out directory, read by review, fix and merge


@dataclass(frozen=True, kw_only=True)
class KnowledgeOrder:
    agent: str
    sector: str
    module: str  # the catalog file, relative to the worktree
    ids: tuple[str, ...]


def module_ids(workspace: Workspace, relative: Path) -> list[str]:
    """The schema ids of one catalog file, ``relative`` to the catalog's directory (the catalog has imported it;
    this reads it back)."""
    name = ".".join([workspace.catalog_module, *relative.with_suffix("").parts])
    found = vars(importlib.import_module(name)).values()
    return sorted(v.schema_id for v in found if isinstance(v, WorldSchema))


def orders(
    workspace: Workspace, sector: str, *, root: Path, only_missing: bool
) -> list[KnowledgeOrder]:
    """One order per catalog file of ``sector`` holding a schema to write; ``only_missing`` leaves out
    the schemas whose world knowledge is already in the catalog."""
    written = workspace.knowledge() if only_missing else {}
    catalog = root / workspace.catalog_dir
    workspace.schemas()  # the catalog imports its files
    out: list[KnowledgeOrder] = []
    for path in sorted((catalog / sector).glob("*/*/*.py")):
        if path.name == "__init__.py":
            continue
        ids = tuple(i for i in module_ids(workspace, path.relative_to(catalog)) if i not in written)
        if ids:
            out.append(KnowledgeOrder(agent=f"knowledge{len(out) + 1:02d}", sector=sector, module=str(path.relative_to(root)), ids=ids))  # fmt: skip
    return out


def read_orders(out: Path) -> dict[str, KnowledgeOrder]:
    return {o["agent"]: KnowledgeOrder(**{**o, "ids": tuple(o["ids"])}) for o in json.loads((out / INDEX).read_text())}  # fmt: skip


def draft_path(out: Path, order: KnowledgeOrder) -> Path:
    return out / f"{order.agent}.jsonl"


_BRIEFS: dict[str, str] = {}  # briefs computed ahead by :func:`prefetch`, by schema id


def _brief_line(workspace: Workspace, schema_id: str) -> str:
    return json.dumps(brief(workspace.schemas()[schema_id]), ensure_ascii=False)


def prefetch(workspace: Workspace, ids: list[str], *, workers: int | None = None) -> None:
    """Compute the briefs of ``ids`` over processes, for the prompts rendered next (by id: a
    schema's formulas are lambdas, which do not pickle)."""
    todo = [i for i in ids if i not in _BRIEFS]
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        lines = pool.map(partial(_brief_line, workspace), todo, chunksize=8)
        _BRIEFS.update(zip(todo, lines, strict=True))


def briefs(workspace: Workspace, ids: tuple[str, ...]) -> str:
    """The briefs of ``ids``, one JSON line each, in order."""
    return "\n".join(_BRIEFS.get(i) or _brief_line(workspace, i) for i in ids)


def _fill(stage: str, **fields: str) -> str:
    text = (PROMPTS / f"knowledge_{stage}.md").read_text()
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text


def _gate_command(workspace: Workspace, order: KnowledgeOrder, draft: Path, run: str) -> str:
    return f"{run} -m {workspace.module} knowledge-gate {draft} {' '.join(order.ids)}"


def render_write(workspace: Workspace, order: KnowledgeOrder, *, draft: Path, worktree: str, run: str) -> str:  # fmt: skip
    return _fill("write", count=str(len(order.ids)), module=order.module, sector=order.sector,
                 worktree=worktree, agent=order.agent, draft=str(draft), briefs=briefs(workspace, order.ids),
                 gate=_gate_command(workspace, order, draft, run))  # fmt: skip


def render_review(
    workspace: Workspace, order: KnowledgeOrder, *, draft: Path, worktree: str
) -> str:
    return _fill("review", count=str(len(order.ids)), module=order.module, sector=order.sector,
                 worktree=worktree, draft=str(draft), briefs=briefs(workspace, order.ids))  # fmt: skip


def render_fix(workspace: Workspace, order: KnowledgeOrder, *, draft: Path, findings: list[str], worktree: str, run: str) -> str:  # fmt: skip
    return _fill("fix", module=order.module, sector=order.sector, worktree=worktree, draft=str(draft),
                 findings="\n".join(f"- {f}" for f in findings), briefs=briefs(workspace, order.ids),
                 gate=_gate_command(workspace, order, draft, run))  # fmt: skip


def write_orders(workspace: Workspace, sector: str, *, out: Path, root: Path, only_missing: bool, worktree: str, run: str) -> list[KnowledgeOrder]:  # fmt: skip
    """One writer prompt per order, the orders' index and a readable list; returns the orders."""
    out.mkdir(parents=True, exist_ok=True)
    todo = orders(workspace, sector, root=root, only_missing=only_missing)
    prefetch(workspace, [i for order in todo for i in order.ids])
    for order in todo:
        prompt = render_write(
            workspace, order, draft=draft_path(out, order).resolve(), worktree=worktree, run=run
        )
        (out / f"{order.agent}.md").write_text(prompt)
    (out / INDEX).write_text(json.dumps([asdict(o) for o in todo], indent=1) + "\n")
    lines = [
        f"- {o.agent}: `{o.module}`, {len(o.ids)} schemas -> `{draft_path(out, o)}`" for o in todo
    ]
    (out / "knowledge_agents.md").write_text("\n".join(lines) + "\n")
    return todo


def _draft(draft: Path) -> dict[str, dict]:
    """A draft's entries by schema id (the first of a repeated id)."""
    written: dict[str, dict] = {}
    for line in draft.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            written.setdefault(entry.get("schema_id"), entry)
    return written


def _problems(workspace: Workspace, entry: dict | None, schema_id: str) -> list[str]:
    if entry is None:
        return ["not in the draft"]
    schema = workspace.schemas()[schema_id]
    return lint(entry, schema, brief(schema))


def gate(workspace: Workspace, draft: Path, ids: list[str]) -> tuple[bool, list[str]]:
    """Every named schema in ``draft`` once, and its world knowledge passing the lint; the lines to
    print and whether all passed."""
    written = _draft(draft)
    lines, ok = [], True
    for schema_id in ids:
        problems = _problems(workspace, written.get(schema_id), schema_id)
        lines.append(f"{schema_id}: {'OK' if not problems else 'FAIL ' + '; '.join(problems)}")
        ok &= not problems
    extra = sorted(set(written) - set(ids))
    if extra:
        lines.append(f"FAIL: schemas the order does not name: {extra}")
        ok = False
    lines.append(
        f"every schema in the draft passes ({len(ids)} schemas)"
        if ok
        else "FAIL: the draft does not pass"
    )
    return ok, lines


def merge(workspace: Workspace, out: Path) -> tuple[int, dict[str, list[str]]]:
    """Every order's draft lines that pass the gate, into the catalog's ``<sector>.json``; the count merged and
    what was left out, by schema id."""
    into = workspace.knowledge_dir
    merged: dict[str, dict] = {}
    left_out: dict[str, list[str]] = {}
    for order in read_orders(out).values():
        draft = draft_path(out, order)
        if not draft.exists():
            left_out.update({i: ["no draft"] for i in order.ids})
            continue
        written = _draft(draft)
        for schema_id in order.ids:
            entry = written.get(schema_id)
            if problems := _problems(workspace, entry, schema_id):
                left_out[schema_id] = problems
            else:
                merged.setdefault(order.sector, {})[schema_id] = {"description": entry["description"], "columns": entry["columns"]}  # fmt: skip
    into.mkdir(parents=True, exist_ok=True)
    for sector, entries in merged.items():
        path = into / f"{sector}.json"
        known = {**(json.loads(path.read_text()) if path.exists() else {}), **entries}
        path.write_text(
            json.dumps(dict(sorted(known.items())), indent=1, ensure_ascii=False) + "\n"
        )
    written_knowledge.cache_clear()
    return sum(map(len, merged.values())), left_out


def _sweep_one(workspace: Workspace, schema_id: str) -> tuple[str, list[str]]:
    entry = workspace.knowledge().get(schema_id)
    if entry is None:
        return schema_id, ["no world knowledge"]
    schema = workspace.schemas()[schema_id]
    return schema_id, lint(entry, schema, brief(schema))


def sweep(workspace: Workspace, sector: str, *, workers: int) -> tuple[bool, list[str]]:
    """Every schema of ``sector`` has world knowledge and it passes the lint; the lines to print."""
    ids = sorted(s.schema_id for s in workspace.schemas().values() if s.taxonomy[0] == sector)
    one = partial(_sweep_one, workspace)
    if workers:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, ids, chunksize=8))
    else:
        results = [one(i) for i in ids]
    failing = [(i, p) for i, p in results if p]
    lines = [f"{i}: FAIL {'; '.join(p)}" for i, p in failing]
    lines.append(f"world knowledge: {len(ids) - len(failing)} OK, {len(failing)} missing or failing of {len(ids)} schemas in {sector}")  # fmt: skip
    return not failing, lines
