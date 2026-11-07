"""Catalog authoring commands, one per step of the protocol (``GUIDE.md``): plan, orders, gate, dedupe, review, fix,
sweep and census for the schemas (part one), knowledge-orders, knowledge-gate, knowledge-review, knowledge-fix,
knowledge-merge and knowledge-sweep for their world knowledge (part two).

    python -m plurel.world.authoring --workspace <module>:<name> <step> ...

A repository can instead call :func:`main` with its :class:`~plurel.world.authoring.workspace.Workspace`."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
from pathlib import Path

from plurel import Table
from plurel.world.authoring import dedupe, knowledge_orders, orders, review
from plurel.world.authoring.realism import (
    census,
    check,
    hygiene,
    phenomena,
    summary,
    target_kind,
    validation,
)
from plurel.world.authoring.workspace import Workspace
from plurel.world.plans import describe, load_plan
from plurel.world.record import level_names, validate

DASH = re.compile("[\u2013\u2014]")


def load_file(path: Path) -> dict[str, Table]:
    """Every schema of one catalog file, validated, with its taxonomy derived from the path."""
    if DASH.search(path.read_text()):
        sys.exit("FAIL: em or en dash in the file")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schemas = {}
    for schema in (v for v in vars(module).values() if isinstance(v, Table)):
        schema.taxonomy = (path.parents[2].name, path.parents[1].name, path.parent.name)
        validate(schema)
        schemas[schema.name] = schema
    return schemas


def gate_line(schema: Table, *, realize: bool) -> tuple[bool, str]:
    t = schema.declared[schema.target]
    reasons = check(schema, realize=realize)
    k = len(level_names(t))
    verdict = "OK" if not reasons else "FAIL " + "; ".join(reasons)
    return (
        not reasons,
        f"{schema.name}: {verdict}  [{target_kind(t)}, {k} classes, {len(schema.declared) - 1} columns, phenomena={sorted(phenomena(schema))}]",
    )


def gate(*, path: Path, ids: list[str]) -> int:
    schemas = load_file(path)
    missing = [i for i in ids if i not in schemas]
    if missing:
        sys.exit(f"FAIL: not in the file: {missing}")
    ok = True
    for schema_id in ids:
        passed, line = gate_line(schemas[schema_id], realize=True)
        print(line)
        ok &= passed
    print(f"every schema in the file validates ({len(schemas)} schemas)")
    return 0 if ok else 1


def sweep(workspace: Workspace, *, sector: str) -> int:
    """Validate every schema of the sector with realization, list its duplicate candidates, count its hygiene
    findings and print its census; exit 1 on an invalid schema or a candidate pair."""
    catalog = list(workspace.schemas().values())
    schemas = sorted((s for s in catalog if s.taxonomy[0] == sector), key=lambda s: s.taxonomy)
    if not schemas:
        sys.exit(f"FAIL: no schemas under sector {sector!r}")
    invalid = 0
    for schema in schemas:
        reasons = validation(schema, realize=True)
        if reasons:
            invalid += 1
            print(f"{schema.name}: INVALID " + "; ".join(reasons))
    print(
        f"validation: {len(schemas) - invalid} OK, {invalid} INVALID of {len(schemas)} schemas in {sector}"
    )
    print(
        f"hygiene findings (authoring rules, not blocking): {sum(bool(hygiene(s)) for s in schemas)} schemas"
    )
    pairs = dedupe.candidates(catalog, threshold=dedupe.THRESHOLD, among={s.name for s in schemas})
    print(f"duplicate candidates: {len(pairs)}")
    print(dedupe.report(pairs, catalog))
    print(summary(census(schemas)))
    return 0 if invalid == 0 and not pairs else 1


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(prog="authoring", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = out.add_subparsers(dest="command", required=True)
    run_help = (
        "the command prefix an agent runs Python with; default: this interpreter in the worktree"
    )
    p = commands.add_parser("plan", help="print a sector plan's tree and which batch modules are missing on disk")  # fmt: skip
    p.add_argument("--sector", required=True)
    p = commands.add_parser("orders", help="one self-contained prompt per batch module of a sector plan")  # fmt: skip
    p.add_argument("--sector", required=True)
    p.add_argument("--out", type=Path, required=True, help="directory for the prompts")
    p.add_argument("--per-file", type=int, required=True, help="schemas per batch module")
    p.add_argument("--worktree", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--run", help=run_help)
    p.add_argument("--only-missing", action="store_true", help="skip batch modules already on disk")
    p = commands.add_parser("gate", help="check the named schemas of one catalog file, OK or FAIL each; exit 1 on any FAIL")  # fmt: skip
    p.add_argument("file", type=Path)
    p.add_argument("ids", nargs="+", metavar="schema_id")
    p = commands.add_parser("dedupe", help="candidate duplicate pairs across the catalog, or those involving one file's schemas")  # fmt: skip
    p.add_argument("--file", type=Path, help="restrict to pairs involving this file's schemas")
    p = commands.add_parser("review", help="one read-only reviewer prompt per catalog file of a sector")  # fmt: skip
    p.add_argument("--sector", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worktree", required=True)
    p = commands.add_parser("fix", help="one fix prompt for a catalog file from a findings file, one finding per line")  # fmt: skip
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--findings", type=Path, required=True, help="text file, one finding per line, each naming a schema id")  # fmt: skip
    p.add_argument("--out", type=Path, required=True, help="the prompt file to write")
    p.add_argument("--worktree", required=True)
    p.add_argument("--run", help=run_help)
    p = commands.add_parser("sweep", help="validation with realization, dedupe and census over one sector; exit 1 on an invalid schema or a candidate pair")  # fmt: skip
    p.add_argument("--sector", required=True)
    p = commands.add_parser("census", help="tabulate every catalog schema and print the summary")
    p.add_argument("--out", type=Path, required=True, help="census csv")
    p = commands.add_parser("knowledge-orders", help="one world knowledge writer prompt per catalog file of a sector")  # fmt: skip
    p.add_argument("--sector", required=True)
    p.add_argument("--out", type=Path, required=True, help="directory for the prompts and drafts")
    p.add_argument("--worktree", required=True)
    p.add_argument("--run", help=run_help)
    p.add_argument("--only-missing", action="store_true", help="skip schemas with world knowledge")
    p = commands.add_parser("knowledge-gate", help="lint the named schemas' world knowledge in one draft, OK or FAIL each; exit 1 on any FAIL")  # fmt: skip
    p.add_argument("draft", type=Path)
    p.add_argument("ids", nargs="+", metavar="schema_id")
    p = commands.add_parser("knowledge-review", help="one read-only reviewer prompt per written draft of an --out")  # fmt: skip
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worktree", required=True)
    p.add_argument("--all", action="store_true", help="every order's prompt, written or not, for a pipeline that reviews each draft as it lands")  # fmt: skip
    p = commands.add_parser("knowledge-fix", help="one fix prompt for a draft from a findings file, one finding per line")  # fmt: skip
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--agent", required=True, help="the order whose draft to fix, e.g. knowledge03")
    p.add_argument("--findings", type=Path, required=True)
    p.add_argument("--worktree", required=True)
    p.add_argument("--run", help=run_help)
    p = commands.add_parser("knowledge-merge", help="the drafts' passing lines into the catalog's world knowledge; exit 1 if any is left out")  # fmt: skip
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("knowledge-sweep", help="every schema of a sector has world knowledge that passes the lint; exit 1 otherwise")  # fmt: skip
    p.add_argument("--sector", required=True)
    p.add_argument("--workers", type=int, required=True, help="processes; 0 = inline")
    return out


def _run(args: argparse.Namespace) -> str:
    return args.run or f"cd {args.worktree} && {sys.executable}"


def main(workspace: Workspace, argv: list[str]) -> int:
    """Run one authoring step on ``workspace``'s catalog from the worktree root; its exit code."""
    args = parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "plan":
        plan = load_plan(workspace.plans_dir / f"{args.sector}.yml")
        print(describe(plan))
        missing = [a.path for a in orders.assignments(plan, workspace, only_missing=True, root=root)]  # fmt: skip
        print(f"batch modules missing on disk: {len(missing)}")
        print("\n".join(str(m) for m in missing))
        return 0
    if args.command == "orders":
        plan = load_plan(workspace.plans_dir / f"{args.sector}.yml")
        todo = orders.write_orders(
            workspace=workspace,
            plan=plan,
            out=args.out,
            root=root,
            only_missing=args.only_missing,
            per_file=args.per_file,
            worktree=args.worktree,
            branch=args.branch,
            run=_run(args),
        )
        print(f"{len(todo)} agents, {len(todo) * args.per_file} schemas -> {args.out}")
        return 0
    if args.command == "gate":
        return gate(path=args.file.resolve(), ids=args.ids)
    if args.command == "dedupe":
        among = set(load_file(args.file.resolve())) if args.file else None
        catalog = list(workspace.schemas().values())
        pairs = dedupe.candidates(catalog, threshold=dedupe.THRESHOLD, among=among)
        print(dedupe.report(pairs, catalog))
        return 0 if not pairs else 1
    if args.command == "review":
        args.out.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p
            for p in (root / workspace.catalog_dir / args.sector).glob("*/*/*.py")
            if p.name != "__init__.py"
        )
        catalog = list(workspace.schemas().values())
        for n, module in enumerate(files, start=1):
            path, ids = module.relative_to(root), list(load_file(module))
            pairs = dedupe.candidates(catalog, threshold=dedupe.THRESHOLD, among=set(ids))
            (args.out / f"review{n:02d}.md").write_text(
                review.render(workspace=workspace, path=path, ids=ids, pairs=pairs, worktree=args.worktree)
            )  # fmt: skip
        print(f"{len(files)} reviewer prompts -> {args.out}")
        return 0
    if args.command == "fix":
        items = [line.strip() for line in args.findings.read_text().splitlines() if line.strip()]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            review.render_fix(
                workspace=workspace,
                path=args.file,
                items=items,
                worktree=args.worktree,
                run=_run(args),
                scratch=str(args.out.parent.resolve()),
            )
        )
        print(f"{len(items)} findings -> {args.out}")
        return 0
    if args.command == "sweep":
        return sweep(workspace, sector=args.sector)
    if args.command.startswith("knowledge-"):
        return knowledge(workspace, args, root)
    table = census(workspace.schemas().values())
    table.to_csv(args.out, index=False)
    print(summary(table))
    return 0


def knowledge(workspace: Workspace, args: argparse.Namespace, root: Path) -> int:
    """The world knowledge steps (``GUIDE.md``, part two)."""
    ko = knowledge_orders
    if args.command == "knowledge-orders":
        todo = ko.write_orders(workspace, args.sector, out=args.out, root=root, only_missing=args.only_missing, worktree=args.worktree, run=_run(args))  # fmt: skip
        print(f"{len(todo)} agents, {sum(len(o.ids) for o in todo)} schemas -> {args.out}")
        return 0
    if args.command == "knowledge-gate":
        ok, lines = ko.gate(workspace, args.draft.resolve(), args.ids)
        print("\n".join(lines))
        return 0 if ok else 1
    if args.command == "knowledge-review":
        written = [o for o in ko.read_orders(args.out).values() if args.all or ko.draft_path(args.out, o).exists()]  # fmt: skip
        ko.prefetch(workspace, [i for order in written for i in order.ids])
        for order in written:
            prompt = ko.render_review(workspace, order, draft=ko.draft_path(args.out, order).resolve(), worktree=args.worktree)  # fmt: skip
            (args.out / f"{order.agent}_review.md").write_text(prompt)
        print(f"{len(written)} reviewer prompts -> {args.out}")
        return 0
    if args.command == "knowledge-fix":
        order = ko.read_orders(args.out)[args.agent]
        findings = [line.strip() for line in args.findings.read_text().splitlines() if line.strip()]
        prompt = ko.render_fix(workspace, order, draft=ko.draft_path(args.out, order).resolve(), findings=findings, worktree=args.worktree, run=_run(args))  # fmt: skip
        (args.out / f"{order.agent}_fix.md").write_text(prompt)
        print(f"{len(findings)} findings -> {args.out / f'{order.agent}_fix.md'}")
        return 0
    if args.command == "knowledge-merge":
        merged, left_out = ko.merge(workspace, args.out)
        for schema_id, problems in sorted(left_out.items()):
            print(f"{schema_id}: LEFT OUT {'; '.join(problems)}")
        print(f"merged {merged}, left out {len(left_out)}")
        return 0 if not left_out else 1
    ok, lines = ko.sweep(workspace, args.sector, workers=args.workers)
    print("\n".join(lines))
    return 0 if ok else 1


def load_workspace(spec: str) -> Workspace:
    """The :class:`Workspace` named ``<module>:<name>``, with the worktree root importable."""
    module, _, name = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), name)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--workspace":
        sys.exit("usage: python -m plurel.world.authoring --workspace <module>:<name> <step> ...")
    sys.exit(main(load_workspace(sys.argv[2]), sys.argv[3:]))
