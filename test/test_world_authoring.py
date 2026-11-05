import json
import pickle
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from plurel.world.authoring import knowledge_orders as ko
from plurel.world.authoring.__main__ import load_workspace, main
from plurel.world.authoring.workspace import catalog_schemas, written_knowledge

EXAMPLE = Path(__file__).parents[1] / "examples" / "world"
SECTOR = "community_services"
SCHEMA = "repair_cafe_item_visit_outcome"
MODULE = Path("catalog/community_services/repair_and_reuse/repair_cafes/item_visits.py")
UNFILLED = re.compile(r"\{[a-z_]+\}")


def forget_catalog():
    for name in [
        m for m in sys.modules if m in ("catalog", "workspace") or m.startswith("catalog.")
    ]:
        del sys.modules[name]
    catalog_schemas.cache_clear()
    written_knowledge.cache_clear()
    ko._BRIEFS.clear()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """The example's catalog copied to a worktree of its own, importable from there."""
    worktree = tmp_path / "worktree"
    shutil.copytree(EXAMPLE, worktree, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.chdir(worktree)
    monkeypatch.syspath_prepend(worktree)
    forget_catalog()
    yield load_workspace("workspace:WORKSPACE")
    forget_catalog()


def add_batch(name: str = "new_records"):
    plan = Path("plans") / f"{SECTOR}.yml"
    plan.write_text(plan.read_text() + f'          {name}:\n          - "a record: its target"\n')


def test_a_workspace_pickles_for_worker_processes(workspace):
    assert pickle.loads(pickle.dumps(workspace)) == workspace
    assert list(workspace.schemas()) == [SCHEMA]
    assert list(workspace.knowledge()) == [SCHEMA]


def test_plan_prints_the_tree_and_what_is_missing(workspace, capsys):
    assert main(workspace, ["plan", "--sector", SECTOR]) == 0
    assert "batch modules missing on disk: 0" in capsys.readouterr().out
    add_batch()
    assert main(workspace, ["plan", "--sector", SECTOR]) == 0
    printed = capsys.readouterr().out
    assert "batch modules missing on disk: 1" in printed and "new_records.py" in printed


def test_orders_render_fully_for_the_missing_batches(workspace, tmp_path):
    add_batch()
    out = tmp_path / "orders"
    arguments = ["orders", "--sector", SECTOR, "--out", str(out), "--per-file", "6", "--worktree", "/w", "--branch", "b", "--only-missing", "--run", "my-launcher python"]  # fmt: skip
    assert main(workspace, arguments) == 0
    prompt = (out / "agent01.md").read_text()
    assert UNFILLED.search(prompt) is None and "a record: its target" in prompt
    assert "my-launcher python -m plurel.world.authoring --workspace workspace:WORKSPACE gate catalog/community_services/repair_and_reuse/repair_cafes/new_records.py" in prompt  # fmt: skip
    assert str(workspace.example_file) in prompt and "Subjects no schema" not in prompt
    assert "new_records.py" in (out / "agents.md").read_text()
    assert not (out / "agent02.md").exists()


def test_a_workspace_adds_its_own_rules_and_checks(workspace, tmp_path):
    add_batch()
    own = replace(workspace, subjects=("bicycle repair",), test_command="pytest -q tests/test_catalog.py", extra_rules="- Own rule: keep it short.")  # fmt: skip
    out = tmp_path / "orders"
    arguments = ["orders", "--sector", SECTOR, "--out", str(out), "--per-file", "6", "--worktree", "/w", "--branch", "b", "--only-missing", "--run", "python"]  # fmt: skip
    assert main(own, arguments) == 0
    prompt = (out / "agent01.md").read_text()
    assert "bicycle repair" in prompt and "Own rule: keep it short." in prompt
    assert "python -m pytest -q tests/test_catalog.py" in prompt


def test_the_gate_passes_the_example_and_names_a_failure(workspace, capsys, tmp_path):
    assert main(workspace, ["gate", str(MODULE), SCHEMA]) == 0
    printed = capsys.readouterr().out
    assert (
        f"{SCHEMA}: OK" in printed and "every schema in the file validates (1 schemas)" in printed
    )
    with pytest.raises(SystemExit, match="not in the file"):
        main(workspace, ["gate", str(MODULE), "nobody"])
    text = MODULE.read_text()
    MODULE.write_text(text.replace("about 25 percent failed", "and the rest failed"))
    assert main(workspace, ["gate", str(MODULE), SCHEMA]) == 0  # two facts are still there
    MODULE.write_text(text.replace("Each row is one household item", "One household item"))
    assert main(workspace, ["gate", str(MODULE), SCHEMA]) == 1
    MODULE.write_text(text.replace("how the visit ended", "how the visit — ended"))
    with pytest.raises(SystemExit, match="dash"):
        main(workspace, ["gate", str(MODULE), SCHEMA])


def test_dedupe_reads_the_catalog(workspace, capsys):
    assert main(workspace, ["dedupe"]) == 0
    assert "no candidate pairs" in capsys.readouterr().out
    assert main(workspace, ["dedupe", "--file", str(MODULE)]) == 0


def test_review_and_fix_prompts_render_fully(workspace, tmp_path):
    out = tmp_path / "review"
    assert (
        main(workspace, ["review", "--sector", SECTOR, "--out", str(out), "--worktree", "/w"]) == 0
    )
    prompt = (out / "review01.md").read_text()
    assert UNFILLED.search(prompt) is None and SCHEMA in prompt and "- none" in prompt
    findings = tmp_path / "findings.txt"
    findings.write_text(f"{SCHEMA}: share of 0.283 copied to the decimal\n\n")
    fix = tmp_path / "fix" / "prompt.md"
    arguments = ["fix", "--file", str(MODULE), "--findings", str(findings), "--out", str(fix), "--worktree", "/w", "--run", "python"]  # fmt: skip
    assert main(workspace, arguments) == 0
    text = fix.read_text()
    assert UNFILLED.search(text) is None and "0.283 copied" in text and f"gate {MODULE}" in text


def test_a_fix_prompt_takes_the_workspaces_own_rules(workspace, tmp_path):
    findings = tmp_path / "findings.txt"
    findings.write_text(f"{SCHEMA}: too easy\n")
    fix = tmp_path / "prompt.md"
    own = replace(workspace, extra_fix_rules="- Also run `{run} -m own.score {path}`.")
    arguments = ["fix", "--file", str(MODULE), "--findings", str(findings), "--out", str(fix), "--worktree", "/w", "--run", "python"]  # fmt: skip
    assert main(own, arguments) == 0
    assert f"Also run `python -m own.score {MODULE}`." in fix.read_text()


def test_sweep_and_census_cover_the_sector(workspace, capsys, tmp_path):
    assert main(workspace, ["sweep", "--sector", SECTOR]) == 0
    printed = capsys.readouterr().out
    assert "validation: 1 OK, 0 INVALID of 1 schemas" in printed and "no candidate pairs" in printed
    with pytest.raises(SystemExit, match="no schemas under sector"):
        main(workspace, ["sweep", "--sector", "nothing"])
    assert main(workspace, ["census", "--out", str(tmp_path / "census.csv")]) == 0
    assert SCHEMA in (tmp_path / "census.csv").read_text()


def test_knowledge_orders_review_and_fix_render_fully(workspace, tmp_path):
    out = tmp_path / "knowledge"
    arguments = ["knowledge-orders", "--sector", SECTOR, "--out", str(out), "--worktree", "/w", "--run", "python"]  # fmt: skip
    assert main(workspace, arguments) == 0
    order = ko.read_orders(out)["knowledge01"]
    assert order.ids == (SCHEMA,) and order.module == str(MODULE)
    write = (out / "knowledge01.md").read_text()
    assert UNFILLED.search(write) is None
    assert f'"schema_id": "{SCHEMA}"' in write
    assert (
        "python -m plurel.world.authoring --workspace workspace:WORKSPACE knowledge-gate" in write
    )
    assert main(workspace, ["knowledge-review", "--out", str(out), "--worktree", "/w"]) == 0
    assert not (out / "knowledge01_review.md").exists()  # no draft written yet
    assert (
        main(workspace, ["knowledge-review", "--out", str(out), "--worktree", "/w", "--all"]) == 0
    )
    assert f'"schema_id": "{SCHEMA}"' in (out / "knowledge01_review.md").read_text()
    findings = tmp_path / "findings.txt"
    findings.write_text(f"{SCHEMA}: B: x: y\n")
    arguments = ["knowledge-fix", "--out", str(out), "--agent", "knowledge01", "--findings", str(findings), "--worktree", "/w", "--run", "python"]  # fmt: skip
    assert main(workspace, arguments) == 0
    assert f"{SCHEMA}: B: x: y" in (out / "knowledge01_fix.md").read_text()


def test_only_missing_leaves_out_written_knowledge(workspace, tmp_path):
    assert ko.orders(workspace, SECTOR, root=Path.cwd(), only_missing=True) == []
    assert len(ko.orders(workspace, SECTOR, root=Path.cwd(), only_missing=False)) == 1


def test_the_knowledge_gate_and_merge_hold_drafts(workspace, tmp_path, capsys):
    written = workspace.knowledge()[SCHEMA]
    out = tmp_path / "knowledge"
    out.mkdir()
    draft = out / "knowledge01.jsonl"
    draft.write_text(json.dumps({"schema_id": SCHEMA, **written}) + "\n")
    assert main(workspace, ["knowledge-gate", str(draft), SCHEMA]) == 0
    assert main(workspace, ["knowledge-gate", str(draft), SCHEMA, "nobody"]) == 1
    assert not ko.gate(workspace, draft, [])[0]  # the draft holds a schema the order does not name
    order = {"agent": "knowledge01", "sector": SECTOR, "module": str(MODULE), "ids": [SCHEMA]}
    (out / ko.INDEX).write_text(json.dumps([order]))
    target = Path("world_knowledge") / f"{SECTOR}.json"
    target.unlink()
    assert (
        main(workspace, ["knowledge-sweep", "--sector", SECTOR, "--workers", "0"]) == 0
    )  # still cached
    written_knowledge.cache_clear()
    assert main(workspace, ["knowledge-sweep", "--sector", SECTOR, "--workers", "0"]) == 1
    assert main(workspace, ["knowledge-merge", "--out", str(out)]) == 0
    assert json.loads(target.read_text()) == {SCHEMA: written}
    assert main(workspace, ["knowledge-sweep", "--sector", SECTOR, "--workers", "0"]) == 0
    capsys.readouterr()
    (out / ko.INDEX).write_text(json.dumps([{**order, "ids": [SCHEMA, "nobody"]}]))
    draft.write_text("")
    assert main(workspace, ["knowledge-merge", "--out", str(out)]) == 1
    assert "LEFT OUT not in the draft" in capsys.readouterr().out


def test_the_guide_and_prompts_ship_inside_the_package():
    package = Path(ko.__file__).parent
    assert (package / "GUIDE.md").exists()
    assert {p.name for p in (package / "prompts").glob("*.md")} == {"knowledge_write.md", "knowledge_review.md", "knowledge_fix.md"}  # fmt: skip
