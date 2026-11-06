import hashlib
import runpy
import sys
from pathlib import Path

import pytest

from plurel.world import brief, lint, load_knowledge, realize_world, roles
from plurel.world.authoring.realism import check

EXAMPLE = Path(__file__).parents[1] / "examples" / "world"


@pytest.fixture
def schema(monkeypatch):
    monkeypatch.syspath_prepend(EXAMPLE)
    for name in [m for m in sys.modules if m == "catalog" or m.startswith("catalog.")]:
        del sys.modules[name]
    from catalog import SCHEMAS

    yield next(s for s in SCHEMAS if s.schema_id == "repair_cafe_item_visit_outcome")
    for name in [m for m in sys.modules if m == "catalog" or m.startswith("catalog.")]:
        del sys.modules[name]


def test_the_example_is_placed_by_its_path(schema):
    assert schema.taxonomy == ("community_services", "repair_and_reuse", "repair_cafes")
    assert schema.domain == "repair cafes"


def test_the_example_passes_its_own_gate(schema):
    assert check(schema, realize=True) == []


def test_the_examples_knowledge_passes_its_lint(schema):
    written = load_knowledge(EXAMPLE / "world_knowledge")[schema.schema_id]
    assert lint(written, schema, brief(schema)) == []


def test_the_examples_shares_and_graph(schema):
    rows = realize_world(schema, 0).rows(4000, 0)
    shares = rows["outcome"].value_counts(normalize=True)
    assert abs(shares["repaired"] - 0.61) < 0.05 and abs(shares["repair_failed"] - 0.25) < 0.05
    assert "volunteer_cover" not in rows.columns
    assert roles(schema)["wait_minutes"] == "shares a cause with the outcome"
    assert roles(schema)["total_minutes"] == "computed from wait_minutes, repair_minutes"


def test_the_examples_draws_are_pinned(schema):
    frame = realize_world(schema, 0).rows(200, 0)
    digest = hashlib.sha256(frame.round(4).to_csv(index=False).encode()).hexdigest()[:16]
    assert digest == "023f5120191af4ca"


def test_the_example_script_runs(capsys, schema):
    runpy.run_path(str(EXAMPLE / "generate.py"), run_name="__main__")
    printed = capsys.readouterr().out
    assert "gate OK" in printed and "world knowledge lint: OK" in printed
