import sys
import textwrap
from pathlib import Path

import pytest

from plurel.world import Taxonomy, load_catalog, twins
from plurel.world.plans import describe, load_plan, load_plans

PLAN = """\
sector: community_services
title: Community services
scope: Records that volunteer groups keep.
row_entities: [a repair visit]
anchors: [an annual report]
exclusions: [paid repair shops]
divisions:
  repair_and_reuse:
    title: Repair and reuse
    groups:
      repair_cafes:
        title: Repair cafes
        batches:
          item_visits:
          - item brought to a repair cafe, how the visit ended
"""
MODULE = """\
from plurel.world.schema import Edge, Node, WorldSchema

{name} = WorldSchema(
    schema_id="{schema_id}",
    target="outcome",
    description="Each row is one visit.",
    nodes={{
        "age": Node(kind="numeric", units="duration", dist="right", scale=(0, 40), desc="its age"),
        "outcome": Node(
            kind="categorical", levels=["fixed", "not_fixed"], level_probs={{"fixed": 0.6, "not_fixed": 0.4}},
            parents={{"age": Edge(effect=+1)}}, noise_scale="medium", noise_dist="gaussian", desc="the end",
        ),
    }},{extra}
)
"""


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


@pytest.fixture
def catalog(tmp_path, monkeypatch, request):
    """An empty catalog's plans and its package directory, importable under a name of its own."""
    package = f"catalog_{abs(hash(request.node.name)) % 10**8}"
    monkeypatch.syspath_prepend(tmp_path)
    write(tmp_path / "plans" / "community_services.yml", PLAN)
    root = tmp_path / package
    root.mkdir()
    yield root, package, Taxonomy.load(tmp_path / "plans")
    for name in [m for m in sys.modules if m == package or m.startswith(f"{package}.")]:
        del sys.modules[name]


def batch(root: Path, name: str = "item_visits") -> Path:
    return root / "community_services" / "repair_and_reuse" / "repair_cafes" / f"{name}.py"


def test_a_plan_declares_its_tree(tmp_path):
    plan = load_plan(write(tmp_path / "community_services.yml", PLAN))
    assert plan.title == "Community services" and plan.row_entities == ("a repair visit",)
    assert plan.groups() == [("repair_and_reuse", "repair_cafes")]
    assert plan.batches() == [("repair_and_reuse", "repair_cafes", "item_visits")]
    assert "repair_cafes: Repair cafes" in describe(plan) and "- item brought" in describe(plan)


def test_a_plan_file_must_name_its_sector(tmp_path):
    with pytest.raises(ValueError, match="file name and sector"):
        load_plan(write(tmp_path / "other.yml", PLAN))
    with pytest.raises(ValueError, match="lower snake case"):
        load_plan(write(tmp_path / "Bad.yml", PLAN.replace("community_services", "Bad")))


def test_a_division_belongs_to_one_sector(tmp_path):
    write(tmp_path / "community_services.yml", PLAN)
    write(tmp_path / "other.yml", PLAN.replace("sector: community_services", "sector: other"))
    with pytest.raises(ValueError, match="registered by both"):
        load_plans(tmp_path)


def test_the_taxonomy_registers_what_the_plans_declare(tmp_path):
    write(tmp_path / "community_services.yml", PLAN)
    taxonomy = Taxonomy.load(tmp_path)
    assert [s.slug for s in taxonomy.sectors] == ["community_services"]
    assert taxonomy.divisions == {"community_services": {"repair_and_reuse": "Repair and reuse"}}
    assert taxonomy.groups == {"repair_and_reuse": {"repair_cafes": "Repair cafes"}}
    assert taxonomy.group_title("repair_cafes") == "Repair cafes"
    taxonomy.validate_path("community_services", "repair_and_reuse", "repair_cafes")
    for path, message in (
        (("nope", "repair_and_reuse", "repair_cafes"), "unknown sector"),
        (("community_services", "nope", "repair_cafes"), "division 'nope'"),
        (("community_services", "repair_and_reuse", "nope"), "group 'nope'"),
    ):
        with pytest.raises(ValueError, match=message):
            taxonomy.validate_path(*path)
    with pytest.raises(KeyError):
        taxonomy.group_title("nope")


def test_a_catalog_places_its_schemas_by_path_and_sorts_them(catalog):
    root, package, taxonomy = catalog
    write(batch(root), MODULE.format(name="B", schema_id="visit_b", extra=""))
    write(batch(root, "more_visits"), MODULE.format(name="A", schema_id="visit_a", extra=""))
    write(batch(root, "__init__"), "")
    schemas = load_catalog(root, package, taxonomy)
    assert [s.schema_id for s in schemas] == ["visit_a", "visit_b"]
    assert schemas[0].taxonomy == ("community_services", "repair_and_reuse", "repair_cafes")
    assert schemas[0].domain == "repair cafes"


@pytest.mark.parametrize(
    ("extra", "module_name", "message"),
    [
        ('\n    taxonomy=("a", "b", "c"),', "item_visits", "never author it inline"),
        ("", "__duplicate__", "duplicate schema_id"),
        ("", "__empty__", "defines no WorldSchema"),
    ],
)
def test_a_catalog_refuses_what_it_cannot_place(catalog, extra, module_name, message):
    root, package, taxonomy = catalog
    write(batch(root), MODULE.format(name="A", schema_id="visit", extra=extra))
    if module_name == "__duplicate__":
        write(batch(root, "again"), MODULE.format(name="B", schema_id="visit", extra=""))
    if module_name == "__empty__":
        write(batch(root, "again"), "VALUE = 1\n")
    with pytest.raises(ValueError, match=message):
        load_catalog(root, package, taxonomy)


def test_a_catalog_refuses_a_path_its_plans_do_not_declare(catalog):
    root, package, taxonomy = catalog
    path = root / "community_services" / "repair_and_reuse" / "invented" / "item_visits.py"
    write(path, MODULE.format(name="A", schema_id="visit", extra=""))
    with pytest.raises(ValueError, match="group 'invented'"):
        load_catalog(root, package, taxonomy)


def test_twins_name_the_subjects_a_schema_hits(catalog):
    root, package, taxonomy = catalog
    write(batch(root), MODULE.format(name="A", schema_id="repair_visit", extra=""))
    schemas = load_catalog(root, package, taxonomy)
    assert twins(schemas, {"bicycles": r"bicycle"}) == {}
    assert twins(schemas, {"repairs": r"repair (visit|cafes)"}) == {"repair_visit": [("repairs", "repair visit")]}  # fmt: skip
