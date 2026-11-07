"""A catalog of schemas: a package whose file layout is its taxonomy. ``<sector>/<division>/<group>/<batch>.py``
holds that group's named :class:`~plurel.Table` literals as module-level values, so a record's placement
has one source of truth and no authored field to drift."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from plurel import Table
from plurel.world.record import domain, validate
from plurel.world.taxonomy import Taxonomy


def load_catalog(root: Path, package: str, taxonomy: Taxonomy) -> tuple[Table, ...]:
    """Every schema under ``root``, the directory of the importable ``package``, sorted by id so
    that draws never depend on the layout: each placed by its path (validated against ``taxonomy``)
    and validated. Raises on a module with no record or a repeated id."""
    schemas: list[Table] = []
    seen: dict[str, str] = {}
    for module_path in sorted(root.glob("*/*/*/*.py")):
        if module_path.name == "__init__.py":
            continue
        sector, division, group = (p.name for p in module_path.parents[2::-1])
        batch = module_path.stem
        taxonomy.validate_path(sector, division, group)
        module = importlib.import_module(f"{package}.{sector}.{division}.{group}.{batch}")
        found = [v for v in vars(module).values() if isinstance(v, Table)]
        if not found:
            raise ValueError(
                f"catalog module {sector}/{division}/{group}/{batch}.py defines no Table"
            )
        for schema in found:
            schema.taxonomy, schema.title = (sector, division, group), taxonomy.group_title(group)
            if schema.name in seen:
                raise ValueError(
                    f"duplicate name {schema.name!r} in {sector}/{division}/{group}/{batch} "
                    f"(also in {seen[schema.name]})"
                )
            seen[schema.name] = f"{sector}/{division}/{group}/{batch}"
            validate(schema)
            schemas.append(schema)
    return tuple(sorted(schemas, key=lambda s: s.name))


def twins(schemas, subjects: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Per schema id, the ``(subject, matched text)`` pairs its words hit: ``subjects`` are patterns
    a catalog must not hold, matched over a schema's id, domain, description and target. Empty for a
    clean catalog."""
    out: dict[str, list[tuple[str, str]]] = {}
    for schema in schemas:
        text = " ".join([
            schema.name.replace("_", " "), domain(schema), schema.doc or "",
            schema.target.replace("_", " "),
        ])  # fmt: skip
        hits = [
            (subject, m.group(0))
            for subject, pattern in subjects.items()
            if (m := re.search(pattern, text, re.I))
        ]
        if hits:
            out[schema.name] = hits
    return out
