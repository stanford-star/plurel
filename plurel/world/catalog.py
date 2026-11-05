"""A catalog of schemas: a package whose file layout is its taxonomy. ``<sector>/<division>/<group>/<batch>.py``
holds that group's :class:`~plurel.world.schema.WorldSchema` literals as module-level values, so a
schema's placement has one source of truth and no authored field to drift."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from plurel.world.schema import WorldSchema
from plurel.world.taxonomy import Taxonomy


def load_catalog(root: Path, package: str, taxonomy: Taxonomy) -> tuple[WorldSchema, ...]:
    """Every schema under ``root``, the directory of the importable ``package``, sorted by id so
    that draws never depend on the layout: each placed by its path (validated against ``taxonomy``)
    and validated. Raises on a module with no schema, an authored placement or a repeated id."""
    schemas: list[WorldSchema] = []
    seen: dict[str, str] = {}
    for module_path in sorted(root.glob("*/*/*/*.py")):
        if module_path.name == "__init__.py":
            continue
        sector, division, group = (p.name for p in module_path.parents[2::-1])
        batch = module_path.stem
        taxonomy.validate_path(sector, division, group)
        module = importlib.import_module(f"{package}.{sector}.{division}.{group}.{batch}")
        found = [v for v in vars(module).values() if isinstance(v, WorldSchema)]
        if not found:
            raise ValueError(
                f"catalog module {sector}/{division}/{group}/{batch}.py defines no WorldSchema"
            )
        for schema in found:
            if schema.taxonomy is not None:
                raise ValueError(
                    f"{schema.schema_id}: taxonomy is loader-derived; never author it inline"
                )
            object.__setattr__(schema, "taxonomy", (sector, division, group))
            object.__setattr__(schema, "title", taxonomy.group_title(group))
            if schema.schema_id in seen:
                raise ValueError(
                    f"duplicate schema_id {schema.schema_id!r} in {sector}/{division}/{group}/{batch} "
                    f"(also in {seen[schema.schema_id]})"
                )
            seen[schema.schema_id] = f"{sector}/{division}/{group}/{batch}"
            schema.validate()
            schemas.append(schema)
    return tuple(sorted(schemas, key=lambda s: s.schema_id))


def twins(schemas, subjects: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Per schema id, the ``(subject, matched text)`` pairs its words hit: ``subjects`` are patterns
    a catalog must not hold, matched over a schema's id, domain, description and target. Empty for a
    clean catalog."""
    out: dict[str, list[tuple[str, str]]] = {}
    for schema in schemas:
        text = " ".join([
            schema.schema_id.replace("_", " "), schema.domain or "", schema.description or "",
            schema.target.replace("_", " "),
        ])  # fmt: skip
        hits = [
            (subject, m.group(0))
            for subject, pattern in subjects.items()
            if (m := re.search(pattern, text, re.I))
        ]
        if hits:
            out[schema.schema_id] = hits
    return out
