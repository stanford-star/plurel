"""Sector plans: one yml per sector in a catalog's plans directory is the single source for the sector's
registration (its scope, divisions, groups and titles) and for the authoring orders (the batches of records
each group still needs, the row entities, anchors and exclusions). The taxonomy registers what the plans
declare; nothing is registered by hand. A group is a directory of batch modules; a batch in a plan is one
order, one module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SLUG = r"^[a-z][a-z0-9_]*$"


@dataclass(frozen=True, kw_only=True)
class Group:
    title: str
    # batch module slug -> "record: target" lines still to author
    batches: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Division:
    title: str
    groups: dict[str, Group]


@dataclass(frozen=True, kw_only=True)
class Plan:
    sector: str
    title: str
    scope: str  # the records this sector holds and the neighbouring sectors that hold the rest
    divisions: dict[str, Division]
    row_entities: tuple[str, ...] = ()
    phenomena: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()

    def groups(self) -> list[tuple[str, str]]:
        """(division, group) pairs in plan order."""
        return [(d, g) for d, division in self.divisions.items() for g in division.groups]

    def batches(self) -> list[tuple[str, str, str]]:
        """(division, group, batch) triples in plan order."""
        return [
            (d, g, b)
            for d, division in self.divisions.items()
            for g, group in division.groups.items()
            for b in group.batches
        ]


def _slug(name: str, where: str) -> str:
    if not re.match(SLUG, name):
        raise ValueError(f"{where}: {name!r} is not a lower snake case slug")
    return name


def _strings(raw: object, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError(f"{where}: expected a list of strings")
    return tuple(raw)


def load_plan(path: Path) -> Plan:
    raw = yaml.safe_load(path.read_text())
    sector = _slug(raw["sector"], path.name)
    if path.stem != sector:
        raise ValueError(f"{path.name}: file name and sector {sector!r} differ")
    divisions: dict[str, Division] = {}
    for d_slug, d_raw in raw["divisions"].items():
        _slug(d_slug, f"{sector} division")
        groups = {}
        for g_slug, g_raw in d_raw["groups"].items():
            _slug(g_slug, f"{sector}/{d_slug} group")
            batches = {
                _slug(b, f"{d_slug}/{g_slug} batch"): _strings(records, f"{g_slug}/{b}")
                for b, records in (g_raw.get("batches") or {}).items()
            }
            groups[g_slug] = Group(title=g_raw["title"], batches=batches)
        divisions[d_slug] = Division(title=d_raw["title"], groups=groups)
    return Plan(
        sector=sector,
        title=raw["title"],
        scope=raw["scope"],
        divisions=divisions,
        row_entities=_strings(raw.get("row_entities"), sector),
        phenomena=_strings(raw.get("phenomena"), sector),
        anchors=_strings(raw.get("anchors"), sector),
        exclusions=_strings(raw.get("exclusions"), sector),
    )


def load_plans(directory: Path) -> dict[str, Plan]:
    """Every plan in the directory, by sector; a division slug used by two sectors is an error."""
    plans = {p.sector: p for p in (load_plan(path) for path in sorted(directory.glob("*.yml")))}
    seen: dict[str, str] = {}
    for plan in plans.values():
        for division in plan.divisions:
            if division in seen:
                raise ValueError(
                    f"division {division!r} is registered by both {seen[division]} and {plan.sector}"
                )
            seen[division] = plan.sector
    return plans


def describe(plan: Plan) -> str:
    """The plan's tree as text: one line per group, then its batches with their records."""
    lines = [f"{plan.sector}: {plan.title}", f"  scope: {plan.scope}"]
    for d_slug, division in plan.divisions.items():
        lines.append(f"  {d_slug}: {division.title}")
        for g_slug, group in division.groups.items():
            lines.append(f"    {g_slug}: {group.title}")
            for b_slug, records in group.batches.items():
                lines.append(f"      {b_slug}:")
                lines.extend(f"        - {record}" for record in records)
    return "\n".join(lines)
