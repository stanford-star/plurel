"""A catalog's taxonomy: sector -> division -> group, declared by its sector plans. A schema's placement is
a path in this tree, never an authored string: it derives from the file location
``<sector>/<division>/<group>/<batch>.py`` and is validated here."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from plurel.world.plans import Plan, load_plans


@dataclass(frozen=True)
class Sector:
    """One sector: its package name in the catalog and its title."""

    slug: str
    title: str


@dataclass(frozen=True)
class Taxonomy:
    """The tree the ``plans`` declare, by sector."""

    plans: dict[str, Plan]

    @classmethod
    def load(cls, directory: Path) -> Taxonomy:
        """The taxonomy of the plans in ``directory``."""
        return cls(load_plans(directory))

    @cached_property
    def sectors(self) -> tuple[Sector, ...]:
        return tuple(Sector(plan.sector, plan.title) for plan in self.plans.values())

    @cached_property
    def divisions(self) -> dict[str, dict[str, str]]:
        """``{sector: {division: title}}``."""
        return {
            plan.sector: {slug: division.title for slug, division in plan.divisions.items()}
            for plan in self.plans.values()
        }

    @cached_property
    def groups(self) -> dict[str, dict[str, str]]:
        """``{division: {group: title}}``."""
        return {
            slug: {group_slug: group.title for group_slug, group in division.groups.items()}
            for plan in self.plans.values()
            for slug, division in plan.divisions.items()
        }

    def group_title(self, group: str) -> str:
        """The title of a registered group slug."""
        for groups in self.groups.values():
            if group in groups:
                return groups[group]
        raise KeyError(f"unregistered group {group!r}")

    def validate_path(self, sector: str, division: str, group: str) -> None:
        """Reject a path the plans do not declare, so no name is invented in the tree."""
        if sector not in self.plans:
            raise ValueError(f"unknown sector {sector!r}: the sectors are {sorted(self.plans)}")
        divisions = self.divisions[sector]
        if division not in divisions:
            raise ValueError(
                f"division {division!r} is not registered under sector {sector!r} "
                f"(registered: {sorted(divisions)})"
            )
        groups = self.groups[division]
        if group not in groups:
            raise ValueError(
                f"group {group!r} is not registered under division {division!r} "
                f"(registered: {sorted(groups)})"
            )
