"""Where a catalog lives and how its agents run commands: everything the authoring tools need to know about a
repository, so a catalog can sit in any of them."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from plurel import Table
from plurel.world.knowledge import load_knowledge


@dataclass(frozen=True, kw_only=True)
class Workspace:
    """A catalog's places, paths relative to the worktree root the commands run from. ``catalog_module`` is the
    importable package whose ``SCHEMAS`` is the catalog (:func:`plurel.world.load_catalog` builds it) and
    ``catalog_dir`` its directory; ``example_file`` is the one file an author reads for style. ``module`` is what
    agents pass to ``python -m`` to run these commands, ``test_command`` a check they run after the gate, when there
    is one. ``subjects`` names what no schema may be about, and ``extra_rules`` and ``extra_fix_rules`` add a
    catalog's own rules to the prompts. A workspace pickles, so worker processes can load the catalog by name."""

    catalog_module: str
    catalog_dir: Path
    plans_dir: Path
    knowledge_dir: Path
    example_file: Path
    module: str
    test_command: str | None = None
    subjects: tuple[str, ...] = ()
    extra_rules: str = ""
    extra_fix_rules: str = ""

    def schemas(self) -> dict[str, Table]:
        """The catalog's schemas, by id."""
        return catalog_schemas(self.catalog_module)

    def knowledge(self) -> dict[str, dict]:
        """The catalog's written world knowledge, by schema id."""
        return written_knowledge(str(self.knowledge_dir))

    def check_commands(self, run: str, path: Path | str, ids: str) -> str:
        """The commands an agent runs on its file, one per line of a prompt's rule list."""
        lines = [f"{run} -m {self.module} gate {path} {ids}", f"{run} -m {self.module} dedupe --file {path}"]  # fmt: skip
        if self.test_command:
            lines.append(f"{run} -m {self.test_command}")
        return "\n".join(f"  `{line}`" for line in lines)


@cache
def catalog_schemas(module: str) -> dict[str, Table]:
    return {schema.name: schema for schema in importlib.import_module(module).SCHEMAS}


@cache
def written_knowledge(directory: str) -> dict[str, dict]:
    return load_knowledge(Path(directory))
