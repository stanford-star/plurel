"""Where the example's catalog lives, for the authoring commands. From this directory:

python -m plurel.world.authoring --workspace workspace:WORKSPACE plan --sector community_services
"""

from pathlib import Path

from plurel.world.authoring.workspace import Workspace

WORKSPACE = Workspace(
    catalog_module="catalog",
    catalog_dir=Path("catalog"),
    plans_dir=Path("plans"),
    knowledge_dir=Path("world_knowledge"),
    example_file=Path("catalog/community_services/repair_and_reuse/repair_cafes/item_visits.py"),
    module="plurel.world.authoring --workspace workspace:WORKSPACE",
)
