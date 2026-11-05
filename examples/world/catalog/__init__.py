"""An example catalog: its file layout is its taxonomy, which its plans declare."""

from pathlib import Path

from plurel.world import Taxonomy, load_catalog

ROOT = Path(__file__).parent
SCHEMAS = load_catalog(ROOT, __name__, Taxonomy.load(ROOT.parent / "plans"))
