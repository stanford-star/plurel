"""Run the concept examples and domain recipes, each in its own isolated process with the Python
running this script, from a temporary directory: the plurel it imports is whichever that Python
has installed, an editable checkout or a wheel. Exits non-zero on the first example that fails."""

import subprocess
import sys
import tempfile
from pathlib import Path

if not __debug__:
    sys.exit("examples use assertions: run without -O")
examples = Path(__file__).resolve().parents[1] / "examples"
recipes = []
for directory in ("concepts", "generation"):
    found = sorted((examples / directory).glob("*.py"))
    if not found:
        sys.exit(f"no examples found in examples/{directory}")
    recipes.extend(found)
with tempfile.TemporaryDirectory() as scratch:
    for recipe in recipes:
        done = subprocess.run(
            [sys.executable, "-I", str(recipe)], cwd=scratch, capture_output=True, text=True
        )
        if done.returncode:
            sys.stdout.write(done.stdout)
            sys.exit(f"{recipe.name} failed:\n{done.stderr}")
        print(done.stdout.strip().splitlines()[-1])
print(f"{len(recipes)} concept examples and domain recipes ran with {sys.executable}")
