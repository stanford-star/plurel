"""Run every recipe in examples/generation, small, each in its own isolated process with the Python
running this script, from a temporary directory: the plurel it imports is whichever that Python
has installed, an editable checkout or a wheel. Exits non-zero on the first recipe that fails."""

import subprocess
import sys
import tempfile
from pathlib import Path

recipes = sorted((Path(__file__).resolve().parents[1] / "examples" / "generation").glob("*.py"))
if not recipes:
    sys.exit("no recipes found in examples/generation")
with tempfile.TemporaryDirectory() as scratch:
    for recipe in recipes:
        done = subprocess.run(
            [sys.executable, "-I", str(recipe)], cwd=scratch, capture_output=True, text=True
        )
        if done.returncode:
            sys.stdout.write(done.stdout)
            sys.exit(f"{recipe.name} failed:\n{done.stderr}")
        print(done.stdout.strip().splitlines()[-1])
print(f"{len(recipes)} recipes ran with {sys.executable}")
