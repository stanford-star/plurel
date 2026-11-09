"""Run the concept examples, domain recipes and analyses, each in its own isolated process with the Python
running this script, from a temporary directory: the plurel it imports is whichever that Python
has installed, an editable checkout or a wheel. PLUREL_QUICK=1 asks the analyses for their small
configuration. Exits non-zero on the first example that fails."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

examples = Path(__file__).resolve().parents[1] / "examples"
recipes = []
for directory in ("concepts", "generation", "analysis"):
    found = sorted((examples / directory).glob("*.py"))
    if not found:
        sys.exit(f"no examples found in examples/{directory}")
    recipes.extend(found)
with tempfile.TemporaryDirectory() as scratch:
    for recipe in recipes:
        done = subprocess.run(
            [sys.executable, "-I", str(recipe)],
            cwd=scratch,
            capture_output=True,
            text=True,
            env=os.environ | {"PLUREL_QUICK": "1"},
        )
        if done.returncode:
            sys.stdout.write(done.stdout)
            sys.exit(f"{recipe.name} failed:\n{done.stderr}")
        print(done.stdout.strip().splitlines()[-1])
print(f"{len(recipes)} concept examples, domain recipes and analyses ran with {sys.executable}")
