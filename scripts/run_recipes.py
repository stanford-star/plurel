"""Run the concept examples, domain recipes, analyses and figures, and realism.py's self-test, each in its
own isolated process with the Python running this script, from a temporary directory: the plurel
it imports is whichever that Python has installed, an editable checkout or a wheel. PLUREL_QUICK=1
asks the analyses for their small configuration. Exits non-zero on the first example that fails."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

examples = Path(__file__).resolve().parents[1] / "examples"
recipes = []
for directory in ("concepts", "generation", "analysis", "figures"):
    found = sorted(p for p in (examples / directory).glob("*.py") if not p.name.startswith("_"))
    if not found:
        sys.exit(f"no examples found in examples/{directory}")
    recipes.extend(found)
runs = [[recipe] for recipe in recipes] + [[examples / "reference" / "realism.py", "--self-test"]]
with tempfile.TemporaryDirectory() as scratch:
    for recipe, *flags in runs:
        done = subprocess.run(
            [sys.executable, "-I", str(recipe), *flags],
            cwd=scratch,
            capture_output=True,
            text=True,
            env=os.environ | {"PLUREL_QUICK": "1"},
        )
        if done.returncode:
            sys.stdout.write(done.stdout)
            sys.exit(f"{recipe.name} {' '.join(flags)} failed:\n{done.stderr}")
        print(done.stdout.strip().splitlines()[-1])
print(f"{len(runs)} examples, recipes, analyses and self-tests ran with {sys.executable}")
