"""Run every Python block of a Markdown file in order, in one shared namespace, as a reader running
them one after another would; then run each block marked `<!-- standalone -->` again on its own, in
a fresh isolated Python process, as a reader following one recipe would. Every block runs in a
temporary directory with a temporary HOME, so the files it writes go nowhere. Exits non-zero on the
first block that fails."""

import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

BLOCK = re.compile(r"(<!-- standalone -->\s*)?```python\n(.*?)```", re.S)

path = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md").resolve()
blocks = [(bool(mark), code) for mark, code in BLOCK.findall(path.read_text())]
with tempfile.TemporaryDirectory() as scratch:
    os.chdir(scratch)
    os.environ["HOME"] = scratch
    namespace: dict[str, object] = {"__name__": "__main__"}
    for number, (_, code) in enumerate(blocks, 1):
        try:
            exec(compile(code, f"{path.name} block {number}", "exec"), namespace)
        except Exception:
            traceback.print_exc()
            sys.exit(f"{path.name}: block {number} of {len(blocks)} failed")
    standalone = [(number, code) for number, (alone, code) in enumerate(blocks, 1) if alone]
    for number, code in standalone:
        done = subprocess.run([sys.executable, "-I", "-c", code], cwd=scratch, env=os.environ)
        if done.returncode:
            sys.exit(f"{path.name}: block {number}, run on its own, failed")
print(f"{path.name}: {len(blocks)} blocks ran together, {len(standalone)} also on their own")
