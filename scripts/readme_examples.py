"""Run every Python block of a Markdown file in order, in one shared namespace, as a reader running
them one after another would; the blocks run in a temporary directory with a temporary HOME, so
the files they write go nowhere. Exits non-zero on the first block that fails."""

import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md").resolve()
blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
with tempfile.TemporaryDirectory() as scratch:
    os.chdir(scratch)
    os.environ["HOME"] = scratch
    namespace: dict[str, object] = {"__name__": "__main__"}
    for number, block in enumerate(blocks, 1):
        try:
            exec(compile(block, f"{path.name} block {number}", "exec"), namespace)
        except Exception:
            traceback.print_exc()
            sys.exit(f"{path.name}: block {number} of {len(blocks)} failed")
print(f"{path.name}: {len(blocks)} blocks ran")
