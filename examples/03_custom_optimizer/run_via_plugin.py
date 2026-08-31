"""Same optimizer as run.py, discovered the way an external package would be —
via APOBENCH_PLUGINS, with no import of my_optimizer in this script at all and
no edit inside pof/.

run.py's in-process `import my_optimizer` works, but it demonstrates the
built-in registry's requirement (something has to import your module), not the
plugin path a third-party package actually uses. This script demonstrates that
path for real: it runs `pof list` as a subprocess with APOBENCH_PLUGINS set,
so the only way "my_method" can appear is through pof/plugins.py's discovery —
the same mechanism an installed apobench.optimizers entry point would use.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

if __name__ == "__main__":
    env = dict(os.environ)
    env["APOBENCH_PLUGINS"] = "my_optimizer"
    # my_optimizer.py's directory on the path, not the repo root -- proving
    # this works from a location that isn't inside pof/ at all.
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-m", "pof.cli", "list"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    print(result.stdout)
    if "my_method" not in result.stdout:
        print("my_method did not appear -- plugin discovery did not fire.",
              file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    print("my_method discovered via APOBENCH_PLUGINS, zero edits inside pof/.")
