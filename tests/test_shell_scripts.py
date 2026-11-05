"""The Linux launchers (install.sh, run.sh, ...) are executable in a clean checkout.

git records the mode of each file; with 100644 a fresh clone's './install.sh' fails with "Permission denied"
(exit 126). Every tracked .sh must be 100755 in the index and start with a shebang.

Run from the backend folder with any Python 3.10+ (skipped outside a git checkout):

    python -m unittest tests.test_shell_scripts
"""

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tracked_shell_scripts() -> dict:
    """path -> git mode of every tracked *.sh, or None outside a git checkout."""
    if shutil.which("git") is None:
        return None
    result = subprocess.run(["git", "ls-files", "-s", "--", "*.sh"], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8")
    if result.returncode != 0:
        return None
    modes = {}
    for line in result.stdout.splitlines():
        meta, path = line.split("\t", 1)
        modes[path] = meta.split()[0]
    return modes


class ShellScriptsTest(unittest.TestCase):
    def test_launchers_are_executable(self):
        modes = tracked_shell_scripts()
        if modes is None:
            self.skipTest("not a git checkout")
        for name in ("install.sh", "run.sh"):
            self.assertIn(name, modes)
        for path, mode in modes.items():
            with self.subTest(path=path):
                self.assertEqual(mode, "100755", f"{path}: git update-index --chmod=+x {path}")
                first_line = (ROOT / path).read_text(encoding="utf-8").splitlines()[0]
                self.assertTrue(first_line.startswith("#!"), first_line)


if __name__ == "__main__":
    unittest.main()
