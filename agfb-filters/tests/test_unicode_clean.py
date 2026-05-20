from __future__ import annotations

import subprocess
import sys


def test_repository_text_files_are_ascii() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_unicode.py"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
