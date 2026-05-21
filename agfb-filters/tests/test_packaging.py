from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_package_imports_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agfb_filters; print(agfb_filters.__file__)",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agfb_filters" in result.stdout
