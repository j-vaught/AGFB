from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SKIPPED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".ty_cache",
    ".venv",
}
SKIPPED_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".pdf",
    ".png",
    ".pyc",
    ".tif",
    ".tiff",
    ".webp",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for path in _repository_files(root):
        if _should_skip(path):
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            data.decode("ascii")
        except UnicodeDecodeError as error:
            failures.append(_format_failure(root, path, data, error.start))

    if failures:
        print("Non-ASCII text found.")
        print("\n".join(failures))
        return 1
    return 0


def _repository_files(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0:
        return tuple(root / line for line in result.stdout.splitlines() if line)
    return tuple(path for path in root.rglob("*") if path.is_file())


def _should_skip(path: Path) -> bool:
    return path.suffix.lower() in SKIPPED_SUFFIXES or any(
        part in SKIPPED_DIRECTORIES for part in path.parts
    )


def _format_failure(root: Path, path: Path, data: bytes, offset: int) -> str:
    prefix = data[:offset]
    line = prefix.count(b"\n") + 1
    column = offset - prefix.rfind(b"\n")
    relative_path = path.relative_to(root)
    return f"{relative_path}:{line}:{column}"


if __name__ == "__main__":
    sys.exit(main())
