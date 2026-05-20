#!/usr/bin/env python3
"""Check repository text files for non-ASCII characters."""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

TEXT_FILENAMES = {
    ".gitignore",
}
TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".typ",
    ".yaml",
    ".yml",
}
SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ty_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def is_text_file(path: Path) -> bool:
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES


def iter_text_files(root: Path) -> list[Path]:
    text_files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.is_file() and is_text_file(path):
            text_files.append(path)
    return sorted(text_files)


def format_non_ascii(character: str) -> str:
    codepoint = f"U+{ord(character):04X}"
    name = unicodedata.name(character, "UNKNOWN")
    return f"{codepoint} {name}"


def scan_path(path: Path, root: Path) -> list[str]:
    relative_path = path.relative_to(root)
    findings: list[str] = []
    for character in str(relative_path):
        if ord(character) > 127:
            findings.append(f"{relative_path}: path contains {format_non_ascii(character)}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        return [f"{relative_path}: invalid UTF-8 text at byte {error.start}"]

    for line_number, line in enumerate(text.splitlines(), start=1):
        for column_number, character in enumerate(line, start=1):
            if ord(character) > 127:
                findings.append(
                    f"{relative_path}:{line_number}:{column_number}: {format_non_ascii(character)}"
                )
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings: list[str] = []
    for path in iter_text_files(root):
        findings.extend(scan_path(path, root))

    if findings:
        print("Unicode check failed. Non-ASCII characters were found.")
        for finding in findings:
            print(finding)
        return 1

    print("Unicode check passed. Text files are ASCII-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
