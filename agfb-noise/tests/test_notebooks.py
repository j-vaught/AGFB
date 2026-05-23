from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from pytest import MonkeyPatch

import agfb_noise.helpers.notebook as notebook_helpers
from agfb_noise.helpers.catalog import shipped_noise_specs


def _noise_notebook_dir(root: Path) -> Path:
    nested = root / "notebooks" / "noise"
    if nested.is_dir():
        return nested
    return root / "notebooks"


def test_noise_notebooks_exist_for_each_shipped_model() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_dir = _noise_notebook_dir(root)

    missing = [
        spec.name
        for spec in shipped_noise_specs()
        if not (notebook_dir / f"{spec.name}.ipynb").is_file()
    ]

    assert missing == []


def test_noise_notebooks_are_source_only_and_use_1024_image() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted(_noise_notebook_dir(root).glob("*.ipynb"))

    assert notebooks
    for path in notebooks:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["nbformat"] == 4
        assert data["metadata"]["kernelspec"] == {
            "display_name": "agfb-noise (.venv)",
            "language": "python",
            "name": "agfb-noise",
        }
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in data["cells"] if cell["cell_type"] == "code"
        )
        assert "synthetic_1024_image" in source
        markdown_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in data["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "## Noise Context" in markdown_source
        assert "https://" in markdown_source
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None


def test_show_noise_preview_displays_once_without_rich_return(
    monkeypatch: MonkeyPatch,
) -> None:
    displayed: list[object] = []

    def fake_import_module(name: str) -> object:
        assert name == "IPython.display"
        return SimpleNamespace(
            HTML=lambda text: {"html": text},
            display=displayed.append,
        )

    monkeypatch.setattr(notebook_helpers.importlib, "import_module", fake_import_module)
    image = torch.zeros(1, 4, 4)

    result = notebook_helpers.show_noise_preview(image, image, title="Preview")

    assert result is None
    assert len(displayed) == 1
