from __future__ import annotations

import json
from pathlib import Path

from agfb_noise.helpers.catalog import shipped_noise_specs


def test_noise_notebooks_exist_for_each_shipped_model() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks" / "noise"

    missing = [
        spec.name
        for spec in shipped_noise_specs()
        if not (notebook_dir / f"{spec.name}.ipynb").is_file()
    ]

    assert missing == []


def test_noise_notebooks_are_source_only_and_use_1024_image() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks" / "noise").glob("*.ipynb"))

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
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
