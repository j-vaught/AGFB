"""Generate source-only notebooks for AGFB noise models."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent

NOTEBOOKS = (
    {
        "name": "gaussian",
        "title": "Gaussian Noise",
        "function": "add_gaussian",
        "summary": "Additive independent Gaussian noise.",
        "expr": "add_gaussian(clean, sigma=0.035, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "local_variance",
        "title": "Local-Variance Gaussian Noise",
        "function": "add_local_variance",
        "summary": "Gaussian noise with a spatially varying variance map.",
        "setup": dedent(
            """
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, clean.shape[-2], device=clean.device),
                torch.linspace(-1.0, 1.0, clean.shape[-1], device=clean.device),
                indexing="ij",
            )
            variance = (0.0002 + 0.004 * ((xx + 1.0) / 2.0) ** 2).unsqueeze(0)
            """
        ).strip(),
        "expr": "add_local_variance(clean, variance=variance, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "uniform",
        "title": "Uniform Noise",
        "function": "add_uniform",
        "summary": "Additive uniform noise over a centered interval.",
        "expr": "add_uniform(clean, half_width=0.06, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "poisson",
        "title": "Poisson Shot Noise",
        "function": "add_poisson",
        "summary": "Photon-counting shot noise generated from nonnegative intensity.",
        "expr": "add_poisson(clean, peak=512.0, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "poisson_gaussian",
        "title": "Poisson-Gaussian Noise",
        "function": "add_poisson_gaussian",
        "summary": "Poisson shot noise with signal-independent Gaussian read noise.",
        "expr": (
            "add_poisson_gaussian(\n"
            "    clean,\n"
            "    peak=512.0,\n"
            "    read_sigma=0.015,\n"
            "    seed=seed,\n"
            "    clamp=(0.0, 1.0),\n"
            ")"
        ),
    },
    {
        "name": "dark_current",
        "title": "Dark-Current Noise",
        "function": "add_dark_current",
        "summary": "Dark-current background counts plus optional Gaussian read noise.",
        "expr": (
            "add_dark_current(\n"
            "    clean,\n"
            "    dark_rate=6.0,\n"
            "    exposure_time=0.5,\n"
            "    peak=512.0,\n"
            "    read_sigma=0.004,\n"
            "    seed=seed,\n"
            "    clamp=(0.0, 1.0),\n"
            ")"
        ),
    },
    {
        "name": "salt",
        "title": "Salt Noise",
        "function": "add_salt",
        "summary": "Random high-valued impulse replacement.",
        "expr": "add_salt(clean, amount=0.02, salt_value=1.0, seed=seed)",
    },
    {
        "name": "pepper",
        "title": "Pepper Noise",
        "function": "add_pepper",
        "summary": "Random low-valued impulse replacement.",
        "expr": "add_pepper(clean, amount=0.02, pepper_value=0.0, seed=seed)",
    },
    {
        "name": "salt_pepper",
        "title": "Salt-And-Pepper Noise",
        "function": "add_salt_pepper",
        "summary": "Random low- and high-valued impulse replacement.",
        "expr": (
            "add_salt_pepper(\n"
            "    clean,\n"
            "    amount=0.03,\n"
            "    salt_vs_pepper=0.5,\n"
            "    salt_value=1.0,\n"
            "    pepper_value=0.0,\n"
            "    seed=seed,\n"
            ")"
        ),
    },
    {
        "name": "random_impulse",
        "title": "Random-Valued Impulse Noise",
        "function": "add_random_impulse",
        "summary": "Impulse replacement with random values sampled from a range.",
        "expr": "add_random_impulse(clean, amount=0.03, low=0.0, high=1.0, seed=seed)",
    },
    {
        "name": "dead_pixel",
        "title": "Dead-Pixel Noise",
        "function": "add_dead_pixels",
        "summary": "Random dead and hot pixel defects.",
        "expr": (
            "add_dead_pixels(\n"
            "    clean,\n"
            "    amount=0.015,\n"
            "    hot_fraction=0.15,\n"
            "    dead_value=0.0,\n"
            "    hot_value=1.0,\n"
            "    seed=seed,\n"
            ")"
        ),
    },
    {
        "name": "speckle",
        "title": "Gaussian Speckle Noise",
        "function": "add_speckle",
        "summary": "Multiplicative Gaussian speckle noise.",
        "expr": "add_speckle(clean, sigma=0.18, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "gamma_speckle",
        "title": "Gamma Speckle Noise",
        "function": "add_gamma_speckle",
        "summary": "Unit-mean gamma multiplicative speckle for integer-look simulation.",
        "expr": "add_gamma_speckle(clean, looks=4, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "rician",
        "title": "Rician Noise",
        "function": "add_rician",
        "summary": "Magnitude image noise from two independent Gaussian channels.",
        "expr": "add_rician(clean, sigma=0.05, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "rayleigh",
        "title": "Rayleigh Noise",
        "function": "add_rayleigh",
        "summary": "Positive Rayleigh-distributed additive noise.",
        "expr": "add_rayleigh(clean, sigma=0.04, seed=seed, clamp=(0.0, 1.0))",
    },
    {
        "name": "quantization",
        "title": "Quantization Noise",
        "function": "add_quantization",
        "summary": "Uniform scalar quantization over a finite intensity range.",
        "expr": "add_quantization(clean, levels=16, min_value=0.0, max_value=1.0)",
    },
    {
        "name": "fixed_pattern",
        "title": "Fixed-Pattern Noise",
        "function": "add_fixed_pattern",
        "summary": "Pixelwise offset and gain nonuniformity.",
        "expr": (
            "add_fixed_pattern(\n"
            "    clean,\n"
            "    offset_sigma=0.025,\n"
            "    gain_sigma=0.04,\n"
            "    seed=seed,\n"
            "    clamp=(0.0, 1.0),\n"
            ")"
        ),
    },
    {
        "name": "stripe",
        "title": "Stripe Noise",
        "function": "add_stripe",
        "summary": "Row and column correlated offset noise.",
        "expr": (
            "add_stripe(\n"
            "    clean,\n"
            "    row_sigma=0.025,\n"
            "    column_sigma=0.018,\n"
            "    seed=seed,\n"
            "    clamp=(0.0, 1.0),\n"
            ")"
        ),
    },
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks" / "noise"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    for item in NOTEBOOKS:
        notebook = _notebook(item)
        (notebook_dir / f"{item['name']}.ipynb").write_text(
            json.dumps(notebook, indent=2) + "\n",
            encoding="utf-8",
        )


def _notebook(item: dict[str, str]) -> dict[str, object]:
    title = item["title"]
    function = item["function"]
    setup = item.get("setup", "")
    expr = item["expr"]
    apply_model = f"""def apply_model() -> torch.Tensor:
{_return_statement(expr)}


noisy = apply_model()
delta = noisy - clean"""
    return {
        "cells": [
            _markdown(
                f"# {title} 1024 synthetic-image example\n\n"
                f"{item['summary']} This notebook starts from the same synthetic "
                "1024 x 1024 image, applies one AGFB noise model, and displays the "
                "clean image, noisy image, and residual."
            ),
            _markdown(
                "## Setup\n\n"
                "Import the model function and notebook helpers. The autoreload lines "
                "help Jupyter pick up local source edits without restarting the kernel."
            ),
            _code(
                f"""# Imports and local reloads.
%load_ext autoreload
%autoreload 2

import time

import torch

from agfb_noise import {function}
from agfb_noise.notebook import show_noise_preview, summarize_tensors, synthetic_1024_image"""
            ),
            _markdown(
                "## Synthetic Image\n\n"
                "Create the shared 1024 x 1024 synthetic input on CUDA when available."
            ),
            _code(
                """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 123

clean = synthetic_1024_image(device=device)
clean.shape, clean.device, clean.dtype"""
            ),
            _markdown("## Apply Noise\n\nSet the model parameters and corrupt the image."),
            _code("\n\n".join(part for part in (setup, apply_model) if part)),
            _markdown(
                "## Summary\n\n"
                "Report compact tensor statistics for the clean, noisy, and residual images."
            ),
            _code(
                """summarize_tensors({
    "clean": clean,
    "noisy": noisy,
    "delta": delta,
})"""
            ),
            _markdown(
                "## Preview\n\n"
                "The preview down-samples the 1024 image for display. The residual uses a "
                "blue-white-garnet diverging map."
            ),
            _code(f"""show_noise_preview(clean, noisy, title="{title}")"""),
            _markdown("## Timing\n\nTime one hot-path model application on the selected device."),
            _code(
                """if clean.is_cuda:
    torch.cuda.synchronize()
start = time.perf_counter()
_ = apply_model()
if clean.is_cuda:
    torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1000.0
elapsed_ms"""
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "agfb-noise (.venv)",
                "language": "python",
                "name": "agfb-noise",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def _code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(source: str) -> list[str]:
    return [f"{line}\n" for line in source.strip().splitlines()]


def _return_statement(expr: str) -> str:
    expr = dedent(expr).strip()
    if "\n" not in expr:
        return f"    return {expr}"
    first, rest = expr.split("\n", 1)
    rest = "\n".join(line.strip() for line in rest.splitlines())
    return f"    return {first}\n{indent(rest, '        ')}"


if __name__ == "__main__":
    main()
