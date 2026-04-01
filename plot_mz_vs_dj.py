#!/usr/bin/env python3
"""Generate Mz vs disorder plots from aggregated NPZ files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
ZOOM_X = (0.0, 4.0)
ZOOM_Y = (-0.05, 0.25)


@dataclass
class Dataset:
    label: str
    nrea: int
    js: np.ndarray
    mz: np.ndarray
    std_mz: np.ndarray

    @property
    def sem(self) -> np.ndarray:
        return self.std_mz / math.sqrt(self.nrea)


def _find_recursive(base: Path, name: str) -> Path:
    matches = sorted(base.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Required file not found: {name} under {base}")
    return matches[0]


def _load_dataset(path: Path, label: str, nrea: int) -> Dataset:
    with np.load(path, allow_pickle=False) as npz:
        js = np.asarray(npz["Js"], dtype=float).squeeze()
        mz = np.asarray(npz["Magnetization"], dtype=float).squeeze()
        std_mz = np.asarray(npz["Std_Mz"], dtype=float).squeeze()

    js = np.atleast_1d(js)
    mz = np.atleast_1d(mz)
    std_mz = np.atleast_1d(std_mz)
    if js.shape != mz.shape or js.shape != std_mz.shape:
        raise ValueError(f"Incompatible shapes in {path}: Js {js.shape}, Magnetization {mz.shape}, Std_Mz {std_mz.shape}")

    order = np.argsort(js)
    return Dataset(label=label, nrea=nrea, js=js[order], mz=mz[order], std_mz=std_mz[order])


def _zoom_limits(curves: Dict[str, Dataset]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    all_x = np.concatenate([d.js for d in curves.values()])
    all_y = np.concatenate([d.mz for d in curves.values()])

    x0, x1 = float(np.min(all_x)), float(np.max(all_x))
    y0, y1 = float(np.min(all_y)), float(np.max(all_y))
    x_pad = 0.08 * (x1 - x0 if x1 > x0 else 1.0)
    y_pad = 0.15 * (y1 - y0 if y1 > y0 else 1.0)

    x_mid = 0.5 * (x0 + x1)
    x_half = 0.25 * (x1 - x0 if x1 > x0 else 1.0)
    return (x_mid - x_half, x_mid + x_half), (y0 - y_pad, y1 + y_pad)


def _plot_magnetization_pdf(curves: Dict[str, Dataset], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        for name, d in curves.items():
            ax.plot(d.js, d.mz, marker="o", linewidth=2, label=name)
            ax.fill_between(d.js, d.mz - d.sem, d.mz + d.sem, alpha=0.2)
        ax.set_xlabel("Delta J")
        ax.set_ylabel("Magnetizacao")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        xlim, ylim = _zoom_limits(curves)
        fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        for name, d in curves.items():
            ax.plot(d.js, d.mz, marker="o", linewidth=2, label=name)
            ax.fill_between(d.js, d.mz - d.sem, d.mz + d.sem, alpha=0.2)
        ax.set_xlim(*ZOOM_X)
        ax.set_ylim(*ZOOM_Y)
        ax.set_xlabel("Delta J")
        ax.set_ylabel("Magnetizacao")
        ax.set_title(f"{title} (Zoom)")
        ax.grid(alpha=0.25)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def _align_curves(js_a: np.ndarray, y_a: np.ndarray, js_b: np.ndarray, y_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer direct matching grid if available; otherwise interpolate on overlap.
    if js_a.shape == js_b.shape and np.allclose(js_a, js_b):
        return js_a, y_a, y_b

    lo = max(float(np.min(js_a)), float(np.min(js_b)))
    hi = min(float(np.max(js_a)), float(np.max(js_b)))
    mask = (js_a >= lo) & (js_a <= hi)
    x = js_a[mask]
    if x.size == 0:
        raise ValueError("No overlapping J range for convergence calculation.")
    yb = np.interp(x, js_b, y_b)
    return x, y_a[mask], yb


def _plot_convergence_pdf(l10_1000: Dataset, l10_2000: Dataset, l11_1000: Dataset, l11_2000: Dataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x10, y10_1000, y10_2000 = _align_curves(l10_1000.js, l10_1000.mz, l10_2000.js, l10_2000.mz)
    x11, y11_1000, y11_2000 = _align_curves(l11_1000.js, l11_1000.mz, l11_2000.js, l11_2000.mz)
    d10 = y10_2000 - y10_1000
    d11 = y11_2000 - y11_1000

    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        ax.plot(x10, np.abs(d10), marker="o", linewidth=2, label="L=10")
        ax.plot(x11, np.abs(d11), marker="s", linewidth=2, label="L=11")
        ax.set_xlabel("Desordem (Delta J)")
        ax.set_ylabel("|Mz(2000) - Mz(1000)|")
        ax.set_title("Convergencia 1000 vs 2000 (L=10,11)")
        ax.grid(alpha=0.25)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    # Exact datasets requested by user.
    l11_1000 = _load_dataset(ROOT / "data" / "Mz_Nqu=11_Nrea=1000.npz", label="L=11", nrea=1000)
    l11_2000 = _load_dataset(ROOT / "data" / "Mz_Nqu=11_Nrea=2000.npz", label="L=11", nrea=2000)
    l12_1000 = _load_dataset(ROOT / "data" / "Mz_Nqu=12_Nrea=1000.npz", label="L=12", nrea=1000)

    extracted10 = ROOT / "data" / "extracted10"
    l10_1000_path = _find_recursive(extracted10, "Mz_Nqu=10_Nrea=1000.npz")
    l10_2000_path = _find_recursive(extracted10, "Mz_Nqu=10_Nrea=2000.npz")
    l10_1000 = _load_dataset(l10_1000_path, label="L=10", nrea=1000)
    l10_2000 = _load_dataset(l10_2000_path, label="L=10", nrea=2000)

    _plot_magnetization_pdf(
        curves={"L=12": l12_1000},
        title="Magnetizacao vs Desordem (Nrea=1000, L=12)",
        out_path=OUT_DIR / "magnetizacao_vs_desordem_L12_Nrea1000.pdf",
    )
    _plot_magnetization_pdf(
        curves={"L=10": l10_1000, "L=11": l11_1000, "L=12": l12_1000},
        title="Magnetizacao vs Desordem (Nrea=1000)",
        out_path=OUT_DIR / "magnetizacao_vs_desordem_Nrea1000.pdf",
    )
    _plot_magnetization_pdf(
        curves={"L=10": l10_2000, "L=11": l11_2000},
        title="Magnetizacao vs Desordem (Nrea=2000)",
        out_path=OUT_DIR / "magnetizacao_vs_desordem_Nrea2000.pdf",
    )
    _plot_convergence_pdf(
        l10_1000=l10_1000,
        l10_2000=l10_2000,
        l11_1000=l11_1000,
        l11_2000=l11_2000,
        out_path=OUT_DIR / "convergencia_1000_vs_2000_L10_L11.pdf",
    )

if __name__ == "__main__":
    main()
