#!/usr/bin/env python3
"""Plota Mz x DeltaJ para o resultado QAS fechado de L=10."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "ClosedBC" / "Mz_Nqu=10_Nrea=1000_K=8_ClosedBC_QAS.npz"
OUT_PATH = ROOT / "outputs" / "magnetizacao_vs_desordem_closedbc_qas_L10_K8_Nrea1000.pdf"

ZOOM_X = (0.0, 4.0)
ZOOM_Y = (-0.05, 10.5)
FIGSIZE = (8.2, 5.2)
DPI = 150
GRID_ALPHA = 0.25


def main() -> int:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {IN_PATH}")

    data = np.load(IN_PATH, allow_pickle=False)
    for key in ("Js", "Magnetization", "Std_Mz"):
        if key not in data.files:
            raise KeyError(f"Chave obrigatoria ausente: {key}")

    js = np.asarray(data["Js"], dtype=float).reshape(-1)
    mz = np.asarray(data["Magnetization"], dtype=float).reshape(-1)
    std = np.asarray(data["Std_Mz"], dtype=float).reshape(-1)
    nrea = int(data["Nrea"]) if "Nrea" in data.files else 1000
    L = int(data["L"]) if "L" in data.files else 10
    K = int(data["K"]) if "K" in data.files else 8

    if not (js.shape == mz.shape == std.shape):
        raise ValueError(f"Shapes inconsistentes: Js={js.shape}, Mz={mz.shape}, Std={std.shape}")

    order = np.argsort(js)
    js = js[order]
    mz = mz[order]
    std = std[order]
    sem = std / math.sqrt(float(nrea))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(js, mz, linewidth=2.0, label=f"L={L} (QAS, K={K})")
        ax.fill_between(js, mz - sem, mz + sem, alpha=0.2)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(f"Magnetizacao vs desordem (QAS, cadeia fechada) - Nrea={nrea}")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(js, mz, linewidth=2.0, label=f"L={L} (QAS, K={K})")
        ax.fill_between(js, mz - sem, mz + sem, alpha=0.2)
        ax.set_xlim(*ZOOM_X)
        ax.set_ylim(*ZOOM_Y)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(f"Zoom em baixa desordem (QAS, cadeia fechada) - Nrea={nrea}")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF gerado em: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
