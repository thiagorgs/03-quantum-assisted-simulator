#!/usr/bin/env python3
"""Gera PDF Mz x DeltaJ para um arquivo QAS ClosedBC."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot QAS ClosedBC NPZ -> PDF (full + zoom).")
    parser.add_argument("--input", required=True, help="Caminho do .npz de entrada")
    parser.add_argument("--output", required=True, help="Caminho do PDF de saida")
    parser.add_argument("--zoom-x", type=float, nargs=2, default=(0.0, 4.0))
    parser.add_argument("--zoom-y", type=float, nargs=2, default=(-0.05, 8.5))
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {in_path}")

    with np.load(in_path, allow_pickle=False) as data:
        for key in ("Js", "Magnetization", "Std_Mz"):
            if key not in data.files:
                raise KeyError(f"Chave obrigatoria ausente: {key}")
        js = np.asarray(data["Js"], dtype=float).reshape(-1)
        mz = np.asarray(data["Magnetization"], dtype=float).reshape(-1)
        std = np.asarray(data["Std_Mz"], dtype=float).reshape(-1)
        nrea = int(data["Nrea"]) if "Nrea" in data.files else 1000
        L = int(data["L"]) if "L" in data.files else -1
        K = int(data["K"]) if "K" in data.files else -1

    if not (js.shape == mz.shape == std.shape):
        raise ValueError(f"Shapes inconsistentes: Js={js.shape}, Mz={mz.shape}, Std={std.shape}")

    order = np.argsort(js)
    js, mz, std = js[order], mz[order], std[order]
    sem = std / math.sqrt(float(nrea))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150, constrained_layout=True)
        ax.plot(js, mz, linewidth=2.0, label=f"L={L} (QAS, K={K})")
        ax.fill_between(js, mz - sem, mz + sem, alpha=0.2)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(f"Magnetizacao vs desordem (QAS, cadeia fechada) - Nrea={nrea}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150, constrained_layout=True)
        ax.plot(js, mz, linewidth=2.0, label=f"L={L} (QAS, K={K})")
        ax.fill_between(js, mz - sem, mz + sem, alpha=0.2)
        ax.set_xlim(*args.zoom_x)
        ax.set_ylim(*args.zoom_y)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(f"Zoom em baixa desordem (QAS, cadeia fechada) - Nrea={nrea}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF gerado em: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
