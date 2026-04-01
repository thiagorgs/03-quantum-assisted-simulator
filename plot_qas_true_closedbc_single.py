#!/usr/bin/env python3
"""Gera PDF de magnetizacao vs desordem para um unico resultado QAS ClosedBC."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "ClosedBC" / "Mz_Nqu=11_Nrea=1000_K=3_ClosedBC_QAS_true.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "magnetizacao_vs_desordem_closedbc_qas_true_L11_K3_Nrea1000.pdf"

ZOOM_X = (0.0, 4.0)
ZOOM_Y = (-0.05, 0.25)
FIGSIZE = (8.2, 5.2)
DPI = 150
GRID_ALPHA = 0.25


def load_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = ("Js", "Magnetization", "Std_Mz")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Chaves obrigatorias ausentes em {path}: {missing}")

        js = np.asarray(data["Js"], dtype=float).reshape(-1)
        mz = np.asarray(data["Magnetization"], dtype=float).reshape(-1)
        std = np.asarray(data["Std_Mz"], dtype=float).reshape(-1)
        nrea = int(data["Nrea"]) if "Nrea" in data.files else 1000
        L = int(data["L"]) if "L" in data.files else -1
        K = int(data["K"]) if "K" in data.files else -1

    if not (js.ndim == mz.ndim == std.ndim == 1):
        raise ValueError(f"Arrays devem ser 1D em {path}")
    if not (js.shape == mz.shape == std.shape):
        raise ValueError(f"Shapes inconsistentes em {path}: Js={js.shape}, Mz={mz.shape}, Std_Mz={std.shape}")

    order = np.argsort(js)
    js = js[order]
    mz = mz[order]
    std = std[order]
    sem = std / math.sqrt(float(nrea))
    return {"js": js, "mz": mz, "std": std, "sem": sem, "nrea": nrea, "L": L, "K": K}


def plot_pdf(dataset: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    title_full = f"Magnetizacao vs desordem (QAS, cadeia fechada) - L={dataset['L']}, K={dataset['K']}, {dataset['nrea']} realizacoes"
    title_zoom = f"Zoom em baixa desordem (QAS, cadeia fechada) - L={dataset['L']}, K={dataset['K']}, {dataset['nrea']} realizacoes"
    label = f"L={dataset['L']} (QAS, K={dataset['K']})"

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(dataset["js"], dataset["mz"], linewidth=2.0, label=label)
        ax.fill_between(dataset["js"], dataset["mz"] - dataset["sem"], dataset["mz"] + dataset["sem"], alpha=0.20)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(title_full)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(dataset["js"], dataset["mz"], linewidth=2.0, label=label)
        ax.fill_between(dataset["js"], dataset["mz"] - dataset["sem"], dataset["mz"] + dataset["sem"], alpha=0.20)
        ax.set_xlim(*ZOOM_X)
        ax.set_ylim(*ZOOM_Y)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(title_zoom)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plota um resultado QAS ClosedBC (.npz) em PDF.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Arquivo .npz de entrada")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Arquivo PDF de saida")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    dataset = load_dataset(input_path)
    plot_pdf(dataset, output_path)
    print(f"PDF gerado em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
