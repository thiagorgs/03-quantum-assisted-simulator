#!/usr/bin/env python3
"""Compara QAS verdadeiro e resultado exato para ClosedBC em L=11."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXACT = ROOT / "data" / "ClosedBC" / "Mz_Nqu=11_Nrea=1000_K=3_ClosedBC_ExactRef.npz"
DEFAULT_QAS = ROOT / "data" / "ClosedBC" / "Mz_Nqu=11_Nrea=1000_K=3_ClosedBC_QAS_true.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "comparacao_closedbc_exactref_vs_qas_true_L11_K3_Nrea1000.pdf"

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

    if not (js.shape == mz.shape == std.shape):
        raise ValueError(f"Shapes inconsistentes em {path}: Js={js.shape}, Mz={mz.shape}, Std_Mz={std.shape}")

    order = np.argsort(js)
    js = js[order]
    mz = mz[order]
    std = std[order]
    sem = std / math.sqrt(float(nrea))
    return {"path": path, "js": js, "mz": mz, "std": std, "sem": sem, "nrea": nrea, "L": L, "K": K}


def align_by_js(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common, idx_a, idx_b = np.intersect1d(a["js"], b["js"], return_indices=True)
    if common.size == 0:
        raise ValueError("Nao ha intersecao entre as grades de ΔJ.")
    return common, a["mz"][idx_a], a["sem"][idx_a], b["mz"][idx_b], b["sem"][idx_b]


def plot_pdf(exact: dict, qas: dict, output: Path) -> None:
    js, mz_exact, sem_exact, mz_qas, sem_qas = align_by_js(exact, qas)
    delta = mz_qas - mz_exact
    max_abs = float(np.max(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta**2)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(js, mz_exact, linewidth=2.0, label="Exato")
        ax.fill_between(js, mz_exact - sem_exact, mz_exact + sem_exact, alpha=0.18)
        ax.plot(js, mz_qas, linewidth=2.0, label=f"QAS (K={qas['K']})")
        ax.fill_between(js, mz_qas - sem_qas, mz_qas + sem_qas, alpha=0.18)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title("ExactRef vs QAS (cadeia fechada) - L=11, 1000 realizacoes")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        ax.text(
            0.02,
            0.02,
            f"max|Δ|={max_abs:.4e}\nRMS(Δ)={rms:.4e}",
            transform=ax.transAxes,
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(js, mz_exact, linewidth=2.0, label="Exato")
        ax.fill_between(js, mz_exact - sem_exact, mz_exact + sem_exact, alpha=0.18)
        ax.plot(js, mz_qas, linewidth=2.0, label=f"QAS (K={qas['K']})")
        ax.fill_between(js, mz_qas - sem_qas, mz_qas + sem_qas, alpha=0.18)
        ax.set_xlim(*ZOOM_X)
        ax.set_ylim(*ZOOM_Y)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title("Zoom em baixa desordem - ExactRef vs QAS (cadeia fechada) - L=11")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(js, delta, linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Δ(ΔJ) = MQAS - Mexato")
        ax.set_title("Diferenca QAS - ExactRef (cadeia fechada) - L=11")
        ax.grid(True, alpha=GRID_ALPHA)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara resultado exato e QAS verdadeiro em L=11.")
    parser.add_argument("--exact", default=str(DEFAULT_EXACT), help="Arquivo .npz exato")
    parser.add_argument("--qas", default=str(DEFAULT_QAS), help="Arquivo .npz QAS")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Arquivo PDF de saida")
    args = parser.parse_args()

    exact = load_dataset(Path(args.exact).resolve())
    qas = load_dataset(Path(args.qas).resolve())
    plot_pdf(exact, qas, Path(args.output).resolve())
    print(f"PDF gerado em: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
