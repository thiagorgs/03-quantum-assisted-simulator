#!/usr/bin/env python3
"""Gera graficos de magnetizacao vs desordem para cadeia fechada (ClosedBC)."""

from __future__ import annotations

import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
ZIP_PATH = DATA_DIR / "ClosedBC.zip"
EXTRACT_DIR = DATA_DIR / "ClosedBC"

ZOOM_X = (0.0, 4.0)
ZOOM_Y = (-0.05, 0.25)
FIGSIZE = (8.2, 5.2)
DPI = 150
GRID_ALPHA = 0.25

NAME_RE = re.compile(r"^Mz_Nqu=(\d+)_Nrea=(\d+)_ClosedBC\.npz$")


@dataclass
class Dataset:
    L: int
    nrea: int
    js: np.ndarray
    mz: np.ndarray
    std_mz: np.ndarray

    @property
    def label(self) -> str:
        return f"L={self.L}"

    @property
    def sem(self) -> np.ndarray:
        return sem_from_std(self.std_mz, self.nrea)


def extract_zip_if_needed(zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists():
        return
    existing = list(extract_dir.rglob("*_ClosedBC.npz")) if extract_dir.exists() else []
    if existing:
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        required = ("Js", "Magnetization", "Std_Mz")
        missing = [k for k in required if k not in npz]
        if missing:
            raise ValueError(f"Arquivo sem chaves obrigatorias {missing}: {path}")

        js = np.asarray(npz["Js"], dtype=float).reshape(-1)
        mz = np.asarray(npz["Magnetization"], dtype=float).reshape(-1)
        std = np.asarray(npz["Std_Mz"], dtype=float).reshape(-1)

    if js.ndim != 1 or mz.ndim != 1 or std.ndim != 1:
        raise ValueError(f"Arrays devem ser 1D em {path}")
    if not (len(js) == len(mz) == len(std)):
        raise ValueError(
            f"Tamanhos inconsistentes em {path}: Js={js.shape}, Magnetization={mz.shape}, Std_Mz={std.shape}"
        )

    order = np.argsort(js)
    return js[order], mz[order], std[order]


def sem_from_std(std: np.ndarray, nrea: int) -> np.ndarray:
    return std / math.sqrt(float(nrea))


def align_by_js(
    js_a: np.ndarray, arr_a: np.ndarray, js_b: np.ndarray, arr_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    common, idx_a, idx_b = np.intersect1d(js_a, js_b, return_indices=True)
    if common.size == 0:
        raise ValueError("Nao ha intersecao entre as grades de DeltaJ para comparacao.")
    return common, arr_a[idx_a], arr_b[idx_b]


def discover_closedbc_datasets(search_root: Path) -> Dict[int, List[Dataset]]:
    by_nrea: Dict[int, List[Dataset]] = {}
    for path in sorted(search_root.rglob("*_ClosedBC.npz")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        L = int(m.group(1))
        nrea = int(m.group(2))
        js, mz, std = load_npz(path)
        by_nrea.setdefault(nrea, []).append(Dataset(L=L, nrea=nrea, js=js, mz=mz, std_mz=std))

    for nrea in by_nrea:
        by_nrea[nrea].sort(key=lambda d: d.L)
    return by_nrea


def plot_pdf_full_and_zoom(datasets: List[Dataset], nrea: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        for d in datasets:
            ax.plot(d.js, d.mz, linewidth=2.0, label=d.label)
            ax.fill_between(d.js, d.mz - d.sem, d.mz + d.sem, alpha=0.20)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(f"Magnetizacao vs desordem (exato, cadeia fechada) - {nrea} realizacoes")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        for d in datasets:
            ax.plot(d.js, d.mz, linewidth=2.0, label=d.label)
            ax.fill_between(d.js, d.mz - d.sem, d.mz + d.sem, alpha=0.20)
        ax.set_xlim(*ZOOM_X)
        ax.set_ylim(*ZOOM_Y)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title(f"Zoom em baixa desordem (exato, cadeia fechada) - {nrea} realizacoes")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def plot_convergence(by_nrea: Dict[int, List[Dataset]], out_path: Path) -> bool:
    if 1000 not in by_nrea or 2000 not in by_nrea:
        return False

    by_l_1000 = {d.L: d for d in by_nrea[1000]}
    by_l_2000 = {d.L: d for d in by_nrea[2000]}
    common_ls = sorted(set(by_l_1000).intersection(by_l_2000))
    if not common_ls:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for L in common_ls:
            d1000 = by_l_1000[L]
            d2000 = by_l_2000[L]
            js, m1000, m2000 = align_by_js(d1000.js, d1000.mz, d2000.js, d2000.mz)
            delta = m2000 - m1000
            max_abs = float(np.max(np.abs(delta)))
            rms = float(np.sqrt(np.mean(delta**2)))

            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
            ax.plot(js, m1000, linewidth=2.0, label="Nrea=1000")
            ax.plot(js, m2000, linewidth=2.0, label="Nrea=2000")
            ax.set_xlabel("ΔJ")
            ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
            ax.set_title(f"Convergencia 1000 vs 2000 (cadeia fechada) - L={L}")
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
            ax.plot(js, delta, linewidth=2.0)
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
            ax.set_xlabel("ΔJ")
            ax.set_ylabel("Δ(ΔJ) = M2000 - M1000")
            ax.set_title(f"Diferenca 2000-1000 (cadeia fechada) - L={L}")
            ax.grid(True, alpha=GRID_ALPHA)
            pdf.savefig(fig)
            plt.close(fig)

    return True


def main() -> int:
    try:
        extract_zip_if_needed(ZIP_PATH, EXTRACT_DIR)
        by_nrea = discover_closedbc_datasets(DATA_DIR)
        if not by_nrea:
            raise FileNotFoundError(
                "Nenhum arquivo '*_ClosedBC.npz' foi encontrado em data/ ou subpastas. "
                "Verifique data/ClosedBC.zip ou data/ClosedBC."
            )

        generated: List[Path] = []
        for nrea in sorted(by_nrea):
            out = OUTPUT_DIR / f"magnetizacao_vs_desordem_closedbc_Nrea{nrea}.pdf"
            plot_pdf_full_and_zoom(by_nrea[nrea], nrea, out)
            generated.append(out)

        conv_out = OUTPUT_DIR / "convergencia_closedbc_1000_vs_2000.pdf"
        if plot_convergence(by_nrea, conv_out):
            generated.append(conv_out)
        else:
            print("Aviso: comparacao de convergencia (1000 vs 2000) nao gerada por falta de pares completos.")

        print("\nArquivos gerados:")
        for path in generated:
            print(f"- {path}")
        return 0
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
