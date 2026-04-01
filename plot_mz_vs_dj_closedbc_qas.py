#!/usr/bin/env python3
"""Plota Magnetizacao vs DeltaJ para cadeia fechada usando resultados QAS."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ClosedBC"
OUTPUT_DIR = ROOT / "outputs"

ZOOM_X = (0.0, 4.0)
ZOOM_Y = (-0.05, 0.25)
FIGSIZE = (8.2, 5.2)
DPI = 150
GRID_ALPHA = 0.25

NAME_QAS_SUMMARY_RE = re.compile(r"Mz_Nqu=(\d+)_Nrea=(\d+).*QAS.*\.npz$", re.IGNORECASE)


@dataclass
class Dataset:
    L: int
    nrea: int
    js: np.ndarray
    mz: np.ndarray
    std_mz: np.ndarray
    source: Path

    @property
    def sem(self) -> np.ndarray:
        return self.std_mz / math.sqrt(float(self.nrea))

    @property
    def label(self) -> str:
        return f"L={self.L}"


def _as_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("Array nao eh 1D")
    return arr


def _validate_shapes(js: np.ndarray, mz: np.ndarray, std: np.ndarray, path: Path) -> None:
    if not (js.ndim == mz.ndim == std.ndim == 1):
        raise ValueError(f"Arrays devem ser 1D em {path}")
    if not (len(js) == len(mz) == len(std)):
        raise ValueError(
            f"Tamanhos inconsistentes em {path}: Js={js.shape}, Magnetization={mz.shape}, Std_Mz={std.shape}"
        )


def _dataset_from_summary(path: Path, L: int, nrea: int) -> Dataset:
    with np.load(path, allow_pickle=False) as npz:
        required = ("Js", "Magnetization", "Std_Mz")
        missing = [k for k in required if k not in npz.files]
        if missing:
            raise ValueError(f"Faltam chaves {missing} em {path}")
        js = _as_1d(npz["Js"])
        mz = _as_1d(npz["Magnetization"])
        std = _as_1d(npz["Std_Mz"])

    _validate_shapes(js, mz, std, path)
    order = np.argsort(js)
    return Dataset(L=L, nrea=nrea, js=js[order], mz=mz[order], std_mz=std[order], source=path)


def _dataset_from_aggregated(path: Path) -> Dataset:
    with np.load(path, allow_pickle=False) as npz:
        required = ("delta_J", "magnetization_mean", "NQ", "N_REALIZ")
        missing = [k for k in required if k not in npz.files]
        if missing:
            raise ValueError(f"Faltam chaves {missing} em {path}")

        L = int(npz["NQ"])
        nrea = int(npz["N_REALIZ"])
        dj = _as_1d(npz["delta_J"])
        mz_raw = _as_1d(npz["magnetization_mean"])

    if dj.shape != mz_raw.shape:
        raise ValueError(f"delta_J e magnetization_mean com shapes diferentes em {path}")

    js_unique = np.unique(dj)
    mz_mean = np.empty(js_unique.shape[0], dtype=float)
    mz_std = np.empty(js_unique.shape[0], dtype=float)
    for i, val in enumerate(js_unique):
        samples = mz_raw[np.isclose(dj, val)]
        if samples.size == 0:
            raise ValueError(f"Sem amostras para DeltaJ={val} em {path}")
        mz_mean[i] = float(np.mean(samples))
        mz_std[i] = float(np.std(samples, ddof=1 if samples.size > 1 else 0))

    return Dataset(L=L, nrea=nrea, js=js_unique, mz=mz_mean, std_mz=mz_std, source=path)


def discover_qas_datasets(data_dir: Path) -> List[Dataset]:
    found: List[Dataset] = []

    for path in sorted(data_dir.rglob("*.npz")):
        name = path.name

        m = NAME_QAS_SUMMARY_RE.match(name)
        if m:
            found.append(_dataset_from_summary(path, L=int(m.group(1)), nrea=int(m.group(2))))
            continue

        if name.lower() == "aggregated_up_sv.npz":
            found.append(_dataset_from_aggregated(path))
            continue

    return found


def plot_pdf_full_and_zoom(datasets: List[Dataset], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(datasets, key=lambda d: d.L)

    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        for d in ordered:
            ax.plot(d.js, d.mz, linewidth=1.8, label=d.label)
            ax.fill_between(d.js, d.mz - d.sem, d.mz + d.sem, alpha=0.18)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title("Magnetizacao vs desordem (QAS, cadeia fechada) - 1000 realizacoes")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(ncol=3)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        for d in ordered:
            ax.plot(d.js, d.mz, linewidth=1.8, label=d.label)
            ax.fill_between(d.js, d.mz - d.sem, d.mz + d.sem, alpha=0.18)
        ax.set_xlim(*ZOOM_X)
        ax.set_ylim(*ZOOM_Y)
        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetizacao (⟨Mz⟩)")
        ax.set_title("Zoom em baixa desordem (QAS, cadeia fechada) - 1000 realizacoes")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(ncol=3)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plota resultados QAS (ClosedBC) para L=4..12 e Nrea=1000.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Pasta com arquivos QAS .npz")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "magnetizacao_vs_desordem_closedbc_qas_Nrea1000_L4aL12.pdf"),
        help="PDF de saida",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()

    try:
        datasets = discover_qas_datasets(data_dir)
        if not datasets:
            raise FileNotFoundError(
                f"Nenhum dataset QAS encontrado em {data_dir}. "
                "Esperado: arquivo com 'QAS' no nome e chaves Js/Magnetization/Std_Mz, "
                "ou arquivos aggregated_up_sv.npz."
            )

        target_nrea = 1000
        datasets = [d for d in datasets if d.nrea == target_nrea and 4 <= d.L <= 12]
        if not datasets:
            raise RuntimeError(
                "Nenhum dataset QAS com Nrea=1000 e L entre 4 e 12 foi encontrado."
            )

        by_l = {}
        for d in datasets:
            by_l[d.L] = d
        needed_ls = list(range(4, 13))
        missing_ls = [L for L in needed_ls if L not in by_l]
        if missing_ls:
            raise RuntimeError(
                f"Faltam resultados QAS para L={missing_ls} (Nrea=1000) em {data_dir}."
            )

        ordered = [by_l[L] for L in needed_ls]
        plot_pdf_full_and_zoom(ordered, output)

        print("Arquivo gerado:")
        print(f"- {output}")
        return 0
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
