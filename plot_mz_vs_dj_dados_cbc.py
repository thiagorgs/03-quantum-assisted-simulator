#!/usr/bin/env python3
"""Gera plots de magnetizacao para o conjunto em data/Dados - CBC."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from plot_mz_vs_dj_closedbc import align_by_js, plot_convergence, plot_pdf_full_and_zoom


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "Dados - CBC"
OUTPUT_DIR = ROOT / "outputs"
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
        return self.std_mz / math.sqrt(float(self.nrea))


def load_npz_flexible(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "Js" not in data.files:
        raise ValueError(f"Arquivo sem Js: {path}")

    js = np.asarray(data["Js"], dtype=float).reshape(-1)
    if "Magnetization" in data.files and "Std_Mz" in data.files:
        mz = np.asarray(data["Magnetization"], dtype=float).reshape(-1)
        std = np.asarray(data["Std_Mz"], dtype=float).reshape(-1)
    elif "Mean_Mz" in data.files and "Var_Mz" in data.files:
        mz = np.asarray(data["Mean_Mz"], dtype=float).reshape(-1)
        var = np.asarray(data["Var_Mz"], dtype=float).reshape(-1)
        std = np.sqrt(np.clip(var, 0.0, None))
    else:
        raise ValueError(
            f"Schema nao suportado em {path}. Chaves encontradas: {list(data.files)}"
        )

    if not (js.shape == mz.shape == std.shape):
        raise ValueError(f"Shapes inconsistentes em {path}: Js={js.shape}, Mz={mz.shape}, Std={std.shape}")
    order = np.argsort(js)
    return js[order], mz[order], std[order]


def discover_dados_cbc(search_root: Path) -> Dict[int, List[Dataset]]:
    by_nrea: Dict[int, List[Dataset]] = {}
    for path in sorted(search_root.rglob("*_ClosedBC.npz")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        L = int(m.group(1))
        nrea = int(m.group(2))
        js, mz, std = load_npz_flexible(path)
        by_nrea.setdefault(nrea, []).append(Dataset(L=L, nrea=nrea, js=js, mz=mz, std_mz=std))
    for nrea in by_nrea:
        by_nrea[nrea].sort(key=lambda d: d.L)
    return by_nrea


def main() -> int:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {DATASET_DIR}")

    by_nrea = discover_dados_cbc(DATASET_DIR)
    if not by_nrea:
        raise FileNotFoundError(f"Nenhum arquivo *_ClosedBC.npz encontrado em: {DATASET_DIR}")

    generated: list[Path] = []
    for nrea in sorted(by_nrea):
        out = OUTPUT_DIR / f"magnetizacao_vs_desordem_closedbc_dados_cbc_Nrea{nrea}.pdf"
        plot_pdf_full_and_zoom(by_nrea[nrea], nrea, out)
        generated.append(out)

    conv_out = OUTPUT_DIR / "convergencia_closedbc_dados_cbc_1000_vs_2000.pdf"
    if plot_convergence(by_nrea, conv_out):
        generated.append(conv_out)

    print("Arquivos gerados:")
    for p in generated:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
