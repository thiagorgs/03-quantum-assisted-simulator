#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # bom pra cluster/sem display
import matplotlib.pyplot as plt


NPZ_PATTERN = re.compile(
    r".*/results_L(?P<L>\d+)_R(?P<R>\d+)/experiment_deltaJ_(?P<dj>[0-9]+\.[0-9]+)\.npz$"
)

REQ_KEYS = ("delta_J", "magnetization_mean", "magnetization_std", "n_realizations", "NQ", "K", "FINAL_T")


def _load_npz_from_bytes(b: bytes) -> dict:
    npz = np.load(io.BytesIO(b))
    missing = [k for k in REQ_KEYS if k not in npz.files]
    if missing:
        raise KeyError(f"Faltando chaves no .npz: {missing}. Chaves disponíveis: {npz.files}")
    return {
        "L": int(npz["NQ"]),
        "R": int(npz["n_realizations"]),
        "delta_J": float(npz["delta_J"]),
        "mag_mean": float(npz["magnetization_mean"]),
        "mag_std": float(npz["magnetization_std"]),
        "K": int(npz["K"]),
        "FINAL_T": float(npz["FINAL_T"]),
        # extras (se existirem)
        "J": float(npz["J"]) if "J" in npz.files else np.nan,
        "h": float(npz["h"]) if "h" in npz.files else np.nan,
        "J2": float(npz["J2"]) if "J2" in npz.files else np.nan,
    }


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Lê tanto:
      - um .zip (como o seu Dados-Thiago.zip)
      - quanto um diretório já extraído (contendo results_L*_R*/experiment_deltaJ_*.npz)
    """
    rows = []

    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            for name in zf.namelist():
                if not NPZ_PATTERN.match(name):
                    continue
                rows.append(_load_npz_from_bytes(zf.read(name)))
    else:
        root = input_path
        for npz_path in root.rglob("results_L*_R*/experiment_deltaJ_*.npz"):
            b = npz_path.read_bytes()
            rows.append(_load_npz_from_bytes(b))

    if not rows:
        raise RuntimeError(
            f"Não achei .npz no formato esperado em: {input_path}\n"
            "Esperava algo como results_L4_R1000/experiment_deltaJ_2.00.npz"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["L", "R", "delta_J"]).reset_index(drop=True)
    return df


def _band_from_std(std: np.ndarray, R: int, mode: str) -> np.ndarray:
    if mode == "none":
        return None
    if mode == "std":
        return std
    if mode == "sem":
        return std / np.sqrt(R)
    raise ValueError(f"Modo de erro inválido: {mode}")


def plot_by_L(df: pd.DataFrame, outdir: Path, err_mode: str, formats: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L].sort_values("delta_J")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        for R in sorted(sub["R"].unique()):
            s = sub[sub["R"] == R].sort_values("delta_J")
            x = s["delta_J"].to_numpy()
            y = s["mag_mean"].to_numpy()
            std = s["mag_std"].to_numpy()

            band = _band_from_std(std, int(R), err_mode)
            line = ax.plot(x, y, label=f"R={R}")[0]
            if band is not None:
                c = line.get_color()
                ax.fill_between(x, y - band, y + band, alpha=0.15, color=c)

        T = float(sub["FINAL_T"].iloc[0])
        K = int(sub["K"].iloc[0])

        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetização média")
        ax.set_title(f"L={L} — QAS statevector (K={K}, T={T})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        for fmt in formats:
            fig.savefig(outdir / f"L{L:02d}_mag_vs_DeltaJ.{fmt}", dpi=200)
        plt.close(fig)


def plot_by_R(df: pd.DataFrame, outdir: Path, formats: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for R in sorted(df["R"].unique()):
        sub = df[df["R"] == R].sort_values("delta_J")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.6, 4.6))

        for L in sorted(sub["L"].unique()):
            s = sub[sub["L"] == L].sort_values("delta_J")
            x = s["delta_J"].to_numpy()
            y = s["mag_mean"].to_numpy()
            ax.plot(x, y, label=f"L={L}")

        T = float(sub["FINAL_T"].iloc[0])
        K = int(sub["K"].iloc[0])

        ax.set_xlabel("ΔJ")
        ax.set_ylabel("Magnetização média")
        ax.set_title(f"R={R} — QAS statevector (K={K}, T={T})")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)

        fig.tight_layout()
        for fmt in formats:
            fig.savefig(outdir / f"R{R}_mag_vs_DeltaJ.{fmt}", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="Dados-Thiago.zip",
                    help="Caminho para o .zip OU para a pasta extraída (ex: Dados-Thiago/).")
    ap.add_argument("--out", type=str, default="plots",
                    help="Pasta de saída dos plots.")
    ap.add_argument("--mode", type=str, default="both", choices=["by_L", "by_R", "both"],
                    help="by_L: uma figura por L com curvas de R; by_R: uma figura por R com curvas de L; both: os dois.")
    ap.add_argument("--error", type=str, default="sem", choices=["none", "std", "sem"],
                    help="Banda de erro em by_L: none, std, ou sem=std/sqrt(R).")
    ap.add_argument("--formats", type=str, default="png",
                    help="Lista separada por vírgula. Ex: png ou png,pdf")
    ap.add_argument("--dump_csv", action="store_true",
                    help="Se ligado, salva também um CSV (data_long.csv) com todos os pontos.")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    df = load_data(input_path)

    if args.dump_csv:
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "data_long.csv", index=False)

    if args.mode in ("by_L", "both"):
        plot_by_L(df, outdir / "by_L", err_mode=args.error, formats=formats)
    if args.mode in ("by_R", "both"):
        plot_by_R(df, outdir / "by_R", formats=formats)

    print(f"OK. Plots salvos em: {outdir}")


if __name__ == "__main__":
    main()