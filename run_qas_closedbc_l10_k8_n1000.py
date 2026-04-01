#!/usr/bin/env python3
"""Executa QAS (somente) para cadeia fechada em L=10, K=8, Nrea=1000.

Usa o mesmo grid Js do arquivo exato ClosedBC e salva:
- Js
- Magnetization (media por DeltaJ)
- Std_Mz (desvio padrao por DeltaJ)

Com checkpoint/resume para execucoes longas.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse.linalg import expm_multiply

from calibrate_qas_closedbc import (
    build_h_sparse,
    make_precomp,
    sample_j_bonds_closed,
    stable_seed,
)


# Fixed request parameters
L = 10
K = 8
N_REALIZ = 1000
J = 1.0
H = 0.6
J2 = 0.3
FINAL_T = 400.0

ROOT = Path(__file__).resolve().parents[1]
GRID_SOURCE = ROOT / "data" / "ClosedBC" / "Mz_Nqu=10_Nrea=1000_ClosedBC.npz"
OUT_DIR = ROOT / "data" / "ClosedBC"
OUT_PATH = OUT_DIR / "Mz_Nqu=10_Nrea=1000_K=8_ClosedBC_QAS.npz"

N_WORKERS = int(os.environ.get("N_WORKERS", "10"))
CHUNKSIZE = int(os.environ.get("CHUNKSIZE", "6"))
BASE_SEED = int(os.environ.get("GLOBAL_SEED", "1234"))

_G: Dict[str, object] = {}


def _init_worker(precomp, J_value: float, J2_value: float, final_t: float) -> None:
    _G["pre"] = precomp
    _G["J"] = float(J_value)
    _G["J2"] = float(J2_value)
    _G["final_t"] = float(final_t)


def _run_one(job: Tuple[float, int]) -> float:
    delta_j, seed = job
    pre = _G["pre"]
    J_value = _G["J"]
    J2_value = _G["J2"]
    final_t = _G["final_t"]

    rng = np.random.default_rng(int(seed))
    j_bonds = sample_j_bonds_closed(pre.L, float(J_value), float(delta_j), rng)
    Hs = build_h_sparse(pre, j_bonds, float(J2_value))

    v0 = np.zeros(Hs.shape[0], dtype=complex)
    v0[0] = 1.0  # |0...0>
    psi_t = expm_multiply((-1j * float(final_t)) * Hs, v0)
    probs = np.abs(psi_t) ** 2
    return float(probs @ pre.mz_eigs)


def load_js_grid(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de grade nao encontrado: {path}")
    data = np.load(path, allow_pickle=False)
    if "Js" not in data:
        raise KeyError(f"Arquivo sem chave 'Js': {path}")
    js = np.asarray(data["Js"], dtype=float).reshape(-1)
    js = np.unique(js)
    js.sort()
    return js


def load_or_init_buffers(js: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(js)
    means = np.full(n, np.nan, dtype=float)
    stds = np.full(n, np.nan, dtype=float)
    done = np.zeros(n, dtype=bool)

    if OUT_PATH.exists():
        old = np.load(OUT_PATH, allow_pickle=False)
        if "Js" in old and np.array_equal(np.asarray(old["Js"], dtype=float), js):
            if "Magnetization" in old:
                means = np.asarray(old["Magnetization"], dtype=float).reshape(-1)
            if "Std_Mz" in old:
                stds = np.asarray(old["Std_Mz"], dtype=float).reshape(-1)
            if "done_mask" in old:
                done = np.asarray(old["done_mask"], dtype=bool).reshape(-1)
            else:
                done = np.isfinite(means)
    return means, stds, done


def save_checkpoint(js: np.ndarray, means: np.ndarray, stds: np.ndarray, done: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        Js=js,
        Magnetization=means,
        Std_Mz=stds,
        done_mask=done,
        Nrea=int(N_REALIZ),
        L=int(L),
        K=int(K),
        boundary="closed",
        method="QAS",
        initial_state="0" * L,
        J=float(J),
        h=float(H),
        J2=float(J2),
        FINAL_T=float(FINAL_T),
    )
    os.replace(tmp, OUT_PATH)


def main() -> int:
    mp.freeze_support()
    js = load_js_grid(GRID_SOURCE)
    means, stds, done = load_or_init_buffers(js)

    print(f"Grid size: {len(js)} | Js min={js.min():.3f} max={js.max():.3f}")
    print(f"Params: L={L} K={K} Nrea={N_REALIZ} FINAL_T={FINAL_T}")
    print(f"Workers: {N_WORKERS} chunksize={CHUNKSIZE}")
    print(f"Output: {OUT_PATH}")

    pre = make_precomp(L, K, H)
    pending = [i for i in range(len(js)) if not done[i]]
    print(f"Pendentes: {len(pending)} / {len(js)}")

    if not pending:
        print("Nada a fazer: arquivo ja completo.")
        return 0

    t0_all = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=max(1, N_WORKERS),
        initializer=_init_worker,
        initargs=(pre, J, J2, FINAL_T),
    ) as pool:
        for k, idx in enumerate(pending, start=1):
            dj = float(js[idx])
            t0 = time.time()
            jobs = [(dj, stable_seed(L, dj, r, BASE_SEED)) for r in range(N_REALIZ)]
            vals = np.fromiter(
                pool.imap_unordered(_run_one, jobs, chunksize=max(1, CHUNKSIZE)),
                dtype=float,
                count=N_REALIZ,
            )

            means[idx] = float(np.mean(vals))
            stds[idx] = float(np.std(vals, ddof=1 if len(vals) > 1 else 0))
            done[idx] = True

            if (k % 3) == 0 or k == len(pending):
                save_checkpoint(js, means, stds, done)

            dt = time.time() - t0
            print(
                f"[{k:03d}/{len(pending):03d}] dJ={dj:5.2f} "
                f"mean={means[idx]: .6f} std={stds[idx]: .6f} dt={dt:6.1f}s"
            )

    save_checkpoint(js, means, stds, done)
    print(f"Concluido em {(time.time() - t0_all)/3600.0:.2f} h")
    print(f"Arquivo salvo em: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
