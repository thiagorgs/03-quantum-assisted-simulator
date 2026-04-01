#!/usr/bin/env python3
"""Executa QAS (somente) para um unico L/K em cadeia fechada com checkpoint/resume."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse.linalg import expm_multiply

from calibrate_qas_closedbc import build_h_sparse, make_precomp, sample_j_bonds_closed, stable_seed


_G: Dict[str, object] = {}


def _init_worker(precomp, j_value: float, j2_value: float, final_t: float) -> None:
    _G["pre"] = precomp
    _G["J"] = float(j_value)
    _G["J2"] = float(j2_value)
    _G["final_t"] = float(final_t)


def _run_one(job: Tuple[float, int]) -> float:
    delta_j, seed = job
    pre = _G["pre"]
    J = _G["J"]
    J2 = _G["J2"]
    final_t = _G["final_t"]

    rng = np.random.default_rng(int(seed))
    j_bonds = sample_j_bonds_closed(pre.L, float(J), float(delta_j), rng)
    Hs = build_h_sparse(pre, j_bonds, float(J2))
    v0 = np.zeros(Hs.shape[0], dtype=complex)
    v0[0] = 1.0
    psi_t = expm_multiply((-1j * float(final_t)) * Hs, v0)
    probs = np.abs(psi_t) ** 2
    return float(probs @ pre.mz_eigs)


def _run_serial_one(pre, delta_j: float, seed: int, j_value: float, j2_value: float, final_t: float) -> float:
    rng = np.random.default_rng(int(seed))
    j_bonds = sample_j_bonds_closed(pre.L, float(j_value), float(delta_j), rng)
    Hs = build_h_sparse(pre, j_bonds, float(j2_value))
    v0 = np.zeros(Hs.shape[0], dtype=complex)
    v0[0] = 1.0
    psi_t = expm_multiply((-1j * float(final_t)) * Hs, v0)
    probs = np.abs(psi_t) ** 2
    return float(probs @ pre.mz_eigs)


def main() -> int:
    parser = argparse.ArgumentParser(description="QAS single-run for closed BC.")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--nrea", type=int, default=1000)
    parser.add_argument("--final-t", type=float, default=400.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.6)
    parser.add_argument("--J2", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("N_WORKERS", "10")))
    parser.add_argument("--chunksize", type=int, default=int(os.environ.get("CHUNKSIZE", "6")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("GLOBAL_SEED", "1234")))
    parser.add_argument("--data-dir", type=str, default="data/ClosedBC")
    parser.add_argument("--basis-mode", type=str, default="k_moment", choices=["k_moment", "x_weight"])
    parser.add_argument("--normalize-by-l", action="store_true", help="Salva Mz e Std_Mz normalizados por L.")
    parser.add_argument("--parity-test", action="store_true", help="Roda teste rapido QAS vs exato antes da producao.")
    parser.add_argument(
        "--parity-dj",
        type=float,
        nargs="*",
        default=[0.5, 2.0, 4.0],
        help="DeltaJ usados no parity test.",
    )
    parser.add_argument("--parity-nrea", type=int, default=20, help="Numero de realizacoes no parity test.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve()
    grid_source = data_dir / f"Mz_Nqu={args.L}_Nrea=1000_ClosedBC.npz"
    out_path = data_dir / f"Mz_Nqu={args.L}_Nrea={args.nrea}_K={args.K}_ClosedBC_QAS.npz"

    if not grid_source.exists():
        raise FileNotFoundError(f"Grid source nao encontrado: {grid_source}")

    with np.load(grid_source, allow_pickle=False) as dsrc:
        if "Js" not in dsrc.files:
            raise KeyError(f"Arquivo sem chave Js: {grid_source}")
        js = np.asarray(dsrc["Js"], dtype=float).reshape(-1)
    js = np.unique(js)
    js.sort()

    means = np.full(len(js), np.nan, dtype=float)
    stds = np.full(len(js), np.nan, dtype=float)
    done = np.zeros(len(js), dtype=bool)
    if out_path.exists():
        with np.load(out_path, allow_pickle=False) as dold:
            if "Js" in dold.files and np.array_equal(np.asarray(dold["Js"], dtype=float), js):
                if "Magnetization" in dold.files:
                    means = np.asarray(dold["Magnetization"], dtype=float).reshape(-1)
                if "Std_Mz" in dold.files:
                    stds = np.asarray(dold["Std_Mz"], dtype=float).reshape(-1)
                if "done_mask" in dold.files:
                    done = np.asarray(dold["done_mask"], dtype=bool).reshape(-1)
                else:
                    done = np.isfinite(means)

    pending = [i for i in range(len(js)) if not done[i]]
    print(f"Grid size: {len(js)} | pending: {len(pending)}")
    print(f"L={args.L} K={args.K} Nrea={args.nrea} FINAL_T={args.final_t}")
    print(f"Output: {out_path}")

    if not pending:
        print("Nada a fazer: output ja completo.")
        return 0

    pre = make_precomp(
        args.L,
        args.K,
        args.h,
        basis_mode=args.basis_mode,
        include_j2=(abs(args.J2) > 0.0),
    )
    print(f"Basis mode: {args.basis_mode} | basis_dim: {len(pre.basis_masks)}")

    if args.parity_test:
        pre_exact = make_precomp(
            args.L,
            args.L,
            args.h,
            basis_mode="x_weight",
            include_j2=(abs(args.J2) > 0.0),
        )
        print(f"Parity test: exact basis_dim={len(pre_exact.basis_masks)}")
        abs_diffs: List[float] = []
        for dj in args.parity_dj:
            q_vals = []
            e_vals = []
            for r in range(args.parity_nrea):
                seed = stable_seed(args.L, float(dj), r, args.seed)
                q_vals.append(_run_serial_one(pre, float(dj), seed, args.J, args.J2, args.final_t))
                e_vals.append(_run_serial_one(pre_exact, float(dj), seed, args.J, args.J2, args.final_t))
            q_vals = np.asarray(q_vals, dtype=float)
            e_vals = np.asarray(e_vals, dtype=float)
            md = float(np.mean(np.abs(q_vals - e_vals)))
            abs_diffs.append(md)
            print(
                f"[parity] dJ={dj:.2f} "
                f"mean_qas={float(np.mean(q_vals)):.6f} "
                f"mean_exact={float(np.mean(e_vals)):.6f} "
                f"mean_abs_diff={md:.6e}"
            )
        print(f"[parity] worst_mean_abs_diff={float(np.max(abs_diffs)):.6e}")

    def save_ckpt() -> None:
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            Js=js,
            Magnetization=means,
            Std_Mz=stds,
            done_mask=done,
            Nrea=int(args.nrea),
            L=int(args.L),
            K=int(args.K),
            basis_mode=str(args.basis_mode),
            basis_dim=int(len(pre.basis_masks)),
            boundary="closed",
            method="QAS",
            initial_state="0" * int(args.L),
            J=float(args.J),
            h=float(args.h),
            J2=float(args.J2),
            FINAL_T=float(args.final_t),
        )
        os.replace(tmp, out_path)

    t0_all = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=max(1, args.workers),
        initializer=_init_worker,
        initargs=(pre, args.J, args.J2, args.final_t),
    ) as pool:
        for k, idx in enumerate(pending, start=1):
            dj = float(js[idx])
            t0 = time.time()
            jobs = [(dj, stable_seed(args.L, dj, r, args.seed)) for r in range(args.nrea)]
            vals = np.fromiter(
                pool.imap_unordered(_run_one, jobs, chunksize=max(1, args.chunksize)),
                dtype=float,
                count=args.nrea,
            )
            means[idx] = float(np.mean(vals))
            stds[idx] = float(np.std(vals, ddof=1 if len(vals) > 1 else 0))
            done[idx] = True

            if (k % 3) == 0 or k == len(pending):
                save_ckpt()
            print(
                f"[{k:03d}/{len(pending):03d}] dJ={dj:5.2f} "
                f"mean={means[idx]: .6f} std={stds[idx]: .6f} dt={time.time()-t0:6.1f}s"
            )

    save_ckpt()
    if args.normalize_by_l:
        with np.load(out_path, allow_pickle=False) as d:
            js_n = np.asarray(d["Js"], dtype=float).reshape(-1)
            means_n = np.asarray(d["Magnetization"], dtype=float).reshape(-1) / float(args.L)
            stds_n = np.asarray(d["Std_Mz"], dtype=float).reshape(-1) / float(args.L)
            done_n = np.asarray(d["done_mask"], dtype=bool).reshape(-1)
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            Js=js_n,
            Magnetization=means_n,
            Std_Mz=stds_n,
            done_mask=done_n,
            Nrea=int(args.nrea),
            L=int(args.L),
            K=int(args.K),
            basis_mode=str(args.basis_mode),
            basis_dim=int(len(pre.basis_masks)),
            boundary="closed",
            method="QAS",
            initial_state="0" * int(args.L),
            J=float(args.J),
            h=float(args.h),
            J2=float(args.J2),
            FINAL_T=float(args.final_t),
            normalized_by_L=1,
        )
        os.replace(tmp, out_path)
        print("Output normalizado por L.")

    print(f"Concluido em {(time.time()-t0_all)/3600.0:.2f} h")
    print(f"Arquivo salvo em: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
