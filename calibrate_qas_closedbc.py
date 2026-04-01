#!/usr/bin/env python3
"""Calibra K do QAS em cadeia fechada comparando com evolucao exata no mesmo protocolo.

Protocolo:
- Estado inicial |0...0>
- Mesmo Hamiltoniano desordenado para QAS e Exato (mesma realizacao)
- Cadeia fechada (periodica) para termos ZZ de 1a e 2a vizinhanca

Saidas:
- CSV resumo com metricas por (L, K)
- CSV com K recomendado por L
- NPZ agregados por (L, K) para QAS e referencia exata
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply


# ----------------------------
# Configuracao (edite aqui)
# ----------------------------
L_VALUES = list(range(4, 13))  # L = 4..12
K_CANDIDATES = [2, 3, 4, 5, 6, 7, 8]
DELTAJ_VALUES = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

N_REALIZ = 60
J = 1.0
H = 0.6
J2 = 0.3
FINAL_T = 400.0
N_TIME = 101
BASE_SEED = 1234

N_WORKERS = int(os.environ.get("N_WORKERS", "6"))
CHUNKSIZE = int(os.environ.get("CHUNKSIZE", "4"))

TOL_MAX_ABS_T = 0.35
TOL_RMSE_T = 0.18
TOL_FINAL_ABS = 0.18

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "qas_closedbc_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_L_VALUES = [4, 6, 8]
PILOT_K_CANDIDATES = [2, 3, 4, 5, 6]
PILOT_DELTAJ_VALUES = [0.5, 4.0, 8.0, 12.0]
PILOT_N_REALIZ = 12
PILOT_N_TIME = 41
PILOT_N_WORKERS = min(4, N_WORKERS)
PILOT_CHUNKSIZE = max(1, CHUNKSIZE)


@dataclass
class Precomp:
    L: int
    K: int
    basis_masks: np.ndarray
    mz_eigs: np.ndarray
    zz1: np.ndarray
    zz2_sum: np.ndarray
    hx: coo_matrix
    basis_mode: str


def stable_seed(L: int, delta_j: float, r: int, base_seed: int) -> int:
    dj_key = int(round(delta_j * 100))
    return int((base_seed * 1_000_003 + L * 10_007 + dj_key * 503 + r) % (2**31 - 1))


def basis_bitmasks_x_only(L: int, K: int) -> np.ndarray:
    K = max(0, min(K, L))
    if K == L:
        return np.arange(1 << L, dtype=np.uint32)

    masks: List[int] = []
    for r in range(K + 1):
        for sites in combinations(range(L), r):
            m = 0
            for s in sites:
                m |= 1 << s
            masks.append(m)
    return np.asarray(masks, dtype=np.uint32)


def _hamiltonian_pauli_terms_closed(L: int, include_j2: bool = True) -> List[np.ndarray]:
    """Pauli strings dos termos do Hamiltoniano (estrutura, sem coeficientes)."""
    # Encoding: 0=I, 1=X, 2=Y, 3=Z (mesmo padrao do codigo do autor).
    terms: List[np.ndarray] = []
    for i in range(L):
        s = np.zeros(L, dtype=np.int8)
        s[i] = 1
        terms.append(s)
    for i in range(L):
        s = np.zeros(L, dtype=np.int8)
        s[i] = 3
        s[(i + 1) % L] = 3
        terms.append(s)
    if include_j2:
        for i in range(L):
            s = np.zeros(L, dtype=np.int8)
            s[i] = 3
            s[(i + 2) % L] = 3
            terms.append(s)
    return terms


def _multiply_pauli_strings(curr: Sequence[np.ndarray], to_mult: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Multiplica strings de Pauli ignorando fase global (como no repo original)."""
    out: List[np.ndarray] = []
    for a in curr:
        for b in to_mult:
            c = np.zeros_like(a)
            for k in range(len(a)):
                ak = int(a[k]); bk = int(b[k])
                if ak == 1 and bk == 2:
                    c[k] = 3
                elif ak == 2 and bk == 1:
                    c[k] = 3
                else:
                    c[k] = abs(ak - bk)
            out.append(c)
    if not out:
        return []
    unique = np.unique(np.asarray(out, dtype=np.int8), axis=0)
    return [row.copy() for row in unique]


def basis_bitmasks_k_moment_closed(L: int, K: int, include_j2: bool = True) -> np.ndarray:
    """Base por expansao K-moment do QAS aplicada ao estado |0...0>."""
    if K < 0:
        raise ValueError("K deve ser >= 0")

    h_terms = _hamiltonian_pauli_terms_closed(L, include_j2=include_j2)
    expand_strings: List[np.ndarray] = [np.zeros(L, dtype=np.int8)]  # identidade

    for _ in range(K):
        expanded = _multiply_pauli_strings(expand_strings, h_terms)
        if expanded:
            # preserva determinismo por ordem lexicografica do unique.
            expand_strings = expanded

    # Aplica string em |0...0>: somente X/Y flipam qubit (Z nao altera bit em |0>).
    masks = set()
    for s in expand_strings:
        mask = 0
        for i, p in enumerate(s):
            if p == 1 or p == 2:
                mask |= 1 << i
        masks.add(mask)

    arr = np.array(sorted(masks), dtype=np.uint32)
    if arr.size == 0:
        arr = np.array([0], dtype=np.uint32)
    return arr


def precompute_z_and_mz(L: int, basis_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = len(basis_masks)
    z = np.empty((m, L), dtype=np.int8)
    for i in range(L):
        bits = (basis_masks >> i) & 1
        z[:, i] = (1 - 2 * bits).astype(np.int8)
    mz_eigs = z.sum(axis=1).astype(np.int16)
    return z, mz_eigs


def precompute_zz_closed(L: int, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = z.shape[0]
    zz1 = np.empty((L, m), dtype=np.int8)
    for i in range(L):
        zz1[i, :] = (z[:, i] * z[:, (i + 1) % L]).astype(np.int8)

    zz2_sum = np.zeros(m, dtype=np.int16)
    for i in range(L):
        zz2_sum += (z[:, i] * z[:, (i + 2) % L]).astype(np.int16)
    return zz1, zz2_sum


def build_hx_sparse(L: int, basis_masks: np.ndarray, h: float) -> coo_matrix:
    m = len(basis_masks)
    full_ordered = (m == (1 << L)) and np.array_equal(
        basis_masks, np.arange(1 << L, dtype=basis_masks.dtype)
    )
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    if full_ordered:
        for s in range(1 << L):
            for i in range(L):
                sp = s ^ (1 << i)
                rows.append(s)
                cols.append(sp)
                data.append(-h)
        return coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()

    idx_map = {int(mask): idx for idx, mask in enumerate(basis_masks)}
    for a, s in enumerate(basis_masks):
        s_int = int(s)
        for i in range(L):
            sp = s_int ^ (1 << i)
            b = idx_map.get(sp)
            if b is not None:
                rows.append(a)
                cols.append(b)
                data.append(-h)
    return coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()


def make_precomp(L: int, K: int, h: float, *, basis_mode: str = "k_moment", include_j2: bool = True) -> Precomp:
    if basis_mode == "k_moment":
        basis_masks = basis_bitmasks_k_moment_closed(L, K, include_j2=include_j2)
    elif basis_mode == "x_weight":
        basis_masks = basis_bitmasks_x_only(L, K)
    else:
        raise ValueError(f"basis_mode invalido: {basis_mode}")
    z, mz_eigs = precompute_z_and_mz(L, basis_masks)
    zz1, zz2_sum = precompute_zz_closed(L, z)
    hx = build_hx_sparse(L, basis_masks, h)
    return Precomp(
        L=L,
        K=K,
        basis_masks=basis_masks,
        mz_eigs=mz_eigs,
        zz1=zz1,
        zz2_sum=zz2_sum,
        hx=hx,
        basis_mode=basis_mode,
    )


def sample_j_bonds_closed(L: int, J: float, delta_j: float, rng: np.random.Generator) -> np.ndarray:
    return J + rng.uniform(-delta_j, delta_j, size=L)


def build_h_sparse(pre: Precomp, j_bonds: np.ndarray, j2: float) -> coo_matrix:
    diag = -(j_bonds @ pre.zz1) - (j2 * pre.zz2_sum)
    return pre.hx + diags(diag, 0, format="csr")


def evolve_mz_trajectory(Hs: coo_matrix, mz_eigs: np.ndarray, final_t: float, n_time: int) -> np.ndarray:
    v0 = np.zeros(Hs.shape[0], dtype=complex)
    v0[0] = 1.0  # |0...0>
    psi_t = expm_multiply((-1j) * Hs, v0, start=0.0, stop=final_t, num=n_time, endpoint=True)
    probs = np.abs(psi_t) ** 2
    return probs @ mz_eigs


_G: Dict[str, object] = {}


def _init_worker(pre_qas: Precomp, pre_exact: Precomp, J: float, J2: float, final_t: float, n_time: int) -> None:
    _G["pre_qas"] = pre_qas
    _G["pre_exact"] = pre_exact
    _G["J"] = float(J)
    _G["J2"] = float(J2)
    _G["final_t"] = float(final_t)
    _G["n_time"] = int(n_time)


def _run_one_realization(job: Tuple[float, int]) -> Tuple[float, float, float, float, float]:
    delta_j, seed = job
    pre_qas = _G["pre_qas"]
    pre_exact = _G["pre_exact"]
    J = _G["J"]
    J2 = _G["J2"]
    final_t = _G["final_t"]
    n_time = _G["n_time"]

    assert isinstance(pre_qas, Precomp)
    assert isinstance(pre_exact, Precomp)

    rng = np.random.default_rng(int(seed))
    j_bonds = sample_j_bonds_closed(pre_qas.L, float(J), float(delta_j), rng)

    H_qas = build_h_sparse(pre_qas, j_bonds, float(J2))
    H_ex = build_h_sparse(pre_exact, j_bonds, float(J2))

    mz_qas_t = evolve_mz_trajectory(H_qas, pre_qas.mz_eigs, float(final_t), int(n_time))
    mz_ex_t = evolve_mz_trajectory(H_ex, pre_exact.mz_eigs, float(final_t), int(n_time))

    diff = mz_qas_t - mz_ex_t
    max_abs_t = float(np.max(np.abs(diff)))
    rmse_t = float(np.sqrt(np.mean(diff * diff)))
    final_abs = float(abs(diff[-1]))
    return float(mz_qas_t[-1]), float(mz_ex_t[-1]), max_abs_t, rmse_t, final_abs


def _aggregate_metrics(vals: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(vals)), float(np.percentile(vals, 95))


def run_one_l_k(
    L: int,
    K: int,
    delta_j_values: Sequence[float],
    n_realiz: int,
    base_seed: int,
    J: float,
    h: float,
    J2: float,
    final_t: float,
    n_time: int,
    n_workers: int,
    chunksize: int,
) -> Dict[str, object]:
    pre_qas = make_precomp(L, K, h, basis_mode="k_moment", include_j2=(abs(J2) > 0.0))
    pre_exact = make_precomp(L, L, h, basis_mode="x_weight", include_j2=(abs(J2) > 0.0))

    qas_mean_by_dj: List[float] = []
    qas_std_by_dj: List[float] = []
    ex_mean_by_dj: List[float] = []
    ex_std_by_dj: List[float] = []
    max_abs_mean_by_dj: List[float] = []
    max_abs_p95_by_dj: List[float] = []
    rmse_mean_by_dj: List[float] = []
    rmse_p95_by_dj: List[float] = []
    final_abs_mean_by_dj: List[float] = []
    final_abs_p95_by_dj: List[float] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=max(1, n_workers),
        initializer=_init_worker,
        initargs=(pre_qas, pre_exact, J, J2, final_t, n_time),
    ) as pool:
        for dj in delta_j_values:
            jobs = [(float(dj), stable_seed(L, float(dj), r, base_seed)) for r in range(n_realiz)]
            out = list(pool.imap_unordered(_run_one_realization, jobs, chunksize=max(1, chunksize)))
            arr = np.asarray(out, dtype=float)
            qas_final = arr[:, 0]
            ex_final = arr[:, 1]
            max_abs_t = arr[:, 2]
            rmse_t = arr[:, 3]
            final_abs = arr[:, 4]

            qas_mean_by_dj.append(float(np.mean(qas_final)))
            qas_std_by_dj.append(float(np.std(qas_final, ddof=1 if len(qas_final) > 1 else 0)))
            ex_mean_by_dj.append(float(np.mean(ex_final)))
            ex_std_by_dj.append(float(np.std(ex_final, ddof=1 if len(ex_final) > 1 else 0)))

            m1, p1 = _aggregate_metrics(max_abs_t)
            m2, p2 = _aggregate_metrics(rmse_t)
            m3, p3 = _aggregate_metrics(final_abs)
            max_abs_mean_by_dj.append(m1)
            max_abs_p95_by_dj.append(p1)
            rmse_mean_by_dj.append(m2)
            rmse_p95_by_dj.append(p2)
            final_abs_mean_by_dj.append(m3)
            final_abs_p95_by_dj.append(p3)

    js = np.asarray(delta_j_values, dtype=float)
    qas_mean = np.asarray(qas_mean_by_dj, dtype=float)
    qas_std = np.asarray(qas_std_by_dj, dtype=float)
    ex_mean = np.asarray(ex_mean_by_dj, dtype=float)
    ex_std = np.asarray(ex_std_by_dj, dtype=float)

    worst_max_abs_mean = float(np.max(max_abs_mean_by_dj))
    worst_rmse_mean = float(np.max(rmse_mean_by_dj))
    worst_final_abs_mean = float(np.max(final_abs_mean_by_dj))

    return {
        "L": L,
        "K": K,
        "Js": js,
        "QAS_mean": qas_mean,
        "QAS_std": qas_std,
        "Exact_mean": ex_mean,
        "Exact_std": ex_std,
        "max_abs_mean_by_dj": np.asarray(max_abs_mean_by_dj),
        "max_abs_p95_by_dj": np.asarray(max_abs_p95_by_dj),
        "rmse_mean_by_dj": np.asarray(rmse_mean_by_dj),
        "rmse_p95_by_dj": np.asarray(rmse_p95_by_dj),
        "final_abs_mean_by_dj": np.asarray(final_abs_mean_by_dj),
        "final_abs_p95_by_dj": np.asarray(final_abs_p95_by_dj),
        "worst_max_abs_mean": worst_max_abs_mean,
        "worst_rmse_mean": worst_rmse_mean,
        "worst_final_abs_mean": worst_final_abs_mean,
    }


def choose_k(rows: List[Dict[str, object]]) -> int:
    for row in sorted(rows, key=lambda x: int(x["K"])):
        if (
            float(row["worst_max_abs_mean"]) <= TOL_MAX_ABS_T
            and float(row["worst_rmse_mean"]) <= TOL_RMSE_T
            and float(row["worst_final_abs_mean"]) <= TOL_FINAL_ABS
        ):
            return int(row["K"])
    best = min(rows, key=lambda x: (float(x["worst_rmse_mean"]), float(x["worst_final_abs_mean"])))
    return int(best["K"])


def save_npz_outputs(row: Dict[str, object], n_realiz: int) -> None:
    L = int(row["L"])
    K = int(row["K"])
    js = np.asarray(row["Js"], dtype=float)

    qas_path = OUT_DIR / f"Mz_Nqu={L}_Nrea={n_realiz}_K={K}_ClosedBC_QAS.npz"
    np.savez_compressed(
        qas_path,
        Js=js,
        Magnetization=np.asarray(row["QAS_mean"], dtype=float),
        Std_Mz=np.asarray(row["QAS_std"], dtype=float),
        Nrea=int(n_realiz),
        L=L,
        K=K,
        boundary="closed",
        method="QAS",
        initial_state="0" * L,
    )

    ex_path = OUT_DIR / f"Mz_Nqu={L}_Nrea={n_realiz}_K={K}_ClosedBC_ExactRef.npz"
    np.savez_compressed(
        ex_path,
        Js=js,
        Magnetization=np.asarray(row["Exact_mean"], dtype=float),
        Std_Mz=np.asarray(row["Exact_std"], dtype=float),
        Nrea=int(n_realiz),
        L=L,
        K=K,
        boundary="closed",
        method="ExactReference",
        initial_state="0" * L,
    )


def write_csv_summary(all_rows: List[Dict[str, object]]) -> Path:
    out = OUT_DIR / "qas_closedbc_k_calibration_summary.csv"
    fields = [
        "L",
        "K",
        "worst_max_abs_mean",
        "worst_rmse_mean",
        "worst_final_abs_mean",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(all_rows, key=lambda x: (int(x["L"]), int(x["K"]))):
            w.writerow({k: row[k] for k in fields})
    return out


def write_csv_recommendations(reco_rows: List[Dict[str, object]]) -> Path:
    out = OUT_DIR / "qas_closedbc_k_recomendado.csv"
    fields = ["L", "K_recomendado", "criterio"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(reco_rows, key=lambda x: int(x["L"])):
            w.writerow(row)
    return out


def main() -> int:
    mp.freeze_support()
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Calibracao de K para QAS em cadeia fechada (com referencia exata)."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Roda um modo leve para validacao rapida "
            "(L/K/DeltaJ/Nreal/Ntime reduzidos)."
        ),
    )
    parser.add_argument(
        "--l-values",
        type=str,
        default="",
        help="Lista de L separada por virgula (ex.: 9,10). Sobrescreve modo normal/pilot para L.",
    )
    args = parser.parse_args()

    if args.pilot:
        l_values = PILOT_L_VALUES
        k_candidates = PILOT_K_CANDIDATES
        deltaj_values = PILOT_DELTAJ_VALUES
        n_realiz = PILOT_N_REALIZ
        n_time = PILOT_N_TIME
        n_workers = PILOT_N_WORKERS
        chunksize = PILOT_CHUNKSIZE
    else:
        l_values = L_VALUES
        k_candidates = K_CANDIDATES
        deltaj_values = DELTAJ_VALUES
        n_realiz = N_REALIZ
        n_time = N_TIME
        n_workers = N_WORKERS
        chunksize = CHUNKSIZE

    if args.l_values.strip():
        l_values = [int(x.strip()) for x in args.l_values.split(",") if x.strip()]

    print("Calibracao QAS (cadeia fechada) iniciada.")
    print(f"Modo pilot: {args.pilot}")
    print(f"L: {l_values}")
    print(f"K candidatos: {k_candidates}")
    print(f"DeltaJ: {deltaj_values}")
    print(f"N_realiz: {n_realiz} | N_time: {n_time} | N_workers: {n_workers}")

    all_rows: List[Dict[str, object]] = []
    reco_rows: List[Dict[str, object]] = []

    for L in l_values:
        rows_l: List[Dict[str, object]] = []
        print(f"\n[L={L}]")
        for K in [k for k in k_candidates if k <= L]:
            print(f"  - Rodando K={K} ...")
            row = run_one_l_k(
                L=L,
                K=K,
                delta_j_values=deltaj_values,
                n_realiz=n_realiz,
                base_seed=BASE_SEED,
                J=J,
                h=H,
                J2=J2,
                final_t=FINAL_T,
                n_time=n_time,
                n_workers=n_workers,
                chunksize=chunksize,
            )
            save_npz_outputs(row, n_realiz)
            rows_l.append(row)
            all_rows.append(row)
            print(
                "    "
                f"worst(max|DeltaMz(t)|)={row['worst_max_abs_mean']:.4f}, "
                f"worst(RMSE_t)={row['worst_rmse_mean']:.4f}, "
                f"worst(|DeltaMz_final|)={row['worst_final_abs_mean']:.4f}"
            )

        if not rows_l:
            continue
        k_rec = choose_k(rows_l)
        reco_rows.append(
            {
                "L": L,
                "K_recomendado": k_rec,
                "criterio": (
                    f"tol_max={TOL_MAX_ABS_T}, tol_rmse={TOL_RMSE_T}, "
                    f"tol_final={TOL_FINAL_ABS}"
                ),
            }
        )
        print(f"  -> K recomendado para L={L}: {k_rec}")

    summary_path = write_csv_summary(all_rows)
    rec_path = write_csv_recommendations(reco_rows)

    print("\nArquivos gerados:")
    print(f"- {summary_path}")
    print(f"- {rec_path}")
    print(f"- {OUT_DIR}/*.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
