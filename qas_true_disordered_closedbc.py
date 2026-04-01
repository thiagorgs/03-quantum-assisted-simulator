#!/usr/bin/env python3
"""QAS verdadeiro (estilo autor) para Ising desordenado fechado.

Inclui:
- K-scan com fidelidade QAS vs exato (paridade estrita, mesma realizacao)
- Producao QAS e, opcionalmente, ExactRef no mesmo protocolo fisico
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
from qutip_qip.operations import csign, rx, ry, rz
import qutip as qt


PAULI_MATS = {
    0: np.array([[1, 0], [0, 1]], dtype=complex),
    1: np.array([[0, 1], [1, 0]], dtype=complex),
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),
    3: np.array([[1, 0], [0, -1]], dtype=complex),
}
EXPECTED_DJ_MIN = 0.0
EXPECTED_DJ_MAX = 12.0
EXPECTED_DJ_STEP = 0.1
EXPECTED_DJ_N = int(round((EXPECTED_DJ_MAX - EXPECTED_DJ_MIN) / EXPECTED_DJ_STEP)) + 1
ALT_DJ_MIN = 0.1
ALT_DJ_MAX = 11.9
ALT_DJ_N = 151


def kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, mats)


def pauli_string_matrix(code: Sequence[int]) -> np.ndarray:
    return kron_all([PAULI_MATS[int(c)] for c in code])


def multiply_paulis(curr_paulis: Sequence[np.ndarray], to_mult_paulis: Sequence[np.ndarray]) -> List[np.ndarray]:
    new_paulis = []
    for a in curr_paulis:
        for b in to_mult_paulis:
            c = np.zeros(len(a), dtype=int)
            for k in range(len(a)):
                ak, bk = int(a[k]), int(b[k])
                if ak == 1 and bk == 2:
                    c[k] = 3
                elif ak == 2 and bk == 1:
                    c[k] = 3
                else:
                    c[k] = abs(ak - bk)
            new_paulis.append(c)
    if not new_paulis:
        return []
    uniq = np.unique(np.asarray(new_paulis, dtype=int), axis=0)
    return [u.copy() for u in uniq]


def build_model_strings(L: int, include_j2: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Retorna strings de pauli para termos X, ZZ_nn, ZZ_nnn (periodico)."""
    x_terms: List[np.ndarray] = []
    nn_terms: List[np.ndarray] = []
    nnn_terms: List[np.ndarray] = []

    for i in range(L):
        s = np.zeros(L, dtype=int)
        s[i] = 1
        x_terms.append(s)
    for i in range(L):
        s = np.zeros(L, dtype=int)
        s[i] = 3
        s[(i + 1) % L] = 3
        nn_terms.append(s)
    if include_j2:
        for i in range(L):
            s = np.zeros(L, dtype=int)
            s[i] = 3
            s[(i + 2) % L] = 3
            nnn_terms.append(s)
    return x_terms, nn_terms, nnn_terms


def get_initial_state(ini_type: int, n_qubits: int, depth: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if ini_type == 1:
        st = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
    elif ini_type == 2:
        rand_angles = rng.random((depth, n_qubits)) * 2 * np.pi
        rand_pauli = rng.integers(1, 4, size=(depth, n_qubits))
        st = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
        st = qt.tensor([ry(np.pi / 4) for _ in range(n_qubits)]) * st
        ent_layer = reduce(lambda a, b: a * b, [csign(n_qubits, i, (i + 1) % n_qubits) for i in range(n_qubits)])
        for d in range(depth):
            rots = []
            for q in range(n_qubits):
                p = int(rand_pauli[d][q])
                angle = float(rand_angles[d][q])
                rots.append(rx(angle) if p == 1 else ry(angle) if p == 2 else rz(angle))
            st = qt.tensor(rots) * st
            st = ent_layer * st
    else:
        st = qt.tensor([qt.basis(2, 1) + qt.basis(2, 0) for _ in range(n_qubits)])
    st = st / st.norm()
    return np.asarray(st.full()).reshape(-1)


@dataclass
class QASCore:
    L: int
    K: int
    tfinal: float
    basis_strings: List[np.ndarray]
    basis_dim_raw: int
    basis_dim_effective: int
    B: np.ndarray  # columns = basis states
    E: np.ndarray
    s_inv: np.ndarray
    ini_alpha: np.ndarray
    term_mats: List[np.ndarray]
    D_terms: np.ndarray  # [n_terms, n_basis, n_basis]
    psi0: np.ndarray
    mz_op: np.ndarray


def build_qas_core(L: int, K: int, tfinal: float, include_j2: bool, seed: int = 1, inv_cond: float = 1e-10) -> QASCore:
    x_terms, nn_terms, nnn_terms = build_model_strings(L, include_j2=include_j2)
    all_terms = x_terms + nn_terms + nnn_terms

    # K-moment expansion
    base = [np.zeros(L, dtype=int)]
    for _ in range(K):
        base = multiply_paulis(base, all_terms)

    psi_basis = get_initial_state(ini_type=2, n_qubits=L, depth=L, seed=seed)
    psi0 = get_initial_state(ini_type=1, n_qubits=L, depth=L, seed=seed + 123)
    dim = 2**L

    # Basis vectors = P_s |psi_basis>
    basis_vecs = []
    for s in base:
        P = pauli_string_matrix(s)
        basis_vecs.append(P @ psi_basis)
    B_raw = np.column_stack(basis_vecs).astype(complex)  # dim x n_raw
    n_raw = B_raw.shape[1]

    # E matrix on raw basis and rank-reduction by eigencutoff
    E_raw = B_raw.conj().T @ B_raw
    evals_raw, evecs_raw = scipy.linalg.eigh(E_raw)
    keep = evals_raw > inv_cond
    if not np.any(keep):
        raise RuntimeError("Subespaco variacional vazio apos cutoff de E. Reduza inv_cond ou ajuste K.")
    lam = evals_raw[keep]
    U = evecs_raw[:, keep]

    # Orthonormal effective basis: B = B_raw * U * lam^{-1/2}
    transform = U @ np.diag(1.0 / np.sqrt(lam))
    B = B_raw @ transform
    n_basis = B.shape[1]
    E = np.eye(n_basis, dtype=complex)
    s_inv = np.eye(n_basis, dtype=complex)

    # IQAE-style init on orthonormal basis (equivale a projetar psi0 no subespaco)
    u = B.conj().T @ psi0
    ini_matrix = -np.outer(u, u.conj())
    _, vecs = scipy.linalg.eigh(ini_matrix)
    ini_alpha = vecs[:, 0]
    norm = np.linalg.norm(ini_alpha)
    if norm <= 0:
        raise RuntimeError("Falha na inicializacao IQAE: norma zero.")
    ini_alpha = ini_alpha / norm

    # Term matrices and reduced D terms
    term_mats: List[np.ndarray] = []
    for s in all_terms:
        term_mats.append(pauli_string_matrix(s))
    D_terms = np.empty((len(term_mats), n_basis, n_basis), dtype=complex)
    for i, M in enumerate(term_mats):
        D_terms[i] = B.conj().T @ (M @ B)

    # Mz operator (sum Z, normalized later by L)
    mz_op = np.zeros((dim, dim), dtype=complex)
    for q in range(L):
        s = np.zeros(L, dtype=int)
        s[q] = 3
        mz_op += pauli_string_matrix(s)

    return QASCore(
        L=L,
        K=K,
        tfinal=tfinal,
        basis_strings=base,
        basis_dim_raw=n_raw,
        basis_dim_effective=n_basis,
        B=B,
        E=E,
        s_inv=s_inv,
        ini_alpha=ini_alpha,
        term_mats=term_mats,
        D_terms=D_terms,
        psi0=psi0,
        mz_op=mz_op,
    )


def sample_coeffs(L: int, delta_j: float, J: float, h: float, J2: float, rng: np.random.Generator, include_j2: bool) -> np.ndarray:
    coeffs: List[float] = []
    coeffs.extend([-h] * L)  # X terms
    for _ in range(L):
        coeffs.append(-(J + float(rng.uniform(-delta_j, delta_j))))  # disordered ZZ nn
    if include_j2:
        coeffs.extend([-J2] * L)
    return np.asarray(coeffs, dtype=float)


def evolve_qas_final(core: QASCore, coeffs: np.ndarray) -> np.ndarray:
    # D = sum c_i D_i
    D = np.tensordot(coeffs, core.D_terms, axes=(0, 0))
    h_toeig = core.s_inv @ D @ core.s_inv.conj().T
    evals, evecs = scipy.linalg.eigh(h_toeig)
    alpha_eig_vecs = core.s_inv.conj().T @ evecs  # columns
    coeffs_eig = alpha_eig_vecs.conj().T @ (core.E @ core.ini_alpha)
    phase = np.exp(-1j * evals * core.tfinal)
    alpha_t = alpha_eig_vecs @ (coeffs_eig * phase)
    psi_qas = core.B @ alpha_t
    n = np.linalg.norm(psi_qas)
    if n > 0:
        psi_qas = psi_qas / n
    return psi_qas


def evolve_exact_final(core: QASCore, coeffs: np.ndarray) -> np.ndarray:
    H = np.zeros_like(core.term_mats[0])
    for c, M in zip(coeffs, core.term_mats):
        H += c * M
    evals, evecs = scipy.linalg.eigh(H)
    c0 = evecs.conj().T @ core.psi0
    psi_t = evecs @ (np.exp(-1j * evals * core.tfinal) * c0)
    n = np.linalg.norm(psi_t)
    if n > 0:
        psi_t = psi_t / n
    return psi_t


def fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def mz_norm(core: QASCore, psi: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, core.mz_op @ psi)) / float(core.L))


def default_exact_output_path(qas_out_path: Path) -> Path:
    name = qas_out_path.name
    if "QAS_true" in name:
        return qas_out_path.with_name(name.replace("QAS_true", "ExactRef"))
    return qas_out_path.with_name(qas_out_path.stem + "_ExactRef.npz")


def save_result_npz(
    out_path: Path,
    js: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    sums: np.ndarray,
    sumsqs: np.ndarray,
    counts: np.ndarray,
    nrea: int,
    nrea_total: int,
    rea_start: int,
    rea_end: int,
    core: QASCore,
    method: str,
    basis_mode: str,
    J: float,
    h: float,
    J2: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        Js=js,
        Magnetization=means,
        Std_Mz=stds,
        Sum_Mz=sums,
        SumSq_Mz=sumsqs,
        Count_Mz=counts,
        Nrea=int(nrea),
        Nrea_total=int(nrea_total),
        rea_start=int(rea_start),
        rea_end=int(rea_end),
        L=int(core.L),
        K=int(core.K),
        basis_mode=basis_mode,
        basis_dim_raw=int(core.basis_dim_raw),
        basis_dim_effective=int(core.basis_dim_effective),
        boundary="closed",
        method=method,
        J=float(J),
        h=float(h),
        J2=float(J2),
        FINAL_T=float(core.tfinal),
        normalized_by_L=1,
    )


def validate_delta_j_grid(js: np.ndarray) -> None:
    js = np.asarray(js, dtype=float).reshape(-1)
    expected_main = np.round(np.linspace(EXPECTED_DJ_MIN, EXPECTED_DJ_MAX, EXPECTED_DJ_N), 10)
    expected_alt = np.round(np.linspace(ALT_DJ_MIN, ALT_DJ_MAX, ALT_DJ_N), 10)
    got = np.round(js, 10)
    ok_main = got.shape == expected_main.shape and np.array_equal(got, expected_main)
    ok_alt = got.shape == expected_alt.shape and np.array_equal(got, expected_alt)
    if ok_main:
        print(
            f"INFO: grade ΔJ validada no padrao principal "
            f"[{EXPECTED_DJ_MIN:.1f}, {EXPECTED_DJ_MAX:.1f}] passo {EXPECTED_DJ_STEP:.1f}."
        )
        return
    if ok_alt:
        print(
            f"INFO: grade ΔJ validada no padrao alternativo "
            f"[{ALT_DJ_MIN:.1f}, {ALT_DJ_MAX:.1f}] ({ALT_DJ_N} pontos)."
        )
        return
    if js.size == 0:
        raise SystemExit("ERRO: grade de ΔJ vazia.")
    if not np.all(np.isfinite(js)):
        raise SystemExit("ERRO: grade de ΔJ contem valores nao finitos.")
    if np.any(np.diff(js) <= 0):
        raise SystemExit("ERRO: grade de ΔJ deve estar estritamente crescente.")
    print(
        "AVISO: grade de ΔJ fora dos dois padroes esperados, mas aceita para execucao. "
        f"Recebido: min={float(np.min(js)):.10g}, max={float(np.max(js)):.10g}, n={js.size}."
    )


def k_scan(
    L: int,
    K_values: Sequence[int],
    delta_j_values: Sequence[float],
    nrea: int,
    J: float,
    h: float,
    J2: float,
    tfinal: float,
    seed: int,
) -> List[dict]:
    include_j2 = abs(J2) > 0.0
    results = []
    for K in K_values:
        core = build_qas_core(L=L, K=K, tfinal=tfinal, include_j2=include_j2, seed=seed)
        fids = []
        for dj in delta_j_values:
            for r in range(nrea):
                rng = np.random.default_rng(seed + 100000 * int(round(dj * 100)) + r)
                coeffs = sample_coeffs(L, dj, J, h, J2, rng, include_j2=include_j2)
                psi_q = evolve_qas_final(core, coeffs)
                psi_e = evolve_exact_final(core, coeffs)
                fids.append(fidelity(psi_q, psi_e))
        fids = np.asarray(fids, dtype=float)
        results.append(
            {
                "K": int(K),
                "basis_dim_raw": int(core.basis_dim_raw),
                "basis_dim_effective": int(core.basis_dim_effective),
                "fidelity_mean": float(np.mean(fids)),
                "fidelity_min": float(np.min(fids)),
                "fidelity_p05": float(np.percentile(fids, 5)),
            }
        )
        print(
            f"K={K:2d} basis_raw={core.basis_dim_raw:4d} basis_eff={core.basis_dim_effective:4d} "
            f"fid_mean={np.mean(fids):.6f} fid_min={np.min(fids):.6f} fid_p05={np.percentile(fids,5):.6f}"
        )
    return results


def produce_qas(
    L: int,
    K: int,
    js: np.ndarray,
    nrea: int,
    rea_start: int,
    rea_end: int,
    J: float,
    h: float,
    J2: float,
    tfinal: float,
    seed: int,
    out_path: Path,
    exact_out_path: Optional[Path] = None,
    with_exact: bool = False,
) -> None:
    if rea_start < 0:
        raise ValueError("rea_start deve ser >= 0")
    if rea_end <= rea_start:
        raise ValueError("rea_end deve ser > rea_start")
    if rea_end > nrea:
        raise ValueError("rea_end nao pode exceder nrea total")

    chunk_nrea = rea_end - rea_start
    include_j2 = abs(J2) > 0.0
    core = build_qas_core(L=L, K=K, tfinal=tfinal, include_j2=include_j2, seed=seed)
    print(
        "QAS production | "
        f"L={L} K={K} basis_raw={core.basis_dim_raw} basis_eff={core.basis_dim_effective} nrea_total={nrea} "
        f"chunk=[{rea_start},{rea_end}) chunk_nrea={chunk_nrea} points={len(js)}"
    )
    means = np.zeros(len(js), dtype=float)
    stds = np.zeros(len(js), dtype=float)
    sums = np.zeros(len(js), dtype=float)
    sumsqs = np.zeros(len(js), dtype=float)
    means_exact = np.zeros(len(js), dtype=float)
    stds_exact = np.zeros(len(js), dtype=float)
    sums_exact = np.zeros(len(js), dtype=float)
    sumsqs_exact = np.zeros(len(js), dtype=float)
    counts = np.full(len(js), chunk_nrea, dtype=np.int64)
    for i, dj in enumerate(js):
        vals = np.zeros(chunk_nrea, dtype=float)
        vals_exact = np.zeros(chunk_nrea, dtype=float)
        for local_r, r_global in enumerate(range(rea_start, rea_end)):
            rng = np.random.default_rng(seed + 100000 * int(round(float(dj) * 100)) + r_global)
            coeffs = sample_coeffs(L, float(dj), J, h, J2, rng, include_j2=include_j2)
            psi_q = evolve_qas_final(core, coeffs)
            vals[local_r] = mz_norm(core, psi_q)
            if with_exact:
                psi_e = evolve_exact_final(core, coeffs)
                vals_exact[local_r] = mz_norm(core, psi_e)
        means[i] = float(np.mean(vals))
        stds[i] = float(np.std(vals, ddof=1 if chunk_nrea > 1 else 0))
        sums[i] = float(np.sum(vals))
        sumsqs[i] = float(np.sum(vals * vals))
        if with_exact:
            means_exact[i] = float(np.mean(vals_exact))
            stds_exact[i] = float(np.std(vals_exact, ddof=1 if chunk_nrea > 1 else 0))
            sums_exact[i] = float(np.sum(vals_exact))
            sumsqs_exact[i] = float(np.sum(vals_exact * vals_exact))
        if (i + 1) % 10 == 0 or i == len(js) - 1:
            if with_exact:
                print(
                    f"[{i+1:03d}/{len(js):03d}] dJ={float(dj):.2f} "
                    f"QAS_mean={means[i]:.6f} Exact_mean={means_exact[i]:.6f}"
                )
            else:
                print(f"[{i+1:03d}/{len(js):03d}] dJ={float(dj):.2f} mean={means[i]:.6f} std={stds[i]:.6f}")
    save_result_npz(
        out_path,
        js=js,
        means=means,
        stds=stds,
        sums=sums,
        sumsqs=sumsqs,
        counts=counts,
        nrea=chunk_nrea,
        nrea_total=nrea,
        rea_start=rea_start,
        rea_end=rea_end,
        core=core,
        method="QAS",
        basis_mode="true_qas_k_moment",
        J=J,
        h=h,
        J2=J2,
    )
    print(f"saved={out_path}")
    if with_exact:
        if exact_out_path is None:
            exact_out_path = default_exact_output_path(out_path)
        save_result_npz(
            exact_out_path,
            js=js,
            means=means_exact,
            stds=stds_exact,
            sums=sums_exact,
            sumsqs=sumsqs_exact,
            counts=counts,
            nrea=chunk_nrea,
            nrea_total=nrea,
            rea_start=rea_start,
            rea_end=rea_end,
            core=core,
            method="ExactRef",
            basis_mode="full_hilbert_exact",
            J=J,
            h=h,
            J2=J2,
        )
        print(f"saved={exact_out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["kscan", "produce"], default="kscan")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--K-values", type=str, default="1,2,3,4")
    parser.add_argument("--nrea", type=int, default=100)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.6)
    parser.add_argument("--J2", type=float, default=0.3)
    parser.add_argument("--tfinal", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deltaj-values", type=str, default="0.5,2.0,4.0,8.0")
    parser.add_argument("--grid-source", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--exact-output", type=str, default="")
    parser.add_argument("--with-exact", action="store_true")
    parser.add_argument("--rea-start", type=int, default=0)
    parser.add_argument("--rea-end", type=int, default=-1)
    args = parser.parse_args()

    if args.mode == "kscan":
        ks = [int(x.strip()) for x in args.K_values.split(",") if x.strip()]
        djs = [float(x.strip()) for x in args.deltaj_values.split(",") if x.strip()]
        res = k_scan(
            L=args.L,
            K_values=ks,
            delta_j_values=djs,
            nrea=args.nrea,
            J=args.J,
            h=args.h,
            J2=args.J2,
            tfinal=args.tfinal,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2))
        return

    # produce
    root = Path(__file__).resolve().parents[1]
    if args.grid_source:
        grid_src = Path(args.grid_source).resolve()
    else:
        grid_src = root / "data" / "ClosedBC" / f"Mz_Nqu={args.L}_Nrea=1000_ClosedBC.npz"
    with np.load(grid_src, allow_pickle=False) as d:
        js = np.unique(np.asarray(d["Js"], dtype=float).reshape(-1))
    js.sort()
    validate_delta_j_grid(js)
    if args.output:
        out = Path(args.output).resolve()
    else:
        if args.rea_end < 0 or (args.rea_start == 0 and args.rea_end == args.nrea):
            out = root / "data" / "ClosedBC" / f"Mz_Nqu={args.L}_Nrea={args.nrea}_K={args.K}_ClosedBC_QAS_true.npz"
        else:
            out = (
                root
                / "data"
                / "ClosedBC"
                / (
                    f"Mz_Nqu={args.L}_Nrea={args.nrea}_K={args.K}_ClosedBC_QAS_true_"
                    f"chunk_{args.rea_start:04d}_{args.rea_end:04d}.npz"
                )
            )
    if args.exact_output:
        exact_out = Path(args.exact_output).resolve()
    elif args.with_exact:
        exact_out = default_exact_output_path(out)
    else:
        exact_out = None
    rea_end = args.nrea if args.rea_end < 0 else args.rea_end
    produce_qas(
        L=args.L,
        K=args.K,
        js=js,
        nrea=args.nrea,
        rea_start=args.rea_start,
        rea_end=rea_end,
        J=args.J,
        h=args.h,
        J2=args.J2,
        tfinal=args.tfinal,
        seed=args.seed,
        out_path=out,
        exact_out_path=exact_out,
        with_exact=args.with_exact,
    )


if __name__ == "__main__":
    main()
