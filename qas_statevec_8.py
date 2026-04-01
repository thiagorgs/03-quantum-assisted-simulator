# qas_statevec_8.py
#
# QAS e Exact (separados) para L=8, K=L, no basis computacional.
# Evolução via expm_multiply (H esparso).
# Paraleliza por realização (Windows spawn-friendly).
#
# Saída: qas_exact_kjall_statevec_L8_K8_scanDeltaJ_N500.npz
# com deltaJ_values, Mz_qas_timeavg, Mz_exact_timeavg
#
# Env vars úteis:
#   N_WORKERS=6        (default: 6)
#   CHUNKSIZE=5        (default: 5)
#   EXACT_DOUBLE=0/1   (default: 0; se 1, recalcula "Exact" mesmo com K=L)
#

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import numpy as np
from itertools import combinations
import multiprocessing as mp

from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply

import matplotlib.pyplot as plt


def env_true(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "y")


# ------------------------------
# Basis / padrões em Z
# ------------------------------
def basis_bitmasks_x_only(L: int, K: int):
    K = min(K, L)
    if K == L:
        # base completa em ordem natural (índice == bitmask)
        return np.arange(1 << L, dtype=np.uint16)

    masks = []
    for r in range(K + 1):
        for sites in combinations(range(L), r):
            m = 0
            for s in sites:
                m |= (1 << s)
            masks.append(m)
    return np.array(masks, dtype=np.uint16)


def precompute_z_and_mz(L: int, basis_masks: np.ndarray):
    m = len(basis_masks)
    z = np.empty((m, L), dtype=np.int8)
    for i in range(L):
        bits = (basis_masks >> i) & 1
        z[:, i] = (1 - 2 * bits).astype(np.int8)  # 0->+1, 1->-1
    mz_eigs = z.sum(axis=1).astype(np.int16)
    return z, mz_eigs


def precompute_zz_patterns(L: int, z: np.ndarray):
    m = z.shape[0]
    zz1 = np.empty((L - 1, m), dtype=np.int8)
    for i in range(L - 1):
        zz1[i, :] = (z[:, i] * z[:, i + 1]).astype(np.int8)

    zz2_sum = np.zeros(m, dtype=np.int16)
    for i in range(L - 2):
        zz2_sum += (z[:, i] * z[:, i + 2]).astype(np.int16)

    return zz1, zz2_sum


def build_Hx_sparse(L: int, basis_masks: np.ndarray, idx_map: dict | None, h: float):
    m = len(basis_masks)

    full_ordered = (m == (1 << L)) and np.array_equal(
        basis_masks, np.arange(1 << L, dtype=basis_masks.dtype)
    )

    rows, cols, data = [], [], []

    if full_ordered:
        # índice == bitmask
        for s in range(1 << L):
            for i in range(L):
                sp = s ^ (1 << i)
                rows.append(s)
                cols.append(sp)
                data.append(-h)
        return coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()

    assert idx_map is not None
    for a, s in enumerate(basis_masks):
        s = int(s)
        for i in range(L):
            sp = s ^ (1 << i)
            b = idx_map.get(sp, None)
            if b is not None:
                rows.append(a)
                cols.append(b)
                data.append(-h)

    return coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()


def make_precomp(L: int, K: int, h: float):
    basis = basis_bitmasks_x_only(L, K)

    full_ordered = (len(basis) == (1 << L)) and np.array_equal(
        basis, np.arange(1 << L, dtype=basis.dtype)
    )
    idx_map = None if full_ordered else {int(m): i for i, m in enumerate(basis)}

    z, mz_eigs = precompute_z_and_mz(L, basis)
    zz1, zz2_sum = precompute_zz_patterns(L, z)
    Hx = build_Hx_sparse(L, basis, idx_map, h)

    return {
        "L": L,
        "K": K,
        "basis": basis,
        "idx_map": idx_map,
        "mz_eigs": mz_eigs,
        "zz1": zz1,
        "zz2_sum": zz2_sum,
        "Hx": Hx,
    }


# ------------------------------
# Hamiltoniano + evolução
# ------------------------------
def sample_J_bonds(L: int, J: float, DeltaJ: float, rng: np.random.Generator):
    return J + rng.uniform(-DeltaJ, DeltaJ, size=L - 1)


def build_H_sparse(J_bonds, J2, Hx, zz1, zz2_sum):
    diag = -(J_bonds @ zz1) - (J2 * zz2_sum)
    return Hx + diags(diag, 0, format="csr")


def evolve_Mz_final(H, t_final, mz_eigs, idx0=0):
    v0 = np.zeros(H.shape[0], dtype=complex)
    v0[idx0] = 1.0
    psi_t = expm_multiply((-1j * t_final) * H, v0)
    p = np.abs(psi_t) ** 2
    return float(p @ mz_eigs)


# ------------------------------
# Multiprocessing globals
# ------------------------------
_G = {}

def _init_worker(pre_qas, pre_exact, J, J2, h, t_final, exact_double):
    _G["pre_qas"] = pre_qas
    _G["pre_exact"] = pre_exact
    _G["J"] = float(J)
    _G["J2"] = float(J2)
    _G["h"] = float(h)
    _G["t_final"] = float(t_final)
    _G["exact_double"] = bool(exact_double)

def _one_realization(job):
    # job = (DeltaJ, seed)
    DeltaJ, seed = job
    pre_qas = _G["pre_qas"]
    pre_exact = _G["pre_exact"]
    J = _G["J"]; J2 = _G["J2"]
    t_final = _G["t_final"]
    exact_double = _G["exact_double"]

    rng = np.random.default_rng(int(seed))
    L = pre_qas["L"]

    J_bonds = sample_J_bonds(L, J, float(DeltaJ), rng)

    # QAS
    H_qas = build_H_sparse(J_bonds, J2, pre_qas["Hx"], pre_qas["zz1"], pre_qas["zz2_sum"])
    mz_qas = evolve_Mz_final(H_qas, t_final, pre_qas["mz_eigs"])

    # Exact separado (mesmo com K=L):
    # - por padrão NÃO recalculamos (pois é idêntico) para economizar tempo
    # - se EXACT_DOUBLE=1, recalcula (paga o dobro)
    if pre_exact is None:
        mz_ex = np.nan
    else:
        if (pre_exact["K"] == pre_qas["K"]) and (not exact_double):
            mz_ex = mz_qas
        else:
            H_ex = build_H_sparse(J_bonds, J2, pre_exact["Hx"], pre_exact["zz1"], pre_exact["zz2_sum"])
            mz_ex = evolve_Mz_final(H_ex, t_final, pre_exact["mz_eigs"])

    return mz_qas, mz_ex


def main():
    # parâmetros
    J = 1.0
    J2 = 0.3
    h = 0.6
    t_final = 400.0

    L = 8
    K = 8
    n_realizations = 500

    DeltaJ_values = np.round(np.arange(0.0, 12.0 + 0.05, 0.1), 2)

    # multiprocessing knobs (para seu i5-10400)
    N_WORKERS = int(os.environ.get("N_WORKERS", "6"))
    CHUNKSIZE = int(os.environ.get("CHUNKSIZE", "5"))
    EXACT_DOUBLE = env_true("EXACT_DOUBLE", "0")

    print(f"[CPU] workers={N_WORKERS}  chunksize={CHUNKSIZE}  EXACT_DOUBLE={int(EXACT_DOUBLE)}")

    # precomputações
    pre_qas = make_precomp(L, K, h)
    pre_exact = make_precomp(L, L, h)  # separado (mesmo que idêntico quando K=L)

    # pool (Windows spawn)
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
        initargs=(pre_qas, pre_exact, J, J2, h, t_final, EXACT_DOUBLE),
    ) as pool:

        Mz_qas_vs = []
        Mz_exact_vs = []

        t0_all = time.time()

        for idJ, DeltaJ in enumerate(DeltaJ_values):
            t0 = time.time()
            print(f"ΔJ = {DeltaJ:.2f}  ({idJ+1}/{len(DeltaJ_values)})")

            base_seed = 1234 + 100000 * idJ
            jobs = [(float(DeltaJ), int(base_seed + r)) for r in range(n_realizations)]

            mz_qas_all = np.empty(n_realizations, dtype=float)
            mz_ex_all  = np.empty(n_realizations, dtype=float)

            for k, (mz_qas, mz_ex) in enumerate(pool.imap_unordered(_one_realization, jobs, chunksize=CHUNKSIZE)):
                mz_qas_all[k] = mz_qas
                mz_ex_all[k]  = mz_ex

            Mz_qas_vs.append(float(np.mean(mz_qas_all)))
            Mz_exact_vs.append(float(np.mean(mz_ex_all)))

            dt = time.time() - t0
            print(f"   mean QAS={Mz_qas_vs[-1]: .6f} | mean Exact={Mz_exact_vs[-1]: .6f}   (Δt={dt:.1f}s)")

            # checkpoint simples (evita perder dias se der ruim)
            tmpname = f"partial_L{L}_K{K}_N{n_realizations}.npz"
            if (idJ + 1) % 5 ==0:
                np.savez_compressed(
                    tmpname,
                    deltaJ_values=DeltaJ_values[: idJ + 1],
                    Mz_qas_timeavg=np.array(Mz_qas_vs, dtype=float),
                    Mz_exact_timeavg=np.array(Mz_exact_vs, dtype=float),
                )

        total = time.time() - t0_all
        print(f"[DONE] tempo total = {total/3600:.2f} h")

    # salvar final
    outname = f"qas_exact_kjall_statevec_L{L}_K{K}_scanDeltaJ_N{n_realizations}.npz"
    np.savez_compressed(
        outname,
        deltaJ_values=DeltaJ_values,
        Mz_qas_timeavg=np.array(Mz_qas_vs, dtype=float),
        Mz_exact_timeavg=np.array(Mz_exact_vs, dtype=float),
    )
    print(f"Salvo em: {outname}")

    # plot
    plt.figure()
    plt.plot(DeltaJ_values, Mz_qas_vs, "o-", label="QAS")
    plt.plot(DeltaJ_values, Mz_exact_vs, "^-", label="Exato")
    plt.xlabel(r"$\Delta J$")
    plt.ylabel(r"$\langle M_z(t_f)\rangle$")
    plt.title(f"L={L}, K={K}, t_f={t_final}")
    plt.ylim(-L, L)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mp.freeze_support()
    main()
