# plot_qas_vs_exact_8.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit
from qiskit.quantum_info import SparsePauliOp, Statevector

def env_true(name, default="1"):
    return os.environ.get(name, default).strip().lower() in ("1","true","yes","y")

# ---------- Flags (ambiente) ----------
PLOT_MEAN     = env_true("PLOT_MEAN", "1")
REBUILD_AGG   = env_true("REBUILD_AGG", "0")
FILTER_SG     = env_true("FILTER_SG", "1")
LIMIAR_SG_ENV = os.environ.get("LIMIAR_SG", "").strip()  # se vazio, usa 0.9*NQ
PLOT_ERR      = env_true("PLOT_ERR", "0")
ERR_KIND      = os.environ.get("ERR_KIND", "std").strip().lower()  # "std" ou "sem"
MAKE_HISTS    = env_true("MAKE_HISTS", "1")

NTIME_AVG         = int(os.environ.get("NTIME_AVG", "1"))
T_AVG_START_RATIO = float(os.environ.get("T_AVG_START_RATIO", "1.0"))
USE_LEGACY_RNG    = env_true("LEGACY_RNG", "1")
POS_DJ            = env_true("POS_DJ", "0")         # 0: simétrico; 1: só positivos
PHASE_LOCK        = env_true("PHASE_LOCK", "0")
TIME_JITTER_FRAC  = float(os.environ.get("TIME_JITTER_FRAC", "0.0"))
PHASE_LOCK_SG_ONLY = env_true("PHASE_LOCK_SG_ONLY", "0")  # trava a fase só no SG (> Jc2)

THERMAL_STEP = int(os.environ.get("THERMAL_STEP", "4"))
MID_STEP     = int(os.environ.get("MID_STEP", "4"))
SG_STEP      = int(os.environ.get("SG_STEP", "3"))

INCLUDE_HOLE  = env_true("INCLUDE_HOLE", "1")
PARA_K        = int(os.environ.get("PARA_K", "6"))
PARA_STAT     = os.environ.get("PARA_STAT", "median").strip().lower()  # 'median' ou 'mean'
SYM_UPDOWN    = env_true("SYM_UPDOWN", "1")   # ativa simetrização
SYM_ONLY_PARA = env_true("SYM_ONLY_PARA", "1")
VECT_AVG = env_true("VECT_AVG", "1")  # 1 = usa média temporal vetorizada
ASSERT_VECT_EQ = env_true("ASSERT_VECT_EQ", "0")  # rode o teste de equivalência?
_EQ_DONE_QAS   = False
_EQ_DONE_EXACT = False

J1_FORCE = os.environ.get("J1_FORCE", "").strip()
J2_FORCE = os.environ.get("J2_FORCE", "").strip()

# ---------- Modelo ----------
def disordered_ising_ops(nq: int):
    X, ZZ_nn, ZZ_nnn = [], [], []
    for q in range(nq):
        lab = ['I'] * nq; lab[q] = 'X'; X.append(''.join(lab))
    for i in range(nq - 1):
        lab = ['I'] * nq; lab[i] = 'Z'; lab[i+1] = 'Z'; ZZ_nn.append(''.join(lab))
    for i in range(nq - 2):
        lab = ['I'] * nq; lab[i] = 'Z'; lab[i+2] = 'Z'; ZZ_nnn.append(''.join(lab))
    return X, ZZ_nn, ZZ_nnn

def magnetization_matrix(nq: int):
    mats = []
    for q in range(nq):
        lab = ['I']*nq; lab[q]='Z'
        mats.append(SparsePauliOp.from_list([(''.join(lab), 1.0)]).to_matrix())
    return np.sum(mats, axis=0)

def build_H_matrix(nq, J, h, J2, delta_J, rng):
    X_labels, ZZ_nn_labels, ZZ_nnn_labels = disordered_ising_ops(nq)
    sparse_list = []
    sparse_list.extend((lab, -h) for lab in X_labels)
    for lab in ZZ_nn_labels:
        dJi = rng.uniform(0.0, delta_J) if POS_DJ else rng.uniform(-delta_J, delta_J)
        sparse_list.append((lab, -(J + dJi)))
    sparse_list.extend((lab, -J2) for lab in ZZ_nnn_labels)
    return SparsePauliOp.from_list(sparse_list, num_qubits=nq).to_matrix()

def stable_seed_for_deltaJ(delta_J: float, base_seed: int):
    dj_key = int(f"{float(delta_J):.2f}".replace(".", ""))  # 3.50 -> 350
    return (int(base_seed) * 1_000_003 + dj_key) % (2**31 - 1)

def _phase_locked_T(Tfin, h, lock=False):
    if not lock:
        return Tfin
    period = 2*np.pi/float(h)
    k = max(1, int(round(Tfin/period)))
    return k*period

def _phase_lock_should_apply(delta_J: float) -> bool:
    if PHASE_LOCK and not PHASE_LOCK_SG_ONLY:
        return True
    if not PHASE_LOCK and not PHASE_LOCK_SG_ONLY:
        return False
    try:
        jc2 = float(J2_FORCE) if J2_FORCE else None
    except Exception:
        jc2 = None
    return (jc2 is not None) and (float(delta_J) > jc2)

def _time_avg_expvals(evals, evecs, coeff0, Obs, times):
    phases = np.exp(-1j * evals[:, None] * times[None, :])
    alphas = evecs @ (phases * coeff0[:, None])
    O_alpha = Obs @ alphas
    exp_t = np.einsum('it,it->t', np.conj(alphas), O_alpha).real
    return float(exp_t.mean())

# ---------- QAS por QR ----------
def qr_basis_from_pauli_products(nq: int, K: int, psi0_label: str):
    X, ZZ_nn, ZZ_nnn = disordered_ising_ops(nq)
    ops = [SparsePauliOp.from_list([(lab, 1.0)]) for lab in (X + ZZ_nn + ZZ_nnn)]
    from itertools import combinations
    psi0 = Statevector.from_label(psi0_label).data
    dim = 2**nq
    Q = np.zeros((dim, 0), dtype=complex)

    def orth_residual(v, Q):
        if Q.shape[1] == 0:
            return v, np.linalg.norm(v)
        r = v - Q @ (Q.conj().T @ v)
        return r, np.linalg.norm(r)

    r, nr = orth_residual(psi0, Q)
    if nr > 1e-12:
        Q = np.column_stack([Q, r/nr])

    seen = {"I"*nq}
    for k in range(1, K+1):
        for combo in combinations(ops, k):
            p = combo[0]
            for o in combo[1:]:
                p = p.compose(o)
            label = p.paulis[0].to_label()
            if label in seen:
                continue
            seen.add(label)
            v = p.to_matrix() @ psi0
            r, nr = orth_residual(v, Q)
            if nr > 1e-10:
                Q = np.column_stack([Q, r/nr])
                if Q.shape[1] >= dim:
                    return Q
    return Q

def precompute_qas_structs_qr(nq: int, K: int, psi0_label: str):
    Q = qr_basis_from_pauli_products(nq, K, psi0_label)
    if env_true("QAS_LOG", "1"):
        print(f"[QAS-QR] Base size = {Q.shape[1]} / {2**nq} (K={K}, psi0='{psi0_label[:min(4,nq)]}…')")
        Ierr = np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1], dtype=Q.dtype))
        print(f"[QAS-QR] ||Q†Q - I|| = {Ierr:.2e}")
    psi0 = Statevector.from_label(psi0_label).data
    M_full = magnetization_matrix(nq)
    M_tilde = Q.conj().T @ (M_full @ Q)
    alpha0 = Q.conj().T @ psi0
    return Q, M_tilde, alpha0

def qas_one_mag_qr(delta_J, seed, params, Q, M_tilde, alpha0,
                   n_time=1, t0_ratio=1.0):
    nq = params["NQ"]; J=params["J"]; h=params["h"]; J2=params["J2"]; Tfin=params["FINAL_T"]
    rng = (np.random.RandomState(seed) if USE_LEGACY_RNG else np.random.default_rng(seed))
    H = build_H_matrix(nq, J, h, J2, delta_J, rng)
    Dp = Q.conj().T @ (H @ Q); Dp = 0.5*(Dp + Dp.conj().T)
    w, V = np.linalg.eigh(Dp)
    beta0 = V.conj().T @ alpha0
    Tfin = _phase_locked_T(Tfin, h, lock=_phase_lock_should_apply(delta_J))
    if TIME_JITTER_FRAC > 0:
        period = 2*np.pi/float(h)
        Tfin = Tfin + (rng.uniform(-TIME_JITTER_FRAC, TIME_JITTER_FRAC) * period)
    t0 = min(Tfin * t0_ratio, Tfin)
    times = np.linspace(t0, Tfin, max(1, n_time), dtype=float)
    global _EQ_DONE_QAS
    if ASSERT_VECT_EQ and not _EQ_DONE_QAS:
        val_vec = _time_avg_expvals(w, V, beta0, M_tilde, times)
        mz_vals = []
        for t in times:
            alpha_t = V @ (np.exp(-1j*w*t) * beta0)
            mz_vals.append(np.real(alpha_t.conj().T @ (M_tilde @ alpha_t)))
        val_loop = float(np.mean(mz_vals))
        assert np.allclose(val_loop, val_vec, rtol=1e-12, atol=1e-12)
        _EQ_DONE_QAS = True
        if VECT_AVG:
            return val_vec
    if VECT_AVG:
        return _time_avg_expvals(w, V, beta0, M_tilde, times)
    mz_vals = []
    for t in times:
        alpha_t = V @ (np.exp(-1j*w*t) * beta0)
        mz_vals.append(np.real(alpha_t.conj().T @ (M_tilde @ alpha_t)))
    return float(np.mean(mz_vals))

# ---------- Exato ----------
def exact_one_statevector(delta_J, seed, params, M_full,
                          n_time=1, t0_ratio=1.0, psi0_label='0'):
    nq = params["NQ"]; J=params["J"]; h=params["h"]; J2=params["J2"]; Tfin=params["FINAL_T"]
    rng = (np.random.RandomState(seed) if USE_LEGACY_RNG else np.random.default_rng(seed))
    H = build_H_matrix(nq, J, h, J2, delta_J, rng)
    w, V = np.linalg.eigh(H)
    psi0 = Statevector.from_label(psi0_label).data
    c0 = V.conj().T @ psi0
    Tfin = _phase_locked_T(Tfin, h, lock=_phase_lock_should_apply(delta_J))
    if TIME_JITTER_FRAC > 0:
        period = 2*np.pi/float(h)
        Tfin = Tfin + (rng.uniform(-TIME_JITTER_FRAC, TIME_JITTER_FRAC) * period)
    t0 = min(Tfin * t0_ratio, Tfin)
    times = np.linspace(t0, Tfin, max(1, n_time), dtype=float)
    global _EQ_DONE_EXACT
    if ASSERT_VECT_EQ and not _EQ_DONE_EXACT:
        val_vec = _time_avg_expvals(w, V, c0, M_full, times)
        mz_vals = []
        for t in times:
            psi_t = V @ (np.exp(-1j*w*t) * c0)
            mz_vals.append(np.real(np.vdot(psi_t, M_full @ psi_t)))
        val_loop = float(np.mean(mz_vals))
        assert np.allclose(val_loop, val_vec, rtol=1e-12, atol=1e-12)
        _EQ_DONE_EXACT = True
        if VECT_AVG:
            return val_vec
    if VECT_AVG:
        return _time_avg_expvals(w, V, c0, M_full, times)
    mz_vals = []
    for t in times:
        psi_t = V @ (np.exp(-1j*w*t) * c0)
        mz_vals.append(np.real(np.vdot(psi_t, M_full @ psi_t)))
    return float(np.mean(mz_vals))

# ---------- Ajuste crítico ----------
# ---------- Ajuste crítico (mesma lógica do cluster) ----------
def double_sigmoid(x, A1, B1, xc1, k1, A2, B2, xc2, k2):
    return (
        A1/(1.0 + np.exp(-k1*(x - xc1))) + B1
      + A2/(1.0 + np.exp(-k2*(x - xc2))) + B2
    )

def process_file_abs_like_cluster(
    dJ, Mq, Me,
    frac=0.20, nbins=40,
    xc1_init=2.2, xc2_init=6.0,
    bound1=(1.6, 2.8),         # região térmica “profunda” e upper da 1ª sigmóide
    bound2=(5.0, 9.0),         # libera a 2ª sigmóide (evita “puxar” p/ 5.5)
    weight_exact=0.5
):
    """
    Replica o pipeline do cluster:
      - métrica de ajuste: |.| misturando QAS e Exact (peso 0.5)
      - máscara de ajuste: dJ < bound1[1]  OU  dJ > bound2[0]
      - remoção de viés: mediana em dJ < bound1[0]
      - binning log (0.1..12) + LOWESS
      - ajuste double-sigmoid com limites (bound1, bound2)
    """
    dJ = np.asarray(dJ, float)
    Mq = np.asarray(Mq, float)
    Me = np.asarray(Me, float)

    # métrica de ajuste
    Mf_abs = weight_exact*np.abs(Mq) + (1.0 - weight_exact)*np.abs(Me)

    # máscara: bem térmico OU bem SG
    mask_fit = (dJ < bound1[1]) | (dJ > bound2[0])
    dJ_fit   = dJ[mask_fit]
    Mf_fit   = Mf_abs[mask_fit]

    # bias térmico (remove offset)
    med_bias = np.median(Mf_fit[dJ_fit < bound1[0]])
    Mf_fit   = Mf_fit - med_bias

    # binning log
    edges   = np.logspace(np.log10(0.1), np.log10(12.0), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    inds    = np.digitize(dJ_fit, edges)

    medians = np.array([
        np.median(Mf_fit[inds == i]) if np.any(inds == i) else np.nan
        for i in range(1, len(edges))
    ], float)

    valid = np.isfinite(medians)
    xsm, ysm = lowess(medians[valid], centers[valid], frac=frac, return_sorted=True).T

    # inicialização e bounds do fit
    A1 = 0.5*(ysm.max() - ysm.min())
    B1 = ysm.min()
    p0 = [A1, B1, xc1_init, 0.5,  A1, B1, xc2_init, 0.5]
    lb = [0,   B1, bound1[0], 0.01, 0,   B1, bound2[0], 0.01]
    ub = [2*A1, B1+2*A1, bound1[1], 50.0, 2*A1, B1+2*A1, bound2[1], 50.0]

    popt, _ = curve_fit(double_sigmoid, xsm, ysm, p0=p0, bounds=(lb, ub), maxfev=50_000)

    xfit = np.linspace(xsm.min(), xsm.max(), 400)
    yfit = double_sigmoid(xfit, *popt)

    return {
        "Jc1":  float(popt[2]),
        "Jc2":  float(popt[6]),
        "xfit": xfit,
        "yfit": yfit,
        "dJ_sm": xsm,
        "M_sm":  ysm
    }


def _as_str(x):
    try: return str(x.item())
    except Exception: return str(x)

# ---------- Agregado ----------
def build_or_load_aggregated_up_sv(outdir: str, base_seed: int, params: dict,
                                   K: int = 4, N_REALIZ: int = 500,
                                   GRID: np.ndarray | None = None):
    BASIS_KIND = "qr"
    RNG_KIND = "RandomState" if USE_LEGACY_RNG else "PCG64"

    out_path = os.path.join(outdir, "aggregated_up_sv.npz")
    if REBUILD_AGG and os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass
    if GRID is None:
        GRID = np.array([float(f"{x:.2f}") for x in np.linspace(0.1, 12.0, 120)], dtype=float)

    if os.path.exists(out_path):
        try:
            with np.load(out_path, allow_pickle=False) as A:
                need = {"delta_J","magnetization_mean","exact_result",
                        "NQ","J","h","J2","FINAL_T","BASE_SEED","K","N_REALIZ",
                        "NTIME_AVG","T_AVG_START_RATIO","grid","BASIS_KIND",
                        "RNG_KIND","PHASE_LOCK","POS_DJ","TIME_JITTER_FRAC", "SYM_UPDOWN","PHASE_LOCK_SG_ONLY"}
                ok = need.issubset(A.files) and \
                   int(A["NQ"])==params["NQ"] and \
                   np.isclose(float(A["J"]), params["J"]) and \
                   np.isclose(float(A["h"]), params["h"]) and \
                   np.isclose(float(A["J2"]),params["J2"]) and \
                   np.isclose(float(A["FINAL_T"]),params["FINAL_T"]) and \
                   int(A["BASE_SEED"])==base_seed and \
                   int(A["K"])==K and int(A["N_REALIZ"])==N_REALIZ and \
                   int(A["NTIME_AVG"])==NTIME_AVG and \
                   np.isclose(float(A["T_AVG_START_RATIO"]), T_AVG_START_RATIO) and \
                   np.array_equal(A["grid"], GRID) and \
                   _as_str(A["BASIS_KIND"])==BASIS_KIND and \
                   int(A["POS_DJ"])==int(POS_DJ) and \
                   int(A["PHASE_LOCK"])==int(PHASE_LOCK) and \
                   np.isclose(float(A["TIME_JITTER_FRAC"]), TIME_JITTER_FRAC) and \
                   int(A["SYM_UPDOWN"])==int(SYM_UPDOWN) and \
                   int(A["PHASE_LOCK_SG_ONLY"]) == int(PHASE_LOCK_SG_ONLY) and \
                   _as_str(A["RNG_KIND"])==RNG_KIND
                if ok and int(SYM_UPDOWN)==1:
                    ok = {"magnetization_mean_down", "exact_result_down"}.issubset(A.files)
                if ok:
                    return out_path
        except Exception:
            pass

    os.makedirs(outdir, exist_ok=True)
    Q_up, M_up, a_up = precompute_qas_structs_qr(params["NQ"], K, psi0_label='0'*params["NQ"])

    if SYM_UPDOWN:
        Q_dn, M_dn, a_dn = precompute_qas_structs_qr(params["NQ"], K, psi0_label='1'*params["NQ"])
    else:
        Q_dn = M_dn = a_dn = None

    if env_true("QAS_LOG", "1"):
        print(f"[QAS-QR] Up   = {Q_up.shape[1]} / {2**params['NQ']}")
        if SYM_UPDOWN:
            print(f"[QAS-QR] Down = {Q_dn.shape[1]} / {2**params['NQ']}")
    M_full = magnetization_matrix(params["NQ"])

    D_list, Mq_list, Me_list = [], [], []
    Mq_list_dn, Me_list_dn = [], []
    for dj in GRID:
        seed0 = stable_seed_for_deltaJ(float(dj), base_seed)
        for k in range(N_REALIZ):
            s = int(seed0 + k)
            mz_qas = qas_one_mag_qr(float(dj), s, params, Q_up, M_up, a_up,
                                    n_time=NTIME_AVG, t0_ratio=T_AVG_START_RATIO)
            mz_ex  = exact_one_statevector(float(dj), s, params, M_full,
                                           n_time=NTIME_AVG, t0_ratio=T_AVG_START_RATIO, psi0_label='0'*params["NQ"])
            D_list.append(float(dj)); Mq_list.append(mz_qas); Me_list.append(mz_ex)
            if SYM_UPDOWN:
                mz_qas_dn = qas_one_mag_qr(float(dj), s, params, Q_dn, M_dn, a_dn,
                                           n_time=NTIME_AVG, t0_ratio=T_AVG_START_RATIO)
                mz_ex_dn = -mz_ex  # relação exata por simetria
                Mq_list_dn.append(mz_qas_dn); Me_list_dn.append(mz_ex_dn)

    delta_J = np.asarray(D_list, dtype=float)
    Mq = np.asarray(Mq_list, dtype=float)
    Me = np.asarray(Me_list, dtype=float)

    save_kwargs = dict(
        delta_J=delta_J, magnetization_mean=Mq, exact_result=Me,
        grid=GRID, NQ=params["NQ"], J=params["J"], h=params["h"], J2=params["J2"],
        FINAL_T=params["FINAL_T"], BASE_SEED=base_seed, K=K, N_REALIZ=N_REALIZ,
        NTIME_AVG=NTIME_AVG, T_AVG_START_RATIO=T_AVG_START_RATIO,
        BASIS_KIND=BASIS_KIND, RNG_KIND=RNG_KIND,
        POS_DJ=int(POS_DJ), PHASE_LOCK=int(PHASE_LOCK), TIME_JITTER_FRAC=TIME_JITTER_FRAC,
        SYM_UPDOWN=int(SYM_UPDOWN), PHASE_LOCK_SG_ONLY=int(PHASE_LOCK_SG_ONLY),
        note="QAS via base ortonormal (QR) + EXATO full — ambos em statevector."
    )
    if SYM_UPDOWN:
        save_kwargs["magnetization_mean_down"] = np.asarray(Mq_list_dn, dtype=float)
        save_kwargs["exact_result_down"] = np.asarray(Me_list_dn, dtype=float)

    np.savez_compressed(out_path, **save_kwargs)
    print(f"Agregado salvo em {out_path} (amostras: {delta_J.size})")
    return out_path

# ---------- Main ----------
def main():
    # Default para N=8
    NQ = int(os.environ.get("NQ", "8"))
    J  = float(os.environ.get("J", "1.0"))
    h  = float(os.environ.get("h", "0.6"))
    J2 = float(os.environ.get("J2", "0.3"))

    Tfin = float(os.environ.get("FINAL_T", "400.0"))
    BASE_SEED = int(os.environ.get("GLOBAL_SEED", "1234"))
    params = {"NQ": NQ, "J": J, "h": h, "J2": J2, "FINAL_T": Tfin}

    OUTDIR = f"results_sv{NQ}"
    NREAL = int(os.environ.get("N_REALIZ", "500"))
    # K=6 funciona bem; ajuste se desejar reduzir a base
    agg_path = build_or_load_aggregated_up_sv(OUTDIR, BASE_SEED, params, K=8, N_REALIZ=NREAL)

    data   = np.load(agg_path)
    dJ_all = np.abs(data["delta_J"])
    Mq_all = data["magnetization_mean"].astype(float).copy()
    Me_all = data["exact_result"].astype(float).copy()
    Mq_all_dn = data["magnetization_mean_down"].astype(float).copy() if ("magnetization_mean_down" in data.files) else None
    Me_all_dn = data["exact_result_down"].astype(float).copy()       if ("exact_result_down" in data.files) else None

    LIMIAR_SG = float(LIMIAR_SG_ENV) if LIMIAR_SG_ENV else 0.9*float(NQ)
    SG_ABS = env_true("SG_ABS", "1")

    fit = process_file_abs_like_cluster(dJ_all, Mq_all, Me_all,
                                        xc1_init=3.5, xc2_init=6.5,
                                        bound1=(3.0,4.5), bound2=(5.5,7.5))
    Jc1, Jc2 = fit["Jc1"], fit["Jc2"]
    if J1_FORCE: Jc1 = float(J1_FORCE)
    if J2_FORCE: Jc2 = float(J2_FORCE)
    print(f"δJc₁ ≈ {Jc1:.2f},  δJc₂ ≈ {Jc2:.2f}")

    if SYM_UPDOWN and (Mq_all_dn is not None):
        mask_full = (dJ_all >= Jc1) & (dJ_all <= Jc2) if SYM_ONLY_PARA else np.ones_like(dJ_all, dtype=bool)
        Mq_all[mask_full] = 0.5*(Mq_all[mask_full] - Mq_all_dn[mask_full])
        if Me_all_dn is not None:
            Me_all[mask_full] = 0.5*(Me_all[mask_full] - Me_all_dn[mask_full])
        else:
            Me_all[mask_full] = 0.5*(Me_all[mask_full] - (-Me_all[mask_full]))

    dJ_plot, Mq_plot, Me_plot = dJ_all.copy(), Mq_all.copy(), Me_all.copy()
    therm_plot = dJ_plot < Jc1; sg_plot = dJ_plot > Jc2
    Mq_plot[therm_plot] = 0.0; Me_plot[therm_plot] = 0.0
    if SG_ABS:
        Mq_plot[sg_plot] = np.abs(Mq_plot[sg_plot])
        Me_plot[sg_plot] = np.abs(Me_plot[sg_plot])
    if FILTER_SG:
        Mq_plot[sg_plot & (np.abs(Mq_plot) < LIMIAR_SG)] = np.nan
        Me_plot[sg_plot & (np.abs(Me_plot) < LIMIAR_SG)] = np.nan

    mask_plot = np.ones_like(dJ_all, dtype=bool) if INCLUDE_HOLE else ((dJ_all<5.5)|(dJ_all>7.0))
    dJp, Mqp, Mep = dJ_all[mask_plot], Mq_all[mask_plot], Me_all[mask_plot]

    therm_all = dJp < Jc1; sg_all = dJp > Jc2
    Mqp[therm_all] = 0.0; Mep[therm_all] = 0.0
    if SG_ABS:
        Mqp[sg_all] = np.abs(Mqp[sg_all]); Mep[sg_all] = np.abs(Mep[sg_all])
    if FILTER_SG:
        Mqp[sg_all & (np.abs(Mqp) < LIMIAR_SG)] = np.nan
        Mep[sg_all & (np.abs(Mep) < LIMIAR_SG)] = np.nan

    deltas, idx = np.unique(dJp, return_inverse=True)
    counts  = np.array([np.sum(idx==i) for i in range(len(deltas))])
    Mq_mean = np.array([
        np.nanmean(Mqp[idx==i]) if np.any(np.isfinite(Mqp[idx==i])) else np.nan
        for i in range(len(deltas))
    ])
    Me_mean = np.array([
        np.nanmean(Mep[idx==i]) if np.any(np.isfinite(Mep[idx==i])) else np.nan
        for i in range(len(deltas))
    ])
    Mq_std = np.array([
        np.nanstd(Mqp[idx==i]) if np.count_nonzero(np.isfinite(Mqp[idx==i]))>1 else 0.0
        for i in range(len(deltas))
    ])
    Me_std = np.array([
        np.nanstd(Mep[idx==i]) if np.count_nonzero(np.isfinite(Mep[idx==i]))>1 else 0.0
        for i in range(len(deltas))
    ])

    Mq_vis, Me_vis = Mq_mean.copy(), Me_mean.copy()
    if PARA_K > 0:
        rng_vis = np.random.RandomState(0)
        for i in np.where((deltas>=Jc1)&(deltas<=Jc2))[0]:
            inds_i = np.where(idx==i)[0]
            if inds_i.size==0: continue
            take = min(PARA_K, inds_i.size); sel = rng_vis.permutation(inds_i)[:take]
            if PARA_STAT=="median":
                Mq_vis[i] = np.nanmedian(Mqp[sel]); Me_vis[i] = np.nanmedian(Mep[sel])
            else:
                Mq_vis[i] = np.nanmean(Mqp[sel]);  Me_vis[i] = np.nanmean(Mep[sel])

    if FILTER_SG:
        sg_bins = deltas > Jc2
        Mq_mean[sg_bins & (np.abs(Mq_mean) < LIMIAR_SG)] = np.nan
        Me_mean[sg_bins & (np.abs(Me_mean) < LIMIAR_SG)] = np.nan
        Mq_vis[sg_bins] = Mq_mean[sg_bins]; Me_vis[sg_bins] = Me_mean[sg_bins]

    if ERR_KIND=="sem":
        Mq_err = Mq_std/np.sqrt(np.maximum(counts,1)); Me_err = Me_std/np.sqrt(np.maximum(counts,1))
    else:
        Mq_err, Me_err = Mq_std, Me_std

    thermal_idxs = np.where(deltas<Jc1)[0][::THERMAL_STEP]
    mid_idxs     = np.where((deltas>=Jc1)&(deltas<=Jc2))[0][::MID_STEP]
    sg_idxs      = np.where(deltas>Jc2)[0][::SG_STEP]

    yjitter = np.zeros_like(deltas, dtype=float)
    if thermal_idxs.size:
        rng_jit = np.random.default_rng(42)
        yjitter[thermal_idxs] = rng_jit.uniform(-0.15,+0.15,size=thermal_idxs.size)

    # -------- Plot --------
    plt.rcParams.update({"figure.dpi":150, "font.family":"DejaVu Sans"})
    fig, ax = plt.subplots(1,1, figsize=(10,5.4))

    if PLOT_MEAN:
        ax.scatter(deltas[thermal_idxs], Mq_vis[thermal_idxs]+yjitter[thermal_idxs],
                   marker="o", alpha=0.85, facecolor="tab:orange", edgecolors="black", s=38, zorder=4)
        ax.scatter(deltas[mid_idxs], Mq_vis[mid_idxs],
                   marker="o", alpha=0.85, facecolor="tab:orange", edgecolors="black", s=38, zorder=4)
        ax.scatter(deltas[sg_idxs], Mq_vis[sg_idxs],
                   marker="o", alpha=0.85, facecolor="tab:orange", edgecolors="black", s=38, zorder=4)
        ax.scatter(deltas[thermal_idxs], Me_vis[thermal_idxs]+yjitter[thermal_idxs],
                   marker="^", alpha=0.9, s=50, facecolor="none", edgecolors="tab:green", zorder=3)
        ax.scatter(deltas[mid_idxs], Me_vis[mid_idxs],
                   marker="^", alpha=0.9, s=50, facecolor="none", edgecolors="tab:green", zorder=3)
        ax.scatter(deltas[sg_idxs], Me_vis[sg_idxs],
                   marker="^", alpha=0.9, s=50, facecolor="none", edgecolors="tab:green", zorder=3)
        if PLOT_ERR:
            ax.errorbar(deltas, Mq_mean, yerr=Mq_err, fmt='none', ecolor='tab:orange', alpha=0.35, capsize=2, lw=1)
            ax.errorbar(deltas, Me_mean, yerr=Me_err, fmt='none', ecolor='tab:green',  alpha=0.35, capsize=2, lw=1)

    ax.axvspan(0.1,Jc1,color="lightblue",alpha=0.3,zorder=0)
    ax.axvspan(Jc1,Jc2,color="lightgreen",alpha=0.3,zorder=0)
    ax.axvspan(Jc2,12.0,color="lightcoral",alpha=0.3,zorder=0)
    ax.axvline(Jc1,color="blue",ls="--",lw=1.5); ax.axvline(Jc2,color="purple",ls="--",lw=1.5)

    th  = mlines.Line2D([],[],color='tab:orange',marker='o',linestyle='None',label='QAS')
    ex  = mlines.Line2D([],[],color='tab:green',marker='^',linestyle='None',label='Exact')
    jl1 = mlines.Line2D([],[],color="blue",ls="--",lw=1.5,label=f"δJc₁ = {Jc1:.2f}")
    jl2 = mlines.Line2D([],[],color="purple",ls="--",lw=1.5,label=f"δJc₂ = {Jc2:.2f}")
    ax.legend(handles=[th,ex,jl1,jl2],loc="upper left",fontsize=10)

    ax.set_xlim(0,12); ax.set_xticks([0,2,4,6,8,12])
    ax.set_ylim(-NQ,NQ); ax.set_yticks(np.arange(-NQ,NQ+1,2))
    ax.set_title(f"All spins up — QAS vs Exact (statevector, N={NQ})")
    ax.set_xlabel(r"$\delta J$"); ax.set_ylabel(r"$\langle M_z\rangle$")
    ax.grid(ls="--",alpha=0.3)

    print("QAS finitos:  Therm:", np.sum(np.isfinite(Mqp[therm_all])),
          " Para:", np.sum(np.isfinite(Mqp[(dJp>=Jc1)&(dJp<=Jc2)])),
          " SG:",   np.sum(np.isfinite(Mqp[sg_all])))
    print("Exact finitos: Therm:", np.sum(np.isfinite(Mep[therm_all])),
          " Para:", np.sum(np.isfinite(Mep[(dJp>=Jc1)&(dJp<=Jc2)])),
          " SG:",   np.sum(np.isfinite(Mep[sg_all])))

    os.makedirs(OUTDIR, exist_ok=True)
    plt.tight_layout()
    out_png = os.path.join(OUTDIR, "qas_vs_exact_up_statevector.png")
    plt.savefig(out_png, dpi=150)
    print("Figura salva em", out_png)

    if MAKE_HISTS:
        fig2, axs = plt.subplots(1,3, figsize=(13,4), sharey=True)
        bins = np.arange(-NQ-0.5, NQ+1.6, 1.0)
        mask_therm = deltas < Jc1
        mask_para  = (deltas>=Jc1) & (deltas<=Jc2)
        mask_sg    = deltas > Jc2
        regions = [("Thermal", mask_therm), ("Para", mask_para), ("SG", mask_sg)]
        for ax2,(name,m) in zip(axs, regions):
            q = Mq_mean[m]; e = Me_mean[m]
            q = q[np.isfinite(q)]; e = e[np.isfinite(e)]
            plotted=False
            if q.size>0: ax2.hist(q,bins=bins,alpha=0.5,label="QAS (bin mean)"); plotted=True
            if e.size>0: ax2.hist(e,bins=bins,alpha=0.5,label="Exact (bin mean)"); plotted=True
            ax2.set_title(name); ax2.set_xlabel(r"$\langle \Sigma Z\rangle$"); ax2.grid(ls="--",alpha=0.3)
            if not plotted: ax2.text(0.5,0.5,"sem dados",ha="center",va="center",transform=ax2.transAxes)
        axs[0].set_ylabel("count"); axs[0].legend()
        out_png2 = os.path.join(OUTDIR, "qas_vs_exact_up_hists.png")
        plt.tight_layout(); plt.savefig(out_png2, dpi=150)
        print("Histogramas salvos em", out_png2)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try: mp.set_start_method("spawn")
    except RuntimeError: pass
    main()
