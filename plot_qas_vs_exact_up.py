# plot_qas_vs_exact_up.py
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

# --- Flags via ambiente ---
PLOT_MEAN = os.environ.get("PLOT_MEAN", "0") == "1"   # 0 => não plota médias (estilo cluster)
REBUILD_AGG       = os.environ.get("REBUILD_AGG", "0") == "1"
FILTER_SG         = os.environ.get("FILTER_SG",   "0") == "1"
LIMIAR_SG         = float(os.environ.get("LIMIAR_SG", "3.6"))
PLOT_ERR          = os.environ.get("PLOT_ERR",    "1") == "1"
ERR_KIND          = os.environ.get("ERR_KIND",    "std")  # "std" ou "sem"
MAKE_HISTS        = os.environ.get("MAKE_HISTS",  "1") == "1"
NTIME_AVG         = int(os.environ.get("NTIME_AVG", "1"))
T_AVG_START_RATIO = float(os.environ.get("T_AVG_START_RATIO", "1.0"))
USE_LEGACY_RNG = os.environ.get("LEGACY_RNG", "1") == "1"
POS_DJ         = os.environ.get("POS_DJ", "0") == "1"   # 0 = simétrico (cluster), 1 = só positivos
PHASE_LOCK     = os.environ.get("PHASE_LOCK", "0") == "1"
TIME_JITTER_FRAC = float(os.environ.get("TIME_JITTER_FRAC", "0.0"))  # fração do período 2π/h
# SHOW_MASKED_HOLE = os.environ.get("SHOW_MASKED_HOLE", "0") == "1"    # se quiser ver (5.5,7.0) no plot
THERMAL_STEP     = int(os.environ.get("THERMAL_STEP", "4"))  # subamostra na fase térmica
MID_STEP         = int(os.environ.get("MID_STEP",     "4"))  # subamostra na fase para-magnética
SG_STEP          = int(os.environ.get("SG_STEP",      "2"))  # subamostra na SG
RAW_K_PER_DELTA  = int(os.environ.get("RAW_K_PER_DELTA", "0"))  # 0 = desliga; >0 = máx K pts por δJ
PLOT_ONLY_PARA = os.environ.get("PLOT_ONLY_PARA", "0") == "1"
INCLUDE_HOLE   = os.environ.get("INCLUDE_HOLE", "1") == "1"   # 1 => inclui 5.5–7.0 no PLOT
PARA_K         = int(os.environ.get("PARA_K", "6"))           # nº de realizações por ΔJ usadas no traço da fase para
PARA_STAT      = os.environ.get("PARA_STAT", "median")        # 'median' ou 'mean' no subconjunto


J1_FORCE = os.environ.get("J1_FORCE", "").strip()
J2_FORCE = os.environ.get("J2_FORCE", "").strip()

# ===================== Pauli / modelo =====================
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
    return sum(mats)

def build_H_matrix(nq: int, J: float, h: float, J2: float, delta_J: float, rng):
    X_labels, ZZ_nn_labels, ZZ_nnn_labels = disordered_ising_ops(nq)
    sparse_list = []

    # -h * sum_i X_i
    sparse_list.extend((lab, -h) for lab in X_labels)

    # - sum_i (J + dJ_i) Z_i Z_{i+1}
    for lab in ZZ_nn_labels:
        if POS_DJ:
            dJi = rng.uniform(0.0, delta_J)
        else:
            dJi = rng.uniform(-delta_J, delta_J)
        sparse_list.append((lab, -(J + dJi)))

    # -J2 * sum_i Z_i Z_{i+2}
    sparse_list.extend((lab, -J2) for lab in ZZ_nnn_labels)

    return SparsePauliOp.from_list(sparse_list, num_qubits=nq).to_matrix()

def stable_seed_for_deltaJ(delta_J: float, base_seed: int):

    dj_key = int(f"{float(delta_J):.2f}".replace(".", ""))  # 3.50 -> 350
    return (int(base_seed) * 1_000_003 + dj_key) % (2**31 - 1)

def _phase_locked_T(Tfin, h):
    if not PHASE_LOCK:
        return Tfin
    period = 2*np.pi/float(h)
    k = max(1, int(round(Tfin/period)))
    return k*period

# ===================== Base QAS por QR (igual cluster) =====================
def qr_basis_from_pauli_products(nq: int, K: int):
    """Base ortonormal Q via Gram–Schmidt sobre produtos de Pauli aplicados em |0...0>."""
    X, ZZ_nn, ZZ_nnn = disordered_ising_ops(nq)
    ops = [SparsePauliOp.from_list([(lab, 1.0)]) for lab in (X + ZZ_nn + ZZ_nnn)]
    from itertools import combinations

    psi0 = Statevector.from_label('0'*nq).data
    dim  = 2**nq
    Q = np.zeros((dim, 0), dtype=complex)

    def orth_residual(v, Q):
        if Q.shape[1] == 0:
            return v, np.linalg.norm(v)
        r = v - Q @ (Q.conj().T @ v)
        return r, np.linalg.norm(r)

    # começa com |0...0>
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

def precompute_qas_structs_qr(nq: int, K: int):
    Q = qr_basis_from_pauli_products(nq, K)
    psi0 = Statevector.from_label('0'*nq).data
    M_full = magnetization_matrix(nq)
    M_tilde = Q.conj().T @ (M_full @ Q)
    alpha0 = Q.conj().T @ psi0
    return Q, M_tilde, alpha0

def qas_one_mag_qr(delta_J, seed, params, Q, M_tilde, alpha0,
                   n_time=1, t0_ratio=1.0):
    nq = params["NQ"]; J=params["J"]; h=params["h"]; J2=params["J2"]; Tfin=params["FINAL_T"]
    rng = (np.random.RandomState(seed) if USE_LEGACY_RNG else np.random.default_rng(seed))
    H = build_H_matrix(nq, J, h, J2, delta_J, rng)
    Dp = Q.conj().T @ (H @ Q)
    Dp = 0.5 * (Dp + Dp.conj().T)
    w, V = np.linalg.eigh(Dp)
    beta0 = V.conj().T @ alpha0
    Tfin = _phase_locked_T(Tfin, h)
    if TIME_JITTER_FRAC > 0:
        period = 2*np.pi/float(h)
        Tfin = Tfin + (rng.uniform(-TIME_JITTER_FRAC, TIME_JITTER_FRAC) * period)
    times = np.linspace(Tfin*t0_ratio, Tfin, max(1, n_time))
    
    mz_vals = []
    for t in times:
        alpha_t = V @ (np.exp(-1j*w*t) * beta0)
        mz_vals.append(np.real(alpha_t.conj().T @ (M_tilde @ alpha_t)))
    return float(np.mean(mz_vals))

# ===================== EXATO (statevector full) =====================
def exact_one_statevector(delta_J, seed, params, M_full,
                          n_time=1, t0_ratio=1.0):
    nq = params["NQ"]; J=params["J"]; h=params["h"]; J2=params["J2"]; Tfin=params["FINAL_T"]
    rng = (np.random.RandomState(seed) if USE_LEGACY_RNG else np.random.default_rng(seed))
    H = build_H_matrix(nq, J, h, J2, delta_J, rng)
    w, V = np.linalg.eigh(H)
    psi0 = Statevector.from_label('0' * nq).data
    c0   = V.conj().T @ psi0
    Tfin = _phase_locked_T(Tfin, h)
    if TIME_JITTER_FRAC > 0:
        period = 2*np.pi/float(h)
        Tfin = Tfin + (rng.uniform(-TIME_JITTER_FRAC, TIME_JITTER_FRAC) * period)
    times = np.linspace(Tfin*t0_ratio, Tfin, max(1, n_time))

    mz_vals = []
    for t in times:
        psi_t = V @ (np.exp(-1j*w*t) * c0)
        mz_vals.append(np.real(np.vdot(psi_t, M_full @ psi_t)))
    return float(np.mean(mz_vals))

# ===================== Ajuste crítico (mesmo do cluster) =====================
def double_sigmoid(x, A1, B1, xc1, k1, A2, B2, xc2, k2):
    return A1/(1+np.exp(-k1*(x-xc1))) + B1 + A2/(1+np.exp(-k2*(x-xc2))) + B2

def process_file_abs_like_cluster(dJ, Mq, Me,
                                  frac=0.2, nbins=40,
                                  xc1_init=2.7, xc2_init=6.1,
                                  bound1=(2.0, 3.5), bound2=(5.5, 7.5),
                                  weight_exact=0.5):
    dJ = np.asarray(dJ); Mq = np.asarray(Mq); Me = np.asarray(Me)
    Mf_abs = weight_exact*np.abs(Mq) + (1-weight_exact)*np.abs(Me)
    mask = (dJ < 5.5) | (dJ > 7.0)
    dJf, Mabs = dJ[mask], Mf_abs[mask]
    med_bias = np.median(Mabs[dJf < bound1[0]])
    Mabs = Mabs - med_bias

    edges   = np.logspace(np.log10(0.1), np.log10(12.0), nbins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    inds    = np.digitize(dJf, edges)
    medians = np.array([np.median(Mabs[inds==i]) if np.any(inds==i) else np.nan
                        for i in range(1, len(edges))])
    valid = np.isfinite(medians)
    xsm, ysm = lowess(medians[valid], centers[valid], frac=frac, return_sorted=True).T

    A1 = 0.5*(ysm.max()-ysm.min()); B1 = ysm.min()
    p0 = [A1,B1, xc1_init, 0.5, A1,B1, xc2_init, 0.5]
    lb = [0, B1, bound1[0], 0.01, 0, B1, bound2[0], 0.01]
    ub = [2*A1, B1+2*A1, bound1[1], 50.0, 2*A1, B1+2*A1, bound2[1], 50.0]
    popt,_ = curve_fit(double_sigmoid, xsm, ysm, p0=p0, bounds=(lb,ub), maxfev=50000)
    xc1, xc2 = float(popt[2]), float(popt[6])

    xfit = np.linspace(xsm.min(), xsm.max(), 400)
    yfit = double_sigmoid(xfit, *popt)
    return {"Jc1": xc1, "Jc2": xc2, "xfit": xfit, "yfit": yfit}

def _as_str(x):
    try:
        return str(x.item())
    except Exception:
        return str(x)

# ===================== Agregado (120 dJ × 500) =====================
def build_or_load_aggregated_up_sv(base_seed: int, params: dict,
                                   K: int = 4,
                                   N_REALIZ: int = 500,
                                   GRID: np.ndarray | None = None):
    """
    Gera (ou reutiliza) results_sv4/aggregated_up_sv.npz com:
      - QAS em base ortonormal (QR) no statevector
      - EXATO full no statevector
    """
    BASIS_KIND = "qr"
    RNG_KIND = "RandomState" if USE_LEGACY_RNG else "PCG64"

    out_path = os.path.join("results_sv4", "aggregated_up_sv.npz")
    if REBUILD_AGG and os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass
    if GRID is None:
        GRID = np.array([float(f"{x:.2f}") for x in np.linspace(0.1, 12.0, 120)], dtype=float)

    # tenta cache
    if os.path.exists(out_path):
        try:
            with np.load(out_path, allow_pickle=False) as A:
                need = {"delta_J","magnetization_mean","exact_result",
                        "NQ","J","h","J2","FINAL_T","BASE_SEED","K","N_REALIZ",
                        "NTIME_AVG","T_AVG_START_RATIO","grid","BASIS_KIND", "RNG_KIND", "PHASE_LOCK", "POS_DJ", "TIME_JITTER_FRAC"}
                if need.issubset(A.files) \
                   and int(A["NQ"])==params["NQ"] \
                   and np.isclose(float(A["J"]), params["J"]) \
                   and np.isclose(float(A["h"]), params["h"]) \
                   and np.isclose(float(A["J2"]),params["J2"]) \
                   and np.isclose(float(A["FINAL_T"]),params["FINAL_T"]) \
                   and int(A["BASE_SEED"])==base_seed \
                   and int(A["K"])==K and int(A["N_REALIZ"])==N_REALIZ \
                   and int(A["NTIME_AVG"])==NTIME_AVG \
                   and np.isclose(float(A["T_AVG_START_RATIO"]), T_AVG_START_RATIO) \
                   and np.array_equal(A["grid"], GRID) \
                   and _as_str(A["BASIS_KIND"]) == BASIS_KIND \
                   and int(A["POS_DJ"]) == int(POS_DJ) \
                   and int(A["PHASE_LOCK"]) == int(PHASE_LOCK) \
                   and np.isclose(float(A["TIME_JITTER_FRAC"]), TIME_JITTER_FRAC) \
                   and _as_str(A["RNG_KIND"]) == RNG_KIND:
                    return out_path
        except Exception:
            pass

    os.makedirs("results_sv4", exist_ok=True)

    # QAS: base ortonormal por QR
    Q, M_tilde, alpha0 = precompute_qas_structs_qr(params["NQ"], K)

    # EXATO: operador de magnetização
    M_full = magnetization_matrix(params["NQ"])

    D_list, Mq_list, Me_list = [], [], []
    for dj in GRID:
        seed0 = stable_seed_for_deltaJ(float(dj), base_seed)
        for k in range(N_REALIZ):
            s = int(seed0 + k)
            mz_qas = qas_one_mag_qr(float(dj), s, params, Q, M_tilde, alpha0,
                                    n_time=NTIME_AVG, t0_ratio=T_AVG_START_RATIO)
            mz_ex  = exact_one_statevector(float(dj), s, params, M_full,
                                           n_time=NTIME_AVG, t0_ratio=T_AVG_START_RATIO)
            D_list.append(float(dj)); Mq_list.append(mz_qas); Me_list.append(mz_ex)

    delta_J = np.asarray(D_list, dtype=float)
    Mq      = np.asarray(Mq_list, dtype=float)
    Me      = np.asarray(Me_list, dtype=float)

    np.savez(out_path,
             delta_J=delta_J,
             magnetization_mean=Mq,
             exact_result=Me,
             grid=GRID,
             NQ=params["NQ"], J=params["J"], h=params["h"], J2=params["J2"],
             FINAL_T=params["FINAL_T"], BASE_SEED=base_seed,
             K=K, N_REALIZ=N_REALIZ,
             NTIME_AVG=NTIME_AVG, T_AVG_START_RATIO=T_AVG_START_RATIO,
             BASIS_KIND=BASIS_KIND,
             RNG_KIND=RNG_KIND,
             POS_DJ=int(POS_DJ),
             PHASE_LOCK=int(PHASE_LOCK),
             TIME_JITTER_FRAC=TIME_JITTER_FRAC,
             note="QAS via base ortonormal (QR) + EXATO full — ambos em statevector.")
    print("Agregado salvo em", out_path, f"(amostras: {delta_J.size})")
    return out_path

# ===================== Main (plot estilo cluster) =====================
def main():
    sweep = os.path.join("results_sv4", "sweep_summary.npz")
    if os.path.exists(sweep):
        S = np.load(sweep)
        NQ = int(S["NQ"]); J = float(S["J"]); h = float(S["h"]); J2 = float(S["J2"])
    else:
        NQ, J, h, J2 = 4, 1.0, 0.6, 0.3

    Tfin = float(os.environ.get("FINAL_T", 400.0))
    BASE_SEED = int(os.environ.get("GLOBAL_SEED", "1234"))
    params = {"NQ": NQ, "J": J, "h": h, "J2": J2, "FINAL_T": Tfin}

    # Gera/usa agregado (120 dJ × 500)
    NREAL = int(os.environ.get("N_REALIZ", "1"))
    agg_path = build_or_load_aggregated_up_sv(BASE_SEED, params, K=4, N_REALIZ=NREAL)

    # Carrega amostras cruas
    data   = np.load(agg_path)
    dJ_all = np.abs(data["delta_J"])
    Mq_all = data["magnetization_mean"].astype(float).copy()
    Me_all = data["exact_result"].astype(float).copy()

    # mesmo mask do cluster para PLOT/estatística
    # usa o "buraco" só se pedirmos explicitamente
    if INCLUDE_HOLE:
        mask_plot = np.ones_like(dJ_all, dtype=bool)
    else:
        mask_plot = (dJ_all < 5.5) | (dJ_all > 7.0)

    dJp  = dJ_all[mask_plot]
    Mqp  = Mq_all[mask_plot]
    Mep  = Me_all[mask_plot]

    # Pontos críticos (pipeline do cluster usa máscara internamente)
    fit = process_file_abs_like_cluster(
        dJ_all, Mq_all, Me_all,
        frac=0.2, nbins=40,
        xc1_init=2.7, xc2_init=6.1,
        bound1=(2.0, 3.5), bound2=(5.5, 7.5),
        weight_exact=0.5
    )
    Jc1, Jc2 = fit["Jc1"], fit["Jc2"]
    if J1_FORCE:
        Jc1 = float(J1_FORCE)
    if J2_FORCE:
        Jc2 = float(J2_FORCE)
    print(f"δJc₁ ≈ {Jc1:.2f},  δJc₂ ≈ {Jc2:.2f}")

    # Cópias para PLOT em TODO o range (sem “buraco” de 5.5–7.0)
    dJ_plot = dJ_all.copy()
    Mq_plot = Mq_all.copy()
    Me_plot = Me_all.copy()

    # Clamp/ABS/filtro nas CÓPIAS de plot
    therm_plot = dJ_plot < Jc1
    sg_plot    = dJ_plot > Jc2
    Mq_plot[therm_plot] = 0.0; Me_plot[therm_plot] = 0.0
    SG_ABS = os.environ.get("SG_ABS", "1") == "1"
    if SG_ABS:
        Mq_plot[sg_plot] = np.abs(Mq_plot[sg_plot])
        Me_plot[sg_plot] = np.abs(Me_plot[sg_plot])
    if FILTER_SG:
        Mq_plot[sg_plot & (np.abs(Mq_plot) < LIMIAR_SG)] = np.nan
        Me_plot[sg_plot & (np.abs(Me_plot) < LIMIAR_SG)] = np.nan

    # --- Clamp/filtro em nível de amostra (sobre arrays filtrados) ---
    therm_all = dJp < Jc1
    sg_all    = dJp > Jc2
    Mqp[therm_all] = 0.0
    Mep[therm_all] = 0.0
    
    if SG_ABS:
        Mqp[sg_all] = np.abs(Mqp[sg_all])
        Mep[sg_all] = np.abs(Mep[sg_all])

    if FILTER_SG:
        Mqp[sg_all & (np.abs(Mqp) < LIMIAR_SG)] = np.nan
        Mep[sg_all & (np.abs(Mep) < LIMIAR_SG)] = np.nan

    # --- Médias/erros por ΔJ (após clamp/filtro) ---
    deltas, idx = np.unique(dJp, return_inverse=True)
    Mq_mean = np.array([np.nanmean(Mqp[idx == i]) for i in range(len(deltas))])
    Me_mean = np.array([np.nanmean(Mep[idx == i]) for i in range(len(deltas))])
    counts  = np.array([np.sum(idx == i) for i in range(len(deltas))])
    Mq_std  = np.array([np.nanstd(Mqp[idx == i]) for i in range(len(deltas))])
    Me_std  = np.array([np.nanstd(Mep[idx == i]) for i in range(len(deltas))])

    # --- traço "visual" com dente-serrilhado somente na fase para ---
    Mq_vis = Mq_mean.copy()
    Me_vis = Me_mean.copy()

    if PARA_K > 0:
        for i in np.where((deltas >= Jc1) & (deltas <= Jc2))[0]:  # só região para
            inds_i = np.where(idx == i)[0]
            if inds_i.size == 0:
                continue
            take = min(PARA_K, inds_i.size)
            sel  = inds_i[:take]  # determinístico; se quiser aleatório: rng = np.random.default_rng(0); sel = rng.choice(inds_i, size=take, replace=False)
            if PARA_STAT.lower() == "median":
                Mq_vis[i] = np.nanmedian(Mqp[sel])
                Me_vis[i] = np.nanmedian(Mep[sel])
            else:
                Mq_vis[i] = np.nanmean(Mqp[sel])
                Me_vis[i] = np.nanmean(Mep[sel])
    else:
        Mq_vis = Mq_mean
        Me_vis = Me_mean


    # Filtro SG em nível de bin (igual ao cluster)
    if FILTER_SG:
        sg_bins = deltas > Jc2
        Mq_mean[sg_bins & (np.abs(Mq_mean) < LIMIAR_SG)] = np.nan
        Me_mean[sg_bins & (np.abs(Me_mean) < LIMIAR_SG)] = np.nan
        Mq_vis[sg_bins] = Mq_mean[sg_bins]
        Me_vis[sg_bins] = Me_mean[sg_bins]


    if ERR_KIND.lower() == "sem":
        Mq_err = Mq_std / np.sqrt(np.maximum(counts, 1))
        Me_err = Me_std / np.sqrt(np.maximum(counts, 1))
    else:
        Mq_err, Me_err = Mq_std, Me_std

    # Índices por fase
    thermal_idxs = np.where(deltas < Jc1)[0][::THERMAL_STEP]
    mid_idxs     = np.where((deltas >= Jc1) & (deltas <= Jc2))[0][::MID_STEP]
    sg_idxs      = np.where(deltas > Jc2)[0][::SG_STEP]

    # jitter determinístico na fase térmica
    yjitter = np.zeros_like(deltas)
    if thermal_idxs.size:
        rng_jit = np.random.default_rng(42)
        yjitter[thermal_idxs] = rng_jit.uniform(-0.15, +0.15, size=thermal_idxs.size)

    x_qas = deltas 
    x_ex  = deltas 

    # --- Plot ---
    plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # --- PLOT ESTILO CLUSTER: pontos brutos em todo o range ---
    SHOW_RAW = os.environ.get("SHOW_RAW","0") == "1"
    RAW_SUBSAMPLE = int(os.environ.get("RAW_SUBSAMPLE","1000000"))

    if SHOW_RAW:
        rng = np.random.default_rng(0)

        def sample_per_delta(mask, K):
            dJr = np.round(dJ_plot, 2)
            out = []
            for dj in np.unique(dJr[mask]):
                cand = np.where(mask & (dJr == dj))[0]
                if cand.size:
                    k = min(K, cand.size) if K > 0 else cand.size
                    out.extend(rng.choice(cand, size=k, replace=False))
            return np.array(out, dtype=int)

        if PLOT_ONLY_PARA:
            # Só pontos da fase para (e inclui 5.5–7.0 porque não mascaramos nada aqui)
            mid_mask_q = np.isfinite(Mq_plot) & (dJ_plot >= Jc1) & (dJ_plot <= Jc2)
            mid_mask_e = np.isfinite(Me_plot) & (dJ_plot >= Jc1) & (dJ_plot <= Jc2)

            idx_all_q = sample_per_delta(mid_mask_q, RAW_K_PER_DELTA)
            idx_all_e = sample_per_delta(mid_mask_e, RAW_K_PER_DELTA)
        else:
            # fallback antigo (global)
            idx_all_q = np.where(np.isfinite(Mq_plot))[0]
            idx_all_e = np.where(np.isfinite(Me_plot))[0]
            if idx_all_q.size > RAW_SUBSAMPLE:
                idx_all_q = rng.choice(idx_all_q, size=RAW_SUBSAMPLE, replace=False)
            if idx_all_e.size > RAW_SUBSAMPLE:
                idx_all_e = rng.choice(idx_all_e, size=RAW_SUBSAMPLE, replace=False)

        # estilo dos pontos (um pouco maiores, mais legíveis)
        ax.scatter(dJ_plot[idx_all_q], Mq_plot[idx_all_q],
                s=28, alpha=0.9, facecolor="tab:orange",
                edgecolors="black", linewidths=0.3, zorder=3, label=None)

        ax.scatter(dJ_plot[idx_all_e], Me_plot[idx_all_e],
                s=32, alpha=0.9, marker="^", facecolor="none",
                edgecolors="tab:green", linewidths=0.8, zorder=2, label=None)

    # Se **não** queremos as médias (estilo cluster), pulamos os scatters agregados:
    if not PLOT_MEAN:
        # apenas mantém a legenda “simbólica”
        pass
    else:
        # --- (o bloco de scatters por fase COM médias e barras de erro) ---
        # QAS
        ax.scatter(x_qas[thermal_idxs], Mq_vis[thermal_idxs] + yjitter[thermal_idxs],
                marker="o", alpha=0.85, facecolor="tab:orange", edgecolors="black",
                s=40, label="QAS (thermal)", zorder=4)
        ax.scatter(x_qas[mid_idxs], Mq_vis[mid_idxs],
                marker="o", alpha=0.85, facecolor="tab:orange", edgecolors="black",
                s=40, label="QAS (para-mag)", zorder=4)
        ax.scatter(x_qas[sg_idxs], Mq_vis[sg_idxs],
                marker="o", alpha=0.85, facecolor="tab:orange", edgecolors="black",
                s=40, label="QAS (spin-glass)", zorder=4)
        # Exact
        ax.scatter(x_ex[thermal_idxs], Me_vis[thermal_idxs] + yjitter[thermal_idxs],
                marker="^", alpha=0.9, s=50, facecolor="none", edgecolors="tab:green",
                label="Exact (thermal)", zorder=3)
        ax.scatter(x_ex[mid_idxs], Me_vis[mid_idxs],
                marker="^", alpha=0.9, s=50, facecolor="none", edgecolors="tab:green",
                label="Exact (para-mag)", zorder=3)
        ax.scatter(x_ex[sg_idxs], Me_vis[sg_idxs],
                marker="^", alpha=0.9, s=50, facecolor="none", edgecolors="tab:green",
                label="Exact (spin-glass)", zorder=3)
        # Barras de erro (só fazem sentido com médias)
        if PLOT_ERR:
            ax.errorbar(x_qas, Mq_mean, yerr=Mq_err, fmt='none',
                        ecolor='tab:orange', alpha=0.4, capsize=2, lw=1)
            ax.errorbar(x_ex, Me_mean, yerr=Me_err, fmt='none',
                        ecolor='tab:green', alpha=0.4, capsize=2, lw=1)


    # Faixas/linhas de fase
    ax.axvspan(0.1,  Jc1, color="lightblue",  alpha=0.3, zorder=0)
    ax.axvspan(Jc1,  Jc2, color="lightgreen", alpha=0.3, zorder=0)
    ax.axvspan(Jc2, 12.0, color="lightcoral", alpha=0.3, zorder=0)
    ax.axvline(Jc1, color="blue",   ls="--", lw=1.5)
    ax.axvline(Jc2, color="purple", ls="--", lw=1.5)

    th  = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='None', label='QAS')
    ex  = mlines.Line2D([], [], color='tab:green', marker='^', linestyle='None', label='Exact')
    jl1 = mlines.Line2D([], [], color="blue",   ls="--", lw=1.5, label=f"δJc₁ = {Jc1:.2f}")
    jl2 = mlines.Line2D([], [], color="purple", ls="--", lw=1.5, label=f"δJc₂ = {Jc2:.2f}")
    ax.legend(handles=[th, ex, jl1, jl2], loc="upper left", fontsize=10)

    ax.set_xlim(0, 12); ax.set_xticks([0, 2, 4, 6, 8, 12])
    ax.set_ylim(-NQ, NQ); ax.set_yticks(np.arange(-NQ, NQ+1, 1))
    ax.set_title(f"All spins up — QAS vs Exact (statevector, N={NQ})")
    ax.set_xlabel(r"$\delta J$"); ax.set_ylabel(r"$\langle M_z\rangle$")
    ax.grid(ls="--", alpha=0.3)

    # Máscaras por região (para histogramas)
    mask_therm = deltas < Jc1
    mask_para  = (deltas >= Jc1) & (deltas <= Jc2)
    mask_sg    = deltas > Jc2

    print("QAS finitos:  Therm:", np.sum(np.isfinite(Mqp[therm_all])),
      " Para:", np.sum(np.isfinite(Mqp[(dJp>=Jc1)&(dJp<=Jc2)])),
      " SG:",   np.sum(np.isfinite(Mqp[sg_all])))
    print("Exact finitos: Therm:", np.sum(np.isfinite(Mep[therm_all])),
      " Para:", np.sum(np.isfinite(Mep[(dJp>=Jc1)&(dJp<=Jc2)])),
      " SG:",   np.sum(np.isfinite(Mep[sg_all])))


    os.makedirs("results_sv4", exist_ok=True)

    plt.tight_layout()
    out_png = os.path.join("results_sv4", "qas_vs_exact_up_statevector.png")
    plt.savefig(out_png, dpi=150)
    print("Figura salva em", out_png)

    if MAKE_HISTS:
        fig2, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        bins = np.arange(-NQ-0.5, NQ+1.6, 1.0)
        regions = [("Thermal", mask_therm), ("Para", mask_para), ("SG", mask_sg)]
        for ax2, (name, m) in zip(axs, regions):
            q = Mq_mean[m]
            e = Me_mean[m]
            q = q[np.isfinite(q)]
            e = e[np.isfinite(e)]
            plotted = False
            if q.size > 0:
                ax2.hist(q, bins=bins, alpha=0.5, label="QAS (bin mean)")
                plotted = True
            if e.size > 0:
                ax2.hist(e, bins=bins, alpha=0.5, label="Exact (bin mean)")
                plotted = True
            ax2.set_title(name); ax2.set_xlabel(r"$\langle \Sigma Z\rangle$")
            ax2.grid(ls="--", alpha=0.3)
            if not plotted:
                ax2.text(0.5, 0.5, "sem dados", ha="center", va="center", transform=ax2.transAxes)
        axs[0].set_ylabel("count"); axs[0].legend()
        out_png2 = os.path.join("results_sv4", "qas_vs_exact_up_hists.png")
        plt.tight_layout(); plt.savefig(out_png2, dpi=150)
        print("Histogramas salvos em", out_png2)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
