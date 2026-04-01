# plot_from_summaries.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def _load(npz_path: str):
    with np.load(npz_path, allow_pickle=False) as A:
        dJ = A["delta_J_grid"].astype(float)
        order = np.argsort(dJ)
        out = {k: (A[k].astype(float)[order] if A[k].ndim == 1 else A[k]) for k in A.files}
    meta = {}
    for k, v in out.items():
        if isinstance(v, np.ndarray) and v.shape == ():
            meta[k] = v.item()
    return out, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results_sv4/sweep_summary.npz")
    ap.add_argument("--exact", default=None)
    ap.add_argument("--out", default="results_sv4/qas_vs_exact_from_summaries.png")

    ap.add_argument("--jc1", type=float, default=None)
    ap.add_argument("--jc2", type=float, default=None)
    ap.add_argument("--sg-thr", type=float, default=None)

    ap.add_argument("--no-thermal-clamp", action="store_true",
                    help="Não zera a fase térmica (por padrão zera para ficar igual ao cluster).")
    args = ap.parse_args()

    S, metaS = _load(args.sweep)
    dJ = S["delta_J_grid"]
    Mq_mean = S["magnetization_mean"]
    NQ = int(S.get("NQ", metaS.get("NQ", 4)))

    JC1 = args.jc1 if args.jc1 is not None else float(S.get("JC1", metaS.get("JC1", 2.93)))
    JC2 = args.jc2 if args.jc2 is not None else float(S.get("JC2", metaS.get("JC2", 7.50)))
    SG_THR = args.sg_thr if args.sg_thr is not None else float(S.get("SG_THR", metaS.get("SG_THR", 3.6)))

    has_exact = (args.exact is not None) and os.path.exists(args.exact)
    if has_exact:
        E, metaE = _load(args.exact)
        Me_mean = E["magnetization_mean"]

    therm = dJ < JC1
    para  = (dJ >= JC1) & (dJ <= JC2)
    sg    = dJ > JC2

    Mq_plot = Mq_mean.copy()

    if not args.no_thermal_clamp:
        Mq_plot[therm] = 0.0

    if "magnetization_abs_mean_ge_thr" in S:
        sg_y = S["magnetization_abs_mean_ge_thr"].copy()
        if "magnetization_abs_mean" in S:
            sg_y = np.where(np.isfinite(sg_y), sg_y, S["magnetization_abs_mean"])
        sg_y = np.clip(sg_y, 0.0, float(NQ))
        Mq_plot[sg] = sg_y[sg]
    elif "magnetization_abs_mean" in S:
        sg_y = np.clip(S["magnetization_abs_mean"], 0.0, float(NQ))
        Mq_plot[sg] = sg_y[sg]
    else:
        Mq_plot[sg] = np.abs(Mq_plot[sg])

    if has_exact:
        Me_plot = Me_mean.copy()
        if not args.no_thermal_clamp:
            Me_plot[therm] = 0.0

        if "magnetization_abs_mean_ge_thr" in E:
            sg_y_e = E["magnetization_abs_mean_ge_thr"].copy()
            if "magnetization_abs_mean" in E:
                sg_y_e = np.where(np.isfinite(sg_y_e), sg_y_e, E["magnetization_abs_mean"])
            sg_y_e = np.clip(sg_y_e, 0.0, float(NQ))
            Me_plot[sg] = sg_y_e[sg]
        elif "magnetization_abs_mean" in E:
            sg_y_e = np.clip(E["magnetization_abs_mean"], 0.0, float(NQ))
            Me_plot[sg] = sg_y_e[sg]
        else:
            Me_plot[sg] = np.abs(Me_plot[sg])

    THERMAL_STEP = 4
    MID_STEP     = 4
    SG_STEP      = 2

    idx_th = np.where(therm)[0][::THERMAL_STEP]
    idx_pa = np.where(para)[0][::MID_STEP]
    idx_sg = np.where(sg)[0][::SG_STEP]
    idx_plot = np.unique(np.concatenate([idx_th, idx_pa, idx_sg]))

    dJp = dJ[idx_plot]
    MqP = Mq_plot[idx_plot]
    if has_exact:
        MeP = Me_plot[idx_plot]

    # ---------- plot ----------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.axvspan(dJ.min(), JC1,      color="lightblue",  alpha=0.30, zorder=0)
    ax.axvspan(JC1, JC2,          color="lightgreen", alpha=0.30, zorder=0)
    ax.axvspan(JC2, dJ.max(),     color="lightcoral", alpha=0.30, zorder=0)
    ax.axvline(JC1, color="blue",   ls="--", lw=1.5)
    ax.axvline(JC2, color="purple", ls="--", lw=1.5)

    ax.scatter(dJp, MqP, s=60, marker="o",
               facecolor="tab:orange", edgecolors="black", linewidths=0.6, label="QAS", zorder=3)
    if has_exact:
        ax.scatter(dJp, MeP, s=70, marker="^",
                   facecolor="tab:green", edgecolors="tab:green", linewidths=0.6, label="Exact", zorder=2)

    th  = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='None', label='QAS',
                        markerfacecolor='tab:orange', markeredgecolor='black')
    handles = [th]
    if has_exact:
        ex  = mlines.Line2D([], [], color='tab:green', marker='^', linestyle='None', label='Exact',
                            markerfacecolor='tab:green', markeredgecolor='tab:green')
        handles.append(ex)
    jl1 = mlines.Line2D([], [], color="blue",   ls="--", lw=1.5, label=f"ΔJc₁ = {JC1:.2f}")
    jl2 = mlines.Line2D([], [], color="purple", ls="--", lw=1.5, label=f"ΔJc₂ = {JC2:.2f}")
    handles += [jl1, jl2]
    ax.legend(handles=handles, loc="upper left")

    ax.set_xlim(0, 12)
    ax.set_ylim(-NQ, NQ)
    ax.set_yticks(np.arange(-NQ, NQ + 1, 1))
    ax.set_xlabel(r"$\Delta J$")
    ax.set_ylabel(r"$\langle M_z \rangle$")
    ax.set_title(f"All spins up — QAS vs Exact (N={NQ})")
    ax.grid(ls="--", alpha=0.30)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("Figura salva em:", args.out)
    print(f"(info) JC1={JC1:.2f}, JC2={JC2:.2f}, SG_THR={SG_THR:.2f}")

if __name__ == "__main__":
    main()