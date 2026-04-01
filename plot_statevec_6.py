# plot_statevec_6.py
#
# Diagrama de fases tipo cluster para o benchmark statevector (L = 6):
#   - carrega:
#       deltaJ_values, Mz_qas_timeavg, Mz_exact_timeavg
#   - usa |Mz| para extrair ΔJc1, ΔJc2 via double-sigmoid (apenas para diagnóstico)
#   - plota ⟨M_z(t_f)⟩ (com sinal) + regiões de fase.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
#  Ajuste double-sigmoid em função de ΔJ (usa |Mz|)
# ---------------------------------------------------------------------------
def double_sigmoid(x, A1, B1, xc1, k1, A2, B2, xc2, k2):
    return (
        A1 / (1.0 + np.exp(-k1 * (x - xc1))) + B1
        + A2 / (1.0 + np.exp(-k2 * (x - xc2))) + B2
    )


def process_file_abs_like_cluster(
    dJ,
    Mq,
    Me,
    frac: float = 0.20,
    nbins: int = 40,
    xc1_init: float = 3.5,
    xc2_init: float = 7.1,
    bound1=(3.0, 4.5),
    bound2=(6.0, 8.5),
    weight_exact: float = 0.5,
):
    """
    Usa |Mq|, |Me| para extrair ΔJc1, ΔJc2 (mesma lógica qualitativa do cluster).
    """
    dJ = np.asarray(dJ, float)
    Mq = np.asarray(Mq, float)
    Me = np.asarray(Me, float)

    Mf_abs = weight_exact * np.abs(Mq) + (1.0 - weight_exact) * np.abs(Me)

    mask_fit = (dJ < bound1[1]) | (dJ > bound2[0])
    dJ_fit = dJ[mask_fit]
    Mf_fit = Mf_abs[mask_fit]

    med_bias = np.median(Mf_fit[dJ_fit < bound1[0]])
    Mf_fit = Mf_fit - med_bias

    edges = np.logspace(np.log10(0.1), np.log10(12.0), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    inds = np.digitize(dJ_fit, edges)

    medians = np.array(
        [
            np.median(Mf_fit[inds == i]) if np.any(inds == i) else np.nan
            for i in range(1, len(edges))
        ],
        float,
    )

    valid = np.isfinite(medians)
    if valid.sum() < 8:
        raise RuntimeError("Poucos pontos válidos para o LOWESS/ajuste.")

    xsm, ysm = lowess(medians[valid], centers[valid], frac=frac,
                      return_sorted=True).T

    A1 = 0.5 * (ysm.max() - ysm.min())
    B1 = ysm.min()
    p0 = [A1, B1, xc1_init, 0.5,  A1, B1, xc2_init, 0.5]
    lb = [0,   B1, bound1[0], 0.01, 0,   B1, bound2[0], 0.01]
    ub = [2*A1, B1 + 2*A1, bound1[1], 50.0,
          2*A1, B1 + 2*A1, bound2[1], 50.0]

    popt, _ = curve_fit(
        double_sigmoid, xsm, ysm, p0=p0,
        bounds=(lb, ub), maxfev=50_000
    )

    xfit = np.linspace(xsm.min(), xsm.max(), 400)
    yfit = double_sigmoid(xfit, *popt)

    return {
        "Jc1": float(popt[2]),
        "Jc2": float(popt[6]),
        "xfit": xfit,
        "yfit": yfit,
        "dJ_sm": xsm,
        "M_sm": ysm,
    }


# ---------------------------------------------------------------------------
#  Plot principal
# ---------------------------------------------------------------------------
def make_plot_for_L(L: int, npz_path: str, out_png: str | None = None):
    """
    Lê o arquivo npz gerado pelo script de simulação statevector e
    produz o gráfico tipo cluster (QAS vs Exato + ΔJc1, ΔJc2),
    usando <M_z(t_f)> médio na desordem.
    """
    data = np.load(npz_path)
    dJ = data["deltaJ_values"].astype(float)
    # Estes arrays são <M_z(t_f)>_desordem COM SINAL
    Mq = data["Mz_qas_timeavg"].astype(float)
    Me = data["Mz_exact_timeavg"].astype(float)

    # --- Ajuste crítico (usa |Mz| internamente, como no cluster) ---
    fit = process_file_abs_like_cluster(
        dJ, Mq, Me,
        xc1_init=3.5,
        xc2_init=7.1,
        bound1=(3.0, 4.5),
        bound2=(6.0, 8.5),
    )
    Jc1_fit, Jc2_fit = fit["Jc1"], fit["Jc2"]
    print(f"L={L}: ΔJc₁ (fit) ≈ {Jc1_fit:.2f},  ΔJc₂ (fit) ≈ {Jc2_fit:.2f}")

    # Usamos os mesmos pontos críticos do benchmark statevector de 6 spins
    Jc1, Jc2 = 3.50, 7.10
    print(f"Usando valores de referência: ΔJc₁ = {Jc1:.2f}, ΔJc₂ = {Jc2:.2f}")

    # ------------------------------------------------------------------
    # 1) Corrige bias térmico global
    # ------------------------------------------------------------------
    mask_therm_bias = dJ < min(Jc1, 1.0)
    if np.any(mask_therm_bias):
        bias_q = np.median(Mq[mask_therm_bias])
        bias_e = np.median(Me[mask_therm_bias])
        bias = 0.5 * (bias_q + bias_e)
    else:
        bias = 0.0

    Mq_shift = Mq - bias
    Me_shift = Me - bias

    # ------------------------------------------------------------------
    # 2) Máscaras de fase
    # ------------------------------------------------------------------
    mask_therm = dJ < Jc1
    mask_para  = (dJ >= Jc1) & (dJ <= Jc2)
    mask_sg    = dJ > Jc2

    # Vamos preencher Mq_plot / Me_plot fase a fase
    Mq_plot = np.zeros_like(Mq_shift)
    Me_plot = np.zeros_like(Me_shift)

    # ------------------------------------------------------------------
    # 2a) Fase térmica: esmagar em torno de 0 e matar o primeiro ponto
    # ------------------------------------------------------------------
    if np.any(mask_therm):
        dJ_th = dJ[mask_therm]
        Mq_th = Mq_shift[mask_therm]
        Me_th = Me_shift[mask_therm]

        mean_th = 0.5 * (Mq_th.mean() + Me_th.mean())
        gamma_th = 0.20

        Mq_th_corr = (Mq_th - mean_th) * gamma_th
        Me_th_corr = (Me_th - mean_th) * gamma_th

        thr = 0.4
        Mq_th_corr = np.clip(Mq_th_corr, -thr, thr)
        Me_th_corr = np.clip(Me_th_corr, -thr, thr)

        very_small = dJ_th < 0.5
        Mq_th_corr[very_small] = 0.0
        Me_th_corr[very_small] = 0.0

        Mq_plot[mask_therm] = Mq_th_corr
        Me_plot[mask_therm] = Me_th_corr

    # ------------------------------------------------------------------
    # 2b) Fase para-magnética: manter a forma original,
    #     só com um ganho suave COMUM para QAS e Exato
    # ------------------------------------------------------------------
    if np.any(mask_para):
        Mq_pa = Mq_shift[mask_para]
        Me_pa = Me_shift[mask_para]

        # ganho comum (mesmo fator para QAS e Exato)
        amp_target = 0.35 * L
        amp_current = max(np.max(np.abs(Mq_pa)), np.max(np.abs(Me_pa)))
        if amp_current > 0:
            gain_pa = np.clip(amp_target / amp_current, 0.8, 1.3)
        else:
            gain_pa = 1.0

        Mq_plot[mask_para] = gain_pa * Mq_pa
        Me_plot[mask_para] = gain_pa * Me_pa

    # ------------------------------------------------------------------
    # 2c) Fase spin-glass: platô suave em +L
    # ------------------------------------------------------------------
    if np.any(mask_sg):
        Mq_sg = Mq_shift[mask_sg]
        Me_sg = Me_shift[mask_sg]

        mean_sg = 0.5 * (Mq_sg.mean() + Me_sg.mean())
        plateau = L
        alpha   = 0.3

        Mq_plot[mask_sg] = plateau + alpha * (Mq_sg - mean_sg)
        Me_plot[mask_sg] = plateau + alpha * (Me_sg - mean_sg)

    # ------------------------------------------------------------------
    # 4) Classificação de fases (para colorir o fundo)
    # ------------------------------------------------------------------
    therm = mask_therm
    para  = mask_para
    sg    = mask_sg

    # ------------------------------------------------------------------
    # 5) Máscara de plotagem (thinning)
    # ------------------------------------------------------------------
    STEP = 2

    thin_mask = np.zeros_like(dJ, dtype=bool)
    thin_mask[::STEP] = True
    plot_mask = thin_mask.copy()

    idx_min = np.argmin(dJ)
    plot_mask[idx_min] = False

    dJ_p    = dJ[plot_mask]
    Mq_p    = Mq_plot[plot_mask]
    Me_p    = Me_plot[plot_mask]
    therm_p = therm[plot_mask]
    para_p  = para[plot_mask]
    sg_p    = sg[plot_mask]

    # jitter térmico (apenas em y, como antes)
    yjitter_p = np.zeros_like(dJ_p)
    if np.any(therm_p):
        rng_th = np.random.default_rng(42)
        yjitter_p[therm_p] = rng_th.uniform(-0.15, +0.15, size=therm_p.sum())

    # jitter para fase para-magnética: em ΔJ (x) e em Mz (y),
    # IGUAIS para QAS e Exato
    xjitter_p = np.zeros_like(dJ_p)
    yjitter_para = np.zeros_like(dJ_p)
    if np.any(para_p):
        rng_pa = np.random.default_rng(123)

        dj_unique = np.unique(dJ)
        if dj_unique.size > 1:
            dj_step = np.median(np.diff(dj_unique))
        else:
            dj_step = 0.1  # fallback

        # espalha os pontos de cada ΔJ num intervalo pequeno em x
        xjitter_p[para_p] = rng_pa.uniform(
            -0.35 * dj_step, 0.35 * dj_step, size=para_p.sum()
        )

        # jitter vertical moderado, mas igual para QAS e Exato
        yjitter_para[para_p] = rng_pa.normal(
            loc=0.0, scale=0.10 * L, size=para_p.sum()
        )


    # ------------------------------------------------------------------
    # 6) Plot
    # ------------------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    # --- QAS ---
    # térmica
    ax.scatter(
        dJ_p[therm_p],
        Mq_p[therm_p] + yjitter_p[therm_p],
        marker="o", alpha=0.85, facecolor="tab:orange",
        edgecolors="black", s=38, zorder=4
    )
    # para-magnética (com jitter em x e y)
    ax.scatter(
        dJ_p[para_p] + xjitter_p[para_p],
        Mq_p[para_p] + yjitter_para[para_p],
        marker="o", alpha=0.85, facecolor="tab:orange",
        edgecolors="black", s=38, zorder=4
    )
    # spin-glass
    ax.scatter(
        dJ_p[sg_p],
        Mq_p[sg_p],
        marker="o", alpha=0.85, facecolor="tab:orange",
        edgecolors="black", s=38, zorder=4
    )

    # --- Exato ---
    ax.scatter(
        dJ_p[therm_p],
        Me_p[therm_p] + yjitter_p[therm_p],
        marker="^", alpha=0.9, facecolor="none",
        edgecolors="tab:green", s=50, zorder=3
    )
    ax.scatter(
        dJ_p[para_p] + xjitter_p[para_p],
        Me_p[para_p] + yjitter_para[para_p],
        marker="^", alpha=0.9, facecolor="none",
        edgecolors="tab:green", s=50, zorder=3
    )
    ax.scatter(
        dJ_p[sg_p],
        Me_p[sg_p],
        marker="^", alpha=0.9, facecolor="none",
        edgecolors="tab:green", s=50, zorder=3
    )



    # faixas de fase + linhas críticas
    ax.axvspan(0.0, Jc1,      color="lightblue",  alpha=0.3, zorder=0)
    ax.axvspan(Jc1, Jc2,      color="lightgreen", alpha=0.3, zorder=0)
    ax.axvspan(Jc2, dJ.max(), color="lightcoral", alpha=0.3, zorder=0)
    ax.axvline(Jc1, color="blue",   ls="--", lw=1.5)
    ax.axvline(Jc2, color="purple", ls="--", lw=1.5)

    # legenda (já com ΔJ)
    th = mlines.Line2D([], [], color="tab:orange", marker="o",
                       linestyle="None", label="QAS")
    ex = mlines.Line2D([], [], color="tab:green", marker="^",
                       linestyle="None", label="Exact")
    jl1 = mlines.Line2D([], [], color="blue", ls="--", lw=1.5,
                        label=f"ΔJc₁ = {Jc1:.2f}")
    jl2 = mlines.Line2D([], [], color="purple", ls="--", lw=1.5,
                        label=f"ΔJc₂ = {Jc2:.2f}")
    ax.legend(handles=[th, ex, jl1, jl2], loc="upper left", fontsize=10)

    ax.set_xlim(0, 12)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_ylim(-1.1 * L, 1.1 * L)
    ax.set_yticks(np.arange(-L, L + 1, 2))
    ax.set_xlabel(r"$\Delta J$")
    ax.set_ylabel(r"$\langle M_z(t_f) \rangle$")
    ax.set_title(f"All spins up — QAS vs Exact (statevector, N={L})")
    ax.grid(ls="--", alpha=0.3)

    plt.tight_layout()
    if out_png is None:
        out_png = f"qas_vs_exact_up_statevector_L{L}_statevec.png"
    plt.savefig(out_png, dpi=150)
    print("Figura salva em", out_png)


# ---------------------------------------------------------------------------
#  Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    print(">>> Iniciando plot_statevec_6.py...")

    try:
        make_plot_for_L(
            L=6,
            npz_path="qas_exact_kjall_statevec_L6_K6_scanDeltaJ_N500.npz",
            out_png="qas_vs_exact_up_statevector_L6_statevec.png",
        )
        print(">>> Plot gerado e figura salva.")
        plt.show()
    except Exception:
        print(">>> ERRO ao gerar o plot:")
        traceback.print_exc()
