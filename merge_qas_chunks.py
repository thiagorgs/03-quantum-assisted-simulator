#!/usr/bin/env python3
"""Mescla arquivos NPZ parciais de QAS (chunks de realizacoes).

Esperado em cada chunk:
- Js
- Sum_Mz
- SumSq_Mz
- Count_Mz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def load_chunk(path: Path):
    with np.load(path, allow_pickle=False) as d:
        js = np.asarray(d["Js"], dtype=float).reshape(-1)
        s = np.asarray(d["Sum_Mz"], dtype=float).reshape(-1)
        ss = np.asarray(d["SumSq_Mz"], dtype=float).reshape(-1)
        c = np.asarray(d["Count_Mz"], dtype=np.int64).reshape(-1)

        meta = {
            "L": int(d["L"]) if "L" in d else None,
            "K": int(d["K"]) if "K" in d else None,
            "basis_dim_raw": int(d["basis_dim_raw"]) if "basis_dim_raw" in d else None,
            "basis_dim_effective": int(d["basis_dim_effective"]) if "basis_dim_effective" in d else None,
            "J": float(d["J"]) if "J" in d else None,
            "h": float(d["h"]) if "h" in d else None,
            "J2": float(d["J2"]) if "J2" in d else None,
            "FINAL_T": float(d["FINAL_T"]) if "FINAL_T" in d else None,
            "boundary": str(d["boundary"]) if "boundary" in d else "closed",
            "method": str(d["method"]) if "method" in d else "QAS",
            "basis_mode": str(d["basis_mode"]) if "basis_mode" in d else "true_qas_k_moment",
            "normalized_by_L": int(d["normalized_by_L"]) if "normalized_by_L" in d else 1,
            "Nrea_total": int(d["Nrea_total"]) if "Nrea_total" in d else None,
        }
    return js, s, ss, c, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--pattern", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    files: List[Path] = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"ERRO: nenhum arquivo encontrado em {input_dir} com pattern '{args.pattern}'")

    js_ref = None
    sum_all = None
    sumsq_all = None
    count_all = None
    meta_ref = None

    for idx, fp in enumerate(files):
        js, s, ss, c, meta = load_chunk(fp)

        if js_ref is None:
            js_ref = js
            sum_all = np.zeros_like(s, dtype=float)
            sumsq_all = np.zeros_like(ss, dtype=float)
            count_all = np.zeros_like(c, dtype=np.int64)
            meta_ref = meta
        else:
            if js.shape != js_ref.shape or not np.allclose(js, js_ref, atol=0.0, rtol=0.0):
                raise SystemExit(f"ERRO: grade Js incompativel no arquivo {fp}")

        sum_all += s
        sumsq_all += ss
        count_all += c
        print(f"[{idx+1:03d}/{len(files):03d}] chunk={fp.name}")

    if np.any(count_all <= 0):
        raise SystemExit("ERRO: count <= 0 em algum ponto de Js")

    mean = sum_all / count_all
    # variancia amostral por ponto: (sum(x^2) - n*mean^2)/(n-1)
    var = np.zeros_like(mean)
    mask = count_all > 1
    var[mask] = (sumsq_all[mask] - count_all[mask] * mean[mask] * mean[mask]) / (count_all[mask] - 1)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        Js=js_ref,
        Magnetization=mean,
        Std_Mz=std,
        Sum_Mz=sum_all,
        SumSq_Mz=sumsq_all,
        Count_Mz=count_all,
        Nrea=int(count_all[0]),
        L=int(meta_ref["L"]) if meta_ref["L"] is not None else -1,
        K=int(meta_ref["K"]) if meta_ref["K"] is not None else -1,
        basis_dim_raw=int(meta_ref["basis_dim_raw"]) if meta_ref["basis_dim_raw"] is not None else -1,
        basis_dim_effective=int(meta_ref["basis_dim_effective"]) if meta_ref["basis_dim_effective"] is not None else -1,
        basis_mode=meta_ref["basis_mode"],
        boundary=meta_ref["boundary"],
        method=meta_ref["method"],
        J=float(meta_ref["J"]) if meta_ref["J"] is not None else np.nan,
        h=float(meta_ref["h"]) if meta_ref["h"] is not None else np.nan,
        J2=float(meta_ref["J2"]) if meta_ref["J2"] is not None else np.nan,
        FINAL_T=float(meta_ref["FINAL_T"]) if meta_ref["FINAL_T"] is not None else np.nan,
        normalized_by_L=int(meta_ref["normalized_by_L"]),
    )
    print(f"merged_chunks={len(files)}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
