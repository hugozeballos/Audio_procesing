#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graficar.py — Comparación entre backends de enhancement.

Uso típico:
  python graficar.py --root . --outdir figs

Lee (si existen) en <root>/enh/:
  - enh_metrics_pair.csv
  - enh_metrics_pair_summary.csv

Genera:
  - barras con IC95% por backend/preset
  - boxplots por backend/preset
  - ranking compuesto configurable
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PAIR_FILE   = "enh_metrics_pair.csv"
SUM_FILE    = "enh_metrics_pair_summary.csv"

PAIR_COLS = ["delta_peak_db","delta_rms_db","delta_clip_pct","delta_crest",
             "delta_flat","delta_centroid_hz","delta_lufs","delta_srmr"]

def read_df(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def label_backend(df: pd.DataFrame) -> pd.Series:
    b = df["backend"].astype(str).fillna("")
    p = df["preset"].astype(str).fillna("")
    return b + "/" + p

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def bar_with_ci(df: pd.DataFrame, col: str, out: Path):
    d = df.copy()
    d["label"] = label_backend(d)
    m = to_num(d[col])
    c = to_num(d.get("n_pairs", pd.Series([np.nan]*len(d))))
    # IC95% ~ 1.96 * std/sqrt(n)
    # si viene de summary sin std, estimamos std desde pairs si está disponible (mejor usar pair_df en boxplot)
    # aquí usamos solo la media; el ancho de barra no tendrá error si no hay n válido
    stderr = pd.Series([np.nan]*len(d))
    if "n_pairs" in d.columns:
        stderr = (0 * m + np.nan)  # placeholder
        # si no hay std, solo mostramos media
    plt.figure()
    plt.bar(d["label"], m)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(col)
    plt.title(f"{col} by backend/preset")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def boxplot_pairs(pairs: pd.DataFrame, col: str, out: Path):
    ok = pairs[pairs["error"].fillna("") == ""].copy()
    ok["label"] = label_backend(ok)
    ok[col] = to_num(ok[col])
    ok = ok.dropna(subset=[col])
    if ok.empty: return
    groups = [g[col].values for _, g in ok.groupby("label")]
    labels = [k for k, _ in ok.groupby("label")]
    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(col)
    plt.title(f"{col} distribution by backend/preset")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def derive_summary_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    ok = pairs[pairs["error"].fillna("") == ""].copy()
    for c in PAIR_COLS:
        if c in ok.columns: ok[c] = to_num(ok[c])
    g = ok.groupby(["backend","preset"], dropna=False)
    s = g[PAIR_COLS].mean(numeric_only=True).reset_index()
    s.insert(2, "n_pairs", g.size().values)
    s = s.rename(columns={c: f"mean_{c}" for c in PAIR_COLS})
    return s

def compute_rank(summary: pd.DataFrame, w_srmr=0.4, w_flat=0.3, w_centroid=0.3) -> pd.DataFrame:
    d = summary.copy()
    # normalizaciones simples
    d["mean_delta_srmr"] = to_num(d.get("mean_delta_srmr"))
    d["mean_delta_flat"] = to_num(d.get("mean_delta_flat"))
    d["mean_delta_centroid_hz"] = to_num(d.get("mean_delta_centroid_hz"))
    d["score"] = (
        w_srmr * d["mean_delta_srmr"].fillna(0)
        + w_flat * (-d["mean_delta_flat"].fillna(0))
        + w_centroid * (-d["mean_delta_centroid_hz"].abs().fillna(0) / 1000.0)
    )
    d = d.sort_values("score", ascending=False)
    return d

def plot_rank(df: pd.DataFrame, out: Path):
    d = df.copy()
    d["label"] = label_backend(d)
    plt.figure()
    plt.bar(d["label"], d["score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("score (higher is better)")
    plt.title("Backend ranking (no-ref)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Raíz que contiene enh/")
    ap.add_argument("--outdir", default="figs", help="Directorio de salida")
    ap.add_argument("--metrics", nargs="*", default=["delta_rms_db","delta_flat","delta_centroid_hz","delta_peak_db","delta_srmr"],
                    help="Métricas de pairs para boxplot")
    ap.add_argument("--w-srmr", type=float, default=0.4)
    ap.add_argument("--w-flat", type=float, default=0.3)
    ap.add_argument("--w-centroid", type=float, default=0.3)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    enh_dir = root / "enh"
    outdir = Path(args.outdir).resolve(); ensure_dir(outdir)

    pairs = read_df(enh_dir / PAIR_FILE)
    summary = read_df(enh_dir / SUM_FILE)

    if summary is None and pairs is not None:
        summary = derive_summary_from_pairs(pairs)

    # 1) Boxplots por backend/preset a partir de pairs
    if pairs is not None and not pairs.empty:
        for col in args.metrics:
            if col in pairs.columns:
                boxplot_pairs(pairs, col, outdir / f"box_{col}.png")

    # 2) Barras por backend/preset desde summary
    if summary is not None and not summary.empty:
        for mc in [c for c in summary.columns if c.startswith("mean_delta_")]:
            bar_with_ci(summary, mc, outdir / f"bar_{mc}.png")

        # 3) Ranking compuesto
        rank = compute_rank(summary, args["w_srmr"] if isinstance(args, dict) else args.w_srmr,
                                     args["w_flat"]  if isinstance(args, dict) else args.w_flat,
                                     args["w_centroid"] if isinstance(args, dict) else args.w_centroid)
        rank.to_csv(outdir / "backend_ranking.csv", index=False)
        plot_rank(rank, outdir / "backend_ranking.png")

    print("Listo. Figuras en:", outdir)

if __name__ == "__main__":
    main()
