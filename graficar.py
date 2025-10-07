#!/usr/bin/env python3
# viz_metrics.py
# Uso:
#   python viz_metrics.py --root /ruta/al/dataset
#   # por defecto busca en ./enh/ los CSVs:cp metr  
#   #   enh_metrics_pair.csv, enh_metrics_pair_summary.csv, enh_metrics_no_ref.csv
import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_if_exists(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def bar_mean_by_backend(df: pd.DataFrame, metric_col: str, label: str, out: Path):
    d = df.copy()
    d["label"] = d["backend"].astype(str) + "/" + d["preset"].astype(str)
    d = d.sort_values(metric_col)
    plt.figure()
    plt.bar(d["label"], d[metric_col])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} by backend/preset")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def hist_series(vals: pd.Series, title: str, out: Path):
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if len(v) == 0:
        return
    plt.figure()
    plt.hist(v, bins=30)
    plt.xlabel(title)
    plt.ylabel("count")
    plt.title(f"Distribution of {title}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def scatter_xy(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, out: Path):
    xd = pd.to_numeric(x, errors="coerce")
    yd = pd.to_numeric(y, errors="coerce")
    m = (~xd.isna()) & (~yd.isna())
    if m.sum() == 0:
        return
    plt.figure()
    plt.scatter(xd[m], yd[m], s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def derive_summary_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    cols = ["delta_peak_db","delta_rms_db","delta_clip_pct","delta_crest",
            "delta_flat","delta_centroid_hz","delta_lufs","delta_srmr"]
    def mean_safe(x): return pd.to_numeric(x, errors="coerce").mean()
    ok = pairs[pairs["error"].fillna("") == ""]
    g = ok.groupby(["backend","preset"], dropna=False)
    s = g[cols].agg(mean_safe).reset_index()
    s.insert(2, "n_pairs", g["rel_enh"].count().values)
    # Renombra a esquema "mean_*" para uniformar
    rename = {c: f"mean_{c}" for c in cols}
    s = s.rename(columns=rename)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Directorio raíz que contiene 'enh/'")
    ap.add_argument("--outdir", default="metrics_figs", help="Directorio de salida de figuras")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    enh_dir = root / "enh"
    ensure_dir(enh_dir)

    # Ubicaciones por defecto
    pairs_csv   = enh_dir / "enh_metrics_pair.csv"
    summary_csv = enh_dir / "enh_metrics_pair_summary.csv"
    noref_csv   = enh_dir / "enh_metrics_no_ref.csv"

    pairs_df   = read_if_exists(pairs_csv)
    summary_df = read_if_exists(summary_csv)
    noref_df   = read_if_exists(noref_csv)

    # Derivar summary si no existe y hay pairs
    if summary_df is None and pairs_df is not None and "error" in pairs_df.columns:
        summary_df = derive_summary_from_pairs(pairs_df)

    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    # 1) Barras por backend/preset desde summary
    if summary_df is not None and not summary_df.empty:
        # Acepta nombres "mean_*" o sin prefijo
        candidates = [
            "mean_delta_rms_db", "mean_delta_flat",
            "mean_delta_srmr", "mean_delta_centroid_hz",
            "delta_rms_db", "delta_flat", "delta_srmr", "delta_centroid_hz"
        ]
        for mc in candidates:
            if mc in summary_df.columns:
                bar_mean_by_backend(summary_df, mc, mc, outdir / f"{mc}_by_backend.png")

    # 2) Histogramas de deltas por archivo (pairs)
    if pairs_df is not None and not pairs_df.empty:
        for col in ["delta_rms_db","delta_flat","delta_centroid_hz","delta_peak_db"]:
            if col in pairs_df.columns:
                hist_series(pairs_df[col], col, outdir / f"hist_{col}.png")
        # Scatter delta_srmr vs delta_flat
        if "delta_srmr" in pairs_df.columns and "delta_flat" in pairs_df.columns:
            scatter_xy(pairs_df["delta_flat"], pairs_df["delta_srmr"],
                       "delta_flat", "delta_srmr", outdir / "scatter_delta_srmr_vs_delta_flat.png")

    # 3) Distribuciones no-ref si solo hay noref
    if noref_df is not None and not noref_df.empty:
        for col in ["rms_dbfs","spec_flatness","spec_centroid_hz","lufs","srmr"]:
            if col in noref_df.columns:
                hist_series(noref_df[col], f"{col} (no-ref)", outdir / f"noref_hist_{col}.png")

    # 4) Info mínima en consola
    print("Figuras guardadas en:", outdir)
    if pairs_df is None and summary_df is None and noref_df is None:
        print("No se encontraron CSVs esperados en:", enh_dir)

if __name__ == "__main__":
    main()
