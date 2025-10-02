#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def norm01(series, lo=None, hi=None):
    s = series.copy()
    if lo is None: lo = np.nanpercentile(s, 5)
    if hi is None: hi = np.nanpercentile(s, 95)
    if hi <= lo:   hi = lo + 1e-6
    return (s - lo) / (hi - lo)

def load_df(csv_path: Path) -> pd.DataFrame:
    cand = [csv_path, Path("dataset-audio-raw/enh/enh_metrics.csv"), Path("metrics.csv")]
    src = next((p for p in cand if p and p.exists()), None)
    if src is None:
        raise SystemExit("No encuentro CSV. Usa --csv <ruta>")
    df = pd.read_csv(src)
    num_cols = [
        "stoi","si_sdr_db","snr_seg_db","snr_db","srmr",
        "spectral_dist_db","lufs","lufs_ref","delta_lufs",
        "rms_dbfs_out","rms_dbfs_ref","peak_dbfs_out","peak_dbfs_ref",
        "clip_rate","clip_rate_ref","dur_ref_s","dur_out_s","dur_diff_s",
        # no intrusivas
        "nisqa_overall","nisqa_noisiness","nisqa_discontinuity","nisqa_coloration","nisqa_loudness",
        "nisqa_overall_ref","nisqa_noisiness_ref","nisqa_discontinuity_ref","nisqa_coloration_ref","nisqa_loudness_ref",
        "dnsmos_sig","dnsmos_bak","dnsmos_ovrl","dnsmos_sig_ref","dnsmos_bak_ref","dnsmos_ovrl_ref",
    ]
    df = to_num(df, num_cols)
    # limpiar duraciones ≤0
    for c in ["dur_ref_s","dur_out_s"]:
        if c in df.columns:
            df.loc[df[c] <= 0, c] = np.nan
    return df

def choose_xy(df):
    if {"snr_seg_db","stoi"} <= set(df.columns):
        return "snr_seg_db","STOI"
    if {"si_sdr_db","stoi"} <= set(df.columns):
        return "si_sdr_db","STOI"
    if {"dnsmos_ovrl","stoi"} <= set(df.columns):
        return "dnsmos_ovrl","STOI"
    if {"nisqa_overall","stoi"} <= set(df.columns):
        return "nisqa_overall","STOI"
    return None,None

def scatter_xy(df, save_dir):
    x, ylab = choose_xy(df)
    if x is None or "stoi" not in df.columns: return
    plt.figure(figsize=(10,6))
    if "backend" in df.columns:
        for be, sub in df.groupby("backend"):
            plt.scatter(sub[x], sub["stoi"], alpha=0.7, s=40, label=str(be))
        plt.legend()
    else:
        plt.scatter(df[x], df["stoi"], alpha=0.7, s=40)
    plt.xlabel(x)
    plt.ylabel("STOI")
    plt.title(f"Relación {x} vs STOI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = save_dir / f"scatter_{x}_vs_stoi.png"
    plt.savefig(out, dpi=160)
    plt.close()

def bars_normalized(df, save_dir):
    if "backend" not in df.columns: return
    metrics = [m for m in ["stoi","si_sdr_db","snr_seg_db","dnsmos_ovrl","nisqa_overall"] if m in df.columns]
    if not metrics: return
    means = df.groupby("backend")[metrics].mean(numeric_only=True)
    norm = pd.DataFrame(index=means.index)
    for m in metrics:
        if m == "stoi" or m.startswith("nisqa") or m.startswith("dnsmos"):
            norm[m] = means[m]  # ya 0–1 típicamente
        else:
            norm[m] = norm01(means[m])  # dB a 0–1 con percentiles robustos
    x = np.arange(len(norm)); width = 0.8 / max(1,len(metrics))
    plt.figure(figsize=(12,6))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - (len(metrics)-1)/2.0)*width, norm[m].values, width, label=m, alpha=0.85)
    plt.ylabel("Puntaje normalizado (0–1)")
    plt.xlabel("Backend")
    plt.title("Comparación normalizada por backend")
    plt.xticks(x, norm.index, rotation=45)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = save_dir / "bar_comparison_backends.png"
    plt.savefig(out, dpi=160)
    plt.close()
    # guardar CSV de ranking por métrica
    (save_dir / "rankings").mkdir(parents=True, exist_ok=True)
    for m in metrics:
        norm[[m]].sort_values(m, ascending=False).to_csv(save_dir / "rankings" / f"rank_by_{m}.csv")

def violin_stoi(df, save_dir):
    if "backend" not in df.columns or "stoi" not in df.columns: return
    # Matplotlib violinplot por backend
    be_list = list(df["backend"].dropna().unique())
    data = [df.loc[df["backend"]==be, "stoi"].dropna().values for be in be_list]
    if len(be_list) < 1: return
    plt.figure(figsize=(10,6))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(np.arange(1, len(be_list)+1), be_list, rotation=45)
    plt.title("Distribución de STOI por backend")
    plt.ylabel("STOI")
    for y in [0.6, 0.8]:
        plt.axhline(y, linestyle="--", alpha=0.4)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = save_dir / "violin_stoi.png"
    plt.savefig(out, dpi=160)
    plt.close()

def corr_heatmap(df, save_dir):
    cols = [c for c in ["stoi","si_sdr_db","snr_seg_db","snr_db","spectral_dist_db","srmr","lufs","clip_rate","nisqa_overall","dnsmos_ovrl"] if c in df.columns]
    if len(cols) < 2: return
    corr = df[cols].corr()
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlación")
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlación entre métricas")
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i,j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out = save_dir / "corr_metrics.png"
    plt.savefig(out, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="Ruta al metrics.csv")
    ap.add_argument("--backend", type=str, default=None, help="Filtrar a un backend")
    ap.add_argument("--save-dir", type=Path, default=Path("artifacts/figs"))
    args = ap.parse_args()

    df = load_df(args.csv)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.backend and "backend" in df.columns:
        df = df[df["backend"] == args.backend].copy()

    scatter_xy(df, args.save_dir)
    bars_normalized(df, args.save_dir)
    violin_stoi(df, args.save_dir)
    corr_heatmap(df, args.save_dir)

    # resumen tabular por backend
    core = [c for c in ["stoi","si_sdr_db","snr_seg_db","srmr","dnsmos_ovrl","nisqa_overall"] if c in df.columns]
    if "backend" in df.columns and core:
        summary = df.groupby("backend")[core].agg(["count","mean","median","std"])
        out = args.save_dir / "summary_by_backend.csv"
        summary.round(4).to_csv(out)

if __name__ == "__main__":
    main()
