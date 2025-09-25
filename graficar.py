import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1) CSV de entrada
candidates = [Path("dataset-audio-raw/enh/enh_metrics.csv"), Path("metrics.csv")]
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next((p for p in candidates if p.exists()), None)
if csv_path is None:
    raise SystemExit("No encuentro dataset-audio-raw/enh/enh_metrics.csv ni metrics.csv. Pasa la ruta como argumento.")

df = pd.read_csv(csv_path)

# 2) Normalización de tipos
numeric_cols = [
    "stoi","srmr","lufs_ref","lufs","delta_lufs",
    "peak_dbfs_ref","peak_dbfs_out","rms_dbfs_ref","rms_dbfs_out",
    "clip_rate_ref","clip_rate",
    "snr_db","si_sdr_db",
    "dur_ref_s","dur_out_s","dur_diff_s"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 3) Limpiezas mínimas opcionales
#   - Duraciones no positivas a NaN
for c in ["dur_ref_s","dur_out_s"]:
    if c in df.columns:
        df.loc[df[c] <= 0, c] = pd.NA

# 4) Función de boxplot por backend
def boxplot_by_backend(metric: str):
    if metric not in df.columns:
        return
    d = df[["backend", metric]].dropna()
    if d.empty:
        return
    plt.figure()
    # Pandas boxplot con "by" usa Matplotlib por debajo
    d.boxplot(column=metric, by="backend")
    plt.title(f"{metric} por backend")
    plt.suptitle("")  # limpia subtítulo automático
    plt.xlabel("backend")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# 5) Métricas recomendadas para ver en boxplot
to_plot = [
    m for m in [
        "stoi","srmr","snr_db","si_sdr_db",
        "lufs","delta_lufs","peak_dbfs_out","rms_dbfs_out","clip_rate",
        "dur_out_s","dur_diff_s"
    ] if m in df.columns
]

for m in to_plot:
    boxplot_by_backend(m)

# 6) Resumen tabular útil
summary_cols = [c for c in ["stoi","srmr","snr_db","si_sdr_db","lufs","delta_lufs","clip_rate"] if c in df.columns]
if summary_cols:
    summary = df.groupby("backend", dropna=True)[summary_cols].agg(["count","median","mean"])
    print("\nResumen por backend:\n", summary)
    # Guarda opcional
    out_summary = csv_path.with_name("enh_metrics_summary_by_backend.csv")
    summary.to_csv(out_summary)
    print(f"\nResumen guardado en: {out_summary}")
