#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pathlib, sys
import pandas as pd
import numpy as np

DEFAULT_METRICS = pathlib.Path(__file__).resolve().parent / "enh_metrics.csv"

def load_metrics(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: no existe {path}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(path)
    # Normaliza columnas esperadas
    for c in ["backend","preset","rel","ref_path","out_path","stoi","lufs","clip_rate","rtf","dur_out_s"]:
        if c not in df.columns: df[c] = np.nan
    # Coerción numérica
    for c in ["stoi","lufs","clip_rate","rtf","dur_out_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def summarize(df: pd.DataFrame, by=("backend","preset")) -> pd.DataFrame:
    g = df.groupby(list(by), dropna=False)
    agg = g.agg(
        n=("rel","count"),
        stoi_mean=("stoi","mean"),
        stoi_p10=("stoi", lambda s: s.quantile(0.10)),
        lufs_mean=("lufs","mean"),
        clip_rate_mean=("clip_rate","mean"),
        clip_any=("clip_rate", lambda s: float((s>0).any())),
        rtf_mean=("rtf","mean"),
        dur_sum_s=("dur_out_s","sum"),
    ).reset_index()
    # Orden por calidad: mayor STOI, menor clip, menor RTF
    agg = agg.sort_values(
        by=["stoi_mean","clip_rate_mean","rtf_mean"],
        ascending=[False, True, True],
        kind="mergesort"
    )
    return agg

def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Reglas simples: clipping, stoi bajo
    df = df.copy()
    df["flag_clip"] = (df["clip_rate"] > 0.0).fillna(False)
    df["flag_stoi_low"] = (df["stoi"] < 0.75).fillna(False)
    df["flag"] = np.where(df["flag_clip"] | df["flag_stoi_low"], "check", "")
    return df[df["flag"] == "check"].sort_values(["backend","preset","stoi"])

def main():
    ap = argparse.ArgumentParser(description="Reporte de métricas de enhancement")
    ap.add_argument("--metrics-csv", default=str(DEFAULT_METRICS), help="Ruta a enh_metrics.csv")
    ap.add_argument("--group-by", default="backend,preset", help="Columnas separadas por coma")
    ap.add_argument("--topk", type=int, default=10, help="Filas a mostrar en tabla agregada")
    ap.add_argument("--save-prefix", default="", help="Prefijo opcional para guardar CSVs del reporte")
    args = ap.parse_args()

    mpath = pathlib.Path(args.metrics_csv)
    df = load_metrics(mpath)

    by_cols = tuple([c.strip() for c in args.group_by.split(",") if c.strip()])
    agg = summarize(df, by=by_cols)
    outl = flag_outliers(df)

    # Print resumen corto
    print("\n== Resumen por {} ==".format("/".join(by_cols)))
    with pd.option_context("display.max_rows", args.topk, "display.float_format", "{:,.3f}".format):
        print(agg.head(args.topk))

    print("\n== Outliers sugeridos (clipping o STOI<0.75) ==")
    with pd.option_context("display.max_rows", 50, "display.float_format", "{:,.3f}".format):
        print(outl[["backend","preset","rel","stoi","clip_rate","rtf"]].head(50))

    # Guardar CSVs
    prefix = args.save_prefix or (mpath.parent / "report")
    prefix = pathlib.Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(str(prefix) + "_summary.csv", index=False)
    outl.to_csv(str(prefix) + "_outliers.csv", index=False)
    # Tabla detallada ordenada útil para inspección
    detail = df.sort_values(["backend","preset","stoi"], ascending=[True,True,False])
    detail.to_csv(str(prefix) + "_detail.csv", index=False)

if __name__ == "__main__":
    main()
