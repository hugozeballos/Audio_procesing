#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera dev.list y test.list desde manifests/manifest.csv
Reglas por defecto:
- Ordena por duración ascendente y toma N más cortos para dev.
- test opcional aleatorio sin solaparse con dev.
Uso:
  python preprocesing/gen_splits.py --dataset-dir dataset-audio-raw --dev-size 80 --test-size 0
"""

import argparse, pathlib, random
import pandas as pd

def save_list(paths, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Carpeta del dataset")
    ap.add_argument("--manifest", default="manifests/manifest.csv")
    ap.add_argument("--out-dev", default="splits/dev.list")
    ap.add_argument("--out-test", default="splits/test.list")
    ap.add_argument("--dev-size", type=int, default=5)
    ap.add_argument("--test-size", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=0.0)
    ap.add_argument("--max-dur", type=float, default=10_000.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = pathlib.Path(args.dataset_dir)
    mpath = ds / args.manifest
    df = pd.read_csv(mpath)

    # saneo
    df = df.dropna(subset=["path", "duration_s"])
    df = df[(df["duration_s"] >= args.min_dur) & (df["duration_s"] <= args.max_dur)].copy()

    # DEV: los más cortos primero
    df_dev = df.sort_values("duration_s", kind="stable").head(args.dev_size)
    dev_paths = df_dev["path"].tolist()

    # TEST: aleatorio sin solaparse
    remaining = df[~df["path"].isin(dev_paths)].copy()
    rng = random.Random(args.seed)
    test_paths = []
    if args.test_size > 0 and not remaining.empty:
        cand = remaining["path"].tolist()
        rng.shuffle(cand)
        test_paths = cand[:args.test_size]

    save_list(dev_paths, ds / args.out_dev)
    if args.test_size > 0:
        save_list(test_paths, ds / args.out_test)

    print(f"dev: {len(dev_paths)}  -> {ds / args.out_dev}")
    if args.test_size > 0:
        print(f"test: {len(test_paths)} -> {ds / args.out_test}")

if __name__ == "__main__":
    main()
