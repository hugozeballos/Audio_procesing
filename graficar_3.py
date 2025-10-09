#!/usr/bin/env python3
# graficar_in_out.py
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

TOKEN = re.compile(r"^(?P<base>.+?)_(?P<side>in|out)(?P<suf>(_.*)?)$")

def infer_system(df: pd.DataFrame) -> str:
    lc = {c.lower(): c for c in df.columns}
    if "system" in df.columns: return "system"
    if "backend" in lc and "preset" in lc:
        df["system"] = df[lc["backend"]].astype(str)+"/"+df[lc["preset"]].astype(str); return "system"
    df["system"] = "unknown"; return "system"

def infer_key(df: pd.DataFrame) -> str:
    for k in ("basename","file","utt","key","id","name"):
        if k in df.columns: return k
    # deriva desde alguna ruta
    for c in df.columns:
        s = df[c].dropna()
        if s.empty: continue
        v = str(s.iloc[0])
        if "/" in v or "\\" in v:
            df["basename"] = s.astype(str).apply(lambda x: Path(x).stem)
            return "basename"
    df["basename"] = np.arange(len(df)); return "basename"

def find_pairs(cols: list[str]) -> dict[str, tuple[str,str]]:
    pairs = {}
    parsed = {}
    for c in cols:
        m = TOKEN.match(c)
        if not m: continue
        base, side, suf = m.group("base"), m.group("side"), m.group("suf")
        parsed.setdefault((base, suf), {})[side] = c
    for (base, suf), sides in parsed.items():
        if "in" in sides and "out" in sides:
            pairs[f"{base}{suf}"] = (sides["in"], sides["out"])
    return pairs  # key=nombre lógico, val=(col_in, col_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--systems", required=True,
                    help="7 cajas: 'in' + 6 sistemas, en orden. Ej: base/medium,clearervoice/medium,...")
    ap.add_argument("--metrics", nargs="*", help="bases lógicas; si se omite se autodetecta de *_in/*_out")
    ap.add_argument("--system-col", default=None)
    ap.add_argument("--key-col", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sys_col = args.system_col if (args.system_col and args.system_col in df.columns) else infer_system(df)
    key_col = args.key_col if (args.key_col and args.key_col in df.columns) else infer_key(df)

    pairs = find_pairs(df.columns.tolist())
    if args.metrics:
        metrics = [m for m in args.metrics if m in pairs]
    else:
        metrics = sorted(pairs.keys())

    systems = [s.strip() for s in args.systems.split(",")]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # valores "in" deduplicados por archivo
    for m in metrics:
        col_in, col_out = pairs[m]

        # dedup "in" por clave
        din = df[[key_col, col_in]].dropna()
        vin = din.groupby(key_col)[col_in].first().values  # un valor por archivo

        data = [vin]  # primera caja = IN
        labels = ["in"]

        # luego cada backend con OUT
        for s in systems:
            v = df.loc[df[sys_col]==s, col_out].dropna().values
            data.append(v); labels.append(s)

        # dibuja 7 posiciones aunque alguna esté vacía
        plt.figure()
        non_empty = [(i+1, y) for i,y in enumerate(data) if len(y)]
        if non_empty:
            pos, arr = zip(*non_empty)
            plt.boxplot(arr, positions=pos, showfliers=False)
        for i, y in enumerate(data, 1):
            if len(y):
                x = np.random.normal(i, 0.06, size=len(y))
                plt.plot(x, y, ".", alpha=0.35, markersize=3)

        plt.title(m); plt.ylabel(m)
        plt.xticks(range(1, len(labels)+1), labels, rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"box_{m}.png", dpi=150)
        plt.close()

    print(f"OK: {out_dir}")

if __name__ == "__main__":
    main()
