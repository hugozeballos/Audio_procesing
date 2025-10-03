#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute single-ended metrics BEFORE (prep) and AFTER (enh), then deltas.
Expected layout:
  <root>/prep/**/*.wav
  <root>/enh/<backend>/<preset>/*.wav

Outputs:
  <root>/enh/enh_metrics_pair.csv
  <root>/enh/enh_metrics_pair_summary.csv
"""

import argparse, csv, math, os, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import soundfile as sf

# Optional deps
try:
    import pyloudnorm as pyln
except Exception:
    pyln = None
try:
    from srmrpy import srmr as _srmr
except Exception:
    _srmr = None

EPS = 1e-12

# ---------- helpers (no external heavy deps) ----------

def to_mono_float32(x: np.ndarray) -> np.ndarray:
    """Ensure mono float32 clipped to [-1, 1]."""
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32, copy=False)
    return np.clip(x, -1.0, 1.0)

def peak_rms_dbfs(x: np.ndarray) -> Tuple[float, float]:
    """Return (peak_dbfs, rms_dbfs). 0 dBFS at |x|=1."""
    peak = float(np.max(np.abs(x)) + EPS)
    rms  = float(np.sqrt(np.mean(x**2) + EPS))
    return 20.0 * math.log10(peak + EPS), 20.0 * math.log10(rms + EPS)

def clipping_and_crest(x: np.ndarray) -> Tuple[float, float]:
    """Clipping ratio in [0,1] and crest factor = peak/rms."""
    peak = float(np.max(np.abs(x)) + EPS)
    rms  = float(np.sqrt(np.mean(x**2) + EPS))
    clip = float(np.mean(np.abs(x) >= 0.999))
    crest = float(peak / max(rms, EPS))
    return clip, crest

def spectral_features(x: np.ndarray, sr: int) -> Tuple[float, float]:
    """Spectral flatness (geo/arith) and spectral centroid (Hz) averaged over frames."""
    win, hop = 1024, 512
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    w = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    flats, cents = [], []
    for start in range(0, len(x) - win + 1, hop):
        seg = x[start:start + win]
        X = np.fft.rfft(seg * w)
        P = (np.abs(X) ** 2).astype(np.float64) + EPS
        geo   = math.exp(float(np.mean(np.log(P))))
        arith = float(np.mean(P))
        flats.append(float(geo / max(arith, EPS)))
        cents.append(float((freqs * P).sum() / P.sum()))
    return float(np.mean(flats)), float(np.mean(cents))

def loudness_lufs(x: np.ndarray, sr: int) -> Optional[float]:
    """EBU R128 LUFS if pyloudnorm available, else None."""
    if pyln is None:
        return None
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(x))
    except Exception:
        return None

def srmr_score(x: np.ndarray, sr: int) -> Optional[float]:
    """SRMR if srmrpy available, else None."""
    if _srmr is None:
        return None
    try:
        score, _ = _srmr(x, sr, norm=True)
        return float(score)
    except Exception:
        return None

def metrics_for(x: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    """Compute all single-ended metrics for one signal."""
    dur = float(len(x) / max(sr, 1))
    peak_db, rms_db = peak_rms_dbfs(x)
    clip, crest = clipping_and_crest(x)
    flat, cent  = spectral_features(x, sr)
    lufs = loudness_lufs(x, sr)
    srmr = srmr_score(x, sr)
    return {
        "sr_hz": sr,
        "dur_s": round(dur, 6),
        "peak_dbfs": round(peak_db, 3),
        "rms_dbfs": round(rms_db, 3),
        "clip_pct": round(clip, 6),
        "crest": round(crest, 6),
        "spec_flatness": round(flat, 6),
        "spec_centroid_hz": round(cent, 3),
        "lufs": None if lufs is None else round(lufs, 3),
        "srmr": None if srmr is None else round(srmr, 6),
    }

def list_prep(root: Path) -> Dict[str, Path]:
    """Index prep wavs by stem to allow flexible matching."""
    prep_root = root / "prep"
    idx: Dict[str, Path] = {}
    if not prep_root.exists():
        return idx
    for p in prep_root.rglob("*.wav"):
        idx[p.stem] = p
    return idx

def list_enh(root: Path) -> List[Path]:
    """List enh/<backend>/<preset>/*.wav."""
    enh_root = root / "enh"
    if not enh_root.exists():
        return []
    return sorted(enh_root.glob("*/*/*.wav"))

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Compute metrics on PREP vs ENH and deltas.")
    ap.add_argument("--dataset-dir", required=True, help="Root dataset dir")
    ap.add_argument("--csv-out", default=None, help="Pairs CSV output")
    ap.add_argument("--csv-summary", default=None, help="Summary CSV output")
    args = ap.parse_args()

    base = Path(args.dataset_dir).resolve()
    prep_idx = list_prep(base)
    enh_files = list_enh(base)

    if not enh_files:
        print(f"[warn] no enh files under {base}/enh/<backend>/<preset>", file=sys.stderr)

    out_pairs = Path(args.csv_out) if args.csv_out else (base / "enh" / "enh_metrics_pair.csv")
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    out_sum = Path(args.csv_summary) if args.csv_summary else (base / "enh" / "enh_metrics_pair_summary.csv")

    header = [
        "rel_enh","backend","preset","rel_prep",
        "sr_in","sr_out","dur_in_s","dur_out_s",
        "peak_in_dbfs","peak_out_dbfs","delta_peak_db",
        "rms_in_dbfs","rms_out_dbfs","delta_rms_db",
        "clip_in_pct","clip_out_pct","delta_clip_pct",
        "crest_in","crest_out","delta_crest",
        "flat_in","flat_out","delta_flat",
        "centroid_in_hz","centroid_out_hz","delta_centroid_hz",
        "lufs_in","lufs_out","delta_lufs",
        "srmr_in","srmr_out","delta_srmr",
        "error"
    ]

    rows: List[Dict[str, Optional[float]]] = []
    n_ok = 0

    with out_pairs.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()

        for enh in enh_files:
            row = dict.fromkeys(header, None)
            try:
                rel_enh = enh.relative_to(base).as_posix()
                parts = enh.parts
                backend = parts[-3] if len(parts) >= 3 else ""
                preset  = parts[-2] if len(parts) >= 2 else ""
                stem = enh.stem

                prep = prep_idx.get(stem)
                if not prep or not prep.exists():
                    row.update({
                        "rel_enh": rel_enh, "backend": backend, "preset": preset,
                        "rel_prep": "", "error": "Missing prep match"
                    })
                    wr.writerow(row)
                    continue

                rel_prep = prep.relative_to(base).as_posix()

                xin, srin = sf.read(prep, dtype="float32", always_2d=False)
                xout, srout = sf.read(enh,  dtype="float32", always_2d=False)
                xin, xout = to_mono_float32(xin), to_mono_float32(xout)

                m_in  = metrics_for(xin, srin)
                m_out = metrics_for(xout, srout)

                def d(a, b):
                    return None if (a is None or b is None) else round(float(b) - float(a), 6)

                row.update({
                    "rel_enh": rel_enh, "backend": backend, "preset": preset, "rel_prep": rel_prep,
                    "sr_in": m_in["sr_hz"], "sr_out": m_out["sr_hz"],
                    "dur_in_s": m_in["dur_s"], "dur_out_s": m_out["dur_s"],
                    "peak_in_dbfs": m_in["peak_dbfs"], "peak_out_dbfs": m_out["peak_dbfs"],
                    "delta_peak_db": round(m_out["peak_dbfs"] - m_in["peak_dbfs"], 6),
                    "rms_in_dbfs": m_in["rms_dbfs"], "rms_out_dbfs": m_out["rms_dbfs"],
                    "delta_rms_db": round(m_out["rms_dbfs"] - m_in["rms_dbfs"], 6),
                    "clip_in_pct": m_in["clip_pct"], "clip_out_pct": m_out["clip_pct"],
                    "delta_clip_pct": round(m_out["clip_pct"] - m_in["clip_pct"], 6),
                    "crest_in": m_in["crest"], "crest_out": m_out["crest"],
                    "delta_crest": round(m_out["crest"] - m_in["crest"], 6),
                    "flat_in": m_in["spec_flatness"], "flat_out": m_out["spec_flatness"],
                    "delta_flat": round(m_out["spec_flatness"] - m_in["spec_flatness"], 6),
                    "centroid_in_hz": m_in["spec_centroid_hz"], "centroid_out_hz": m_out["spec_centroid_hz"],
                    "delta_centroid_hz": round(m_out["spec_centroid_hz"] - m_in["spec_centroid_hz"], 6),
                    "lufs_in": m_in["lufs"], "lufs_out": m_out["lufs"], "delta_lufs": d(m_in["lufs"], m_out["lufs"]),
                    "srmr_in": m_in["srmr"], "srmr_out": m_out["srmr"], "delta_srmr": d(m_in["srmr"], m_out["srmr"]),
                    "error": ""
                })
                n_ok += 1
            except Exception as e:
                row.update({
                    "rel_enh": enh.as_posix(), "backend": "", "preset": "", "rel_prep": "",
                    "error": f"{type(e).__name__}: {e}"
                })
            wr.writerow(row)

    # Build summary (no pandas dependency)
    # We will compute means of deltas ignoring None and errors, aggregated by backend/preset.
    import collections
    sums = collections.defaultdict(lambda: collections.defaultdict(float))
    counts = collections.defaultdict(lambda: collections.defaultdict(int))

    # read back rows to avoid duplicated loops above
    import csv as _csv
    with out_pairs.open("r", encoding="utf-8") as f:
        rd = _csv.DictReader(f)
        for r in rd:
            if r.get("error"):  # skip errored rows
                continue
            key = (r["backend"], r["preset"])
            for col in ["delta_peak_db","delta_rms_db","delta_clip_pct","delta_crest",
                        "delta_flat","delta_centroid_hz","delta_lufs","delta_srmr"]:
                v = r.get(col)
                if v is None or v == "" or str(v).lower() == "none":
                    continue
                try:
                    val = float(v)
                except Exception:
                    continue
                sums[key][col] += val
                counts[key][col] += 1

    with out_sum.open("w", newline="", encoding="utf-8") as f:
        sum_hdr = ["backend","preset","n_pairs",
                   "mean_delta_peak_db","mean_delta_rms_db","mean_delta_clip_pct","mean_delta_crest",
                   "mean_delta_flat","mean_delta_centroid_hz","mean_delta_lufs","mean_delta_srmr"]
        wr = csv.DictWriter(f, fieldnames=sum_hdr); wr.writeheader()
        # we also need number of pairs per backend/preset
        pairs_per = {}
        with out_pairs.open("r", encoding="utf-8") as pf:
            rd = csv.DictReader(pf)
            for r in rd:
                if r.get("error"): continue
                key = (r["backend"], r["preset"])
                pairs_per[key] = pairs_per.get(key, 0) + 1

        for (b,p), sdict in sums.items():
            row = {"backend": b, "preset": p, "n_pairs": pairs_per.get((b,p), 0)}
            for col in ["delta_peak_db","delta_rms_db","delta_clip_pct","delta_crest",
                        "delta_flat","delta_centroid_hz","delta_lufs","delta_srmr"]:
                c = counts[(b,p)].get(col, 0)
                m = (sdict[col] / c) if c > 0 else None
                row["mean_"+col] = None if m is None else round(m, 6)
            wr.writerow(row)

    print(f"[done] pairs={out_pairs} summary={out_sum} ok={n_ok} total_enh={len(enh_files)}", file=sys.stderr)

if __name__ == "__main__":
    main()
