#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import soundfile as sf

# --- settings ---
EPS = 1e-12
VAD_COLS = [
    "vad_speech_ratio_in","vad_speech_ratio_out","delta_vad_speech_ratio",
    "vad_seg_count_in","vad_seg_count_out","delta_vad_seg_count",
    "vad_seg_mean_dur_in_s","vad_seg_mean_dur_out_s","delta_vad_seg_mean_dur_s",
    "speech_level_in_dbfs","speech_level_out_dbfs","delta_speech_level_db",
    "nonspeech_level_in_dbfs","nonspeech_level_out_dbfs","delta_nonspeech_level_db",
    "speech_nonspeech_snr_in_db","speech_nonspeech_snr_out_db","delta_speech_nonspeech_snr_db",
]

# --- deps opcional: webrtcvad ---
try:
    import webrtcvad
except Exception:
    webrtcvad = None

def to_mono_float32(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32, copy=False)
    return np.clip(x, -1.0, 1.0)

def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / max(sr_in, 1)))
    if n_out <= 1:
        return x
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    fp = x.astype(np.float64, copy=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(x_new, xp, fp).astype(np.float32, copy=False)
    return np.clip(y, -1.0, 1.0)

def _vad_mask_webrtc(x: np.ndarray, sr: int, frame_ms: int = 30, aggr: int = 2):
    if webrtcvad is None or sr not in (8000, 16000, 32000, 48000):
        return None, None
    vad = webrtcvad.Vad(aggr)
    hop = int(sr * frame_ms / 1000)
    if hop <= 0 or len(x) < hop:
        return None, None
    flags = []
    for i in range(0, len(x) - hop + 1, hop):
        seg = x[i:i+hop]
        pcm = (np.clip(seg, -1, 1) * 32767.0).astype(np.int16).tobytes()
        flags.append(vad.is_speech(pcm, sr))
    mask = np.repeat(flags, hop)[:len(x)]
    return mask.astype(bool), frame_ms / 1000.0
#debug
def _vad_mask_wrapped(x: np.ndarray, sr: int, vad_resample: str = "48k"):
    tgt_sr = {"none": sr, "16k": 16000, "48k": 48000}.get(vad_resample, sr)
    print("DBG:_vad_mask_wrapped vad_resample=", vad_resample, "sr=", sr, "tgt_sr=", tgt_sr)
    x_vad = _resample_linear(x, sr, tgt_sr) if tgt_sr != sr else x
    m_tgt, step = _vad_mask_webrtc(x_vad, tgt_sr)
    if m_tgt is None:
        return None, None
    if tgt_sr != sr:
        m = _resample_linear(m_tgt.astype(np.float32), tgt_sr, sr) > 0.5
        return m, step
    return m_tgt, step

def _vad_stats_and_levels(x: np.ndarray, sr: int, mask: Optional[np.ndarray]):
    if mask is None or mask.size == 0:
        return {
            "vad_speech_ratio": None, "vad_seg_count": None, "vad_seg_mean_dur_s": None,
            "speech_level_dbfs": None, "nonspeech_level_dbfs": None, "speech_nonspeech_snr_db": None,
        }
    speech_ratio = float(mask.mean())
    b = mask.astype(np.int8)
    ch = np.diff(b, prepend=0)
    starts = np.where(ch == 1)[0]
    ends   = np.where(ch == -1)[0]
    if mask[-1]:
        ends = np.append(ends, len(mask)-1)
    durs = (ends - starts) / max(sr, 1)
    seg_count = int(len(durs))
    seg_mean = float(durs.mean()) if seg_count > 0 else 0.0

    def _lvl_db(y):
        if y.size == 0: return None
        rms = float(np.sqrt(np.mean(y**2) + EPS))
        return 20.0 * math.log10(rms + EPS)

    sp = x[mask]; ns = x[~mask]
    sp_db = _lvl_db(sp); ns_db = _lvl_db(ns)
    snr = None if (sp_db is None or ns_db is None) else (sp_db - ns_db)
    return {
        "vad_speech_ratio": round(speech_ratio, 6),
        "vad_seg_count": seg_count,
        "vad_seg_mean_dur_s": round(seg_mean, 6),
        "speech_level_dbfs": None if sp_db is None else round(sp_db, 3),
        "nonspeech_level_dbfs": None if ns_db is None else round(ns_db, 3),
        "speech_nonspeech_snr_db": None if snr is None else round(snr, 3),
    }

def _missing_vad(row: Dict[str,str]) -> bool:
    for c in VAD_COLS:
        if row.get(c) in (None, "", "None"):
            return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Fill or recompute VAD-only metrics into an existing metrics CSV.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--csv-in",  default=None, help="Defaults to enh/enh_metrics_pair.csv")
    ap.add_argument("--csv-out", default=None, help="Defaults to enh/enh_metrics_pair_vad.csv (or --inplace)")
    ap.add_argument("--vad-resample", default="48k", choices=["none","16k","48k"])
    ap.add_argument("--only-missing", action="store_true", help="Compute VAD only for rows with missing VAD fields")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input CSV")
    args = ap.parse_args()

    base = Path(args.dataset_dir).resolve()
    csv_in  = Path(args.csv_in)  if args.csv_in  else (base / "enh" / "enh_metrics_pair.csv")
    csv_out = Path(args.csv_out) if args.csv_out else (base / "enh" / "enh_metrics_pair_vad.csv")
    if args.inplace:
        csv_out = csv_in

    if not csv_in.exists():
        raise FileNotFoundError(f"CSV not found: {csv_in}")

    with csv_in.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []

        # garantiza columnas VAD
        new_header = list(header)
        for c in VAD_COLS:
            if c not in new_header:
                new_header.append(c)

        rows = []
        for r in rd:
            need = True
            if args.only-missing:
                need = _missing_vad(r)
            if not need:
                rows.append(r)
                continue

            # paths
            rel_prep = r.get("rel_prep","")
            rel_enh  = r.get("rel_enh","")
            prep_p = (base / rel_prep) if rel_prep else None
            enh_p  = (base / rel_enh)  if rel_enh  else None

            try:
                # carga audio in/out
                xin, srin = (sf.read(prep_p, dtype="float32", always_2d=False) if (prep_p and prep_p.exists()) else (None, None))
                xout, srout = (sf.read(enh_p, dtype="float32", always_2d=False) if (enh_p and enh_p.exists())  else (None, None))
                if xin is not None:  xin  = to_mono_float32(xin)
                if xout is not None: xout = to_mono_float32(xout)

                # VAD + stats
                def compute_for(x, sr):
                    if x is None or sr is None:
                        return None
                    mask, _ = _vad_mask_wrapped(x, sr, vad_resample=args.vad_resample)
                    return _vad_stats_and_levels(x, sr, mask)

                vin  = compute_for(xin,  srin)
                vout = compute_for(xout, srout)

                def d(a,b):
                    return None if (a is None or b is None) else round(float(b) - float(a), 6)

                # escribe en fila
                r["vad_speech_ratio_in"]      = None if vin  is None else vin["vad_speech_ratio"]
                r["vad_speech_ratio_out"]     = None if vout is None else vout["vad_speech_ratio"]
                r["delta_vad_speech_ratio"]   = d(None if vin is None else vin["vad_speech_ratio"],
                                                  None if vout is None else vout["vad_speech_ratio"])

                r["vad_seg_count_in"]         = None if vin  is None else vin["vad_seg_count"]
                r["vad_seg_count_out"]        = None if vout is None else vout["vad_seg_count"]
                r["delta_vad_seg_count"]      = d(None if vin is None else vin["vad_seg_count"],
                                                  None if vout is None else vout["vad_seg_count"])

                r["vad_seg_mean_dur_in_s"]    = None if vin  is None else vin["vad_seg_mean_dur_s"]
                r["vad_seg_mean_dur_out_s"]   = None if vout is None else vout["vad_seg_mean_dur_s"]
                r["delta_vad_seg_mean_dur_s"] = d(None if vin is None else vin["vad_seg_mean_dur_s"],
                                                  None if vout is None else vout["vad_seg_mean_dur_s"])

                r["speech_level_in_dbfs"]     = None if vin  is None else vin["speech_level_dbfs"]
                r["speech_level_out_dbfs"]    = None if vout is None else vout["speech_level_dbfs"]
                r["delta_speech_level_db"]    = d(None if vin is None else vin["speech_level_dbfs"],
                                                  None if vout is None else vout["speech_level_dbfs"])

                r["nonspeech_level_in_dbfs"]  = None if vin  is None else vin["nonspeech_level_dbfs"]
                r["nonspeech_level_out_dbfs"] = None if vout is None else vout["nonspeech_level_dbfs"]
                r["delta_nonspeech_level_db"] = d(None if vin is None else vin["nonspeech_level_dbfs"],
                                                  None if vout is None else vout["nonspeech_level_dbfs"])

                r["speech_nonspeech_snr_in_db"]  = None if vin  is None else vin["speech_nonspeech_snr_db"]
                r["speech_nonspeech_snr_out_db"] = None if vout is None else vout["speech_nonspeech_snr_db"]
                r["delta_speech_nonspeech_snr_db"] = d(None if vin is None else vin["speech_nonspeech_snr_db"],
                                                       None if vout is None else vout["speech_nonspeech_snr_db"])
            except Exception as e:
                # no tocar otras columnas; deja VAD vacías
                pass

            rows.append(r)

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=new_header)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in new_header})

    print(f"[done] wrote {csv_out}")
if __name__ == "__main__":
    import numpy as np
    main()
