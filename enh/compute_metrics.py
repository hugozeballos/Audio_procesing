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

# --- extra opcionales ---
try:
    import webrtcvad
except Exception:
    webrtcvad = None

try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    silhouette_score = calinski_harabasz_score = davies_bouldin_score = None
    AgglomerativeClustering = None

try:
    import torch
    from speechbrain.pretrained import EncoderClassifier  # ECAPA embeddings
except Exception:
    torch = None
    EncoderClassifier = None

# MOS opcional
try:
    from dnsmos import DNSMOS  # pip install dnsmos
except Exception:
    DNSMOS = None

EPS = 1e-12

# ---- completeness: skip whole row only if all required metrics are present ----
KEY_COLS = {"rel_enh","backend","preset","rel_prep","error"}
OPTIONAL_COLS = {
    # optional/costly or env-dependent metrics that should NOT block skipping
    "rtf_out","dnsmos_sig_out","dnsmos_bak_out","dnsmos_ovrl_out",
    "lufs_in","lufs_out","delta_lufs",
    "srmr_in","srmr_out","delta_srmr",
    "emb_temporal_smoothness_in","emb_temporal_smoothness_out","delta_emb_temporal_smoothness",
    "num_clusters_in","num_clusters_out",
    "silhouette_in","silhouette_out",
    "db_index_in","db_index_out",
    "calinski_harabasz_in","calinski_harabasz_out",
    "between_cluster_min_cos_in","between_cluster_min_cos_out",
    "cluster_size_cv_in","cluster_size_cv_out",
}

def _row_is_complete(row: dict, header_cols: List[str]) -> bool:
    """True if all required (non-optional) columns are non-empty and error is empty."""
    if row.get("error", "") != "":
        return False
    for c in header_cols:
        if c in KEY_COLS or c in OPTIONAL_COLS:
            continue
        v = row.get(c, None)
        if v in (None, "", "None"):
            return False
    return True


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

# ---------- NUEVO: VAD + niveles ----------
# --- resample ligero sin deps externas ---
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
    """Devuelve (mask_bool, frame_sec) o (None, None) si no hay webrtcvad o SR no soportado."""
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

def _vad_mask_wrapped(x: np.ndarray, sr: int, backend: str = "webrtc", vad_resample: str = "48k"):
    """
    Aplica VAD con posible re-muestreo temporal.
    backend: 'webrtc' (por ahora), 'silero' opcional si lo añades luego.
    vad_resample: 'none' | '16k' | '48k' (recomendado 48k para WebRTC)
    """
    # elegir SR objetivo para VAD
    tgt_sr = {"none": sr, "16k": 16000, "48k": 48000}.get(vad_resample, sr)
    x_vad = _resample_linear(x, sr, tgt_sr) if tgt_sr != sr else x

    if backend == "webrtc":
        m_tgt, step = _vad_mask_webrtc(x_vad, tgt_sr)
    else:
        # placeholder por si luego añades otros backends
        m_tgt, step = _vad_mask_webrtc(x_vad, tgt_sr)

    if m_tgt is None:
        return None, None

    # mapear máscara de vuelta al SR original si se re-muestreó
    if tgt_sr != sr:
        m = _resample_linear(m_tgt.astype(np.float32), tgt_sr, sr) > 0.5
        return m, step
    return m_tgt, step


def _vad_stats_and_levels(x: np.ndarray, sr: int, mask: Optional[np.ndarray]):
    """Dict con vad_speech_ratio, seg_count, seg_mean_dur_s y niveles speech/noise en dBFS + SNR."""
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

# ---------- NUEVO: embeddings/cluster + MOS ----------
_ECAPA = None
def _get_ecapa():
    global _ECAPA
    if _ECAPA is None and (EncoderClassifier is not None):
        try:
            _ECAPA = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        except Exception:
            _ECAPA = None
    return _ECAPA

def _emb_cluster_metrics(x: np.ndarray, sr: int, mask: Optional[np.ndarray],
                         win_s: float = 1.5, hop_s: float = 0.75):
    """
    Devuelve dict con:
    emb_temporal_smoothness, num_clusters, silhouette, db_index, calinski_harabasz,
    between_cluster_min_cos, cluster_size_cv
    o todos None si faltan deps.
    """
    try:
        if torch is None or EncoderClassifier is None or AgglomerativeClustering is None \
        or silhouette_score is None:
            return {
                "emb_temporal_smoothness": None, "num_clusters": None, "silhouette": None,
                "db_index": None, "calinski_harabasz": None,
                "between_cluster_min_cos": None, "cluster_size_cv": None,
            }
        ecapa = _get_ecapa()
        if ecapa is None:
            return {
                "emb_temporal_smoothness": None, "num_clusters": None, "silhouette": None,
                "db_index": None, "calinski_harabasz": None,
                "between_cluster_min_cos": None, "cluster_size_cv": None,
            }

        win = int(sr * win_s); hop = int(sr * hop_s)
        if win <= 0 or len(x) < win:
            return {k: None for k in [
                "emb_temporal_smoothness","num_clusters","silhouette","db_index",
                "calinski_harabasz","between_cluster_min_cos","cluster_size_cv"]}

        seg_embs = []
        for i in range(0, len(x) - win + 1, hop):
            if mask is not None and not mask[i:i+win].any():
                continue
            seg = x[i:i+win]
            t = torch.from_numpy(seg).float().unsqueeze(0)
            try:
                emb = ecapa.encode_batch(t).squeeze(0).squeeze(0).detach().cpu().numpy()
                seg_embs.append(emb)
            except Exception:
                break
        if len(seg_embs) < 3:
            return {k: None for k in [
                "emb_temporal_smoothness","num_clusters","silhouette","db_index",
                "calinski_harabasz","between_cluster_min_cos","cluster_size_cv"]}

        E = np.vstack(seg_embs)

        def _cos(a,b):
            na = np.linalg.norm(a)+EPS; nb = np.linalg.norm(b)+EPS
            return float(np.dot(a,b)/(na*nb))

        sims = [_cos(E[i], E[i+1]) for i in range(len(E)-1)]
        smooth = float(np.median(sims))

        # clustering con AHC y métrica coseno
        try:
            ahc = AgglomerativeClustering(n_clusters=None, distance_threshold=0.35, metric="cosine", linkage="average")
        except TypeError:
                # sklearn < 1.2
            ahc = AgglomerativeClustering(n_clusters=None, distance_threshold=0.35, affinity="cosine", linkage="average")
        labels = ahc.fit_predict(E)
        num_clusters = int(len(np.unique(labels)))

        try:
            sil = float(silhouette_score(E, labels, metric="cosine")) if num_clusters>1 else None
            dbi = float(davies_bouldin_score(E, labels)) if num_clusters>1 else None
            ch  = float(calinski_harabasz_score(E, labels)) if num_clusters>1 else None
            cents = np.vstack([E[labels==c].mean(axis=0) for c in range(num_clusters)])
            if num_clusters>1:
                cs = []
                for i in range(num_clusters):
                    for j in range(i+1, num_clusters):
                        cs.append(_cos(cents[i], cents[j]))
                between_min_cos = float(min(cs))
            else:
                between_min_cos = None
            sizes = np.array([(labels==c).sum() for c in range(num_clusters)], dtype=float)
            cluster_cv = float(sizes.std()/(sizes.mean()+EPS)) if num_clusters>1 else 0.0
        except Exception:
            sil = dbi = ch = between_min_cos = cluster_cv = None

        return {
            "emb_temporal_smoothness": round(smooth, 6),
            "num_clusters": num_clusters,
            "silhouette": None if sil is None else round(sil, 6),
            "db_index": None if dbi is None else round(dbi, 6),
            "calinski_harabasz": None if ch is None else round(ch, 6),
            "between_cluster_min_cos": None if between_min_cos is None else round(between_min_cos, 6),
            "cluster_size_cv": None if cluster_cv is None else round(cluster_cv, 6),
        }
    except Exception:
        return {
            "emb_temporal_smoothness": None, "num_clusters": None, "silhouette": None,
            "db_index": None, "calinski_harabasz": None,
            "between_cluster_min_cos": None, "cluster_size_cv": None,
        }

def _maybe_dnsmos(x: np.ndarray, sr: int):
    if DNSMOS is None:
        return {"dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovrl": None}
    try:
        m = DNSMOS()
        r = m.predict_signal(x, sr)
        return {
            "dnsmos_sig": round(float(r.get("SIG", None)), 3),
            "dnsmos_bak": round(float(r.get("BAK", None)), 3),
            "dnsmos_ovrl": round(float(r.get("OVRL", None)), 3),
        }
    except Exception:
        return {"dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovrl": None}

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
    """List all WAVs under enh recursively, e.g. enh/<backend>/<preset>/data/raw/YYYY/MM/DD/*.wav"""
    enh_root = root / "enh"
    if not enh_root.exists():
        return []
    exts = {".wav", ".WAV"}  # agrega más si usas .flac
    return sorted(p for p in enh_root.rglob("*") if p.suffix in exts)

def _align_mask(mask: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = int(mask.shape[0])
    if m == n:
        return mask.astype(bool, copy=False)
    if m > n:
        return mask[:n].astype(bool, copy=False)
    # m < n: pad con False
    return np.pad(mask.astype(bool, copy=False), (0, n - m), constant_values=False)

def _vad_mask_wrapped(x: np.ndarray, sr: int, backend: str = "webrtc", vad_resample: str = "48k"):
    ...
    if tgt_sr != sr:
        m = _resample_linear(m_tgt.astype(np.float32), tgt_sr, sr) > 0.5
        return _align_mask(m, len(x)), step
    return _align_mask(m_tgt, len(x)), step


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Compute metrics on PREP vs ENH and deltas.")
    ap.add_argument("--dataset-dir", required=True, help="Root dataset dir")
    ap.add_argument("--csv-out", default=None, help="Pairs CSV output")
    ap.add_argument("--csv-summary", default=None, help="Summary CSV output")

    ap.add_argument("--vad-backend", default="webrtc", choices=["webrtc"],
                    help="Backend VAD. Usaremos WebRTC.")
    ap.add_argument("--vad-resample", default="48k", choices=["none","16k","48k"],
                    help="SR temporal solo para VAD. Recomendado 48k con WebRTC.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="No recalcula filas ya procesadas sin error.")
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

    # ---------- NUEVO: columnas extra ----------
    # VAD + niveles + SNR
    header += [
        "vad_speech_ratio_in","vad_speech_ratio_out","delta_vad_speech_ratio",
        "vad_seg_count_in","vad_seg_count_out","delta_vad_seg_count",
        "vad_seg_mean_dur_in_s","vad_seg_mean_dur_out_s","delta_vad_seg_mean_dur_s",
        "speech_level_in_dbfs","speech_level_out_dbfs","delta_speech_level_db",
        "nonspeech_level_in_dbfs","nonspeech_level_out_dbfs","delta_nonspeech_level_db",
        "speech_nonspeech_snr_in_db","speech_nonspeech_snr_out_db","delta_speech_nonspeech_snr_db",
    ]
    # Embeddings/cluster
    header += [
        "emb_temporal_smoothness_in","emb_temporal_smoothness_out","delta_emb_temporal_smoothness",
        "num_clusters_in","num_clusters_out",
        "silhouette_in","silhouette_out",
        "db_index_in","db_index_out",
        "calinski_harabasz_in","calinski_harabasz_out",
        "between_cluster_min_cos_in","between_cluster_min_cos_out",
        "cluster_size_cv_in","cluster_size_cv_out",
    ]
    # Coste + MOS
    header += ["rtf_out","dnsmos_sig_out","dnsmos_bak_out","dnsmos_ovrl_out"]

    # índice de ya procesados y modo append (skip only if row is complete)
    done = set()
    append_mode = out_pairs.exists()
    if args.skip_existing and out_pairs.exists():
        with out_pairs.open("r", encoding="utf-8") as f_done:
            rd = csv.DictReader(f_done)
            prev_hdr = rd.fieldnames or []
            for r in rd:
                if _row_is_complete(r, prev_hdr):
                    rel = r.get("rel_enh", "")
                    if rel:
                        done.add(rel)


    rows: List[Dict[str, Optional[float]]] = []
    n_ok = 0

    fmode = "a" if append_mode else "w"
    with out_pairs.open(fmode, newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        if not append_mode:
            wr.writeheader()

        for enh in enh_files:
            rel_enh = enh.relative_to(base).as_posix()
            if args.skip_existing and rel_enh in done:
                    continue
            row = dict.fromkeys(header, None)
            try:
                rel_enh = enh.relative_to(base).as_posix()
                # extraer backend y preset respecto a base/enh
                rel_parts = enh.relative_to(base / "enh").parts
                backend = rel_parts[0] if len(rel_parts) >= 2 else ""
                preset  = rel_parts[1] if len(rel_parts) >= 2 else ""
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

                # ---------- NUEVO: VAD + niveles + SNR ----------
                mask_in,  _ = _vad_mask_wrapped(xin,  srin,  backend=args.vad_backend, vad_resample=args.vad_resample)
                mask_out, _ = _vad_mask_wrapped(xout, srout, backend=args.vad_backend, vad_resample=args.vad_resample)
                # asegurar por si acaso
                mask_in  = _align_mask(mask_in,  len(xin))
                mask_out = _align_mask(mask_out, len(xout))
                vad_in  = _vad_stats_and_levels(xin, srin, mask_in)
                vad_out = _vad_stats_and_levels(xout, srout, mask_out)

                # ---------- NUEVO: embeddings/cluster ----------
                emb_in  = _emb_cluster_metrics(xin, srin, mask_in)
                emb_out = _emb_cluster_metrics(xout, srout, mask_out)

                # ---------- NUEVO: MOS no intrusivo (sobre 'out') ----------
                mos_out = _maybe_dnsmos(xout, srout)

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
                    # NUEVO: VAD + niveles + SNR
                    "vad_speech_ratio_in": vad_in["vad_speech_ratio"],
                    "vad_speech_ratio_out": vad_out["vad_speech_ratio"],
                    "delta_vad_speech_ratio": d(vad_in["vad_speech_ratio"], vad_out["vad_speech_ratio"]),
                    "vad_seg_count_in": vad_in["vad_seg_count"],
                    "vad_seg_count_out": vad_out["vad_seg_count"],
                    "delta_vad_seg_count": d(vad_in["vad_seg_count"], vad_out["vad_seg_count"]),
                    "vad_seg_mean_dur_in_s": vad_in["vad_seg_mean_dur_s"],
                    "vad_seg_mean_dur_out_s": vad_out["vad_seg_mean_dur_s"],
                    "delta_vad_seg_mean_dur_s": d(vad_in["vad_seg_mean_dur_s"], vad_out["vad_seg_mean_dur_s"]),
                    "speech_level_in_dbfs": vad_in["speech_level_dbfs"],
                    "speech_level_out_dbfs": vad_out["speech_level_dbfs"],
                    "delta_speech_level_db": d(vad_in["speech_level_dbfs"], vad_out["speech_level_dbfs"]),
                    "nonspeech_level_in_dbfs": vad_in["nonspeech_level_dbfs"],
                    "nonspeech_level_out_dbfs": vad_out["nonspeech_level_dbfs"],
                    "delta_nonspeech_level_db": d(vad_in["nonspeech_level_dbfs"], vad_out["nonspeech_level_dbfs"]),
                    "speech_nonspeech_snr_in_db": vad_in["speech_nonspeech_snr_db"],
                    "speech_nonspeech_snr_out_db": vad_out["speech_nonspeech_snr_db"],
                    "delta_speech_nonspeech_snr_db": d(vad_in["speech_nonspeech_snr_db"], vad_out["speech_nonspeech_snr_db"]),
                    # NUEVO: embeddings/cluster
                    "emb_temporal_smoothness_in": emb_in["emb_temporal_smoothness"],
                    "emb_temporal_smoothness_out": emb_out["emb_temporal_smoothness"],
                    "delta_emb_temporal_smoothness": d(emb_in["emb_temporal_smoothness"], emb_out["emb_temporal_smoothness"]),
                    "num_clusters_in": emb_in["num_clusters"],
                    "num_clusters_out": emb_out["num_clusters"],
                    "silhouette_in": emb_in["silhouette"],
                    "silhouette_out": emb_out["silhouette"],
                    "db_index_in": emb_in["db_index"],
                    "db_index_out": emb_out["db_index"],
                    "calinski_harabasz_in": emb_in["calinski_harabasz"],
                    "calinski_harabasz_out": emb_out["calinski_harabasz"],
                    "between_cluster_min_cos_in": emb_in["between_cluster_min_cos"],
                    "between_cluster_min_cos_out": emb_out["between_cluster_min_cos"],
                    "cluster_size_cv_in": emb_in["cluster_size_cv"],
                    "cluster_size_cv_out": emb_out["cluster_size_cv"],
                    # NUEVO: coste y MOS
                    "rtf_out": None,
                    "dnsmos_sig_out": mos_out["dnsmos_sig"],
                    "dnsmos_bak_out": mos_out["dnsmos_bak"],
                    "dnsmos_ovrl_out": mos_out["dnsmos_ovrl"],
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
                        "delta_flat","delta_centroid_hz","delta_lufs","delta_srmr",
                        "delta_vad_speech_ratio","delta_vad_seg_count","delta_vad_seg_mean_dur_s",
                        "delta_speech_level_db","delta_nonspeech_level_db","delta_speech_nonspeech_snr_db",
                        "delta_emb_temporal_smoothness"]:
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
                   "mean_delta_flat","mean_delta_centroid_hz","mean_delta_lufs","mean_delta_srmr",
                   "mean_delta_vad_speech_ratio","mean_delta_vad_seg_count","mean_delta_vad_seg_mean_dur_s",
                   "mean_delta_speech_level_db","mean_delta_nonspeech_level_db","mean_delta_speech_nonspeech_snr_db",
                   "mean_delta_emb_temporal_smoothness"]
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
                        "delta_flat","delta_centroid_hz","delta_lufs","delta_srmr",
                        "delta_vad_speech_ratio","delta_vad_seg_count","delta_vad_seg_mean_dur_s",
                        "delta_speech_level_db","delta_nonspeech_level_db","delta_speech_nonspeech_snr_db",
                        "delta_emb_temporal_smoothness"]:
                c = counts[(b,p)].get(col, 0)
                m = (sdict[col] / c) if c > 0 else None
                row["mean_"+col] = None if m is None else round(m, 6)
            wr.writerow(row)

    print(f"[done] pairs={out_pairs} summary={out_sum} ok={n_ok} total_enh={len(enh_files)}", file=sys.stderr)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# compute_vad_fill.py
import argparse, os, sys, math, tempfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

try:
    import webrtcvad
except Exception:
    print("Falta webrtcvad. Instala con: pip install webrtcvad", file=sys.stderr)
    sys.exit(1)

# --- util: resample ligero (sin scipy/librosa) ---
def resample_mono_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if sr_in == 16000:
        return x.astype(np.float32, copy=False)
    # interpolación lineal
    n_out = int(round(len(x) * 16000 / sr_in))
    if n_out < 1:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    fp = x.astype(np.float64, copy=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(x_new, xp, fp).astype(np.float32)
    return y

def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def frame_bytes(pcm16: np.ndarray, sr: int = 16000, frame_ms: int = 30):
    # 10/20/30 ms válidos para webrtcvad
    assert frame_ms in (10, 20, 30)
    bytes_per_sample = 2  # int16
    samples_per_frame = sr * frame_ms // 1000
    n_frames = len(pcm16) // samples_per_frame
    pcm16 = pcm16[: n_frames * samples_per_frame]
    b = pcm16.tobytes()
    frame_len = samples_per_frame * bytes_per_sample
    for i in range(n_frames):
        start = i * frame_len
        yield b[start : start + frame_len]

def compute_vad_metrics(wav_path: Path, aggressiveness: int = 2, frame_ms: int = 30):
    if not wav_path.exists():
        return None
    x, sr = sf.read(str(wav_path), always_2d=False)
    x = resample_mono_16k(x, sr)
    if x.size == 0:
        return {
            "vad_total_s": 0.0,
            "vad_total_speech_s": 0.0,
            "vad_speech_ratio": 0.0,
            "vad_num_segments": 0,
            "vad_avg_segment_s": 0.0,
            "vad_longest_segment_s": 0.0,
        }
    pcm16 = float_to_pcm16(x)
    vad = webrtcvad.Vad(aggressiveness)
    frm_bytes = list(frame_bytes(pcm16, 16000, frame_ms))
    if not frm_bytes:
        total_s = len(x) / 16000.0
        return {
            "vad_total_s": total_s,
            "vad_total_speech_s": 0.0,
            "vad_speech_ratio": 0.0,
            "vad_num_segments": 0,
            "vad_avg_segment_s": 0.0,
            "vad_longest_segment_s": 0.0,
        }

    voiced_flags = [vad.is_speech(fb, 16000) for fb in frm_bytes]
    # agrupar segmentos contiguos True
    spk_segments = []
    i = 0
    while i < len(voiced_flags):
        if voiced_flags[i]:
            j = i + 1
            while j < len(voiced_flags) and voiced_flags[j]:
                j += 1
            spk_segments.append((i, j))  # [i, j)
            i = j
        else:
            i += 1

    frame_sec = frame_ms / 1000.0
    seg_durs = [(j - i) * frame_sec for (i, j) in spk_segments]
    total_s = len(x) / 16000.0
    total_speech_s = float(np.sum(seg_durs)) if seg_durs else 0.0
    num_segments = len(seg_durs)
    avg_seg = float(np.mean(seg_durs)) if seg_durs else 0.0
    longest_seg = float(np.max(seg_durs)) if seg_durs else 0.0
    speech_ratio = (total_speech_s / total_s) if total_s > 0 else 0.0

    return {
        "vad_total_s": round(total_s, 6),
        "vad_total_speech_s": round(total_speech_s, 6),
        "vad_speech_ratio": round(speech_ratio, 6),
        "vad_num_segments": int(num_segments),
        "vad_avg_segment_s": round(avg_seg, 6),
        "vad_longest_segment_s": round(longest_seg, 6),
    }

def ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series([np.nan] * len(df), dtype="float64" if c != "vad_num_segments" else "Int64")
    return df

def row_has_all_vad(row, cols):
    for c in cols:
        if pd.isna(row[c]):
            return False
    return True

def main():
    ap = argparse.ArgumentParser(description="Rellenar métricas VAD faltantes en un CSV sin agregar filas.")
    ap.add_argument("csv_path", type=Path, help="Ruta del CSV existente.")
    ap.add_argument("--audio-col", default="enh_path",
                    help="Nombre de la columna con la ruta del audio a evaluar (ej. enh_path, ref_path, in_path).")
    ap.add_argument("--root", type=Path, default=None,
                    help="Prefijo opcional para resolver rutas relativas de audio.")
    ap.add_argument("--aggr", type=int, default=2, choices=[0,1,2,3], help="Agresividad VAD 0-3.")
    ap.add_argument("--frame-ms", type=int, default=30, choices=[10,20,30], help="Tamaño de frame VAD.")
    ap.add_argument("--backup", action="store_true", help="Guardar .bak del CSV original.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    vad_cols = [
        "vad_total_s",
        "vad_total_speech_s",
        "vad_speech_ratio",
        "vad_num_segments",
        "vad_avg_segment_s",
        "vad_longest_segment_s",
    ]
    df = ensure_cols(df, vad_cols)

    if args.audio_col not in df.columns:
        print(f"Columna de audio '{args.audio_col}' no existe en el CSV.", file=sys.stderr)
        sys.exit(2)

    n_total = len(df)
    n_done = 0
    for idx, row in df.iterrows():
        # si ya están todas las VAD, saltar
        if row_has_all_vad(row, vad_cols):
            n_done += 1
            continue

        rel = str(row[args.audio_col]) if not pd.isna(row[args.audio_col]) else ""
        if not rel:
            # dejar NaN si no hay audio
            continue
        wav_path = Path(rel)
        if args.root and not wav_path.is_absolute():
            wav_path = args.root / wav_path

        m = compute_vad_metrics(wav_path, aggressiveness=args.aggr, frame_ms=args.frame_ms)
        if m is None:
            # archivo no encontrado: no rellenar
            continue

        for k, v in m.items():
            if pd.isna(row[k]):
                df.at[idx, k] = v

        n_done += 1

    # respaldo opcional y guardado atómico
    if args.backup:
        shutil.copy2(args.csv_path, str(args.csv_path) + ".bak")

    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(args.csv_path.parent), suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    tmp_path.replace(args.csv_path)

    print(f"Procesadas {n_done}/{n_total} filas. CSV actualizado: {args.csv_path}")

if __name__ == "__main__":
    main()
