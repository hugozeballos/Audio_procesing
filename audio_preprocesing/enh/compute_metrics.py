# enh/compute_metrics.py
#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path
import numpy as np
import soundfile as sf



# --- imports opcionales (cada uno se usa si está disponible) ---
try:
    from pystoi.stoi import stoi as _stoi
except Exception:
    _stoi = None

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None

try:
    from srmrpy import srmr as _srmr
except Exception:
    _srmr = None

# --- util: resample robusto sin depender de scipy/librosa ---
def _resample_to(sr_in: int, x: np.ndarray, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    # lineal por interpolación (suficiente para métricas agregadas)
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 1 or len(x) <= 1:
        return np.zeros((max(n_out, 1),), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=True, dtype=np.float64)
    fp = x.astype(np.float64, copy=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=True, dtype=np.float64)
    y = np.interp(x_new, xp, fp)
    return y.astype(np.float32)

def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32)

def _dur(x: np.ndarray, sr: int) -> float:
    return 0.0 if sr <= 0 else len(x) / float(sr)

def _clip_rate(x: np.ndarray, thr: float = 0.999) -> float:
    if x.size == 0:
        return 0.0
    return float((np.abs(x) >= thr).sum()) / float(x.size)

EPS = 1e-12

def _zero_mean(x):
    return x - np.mean(x)

def _align_by_xcorr(ref, est, sr, max_shift_s=0.5):
    # recorta a la misma longitud primero
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    max_lag = int(max_shift_s * sr)
    if max_lag == 0:
        return ref, est
    # correlación con ventana de desplazamientos
    lags = np.arange(-max_lag, max_lag + 1)
    best_lag = 0
    best_val = -np.inf
    for lag in lags:
        if lag >= 0:
            v = np.dot(ref[lag:], est[:n-lag])
        else:
            v = np.dot(ref[:n+lag], est[-lag:])
        if v > best_val:
            best_val = v
            best_lag = lag
    # aplica desplazamiento
    if best_lag >= 0:
        return ref[best_lag:], est[:n-best_lag]
    else:
        lag = -best_lag
        return ref[:n-lag], est[lag:]


def _peak_dbfs(x: np.ndarray) -> float:
    if x.size == 0: return float("nan")
    return 20 * np.log10(np.max(np.abs(x)) + 1e-12)

def _rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0: return float("nan")
    return 20 * np.log10(np.sqrt(np.mean(x**2) + 1e-12))

def _lufs(x: np.ndarray, sr: int):
    if pyln is None or x.size == 0 or sr <= 0:
        return None
    meter = pyln.Meter(sr)  # EBU R128
    try:
        return float(meter.integrated_loudness(x))
    except Exception:
        return None

def _stoi_score(ref: np.ndarray, deg: np.ndarray, sr_ref: int, sr_deg: int) -> float | None:
    if _stoi is None:
        return None
    # STOI soporta 10 kHz o 16 kHz. Usamos 16 kHz.
    tgt_sr = 16000
    r = _resample_to(sr_ref, _to_mono(ref), tgt_sr)
    d = _resample_to(sr_deg, _to_mono(deg), tgt_sr)
    L = min(len(r), len(d))
    if L < tgt_sr // 2:
        return None
    r = r[:L]; d = d[:L]
    try:
        return float(_stoi(r, d, tgt_sr, extended=False))
    except Exception:
        return None

def _srmr_score(x: np.ndarray, sr: int) -> float | None:
    if _srmr is None or x.size == 0 or sr <= 0:
        return None
    try:
        val, _, _ = _srmr(_to_mono(x), sr)
        # srmrpy devuelve (score, energy, bands). Nos quedamos con score.
        if isinstance(val, (list, tuple, np.ndarray)):
            val = float(np.mean(val))
        return float(val)
    except Exception:
        return None
    
def snr_db(ref, est, sr, align=True):
    """
    SNR = 10*log10( ||ref||^2 / ||est-ref||^2 ), con opcional alineación.
    ref: señal referencia limpia o 'antes'
    est: señal estimada o 'después'
    """
    if align:
        ref, est = _align_by_xcorr(ref, est, sr)
    n = min(len(ref), len(est))
    if n < 160:  # <10 ms @16k
        return None
    ref = ref[:n]
    est = est[:n]
    err = est - ref
    p_sig = np.sum(ref**2) + EPS
    p_err = np.sum(err**2) + EPS
    return 10.0 * np.log10(p_sig / p_err)

def snr_segmental_db(ref, est, sr, segment_length=0.02, align=True):
    """
    SNR segmental - más robusto que SNR normal para evaluación de voz
    """
    if align:
        ref, est = _align_by_xcorr(ref, est, sr)
    
    n = min(len(ref), len(est))
    if n < sr * segment_length * 2:  # mínimo 2 segmentos
        return None
    
    ref = ref[:n]
    est = est[:n]
    
    segment_samples = int(sr * segment_length)  # 20ms típico
    num_segments = n // segment_samples
    
    snr_segments = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        
        ref_seg = ref[start:end]
        est_seg = est[start:end]
        err_seg = est_seg - ref_seg
        
        # Solo calcular SNR en segmentos con suficiente energía
        ref_energy = np.sum(ref_seg**2)
        if ref_energy > EPS and len(ref_seg) > 10:
            p_sig = ref_energy
            p_err = np.sum(err_seg**2) + EPS
            snr_seg = 10.0 * np.log10(p_sig / p_err)
            # Limitar valores extremos (típico en SNRseg)
            snr_seg = max(min(snr_seg, 35), -10)
            snr_segments.append(snr_seg)
    
    if not snr_segments:
        return None
    
    return float(np.mean(snr_segments))

def spectral_distortion_db(ref, est, sr, align=True):
    """
    Distorsión espectral simple - evalúa preservación de características espectrales
    """
    if align:
        ref, est = _align_by_xcorr(ref, est, sr)
    
    n = min(len(ref), len(est))
    if n < 1024:  # Necesitamos suficiente longitud para análisis espectral
        return None
    
    ref = ref[:n]
    est = est[:n]
    
    # Calcular espectros de potencia suavizados
    n_fft = 1024
    hop_length = n_fft // 4
    
    # Espectrograma de referencia
    ref_mag = []
    for i in range(0, n - n_fft, hop_length):
        frame = ref[i:i + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        ref_mag.append(spec)
    
    # Espectrograma de estimación
    est_mag = []
    for i in range(0, n - n_fft, hop_length):
        frame = est[i:i + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        est_mag.append(spec)
    
    if not ref_mag or not est_mag:
        return None
    
    ref_mag = np.array(ref_mag)
    est_mag = np.array(est_mag)
    
    # Asegurar misma longitud
    min_frames = min(len(ref_mag), len(est_mag))
    ref_mag = ref_mag[:min_frames]
    est_mag = est_mag[:min_frames]
    
    # Distorsión espectral en dB (log spectral distance)
    spectral_dist = np.mean(20 * np.log10((np.abs(ref_mag - est_mag) + EPS) / (ref_mag + EPS)))
    return float(spectral_dist)

def si_sdr_db(ref, est, sr, align=True):
    """
    SI-SDR según Le Roux et al. (scale-invariant).
    """
    if align:
        ref, est = _align_by_xcorr(ref, est, sr)
    n = min(len(ref), len(est))
    if n < 160:
        return None
    s = _zero_mean(ref[:n])
    sh = _zero_mean(est[:n])
    denom = np.sum(s**2) + EPS
    alpha = np.sum(sh * s) / denom
    s_target = alpha * s
    e_noise = sh - s_target
    num = np.sum(s_target**2) + EPS
    den = np.sum(e_noise**2) + EPS
    return 10.0 * np.log10(num / den)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)
    return x, int(sr)

def main():
    ap = argparse.ArgumentParser(description="Compute enhancement metrics over dev split")
    ap.add_argument("--dataset-dir", type=Path, default=Path("dataset-audio-raw"))
    ap.add_argument("--dev-list", type=Path, default=None,
                    help="Ruta a splits/dev.list; por defecto usa <dataset-dir>/splits/dev.list")
    ap.add_argument("--ref-source", choices=["prep", "raw"], default="prep",
                    help="Referencia para métricas intrusivas: prep o raw")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Salida CSV; por defecto enh/enh_metrics.csv dentro del dataset")
    args = ap.parse_args()

    ds = args.dataset_dir
    dev_list = args.dev_list or (ds / "splits" / "dev.list")
    out_csv = args.out_csv or (ds / "enh" / "enh_metrics.csv")

    if not dev_list.exists():
        raise SystemExit(f"dev.list no existe: {dev_list}")

    rels = [ln.strip() for ln in dev_list.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]

    # Descubre backends/presets disponibles por carpetas
    enh_root = ds / "enh"
    if not enh_root.exists():
        raise SystemExit(f"No existe {enh_root}. Corre primero run_enh_dev.py")
    combos = []
    for backend_dir in sorted([p for p in enh_root.iterdir() if p.is_dir()]):
        for preset_dir in sorted([p for p in backend_dir.iterdir() if p.is_dir()]):
            combos.append((backend_dir.name, preset_dir.name, preset_dir))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "backend","preset","rel",
            "ref_path","out_path",
            "sr_ref","sr_out",
            "dur_ref_s","dur_out_s","dur_diff_s",
            "stoi","srmr_ref","srmr",
            "lufs_ref","lufs","delta_lufs",
            "peak_dbfs_ref","peak_dbfs_out",
            "rms_dbfs_ref","rms_dbfs_out",
            "clip_rate_ref","clip_rate",
            "snr_db","si_sdr_db",
            "snr_seg_db", "spectral_dist_db",
            "rtf",
        ])

        for rel in rels:
            rel_wav = Path(rel).with_suffix(".wav")
            # referencia
            if args.ref_source == "prep":
                ref_path = ds / "prep" / rel_wav
            else:
                # si raw es mp3/flac, lo intentamos igualmente; si libsndfile no soporta, se omite STOI
                ref_path = ds / rel
            ref_x = ref_sr = None
            if ref_path.exists():
                try:
                    ref_x, ref_sr = _read_wav(ref_path)
                except Exception:
                    ref_x, ref_sr = None, None

            for backend, preset, preset_dir in combos:
                out_path = preset_dir / rel_wav
                if not out_path.exists():
                    # no hay salida para este archivo-backend/preset; continuar
                    continue
                try:
                    out_x, out_sr = _read_wav(out_path)
                except Exception:
                    w.writerow([
                        backend, preset, str(rel),
                        str(ref_path), str(out_path),
                        ref_sr or "", "",          # sr_out vacío
                        f"{_dur(ref_x, ref_sr):.3f}" if (ref_x is not None and ref_sr) else "",  # dur_ref_s
                        "", "",                    # dur_out_s, dur_diff_s
                        "", "",                    # stoi, srmr
                        f"{_lufs(ref_x, ref_sr):.2f}" if (ref_x is not None and ref_sr) else "",  # lufs_ref
                        "", "",                    # lufs, delta_lufs
                        f"{_peak_dbfs(ref_x):.2f}" if (ref_x is not None and ref_sr) else "",
                        "",                        # peak_dbfs_out
                        f"{_rms_dbfs(ref_x):.2f}" if (ref_x is not None and ref_sr) else "",
                        "",                        # rms_dbfs_out
                        f"{_clip_rate(ref_x):.6f}" if (ref_x is not None and ref_sr) else "",
                        "",                        # clip_rate
                        "", "", "",                # stoi, srmr_ref, srmr
                        "", "", "",                # snr_seg, spectral_dist
                        "",                        # rtf
                    ])
                    continue
                # métricas
                dur_out = _dur(out_x, out_sr)
                clip_out = _clip_rate(out_x)
                lufs_out = _lufs(out_x, out_sr)
                srmr_out = _srmr_score(out_x, out_sr)
                peak_out = _peak_dbfs(out_x)
                rms_out  = _rms_dbfs(out_x)
                
                snr_val = sisdr_val = snr_seg_val = spectral_dist_val = None
                

                # métricas de referencia
                dur_ref = lufs_ref = srmr_ref = peak_ref = rms_ref = clip_ref = None
                stoi_val = None
                if ref_x is not None and ref_sr:
                    dur_ref  = _dur(ref_x, ref_sr)
                    lufs_ref = _lufs(ref_x, ref_sr)
                    srmr_ref = _srmr_score(ref_x, ref_sr)
                    peak_ref = _peak_dbfs(ref_x)
                    rms_ref  = _rms_dbfs(ref_x)
                    clip_ref = _clip_rate(ref_x)
                    stoi_val = _stoi_score(ref_x, out_x, ref_sr, out_sr)

                    ref_cmp = ref_x if ref_sr == out_sr else _resample_to(ref_sr, ref_x, out_sr)
                    snr_val   = snr_db(ref_cmp, out_x, out_sr, align=True)
                    sisdr_val = si_sdr_db(ref_cmp, out_x, out_sr, align=True)
                    snr_seg_val = snr_segmental_db(ref_cmp, out_x, out_sr, align=True)  # ← NUEVO CÁLCULO
                    spectral_dist_val = spectral_distortion_db(ref_cmp, out_x, out_sr, align=True)  # ← NUEVO CÁLCULO

                    

                # deltas
                dur_diff = (dur_out - dur_ref) if dur_ref is not None else None
                delta_lufs = (lufs_out - lufs_ref) if (lufs_out is not None and lufs_ref is not None) else None

                w.writerow([
                    backend, preset, str(rel),
                    str(ref_path), str(out_path),
                    ref_sr or "", out_sr,
                    f"{dur_ref:.3f}" if dur_ref is not None else "",
                    f"{dur_out:.3f}",
                    f"{dur_diff:.3f}" if dur_diff is not None else "",
                    f"{stoi_val:.4f}" if stoi_val is not None else "",
                    f"{srmr_ref:.4f}" if srmr_ref is not None else "",
                    f"{srmr_out:.4f}" if srmr_out is not None else "",
                    f"{lufs_ref:.2f}" if lufs_ref is not None else "",
                    f"{lufs_out:.2f}" if lufs_out is not None else "",
                    f"{delta_lufs:+.2f}" if delta_lufs is not None else "",
                    f"{peak_ref:.2f}" if peak_ref is not None else "",
                    f"{peak_out:.2f}",
                    f"{rms_ref:.2f}" if rms_ref is not None else "",
                    f"{rms_out:.2f}",
                    f"{clip_ref:.6f}" if clip_ref is not None else "",
                    f"{clip_out:.6f}",
                    f"{snr_val:.2f}"   if snr_val   is not None else "",
                    f"{sisdr_val:.2f}" if sisdr_val is not None else "",
                    f"{snr_seg_val:.2f}" if snr_seg_val is not None else "",  # ← NUEVA MÉTRICA
                    f"{spectral_dist_val:.2f}" if spectral_dist_val is not None else "",  # ← NUEVA MÉTRICA
                    "",  # rtf pendiente
                ])

    print(f"OK -> {out_csv}")

if __name__ == "__main__":
    main()
