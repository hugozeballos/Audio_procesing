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
            "dur_ref_s","dur_out_s",
            "stoi","srmr","lufs","clip_rate","rtf"
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
                    # no pudimos leer salida
                    w.writerow([backend,preset,str(rel),str(ref_path),str(out_path),
                                ref_sr or "", "", "", "", "", "", "", ""])
                    continue

                # métricas
                dur_ref = _dur(ref_x, ref_sr) if ref_x is not None and ref_sr else None
                dur_out = _dur(out_x, out_sr)
                clip = _clip_rate(out_x)
                lufs = _lufs(out_x, out_sr)
                srmr_val = _srmr_score(out_x, out_sr)

                stoi_val = None
                if ref_x is not None and ref_sr:
                    stoi_val = _stoi_score(ref_x, out_x, ref_sr, out_sr)

                # RTF: intenta leer de enh_log.csv si existe
                rtf = ""
                # (opcional) puedes integrar lectura de enh/enh_log.csv aquí y mapear por out_path

                w.writerow([
                    backend, preset, str(rel),
                    str(ref_path), str(out_path),
                    ref_sr or "", out_sr,
                    f"{dur_ref:.3f}" if dur_ref is not None else "",
                    f"{dur_out:.3f}",
                    f"{stoi_val:.4f}" if stoi_val is not None else "",
                    f"{srmr_val:.4f}" if srmr_val is not None else "",
                    f"{lufs:.2f}" if isinstance(lufs,(int,float)) and not math.isnan(lufs) else "",
                    f"{clip:.6f}",
                    rtf
                ])

    print(f"OK -> {out_csv}")

if __name__ == "__main__":
    main()
