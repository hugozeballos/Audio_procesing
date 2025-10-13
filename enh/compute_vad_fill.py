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

import os, sys
print("DBG:file", __file__)
print("DBG:argv", sys.argv)
print("DBG:cwd", os.getcwd())
print("DBG:python", sys.executable)
print("DBG:sys.path[0:3]", sys.path[0:3])

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

    print("DBG:csv", args.csv_path)
    print("DBG:audio_col", args.audio_col)
    print("DBG:root", args.root)
    print("DBG:aggr", args.aggr, "frame_ms", args.frame_ms)


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
    print("DBG:columns", list(df.columns))
    print("DBG:rows", len(df))

    for idx, row in df.iterrows():
        print(f"DBG:row={idx}")
        # si ya están todas las VAD, saltar
        if row_has_all_vad(row, vad_cols):
            n_done += 1
            continue

        rel = str(row[args.audio_col]) if not pd.isna(row[args.audio_col]) else ""
        print("DBG:rel", rel)
        if not rel:
            # dejar NaN si no hay audio
            continue
        wav_path = Path(rel)
        if args.root and not wav_path.is_absolute():
            wav_path = args.root / wav_path
        print("DBG:wav_path", str(wav_path), "exists?", wav_path.exists())

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
