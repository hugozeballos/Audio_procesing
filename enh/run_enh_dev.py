#!/usr/bin/env python3
import sys, pathlib
from pathlib import Path
import argparse
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))                   # raíz
import os, csv, time, numpy as np, soundfile as sf
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from utils.audio_prep import trim_tails_vad, peak_normalize_minus1_dbfs as peak_norm
from enh.backends.metricgan import MetricGANPlus
from enh.backends.resemble_enh import ResembleEnh
from enh.backends.voicefixer_real import VoiceFixerReal
from enh.backends.sepformer import SepformerEnh
from enh.backends.deepfilternet import DeepFilterNetCLI 
from enh.backends.clearervoice import ClearerVoiceCLI


BACKENDS = {
    "clearervoice": ClearerVoiceCLI,
    "deepfilternet": DeepFilterNetCLI,
    "sepformer": SepformerEnh,
    "resemble_enh": ResembleEnh,
    "voicefixer": VoiceFixerReal,
    "metricgan": MetricGANPlus,
}

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset-audio-raw"
DEV_LIST = DATASET_DIR / "splits" / "dev.list"
OUT_ROOT = DATASET_DIR / "enh"

load_dotenv(find_dotenv())
BACKEND_NAME = os.getenv("ENH_BACKEND", "metricgan")
PRESET       = os.getenv("ENH_PRESET", "medium")

def get_backend(name: str):
    cls = BACKENDS.get(name.lower())
    if cls is None:
        raise ValueError(f"backend desconocido: {name}")
    return cls()

def read_list(p: pathlib.Path):
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]

def ensure_parent(p: pathlib.Path): p.parent.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=BACKEND_NAME, choices=list(BACKENDS.keys()))
parser.add_argument("--preset",  default=PRESET,       choices=["light","medium","aggressive"])
args = parser.parse_args()
BACKEND_NAME = args.backend
PRESET       = args.preset

def main():
    items = read_list(DEV_LIST)
    if not items:
        print("dev.list vacío."); return

    backend = get_backend(BACKEND_NAME)
    log_dir = OUT_ROOT / backend.name / PRESET
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "enh_log.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["in_path","out_path","sr","dur_in_s","dur_prep_s","dur_out_s",
                    "peak_in","peak_prep","peak_out","gain_prep_db","rtf","backend","preset","note"])
        for rel in tqdm(items, desc=f"Enh {backend.name}/{PRESET}"):
            in_rel = Path("prep") / Path(rel).with_suffix(".wav")   # usa WAV de prep
            in_abs = DATASET_DIR / in_rel
            if not in_abs.exists():
                raise FileNotFoundError(f"Falta PREP local: {in_abs}. Ejecuta preprocesing/prep_dev.py primero.")
            x, sr = sf.read(in_abs, dtype="float32", always_2d=False)
            if x.ndim == 2: x = x.mean(axis=1).astype(np.float32)
            dur_in = len(x)/sr if sr else 0.0; peak_in = float(np.max(np.abs(x))) if x.size else 0.0

            y = peak_norm(trim_tails_vad(x, sr))
            dur_prep = len(y)/sr if sr else 0.0; peak_prep = float(np.max(np.abs(y))) if y.size else 0.0
            gain_db = 20*np.log10(peak_prep/peak_in) if (peak_in>0 and peak_prep>0) else 0.0

            t0 = time.time()
            z, info = backend.enhance(y, sr, PRESET, opts={})
            rtf = (time.time()-t0) / (len(y)/sr if sr and len(y)>0 else 1.0)
            peak_out = float(np.max(np.abs(z))) if z.size else 0.0
            dur_out = len(z)/sr if sr else 0.0

            out_rel = Path("enh") / backend.name / PRESET / Path(rel).with_suffix(".wav")
            out_abs = DATASET_DIR / out_rel
            ensure_parent(out_abs); sf.write(out_abs, z, sr, format="WAV", subtype="PCM_16")

            w.writerow([rel, str(out_rel), sr, f"{dur_in:.3f}", f"{dur_prep:.3f}", f"{dur_out:.3f}",
                        f"{peak_in:.6f}", f"{peak_prep:.6f}", f"{peak_out:.6f}",
                        f"{gain_db:.3f}", f"{rtf:.3f}", backend.name, PRESET, info.get("note","")])
    print(f"OK -> {log_path}")

if __name__ == "__main__":
    main()
