# prep_dev.py
import os, io, csv, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import hf_hub_download
from utils.audio_prep import trim_tails_vad
from utils.audio_prep import peak_normalize_minus1_dbfs as peak_norm


# === rutas base ===
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset-audio-raw"         # <-- apunta al repo del dataset

DEV_LIST = DATASET_DIR / "splits" / "dev.list"
OUT_ROOT = DATASET_DIR / "prep"                          # salidas dentro del dataset

# 1) Cargar .env del dataset ANTES de leer variables
from dotenv import load_dotenv
load_dotenv(find_dotenv())

HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN   = os.getenv("HF_TOKEN")


def read_list(p: pathlib.Path):
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]

def load_audio(rel_path: str):
    local = DATASET_DIR / rel_path                    # <-- busca dentro del dataset
    if local.exists():
        x, sr = sf.read(local, dtype="float32", always_2d=False)
        if x.ndim == 2: x = x.mean(axis=1).astype("float32")
        return x, sr, str(local)
    # si no existe local, baja desde HF
    if not HF_REPO_ID:
        raise RuntimeError(f"No local y falta HF_REPO_ID: {rel_path}")
    cache = hf_hub_download(repo_id=HF_REPO_ID, filename=rel_path, token=HF_TOKEN, repo_type="dataset")
    x, sr = sf.read(cache, dtype="float32", always_2d=False)
    if x.ndim == 2: x = x.mean(axis=1).astype("float32")
    return x, sr, cache

def ensure_parent(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    items = read_list(DEV_LIST)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / "prep_log.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["in_path","out_path","sr","dur_in_s","dur_vad_s","peak_in","peak_out","gain_db"])
        for rel in tqdm(items, desc="Prep dev"):
            # load
            x, sr, src = load_audio(rel)
            dur_in = len(x)/sr if sr else 0.0
            peak_in = float(np.max(np.abs(x))) if x.size else 0.0

            # process
            y = trim_tails_vad(x, sr) if x.size else x
            z = peak_norm(y)
            peak_out = float(np.max(np.abs(z))) if z.size else 0.0
            gain_db = 20*np.log10(peak_out/peak_in) if (peak_in > 0 and peak_out > 0) else 0.0

            # write
            in_rel = pathlib.Path(rel)
            out_rel = pathlib.Path("prep") / in_rel.with_suffix(".wav")   # salida en WAV
            out_abs = DATASET_DIR / out_rel
            ensure_parent(out_abs)
            sf.write(out_abs, z, sr, format="WAV", subtype="PCM_16")


            w.writerow([rel, str(out_rel), sr, f"{dur_in:.3f}", f"{len(y)/sr:.3f}",
                        f"{peak_in:.6f}", f"{peak_out:.6f}", f"{gain_db:.3f}"])

    print(f"OK -> {log_path}")

if __name__ == "__main__":
    main()
