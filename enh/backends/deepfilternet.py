# enh/backends/deepfilternet.py
import os, subprocess, tempfile, pathlib
import sys
import numpy as np, soundfile as sf
from dataclasses import dataclass
from scipy.signal import resample_poly

from enh.backends.base import BackendBase
from utils.audio_prep import peak_normalize_minus1_dbfs as peak_norm

def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32)

def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0: return x
    up, down = sr_out, sr_in
    return resample_poly(x, up, down).astype(np.float32)

@dataclass
class DeepFilterNetCLI(BackendBase):
    """
    Backend para DeepFilterNet usando su CLI: `python -m df.enhance in.wav out.wav`.
    Requiere: `pip install deepfilternet` (trae el módulo `df`).
    Presets mapean a parámetros simples de post:
      - light/medium/aggressive -> distinto wet y post_gain_db.
    """
    name: str = "deepfilternet"
    TARGET_SR: int = 48000

    def __post_init__(self):
        self.device = os.getenv("ENH_DEVICE", "cpu").lower()  # solo informativo
        self.presets = {
            "light":      {"wet": 0.7, "post_gain_db": 0.0},
            "medium":     {"wet": 0.85,"post_gain_db": 0.0},
            "aggressive": {"wet": 1.0, "post_gain_db": 0.0},
        }

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: dict) -> tuple[np.ndarray, dict]:
        cfg = self.presets.get(preset, self.presets["medium"])
        wet = float(cfg["wet"])

        xin = _to_mono(x)
        xin_48k = _resample(xin, sr, self.TARGET_SR)

        # tmp wav -> CLI -> tmp wav
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            in_wav  = td / "in.wav"
            out_dir = td 
            sf.write(in_wav, xin_48k, self.TARGET_SR, subtype="PCM_16")

            # Llamada mínima. DeepFilterNet descarga el modelo al cache en el primer uso.
            cmd = [sys.executable, "-m", "df.enhance", str(in_wav), str(out_dir), "--device", os.getenv("ENH_DEVICE","cpu"), "--model", "dfnet3"] + (["--model_dir", os.getenv("DF_MODEL_DIR")] if os.getenv("DF_MODEL_DIR") else [])

            # Nota: algunas versiones aceptan flags extra; mantenemos la invocación mínima por portabilidad.
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"DeepFilterNet falló ({res.returncode}).\nSTDERR:\n{res.stderr}")

            out_file = [p for p in td.glob("*.wav") if p.name != "in.wav"][0]
            y48, sr_out = sf.read(out_file, dtype="float32", always_2d=False)
            if y48.ndim == 2: y48 = y48.mean(axis=1).astype(np.float32)

        # wet/dry mix
        n = min(len(xin_48k), len(y48))
        y48 = (wet * y48[:n] + (1.0 - wet) * xin_48k[:n]).astype(np.float32)

        # post: pico −1 dBFS
        y48 = peak_norm(y48)

        # devolver al SR original
        y = _resample(y48, self.TARGET_SR, sr)

        info = {"note": f"df.enhance cli, wet={wet}, device={self.device}"}
        return y.astype(np.float32), info
