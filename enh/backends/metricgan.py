# enh/backends/metricgan.py
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly
import torch
from .base import BackendBase
from utils.audio_prep import peak_normalize_minus1_dbfs as peak_norm

from speechbrain.inference import SpectralMaskEnhancement  # was: speechbrain.pretrained


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1).astype(np.float32)

def _resample_unsafe(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out: return x
    # polyphase: mejor que naive
    g = np.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    y = resample_poly(x, up, down).astype(np.float32)
    return y

class MetricGANPlus(BackendBase):
    NAME = "metricgan"
    PRESETS = ["light", "medium", "aggressive"]
    TARGET_SR = 16000
    NEEDS_MONO = True
    CHUNK_SEC = None
    HOP_SEC = None

    def __init__(self, preset: str = "medium", device: str | None = None):
        self.name = "metricgan"
        super().__init__(preset)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # modelo preentrenado VoiceBank
        self.enh = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            run_opts={"device": self.device}
        )
        # presets = solo post-gain y normalización final
        self.cfg = {
            "light":      {"wet": 0.70, "post_gain_db": 0.0, "final_peak_dbfs": -1.0},
            "medium":     {"wet": 0.85, "post_gain_db": 1.5, "final_peak_dbfs": -1.0},
            "aggressive": {"wet": 1.00, "post_gain_db": 3.0, "final_peak_dbfs": -1.0},
        }[preset]

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: dict):
        """
        Adaptador para el contrato común: devuelve (audio_enh, info_dict).
        Reusa tu método existente 'process'.
        """
        y, sr_out = self.process(x, sr)   # ya lo tienes implementado
        info = {"note": f"metricgan:{preset}", "sr_out": sr_out}
        return y, info

    def process(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        x = _to_mono(x).astype(np.float32)
        xin = _resample_unsafe(x, sr, self.TARGET_SR)

        with torch.no_grad():
            wav = torch.from_numpy(xin).unsqueeze(0)   # [1, T]
            y = self.enh.enhance_batch(wav, lengths=torch.tensor([1.0], dtype=torch.float32))
            y = y.squeeze(0).cpu().numpy().astype(np.float32)

        # 1) wet/dry
        a = float(self.cfg["wet"])                     # 0..1
        y = (a * y + (1.0 - a) * xin).astype(np.float32)

        # 2) post-gain (dB)
        g = 10.0 ** (float(self.cfg["post_gain_db"]) / 20.0)
        if g != 1.0:
            y = (y * g).astype(np.float32)

        # 3) normalizar pico a final_peak_dbfs
        target = 10.0 ** (float(self.cfg["final_peak_dbfs"]) / 20.0)  # p.ej. -1 dBFS -> ~0.89125
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.0:
            s = target / peak if peak > target else 1.0               # no eleva si ya está por debajo
            y = (y * s).astype(np.float32)

        # 4) devolver al SR original
        y = _resample_unsafe(y, self.TARGET_SR, sr)
        return y.astype(np.float32), sr