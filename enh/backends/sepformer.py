# enh/backends/sepformer.py
import os
import numpy as np
from scipy.signal import resample_poly
import torch
from speechbrain.inference import SpectralMaskEnhancement
from utils.audio_prep import peak_normalize_minus1_dbfs as peak_norm
from enh.backends.base import BackendBase

class SepformerEnh(BackendBase):
    """
    Speech enhancement con SepFormer (SpeechBrain).
    Repo HF: speechbrain/sepformer-wham16k-enhancement
    Trabaja a 16 kHz mono.
    """
    NAME = "sepformer"
    TARGET_SR = 16000

    def __init__(self):
        super().__init__()
        device = os.getenv("ENH_DEVICE", "cpu")
        run_opts = {"device": device}
        # descarga/carga del checkpoint
        self.enh = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/sepformer-wham16k-enhancement",
            savedir="~/.cache/speechbrain/sepformer_wham16k",
            run_opts=run_opts
        )

    @staticmethod
    def _to_mono(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32)

    @staticmethod
    def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out or x.size == 0:
            return x.astype(np.float32)
        g = np.gcd(sr_in, sr_out)
        up, down = sr_out // g, sr_in // g
        return resample_poly(x, up, down).astype(np.float32)

    def process(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        # pre
        x = self._to_mono(x)
        xin = self._resample(x, sr, self.TARGET_SR)

        # inferencia (requiere lengths)
        with torch.no_grad():
            wav = torch.from_numpy(xin).unsqueeze(0)  # [1, T]
            lengths = torch.tensor([1.0], dtype=torch.float32, device=wav.device)
            y = self.enh.enhance_batch(wav, lengths=lengths)  # [1, T]
            y = y.squeeze(0).cpu().numpy().astype(np.float32)

        # post
        y = peak_norm(y)
        y = self._resample(y, self.TARGET_SR, sr)
        return y.astype(np.float32), sr

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: dict):
        """
        preset: 'light'|'medium'|'aggressive' (mismo modelo; presets se usan
        solo para logging/consistencia).
        """
        y, sr_out = self.process(x, sr)
        info = {"note": "speechbrain-sepformer", "sr_model": self.TARGET_SR}
        return y, info
