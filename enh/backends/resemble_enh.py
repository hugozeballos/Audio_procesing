# enh/backends/resemble_enh.py
import os
import numpy as np
import torch
from dataclasses import dataclass
from scipy.signal import resample_poly
from resemble_enhance.enhancer.inference import denoise as r_denoise, enhance as r_enhance
import sys, logging


log = logging.getLogger("resemble_enh")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32)

def _resample_unsafe(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0: return x
    g = np.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    return resample_poly(x, up, down).astype(np.float32)

def _peak_norm_minus1_dbfs(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    peak = float(np.max(np.abs(x)))
    target = 10 ** (-1/20)  # ~0.8912509
    if peak <= 0: return x
    s = min(1.0, target / peak)
    return (x * s).astype(np.float32)

@dataclass
class _Cfg:
    solver: str
    nfe: int
    tau: float
    denoise_first: bool
    post_gain_db: float

class ResembleEnh:
    """
    Backend real para Resemble-Enhance.
    Presets: light | medium | aggressive
    """
    name = "resemble_enh"
    TARGET_SR = 44100

    def __init__(self, device: str | None = None, preset: str = "medium"):
        self.device = device or os.getenv("ENH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = bool(int(os.getenv("ENH_DEBUG", "0")))
        preset = preset.lower()
        self.cfgs = {
            "light": _Cfg(solver="Midpoint", nfe=32,  tau=0.45, denoise_first=False, post_gain_db=0.0),
            "medium": _Cfg(solver="Midpoint", nfe=64,  tau=0.50, denoise_first=False, post_gain_db=0.0),
            "aggressive": _Cfg(solver="Midpoint", nfe=96,  tau=0.55, denoise_first=True,  post_gain_db=0.0),
        }
        self.cfg = self.cfgs.get(preset, self.cfgs["medium"])

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: dict | None = None):
        # permite override rápido del preset
        if preset and preset.lower() in self.cfgs:
            self.cfg = self.cfgs[preset.lower()]
        y, sr_out = self.process(x, sr)
        note = {
            "solver": self.cfg.solver,
            "nfe": self.cfg.nfe,
            "tau": self.cfg.tau,
            "denoise_first": self.cfg.denoise_first,
            "device": self.device,
        }
        return y, note

    def process(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if self.debug:
             log.info(f"[res] start sr={sr} len_in={len(x)} device={self.device} preset={self.cfg}")

        x = _to_mono(x)
        xin = _resample_unsafe(x, sr, self.TARGET_SR)

        # IMPORTANTE: mantener entrada en CPU como NumPy. La lib gestiona el device internamente.
        wav = xin  # np.float32, 1D
        use_device = self.device if (self.device and self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        lambd = 0.9 if self.cfg.denoise_first else 0.1

        try:
            with torch.inference_mode():
                if self.cfg.denoise_first:
                    dwav, _ = r_denoise(wav, self.TARGET_SR, use_device)
                    dwav = dwav.squeeze().to(use_device)
                    enh, _ = r_enhance(dwav, self.TARGET_SR, use_device, solver=self.cfg.solver.lower(),
                                    nfe=self.cfg.nfe,
                                    lambd=lambd, tau=self.cfg.tau)
                else:
                    enh, _ = r_enhance(wav, self.TARGET_SR, use_device, solver=self.cfg.solver.lower(),
                                    nfe=self.cfg.nfe,
                                    lambd=lambd, tau=self.cfg.tau)
                    
        except Exception as e:
            log.error(f"[res] fail: {type(e).__name__}: {e} | sr_in={sr} len_in={len(x)} device={use_device} preset={self.cfg}",
                    exc_info=self.debug)

        y = np.asarray(enh, dtype=np.float32).reshape(-1)
        if self.cfg.post_gain_db != 0.0:
            g = 10 ** (self.cfg.post_gain_db / 20.0)
            y = (y * g).astype(np.float32)

        y = _peak_norm_minus1_dbfs(y)
        y = _resample_unsafe(y, self.TARGET_SR, sr)
        if self.debug:
            log.info(f"[res] done len_out={len(y)} target_sr={self.TARGET_SR}")
        return y.astype(np.float32), sr
