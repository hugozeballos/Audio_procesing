# enh/backends/base.py
from __future__ import annotations
import numpy as np

class BackendBase:
    """Interfaz común para todos los backends de enhacement."""
    name = "base"

    def __init__(self, device: str = "cpu"):
        self.device = device  # "cpu", "cuda:0", etc.

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: dict) -> tuple[np.ndarray, dict]:
        """Wrapper estable: llama a process() y empaqueta metadatos mínimos."""
        y, sr_out = self.process(x, sr, preset, opts)
        return y.astype(np.float32), {"note": preset, "device": self.device}

    def process(self, x: np.ndarray, sr: int, preset: str, opts: dict) -> tuple[np.ndarray, int]:
        """Debe implementarse en cada backend concreto."""
        raise NotImplementedError
