import time, numpy as np
from typing import Dict, Any, Tuple
from .base import BackendBase

class ResembleEnhStub(BackendBase):
    name = "resemble_enhance_stub"

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        y = x.copy()
        k = {"light": 0.005, "medium": 0.01, "aggressive": 0.02}.get(preset, 0.01)
        alpha = 1.0 - k
        for i in range(1, len(y)):
            y[i] = alpha * y[i-1] + (1-alpha) * y[i]
        return y.astype(np.float32), {"backend": self.name, "preset": preset, "note": "stub-lowpass", "proc_sec": time.time()-t0}
