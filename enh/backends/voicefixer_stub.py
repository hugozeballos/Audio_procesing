import time, numpy as np
from typing import Dict, Any, Tuple
from .base import BackendBase

class VoiceFixerStub(BackendBase):
    name = "voicefixer_stub"

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        y = x.copy()
        thr_db = {"light": -50.0, "medium": -45.0, "aggressive": -40.0}.get(preset, -45.0)
        amp_thr = 10 ** (thr_db/20.0)
        y[np.abs(y) < amp_thr] = 0.0
        return y.astype(np.float32), {"backend": self.name, "preset": preset, "note": "stub-gate", "proc_sec": time.time()-t0}
