# enh/backends/clearervoice.py
import os, subprocess, shlex, tempfile, pathlib, soundfile as sf, numpy as np
from .base import BackendBase
from utils.audio_prep import peak_normalize_minus1_dbfs as peak_norm

class ClearerVoiceCLI(BackendBase):
    """Wrapper CLI de ClearerVoice. Configura ENH_CLEARERVOICE_CMD con {in},{out},{preset}."""
    def __init__(self, device=None, preset=None):
        dev = os.getenv("ENH_DEVICE", "cpu")
        super().__init__(device=dev)
        self.NAME = "clearervoice"
        self.preset = preset or "medium"
        self.cmd_tpl = os.getenv("ENH_CLEARERVOICE_CMD")
        if not self.cmd_tpl:
            raise RuntimeError("ENH_CLEARERVOICE_CMD no definido")

    def process(self, x: np.ndarray, sr: int):
        with tempfile.TemporaryDirectory() as td:
            inp = pathlib.Path(td) / "in.wav"
            outp = pathlib.Path(td) / "out.wav"
            sf.write(inp, x.astype(np.float32), sr, subtype="PCM_16")
            cmd = self.cmd_tpl.format(inp=str(inp), outp=str(outp), preset=self.preset)
            subprocess.run(shlex.split(cmd), check=True)
            y, sr_out = sf.read(outp, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1).astype(np.float32)
        y = peak_norm(y)
        return y.astype(np.float32), sr_out

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: dict | None = None):
        self.preset = preset or self.preset
        y, _ = self.process(x, sr)
        return y, {"note": "clearervoice-cli"}
