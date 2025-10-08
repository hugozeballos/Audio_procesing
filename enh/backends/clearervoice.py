# enh/backends/clearervoice.py
import os, tempfile, pathlib, subprocess, shlex
import numpy as np
import soundfile as sf
from .base import BackendBase

CMD = os.getenv("ENH_CLEARERVOICE_CMD")  # ej: 'clearervoice --preset {preset} -i {in} -o {out}'
if not CMD:
    raise RuntimeError("ENH_CLEARERVOICE_CMD no definido (modo CLI)")

# opcional: mejor resample
try:
    import librosa
    def _resample(x, sr_in, sr_out):
        return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out, res_type="kaiser_fast")
except Exception:
    librosa = None
    def _resample(x, sr_in, sr_out):
        if sr_in == sr_out: return x
        # fallback lineal simple
        n_out = int(round(len(x) * sr_out / sr_in))
        idx = np.linspace(0, len(x) - 1, n_out)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

# mapeo de presets a modelos y SR
PRESET2MODEL = {
    "fast":   ("FRCRN_SE_16K", 16000),
    "medium": ("MossFormer2_SE_48K", 48000),
    "hq":     ("MossFormer2_SE_48K", 48000),
}

class ClearerVoiceBackend(BackendBase):
    """Usa ModelScope ClearerVoice vía API Python, sin CLI."""
    name = "clearervoice"       # <-- agrega aquí, justo debajo de la docstring

    def __init__(self, device=None, preset=None):
        dev = os.getenv("ENH_DEVICE", "cuda" if device in ("cuda","gpu") else "cpu")
        super().__init__(device=dev)
        self.NAME = "clearervoice"
        self.preset = (preset or os.getenv("ENH_PRESET") or "medium").lower()

        model_name, target_sr = PRESET2MODEL.get(self.preset, PRESET2MODEL["medium"])
        self._target_sr = target_sr
        self.cmd_tpl = CMD
        self.mode = "cli"


    def process(self, x: np.ndarray, sr: int):
        mono = False
        if x.ndim == 2:  # [T, C] o [C, T]
            # normaliza a [T], usa mezcla a mono
            if x.shape[0] < x.shape[1]:
                x = x.T
            x = x.mean(axis=1)
            mono = True

        # resample si es necesario
        if sr != self._target_sr:
            x = _resample(x.astype(np.float32), sr, self._target_sr)
            sr_in = self._target_sr
        else:
            sr_in = sr

        # ClearVoice (vía paths, robusto sin ffmpeg si usamos WAV)
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            inp = td / "in.wav"
            outp = td / "out.wav"
            sf.write(inp, x.astype(np.float32), sr_in)
            # infer vía CLI
            cmd = self.cmd_tpl.format(**{'in': str(inp), 'out': str(outp), 'preset': self.preset})
            subprocess.run(shlex.split(cmd), check=True)
            y_ret, sr_ret = sf.read(outp, always_2d=False)

        # mantiene mono. si quieres devolver estéreo, duplica canales aquí.
        return y_ret.astype(np.float32), sr_ret

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts=None):
        # permitir override de preset en tiempo de ejecución
        if preset and preset.lower() != self.preset:
            model_name, target_sr = PRESET2MODEL.get(preset.lower(), PRESET2MODEL["medium"])
            self.preset = preset.lower()
            self._target_sr = target_sr
        y, sr_out = self.process(x, sr)
        info = {"preset": preset or self.preset, "sr_out": sr_out, "backend": self.NAME}
        return y, info
