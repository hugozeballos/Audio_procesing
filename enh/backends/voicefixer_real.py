# enh/backends/voicefixer_real.py
import os, io, tempfile, numpy as np, soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly
from utils.audio_prep import peak_normalize_minus1_dbfs as peak_norm
from enh.backends.base import BackendBase

class VoiceFixerReal(BackendBase):
    """
    Wrapper de VoiceFixer (https://github.com/haoheliu/voicefixer).
    Usa la API por archivos: escribe wav temporal, llama restore, lee salida.
    """
    NAME = "voicefixer"
    TARGET_SR = 22050  # VoiceFixer trabaja a 22.05 kHz mono

    def __init__(self):
        super().__init__()
        # Carga perezosa en enhance() para evitar inicializar modelo si no se usa.
        self._vf = None

    def _lazy_load(self):
        if self._vf is None:
            from voicefixer import VoiceFixer
            self._vf = VoiceFixer()

    def _to_mono(self, x: np.ndarray) -> np.ndarray:
        return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32)

    def _resample(self, x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out or x.size == 0:
            return x.astype(np.float32)
        # resample_poly es estable y rápido
        g = np.gcd(sr_in, sr_out)
        up, down = sr_out // g, sr_in // g
        return resample_poly(x, up, down).astype(np.float32)

    def process(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        # pre
        x = self._to_mono(x)
        xin = self._resample(x, sr, self.TARGET_SR)

        # IO temporario
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.wav"
            outp = Path(td) / "out.wav"
            sf.write(inp, xin, self.TARGET_SR, format="WAV", subtype="PCM_16")

            # modelo
            self._lazy_load()
            use_cuda = os.getenv("ENH_DEVICE", "cpu").lower().startswith("cuda")
            # mode=0: full restoration; mode=1: denoise only (ajústalo si quieres presets distintos)
            self._vf.restore(input=str(inp), output=str(outp), cuda=use_cuda, mode=0)

            y, sr_y = sf.read(outp, dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1).astype(np.float32)

        # post
        y = peak_norm(y)
        y = self._resample(y, sr_y, sr)  # devuelvo en SR original
        return y.astype(np.float32), sr

    def enhance(self, x: np.ndarray, sr: int, preset: str, opts: dict):
        """
        preset:
          - 'light'     -> mode=1 (denoise only)
          - 'medium'    -> mode=0 (full restoration)
          - 'aggressive'-> mode=0 + post-gain -1 dBFS ya aplicado por peak_norm
        """
        # map de preset -> modo
        preset = (preset or "medium").lower()
        mode = 0 if preset in ("medium", "aggressive") else 1

        # Ejecuta cambiando transientemente el modo
        # VoiceFixer no expone setter simple por instancia, así que usamos archivos con el modo elegido
        # Implemento cambiando temporalmente vía llamada directa:
        self._lazy_load()
        use_cuda = os.getenv("ENH_DEVICE", "cpu").lower().startswith("cuda")

        # Hago la misma ruta de archivos para respetar API
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.wav"
            outp = Path(td) / "out.wav"

            # pre iguales que process() para mantener consistencia
            x_mono = self._to_mono(x)
            xin = self._resample(x_mono, sr, self.TARGET_SR)
            sf.write(inp, xin, self.TARGET_SR, format="WAV", subtype="PCM_16")

            # correr VF con el modo mapeado por preset
            self._vf.restore(input=str(inp), output=str(outp), cuda=use_cuda, mode=mode)

            y, sr_y = sf.read(outp, dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1).astype(np.float32)

        y = peak_norm(y)
        y = self._resample(y, sr_y, sr)
        info = {"note": f"voicefixer-mode={mode}", "sr_model": self.TARGET_SR}
        return y.astype(np.float32), info
