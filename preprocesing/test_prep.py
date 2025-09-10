import numpy as np
from utils.audio_prep import peak_normalize_minus1_dbfs, trim_tails_vad

sr = 16000
# 0.8 s de silencio + 1 s de tono a -3 dBFS + 0.8 s de silencio
t = np.arange(int(0.8*sr + 1.0*sr + 0.8*sr)) / sr
x = np.zeros_like(t, dtype=np.float32)
tone = 10**(-3/20) * np.sin(2*np.pi*440*np.arange(int(1.0*sr))/sr)
x[int(0.8*sr):int(1.8*sr)] = tone

print("Antes:")
print("  dur_s        =", len(x)/sr)
print("  peak         =", float(np.max(np.abs(x))))

y = trim_tails_vad(x, sr, frame_ms=30, thresh_db=-45.0, min_sil_ms=500, margin_ms=150)
z = peak_normalize_minus1_dbfs(y)

print("Después VAD:")
print("  dur_s        =", len(y)/sr)
print("  peak         =", float(np.max(np.abs(y))))
print("Después Normalizar -1 dBFS:")
print("  peak         =", float(np.max(np.abs(z))))
print("  target_amp   ~", 10**(-1/20))