import numpy as np

def peak_normalize_minus1_dbfs(x: np.ndarray) -> np.ndarray:
    """
    x float32 en [-1, 1]. Devuelve x escalada para que el pico quede en -1 dBFS (~0.89125).
    """
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak <= 0:
        return x
    target = 10 ** (-1/20)  # 0.89125
    s = target / peak if peak > target else 1.0  # no sube si ya está por debajo
    return (x * s).astype(np.float32)

def trim_tails_vad(x: np.ndarray, sr: int,
                   frame_ms: int = 30, thresh_db: float = -45.0,
                   min_sil_ms: int = 500, margin_ms: int = 150) -> np.ndarray:
    """
    VAD simple por energía SOLO en colas. No recorta interior.
    - thresh_db: umbral de energía (dBFS) para considerar silencio.
    - min_sil_ms: silencio continuo mínimo para recortar.
    - margin_ms: colchón que se conserva tras el recorte.
    """
    if x.size == 0:
        return x
    # energía por frame
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0: frame_len = 1
    n_frames = max(1, len(x) // frame_len)
    frames = x[:n_frames*frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    dbfs = 20*np.log10(rms + 1e-12)  # aprox. a dBFS si x está normalizado [-1,1]

    # silencio si dbfs < umbral
    sil = dbfs < thresh_db
    run = lambda arr: np.flatnonzero(np.convolve(arr, np.ones(int(min_sil_ms/frame_ms), dtype=int), 'same')
                                     >= int(min_sil_ms/frame_ms))

    # inicio: buscar tramo silencioso continuo desde frame 0
    start_idx = 0
    if sil[0]:
        # primera zona silenciosa
        k = 0
        while k < n_frames and sil[k]:
            k += 1
        if (k * frame_ms) >= min_sil_ms:
            start_idx = int(max(0, k*frame_ms*sr/1000 - margin_ms*sr/1000))

    # fin: buscar tramo silencioso continuo desde el final
    end_idx = len(x)
    if sil[-1]:
        k = 0
        j = n_frames - 1
        while j >= 0 and sil[j]:
            k += 1; j -= 1
        if (k * frame_ms) >= min_sil_ms:
            end_idx = int(min(len(x), len(x) - (k*frame_ms*sr/1000 - margin_ms*sr/1000)))

    if start_idx >= end_idx:
        return x  # no recortar si desborda
    return x[start_idx:end_idx].astype(np.float32)
