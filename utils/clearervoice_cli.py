#!/usr/bin/env python3
# Usage: python scripts/clearervoice_cli.py --input in.wav --output out.wav --preset medium
import argparse, sys, soundfile as sf, numpy as np

def _try_import_api():
    # Try common ClearVoice entrypoints
    try:
        from clearvoice import enhancer as api  # e.g., api.enhance(y, sr, model=..., device=...)
        return ("module_attr", api)
    except Exception:
        pass
    try:
        from clearvoice.enhancer import enhance as fn  # e.g., fn(y, sr, model=..., device=...)
        return ("callable", fn)
    except Exception:
        pass
    return (None, None)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--preset", default="medium")
    p.add_argument("--device", default="cpu")  # evita conflictos CUDA
    args = p.parse_args()

    mode, api = _try_import_api()
    if api is None:
        sys.stderr.write(
            "[clearervoice_cli] No ClearVoice Python API found. "
            "Ensure the 'clearvoice' package exposes an enhancer function.\n"
        )
        sys.exit(2)

    y, sr = sf.read(args.input, always_2d=False)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    # Map preset->model name (ajusta si tu paquete usa otros nombres)
    model_by_preset = {
        "fast":  "MossFormer2_SE_16K",
        "medium":"MossFormer2_SE_48K",
        "high":  "MossFormer2_SE_48K",
    }
    model = model_by_preset.get(args.preset, "MossFormer2_SE_48K")

    if mode == "module_attr":
        # api.enhance(...)
        out, sr_out = api.enhance(y, sr, model=model, device=args.device)
    else:
        # callable style
        out, sr_out = api(y, sr, model=model, device=args.device)

    sf.write(args.output, out, sr_out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
