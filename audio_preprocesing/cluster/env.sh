#!/usr/bin/env bash
set -euo pipefail

# 1) Activar venv local del repo
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
else
  echo "[env] .venv no encontrado en $(pwd)"; exit 1
fi

# 2) Cargar .env
# Cargar .env correctamente (respeta comillas y espacios)
if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

# 3) Ajustes runtime
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export ENH_DEVICE="${ENH_DEVICE:-cpu}"
export AUDIO_ROOT="${AUDIO_ROOT:-dataset-audio-raw}"

# 4) Diagnóstico mínimo
python -V
which python
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
