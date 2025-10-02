#!/usr/bin/env bash
#SBATCH -J enh_benchmark
#SBATCH --partition=ialab-high
#SBATCH --gres=gpu:1080_ti:2
#SBATCH -t 04:00:00
#SBATCH --mem=16G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Uso:
#   bash pipeline_all_in_one.sh
#   INGEST=1 bash pipeline_all_in_one.sh

set -euo pipefail
mkdir -p logs

# 0) Raíz del repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "cd fallo"; exit 1; }

# 1) Entorno de cluster (vars como GOOGLE_APPLICATION_CREDENTIALS, HF_TOKEN, etc.)
source cluster/env.sh

# Parámetro opcional
INGEST="${INGEST:-1}"

# --- util: leer JSON sin jq ---
CFG="config/experiment_config.json"
json_get () {
python - "$CFG" "$1" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
print(cfg.get(sys.argv[2], ""))
PY
}

DATASET_DIR="$(json_get dataset_dir)"
DEV_SIZE="$(json_get dev_size)"
TEST_SIZE="$(json_get test_size)"
PREP_DIR="$(json_get prepared_dev_dir)"
ENH_DIR="$(json_get enh_out_dir)"
METRICS_OUT="$(json_get metrics_out)"

mkdir -p "$PREP_DIR" "$ENH_DIR" "$(dirname "$METRICS_OUT")"

# Backends
if [[ -f backends.txt ]]; then
  mapfile -t BACKENDS < <(grep -v '^\s*$' backends.txt)
else
  echo "backends.txt no encontrado"; exit 1
fi

echo "[0] Config:"
echo "  dataset_dir=$DATASET_DIR"
echo "  dev_size=$DEV_SIZE test_size=$TEST_SIZE"
echo "  prep_dir=$PREP_DIR"
echo "  enh_dir=$ENH_DIR"
echo "  metrics_out=$METRICS_OUT"
echo "  backends=${#BACKENDS[@]}"
echo "  INGEST=$INGEST  ENH_DEVICE=${ENH_DEVICE:-cpu}"

# ---- helpers de entornos ----
ENH_VENV="${ENH_VENV:-.venv}"              # tu venv actual (enhancement)
METRICS_VENV="${METRICS_VENV:-.venv-metrics}"  # nuevo venv (métricas)

use_venv() {
  local v="$1"
  [[ -f "$v/bin/activate" ]] || { echo "ERROR: venv no encontrado: $v"; exit 1; }
  # shellcheck disable=SC1090
  source "$v/bin/activate"
}

# 2) Ingesta opcional Drive→HF (usa .venv)
use_venv "$ENH_VENV"
if [[ "$INGEST" == "1" ]]; then
  echo "[1] ingest_drive_to_hf.py"
  : "${GOOGLE_APPLICATION_CREDENTIALS:?falta GOOGLE_APPLICATION_CREDENTIALS}"
  : "${DRIVE_FOLDER_ID:?falta DRIVE_FOLDER_ID}"
  : "${HF_TOKEN:?falta HF_TOKEN}"
  : "${HF_REPO_ID:?falta HF_REPO_ID}"
  python dataset-audio-raw/ingest_drive_to_hf.py
fi

# 3) Splits (idempotente)  [.venv]
echo "[2] gen_splits"
python preprocesing/gen_splits.py \
  --dataset-dir "$DATASET_DIR" \
  --dev-size "$DEV_SIZE" \
  --test-size "$TEST_SIZE"

# 4) Prep dev  [.venv]
echo "[3] prep_dev"
python preprocesing/prep_dev.py --config "$CFG"

# 5) Enhancement por backend  [.venv]
echo "[4] enhancement"
for LINE in "${BACKENDS[@]}"; do
  IFS=':' read -r B P <<<"$LINE"
  OUT="$ENH_DIR/${B}_${P}"
  mkdir -p "$OUT"
  echo "  -> $B:$P  out=$OUT  device=${ENH_DEVICE:-cpu}"
  ENH_BACKEND="$B" ENH_PRESET="$P" ENH_DEVICE="${ENH_DEVICE:-cpu}" \
  python enh/run_enh_dev.py
done
deactivate

# 6) Métricas (NISQA/DNSMOS)  [.venv-metrics]
echo "[5] metrics"
use_venv "$METRICS_VENV"
python enh/compute_metrics.py \
  --dataset-dir "$DATASET_DIR" \
  --out-csv "$METRICS_OUT" \
  --ref-source prep
deactivate

echo "[done] Resultado: $METRICS_OUT"
