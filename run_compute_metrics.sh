#!/usr/bin/env bash
#SBATCH -J enh_metrics
#SBATCH --partition=ialab-eph
#SBATCH --gres=gpu:0
#SBATCH -t 01:00:00
#SBATCH --mem=8G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"


# vars del cluster si las tienes
[[ -f cluster/env.sh ]] && source cluster/env.sh

# leer config/experiment_config.json
CFG="/config/experiment_config.json"
json_get () {
python - "$CFG" "$1" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
print(cfg.get(sys.argv[2], ""))
PY
}

DATASET_DIR="$(json_get dataset_dir)"
METRICS_OUT="$(json_get metrics_out)"

mkdir -p "$(dirname "$METRICS_OUT")" logs

# venv de métricas (por defecto .venv-metrics)
METRICS_VENV="${METRICS_VENV:-.venv-metrics}"
[[ -f "$METRICS_VENV/bin/activate" ]] || { echo "venv no encontrado: $METRICS_VENV"; exit 1; }
# shellcheck disable=SC1090
source "$METRICS_VENV/bin/activate"

echo "[metrics] dataset_dir=$DATASET_DIR"
echo "[metrics] csv_out=$METRICS_OUT"

python enh/compute_metrics.py \
  --dataset-dir "$DATASET_DIR" \
  --csv-out "$METRICS_OUT" \
  --csv-summary "$(dirname "$METRICS_OUT")/$(basename "$METRICS_OUT" .csv)_summary.csv"

deactivate
echo "[done] $METRICS_OUT"
