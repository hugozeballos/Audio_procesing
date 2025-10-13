#!/usr/bin/env bash
#SBATCH -J enh_metrics
#SBATCH --partition=ialab-eph
#SBATCH --gres=gpu:2080_super:1
#SBATCH -t 23:59:00
#SBATCH --mem=16G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
elw-WÑE-rt,mhtnbbn}1+
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"


# vars del cluster si las tienes
[[ -f cluster/env.sh ]] && source cluster/env.sh


DATASET_DIR="dataset-audio-raw"
METRICS_OUT="artifacts/metrics.csv"

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
  --csv-summary "$(dirname "$METRICS_OUT")/$(basename "$METRICS_OUT" .csv)_summary.csv" \
  --vad-backend webrtc \
  --vad-resample 48k 

deactivate
echo "[done] $METRICS_OUT"
