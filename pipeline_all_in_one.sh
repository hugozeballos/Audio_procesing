#!/usr/bin/env bash
#SBATCH -J enh_benchmark
#SBATCH --partition=ialab-low        # Se tiene que elegir una partición de nodos con GPU
#SBATCH -p gpu                 # cámbialo a 'cpu' si no usas GPU
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH -t 04:00:00
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


# Uso:
#   bash pipeline_all_in_one.sh               # benchmark sin ingesta
#   INGEST=1 bash pipeline_all_in_one.sh      # con ingesta Drive→HF

set -euo pipefail

# 0) Entrar a raíz del repo
cd "$(dirname "$0")"
mkdir -p logs artifacts

# 1) Entorno
source cluster/env.sh

# Parámetro opcional
INGEST="${INGEST:-0}"

CFG="config/experiment_config.json"
DATASET_DIR=$(jq -r '.dataset_dir' "$CFG")
DEV_SIZE=$(jq -r '.dev_size' "$CFG")
TEST_SIZE=$(jq -r '.test_size' "$CFG")
PREP_DIR=$(jq -r '.prepared_dev_dir' "$CFG")
ENH_DIR=$(jq -r '.enh_out_dir' "$CFG")
METRICS_OUT=$(jq -r '.metrics_out' "$CFG")
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

# 2) Ingesta opcional Drive→HF
if [[ "$INGEST" == "1" ]]; then
  echo "[1] ingest_drive_to_hf.py"
  : "${GOOGLE_APPLICATION_CREDENTIALS:?falta GOOGLE_APPLICATION_CREDENTIALS}"
  : "${DRIVE_FOLDER_ID:?falta DRIVE_FOLDER_ID}"
  : "${HF_TOKEN:?falta HF_TOKEN}"
  : "${HF_REPO_ID:?falta HF_REPO_ID}"
  python tools/ingest_drive_to_hf.py
fi

# 3) Splits (idempotente)
echo "[2] gen_splits"
python preprocesing/gen_splits.py \
  --dataset-dir "$DATASET_DIR" \
  --dev-size "$DEV_SIZE" \
  --test-size "$TEST_SIZE"

# 4) Prep dev
echo "[3] prep_dev"
python preprocesing/prep_dev.py --config "$CFG"

# 5) Enhancement por backend
echo "[4] enhancement"
for LINE in "${BACKENDS[@]}"; do
  IFS=':' read -r B P <<<"$LINE"
  OUT="$ENH_DIR/${B}_${P}"
  mkdir -p "$OUT"
  echo "  -> $B:$P  out=$OUT  device=${ENH_DEVICE:-cpu}"
  ENH_DEVICE="${ENH_DEVICE:-cpu}" python enh/run_enh_dev.py \
    --config "$CFG" \
    --backend "$B" \
    --preset "$P" \
    --out "$OUT"
done

# 6) Métricas
echo "[5] metrics"
python enh/compute_metrics.py \
  --input-dir "$ENH_DIR" \
  --out "$METRICS_OUT"

echo "[done] Resultado: $METRICS_OUT"
