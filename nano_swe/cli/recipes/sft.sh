#!/bin/bash
# Launches the SFT recipe. Override any recipe key with extra flags, e.g.:
#   MODEL_PATH=/path/to/model bash nano_swe/cli/recipes/sft.sh --train.max_epochs 3
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the base checkpoint to fine-tune}"
: "${DATASET:?Set DATASET to the SFT dataset path (see sft.yaml)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m nano_swe.cli.train_sft \
    --config "$DIR/sft.yaml" \
    --model.model_name_or_path "$MODEL_PATH" \
    --data.dataset "$DATASET" \
    "$@"
