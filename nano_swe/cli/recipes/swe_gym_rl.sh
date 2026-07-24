#!/bin/bash
# Launches the SWE-Gym RL recipe. Override any recipe key with extra flags, e.g.:
#   MODEL_PATH=/path/to/model bash nano_swe/cli/recipes/swe_gym_rl.sh --rollout.n_samples_per_prompt 8
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the base checkpoint to train}"
: "${PROMPT_DATASET:?Set PROMPT_DATASET to the prompt manifest (see swe_gym_rl.yaml)}"

python3 -m nano_swe.cli.train_rl_ray \
    --config "$DIR/swe_gym_rl.yaml" \
    --actor.model_name_or_path "$MODEL_PATH" \
    --data.prompt_dataset "$PROMPT_DATASET" \
    "$@"
