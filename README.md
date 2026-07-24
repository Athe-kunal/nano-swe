# nano-swe

Nano library for on-policy distillation / RL on SWE-bench-style coding tasks.

## Setup

```bash
uv sync
cp .env.example .env  # fill in OPENAI_API_KEY, OPENAI_BASE_URL, DAYTONA_API_KEY, WANDB_API_KEY
```

## Datasets

Pulls SWE-Gym / SWE-bench Verified and converts each instance into a Harbor-format task
directory (`task.toml`, `instruction.md`, `solution/`, `tests/`) under `nano_swe/swe_data/data/`.

```bash
make download-dataset          # SWE-Gym-Lite (230 instances)
make download-dataset-full     # full SWE-Gym (~2.4k instances)
make download-verified-eval    # 50-instance repo-stratified SWE-bench Verified sample
```

## Eval

Runs `k` OpenHands rollouts per task in a Daytona sandbox (booted straight from each task's
prebuilt Docker image) against an OpenAI-compatible model, reports pass@k / avg@k to W&B.

```bash
make run-eval-smoke   # one task, one rollout — quick pipeline check
make run-eval         # full eval; override any of these, e.g.:
make run-eval EVAL_MODEL=openai/gpt-5 EVAL_TASK_DIR=nano_swe/swe_data/data/lite EVAL_K=8 EVAL_NUM_WORKERS=16
```

## Training

Entrypoints live under `nano_swe/cli/`, ported from [Molt](https://github.com/NVIDIA-NeMo/labs-molt)
(Ray + vLLM + FSDP2). Each run is a YAML recipe (`nano_swe/cli/recipes/`) + a launch script;
any recipe key can still be overridden on the command line.

```bash
# RL on SWE-Gym via the OpenHands + Daytona rollout runner
MODEL_PATH=/path/to/model PROMPT_DATASET=/path/to/prompts.jsonl \
  bash nano_swe/cli/recipes/swe_gym_rl.sh

# SFT
MODEL_PATH=/path/to/model DATASET=/path/to/sft.jsonl \
  bash nano_swe/cli/recipes/sft.sh

# override a recipe value ad-hoc
MODEL_PATH=... PROMPT_DATASET=... bash nano_swe/cli/recipes/swe_gym_rl.sh --rollout.n_samples_per_prompt 8
```

See the comments at the top of each recipe YAML for what it sets and known gaps (e.g. the
prompt/SFT dataset loaders `nano_swe.datasets.*` aren't built yet).
