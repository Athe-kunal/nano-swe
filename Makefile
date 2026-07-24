.PHONY: download-dataset download-dataset-full download-verified-eval run-eval

download-dataset:
	uv run python -m nano_swe.swe_data.build_dataset --dataset lite

download-dataset-full:
	uv run python -m nano_swe.swe_data.build_dataset --dataset full

download-verified-eval:
	uv run python -m nano_swe.swe_data.build_verified_eval

# run_eval.py loads OPENAI_API_KEY / OPENAI_BASE_URL / WANDB_API_KEY from .env itself;
# override any of these on the command line, e.g. `make run-eval EVAL_MODEL=openai/gpt-5`.
EVAL_TASK_DIR ?= nano_swe/swe_data/data/verified_eval_50
EVAL_MODEL ?= openai/accounts/fireworks/models/gpt-oss-120b
EVAL_K ?= 4
EVAL_WANDB_RUN_NAME ?= eval-$(shell date +%Y%m%d-%H%M%S)

run-eval:
	uv run python -m nano_swe.harness.run_eval \
		--task-dir $(EVAL_TASK_DIR) \
		--model $(EVAL_MODEL) \
		-k $(EVAL_K) \
		--wandb-run-name $(EVAL_WANDB_RUN_NAME)
