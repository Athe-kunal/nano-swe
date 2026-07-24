.PHONY: download-dataset download-dataset-full download-verified-eval

download-dataset:
	uv run python -m nano_swe.swe_data.build_dataset --dataset lite

download-dataset-full:
	uv run python -m nano_swe.swe_data.build_dataset --dataset full

download-verified-eval:
	uv run python -m nano_swe.swe_data.build_verified_eval
