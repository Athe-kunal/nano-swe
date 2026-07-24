"""Smoke test: runs one Harbor task end-to-end in a real Daytona sandbox.

Pipeline: spin up a Daytona sandbox, clone the task's repo at its base commit,
run an OpenHands agent (backed by a Fireworks model) against the task's
instruction, then grade the result with the task's own tests/grade.py.

This is a one-off harness smoke test, not the training loop (see nano_swe/agents
for the RL rollout path, which runs the same episode against the training-time
vLLM policy instead of Fireworks).
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from nano_swe.harness.episode import run_episode

DEFAULT_MODEL = "openai/accounts/fireworks/models/gpt-oss-120b"
DEFAULT_TASK_DIR = Path(__file__).parent.parent / "swe_data" / "data" / "lite" / "getmoto__moto-5752"


def main() -> None:
    """Parses CLI args and runs the dummy episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    load_dotenv()
    run_episode(
        args.task_dir,
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.model,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
