"""Evaluates an OpenAI-compatible model on a Harbor task set: pass@k / avg@k over k rollouts
per task, reported to Weights & Biases.

Reuses nano_swe.harness.episode.run_episode (the same Daytona + OpenHands + grading pipeline
validated in the dummy-episode harness) for each rollout, and
nano_swe.utils.logging_utils.WandbLogger for reporting — no new eval infra, just fan-out + a
pass@k/avg@k reduction over its results.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from nano_swe.harness.episode import run_episode
from nano_swe.utils.logging_utils import WandbLogger, init_logger

logger = init_logger(__name__)


def run_pass_at_k(task_dirs: list[Path], k: int, max_concurrency: int, **episode_kwargs) -> dict[str, list[float]]:
    """Runs k rollouts of every task, bounded by max_concurrency concurrent sandboxes.

    Returns {task_id: [reward, ...]} (one entry per rollout, in completion order).
    """
    rewards_by_task: dict[str, list[float]] = {task_dir.name: [] for task_dir in task_dirs}

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {
            pool.submit(run_episode, task_dir, **episode_kwargs): task_dir.name
            for task_dir in task_dirs
            for _ in range(k)
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                reward = future.result()["reward"]
            except Exception as e:
                logger.warning(f"{task_id}: rollout failed ({e})")
                reward = 0.0
            rewards_by_task[task_id].append(reward)
            logger.info(f"{task_id}: reward={reward} ({len(rewards_by_task[task_id])}/{k})")

    return rewards_by_task


def summarize(rewards_by_task: dict[str, list[float]]) -> dict[str, float]:
    """pass@k (fraction of tasks solved by at least one of k rollouts) and avg@k (mean per-task
    success rate, averaged over tasks), plus the raw rollout success rate for reference."""
    pass_at_k = [max(rewards) for rewards in rewards_by_task.values()]
    avg_at_k = [sum(rewards) / len(rewards) for rewards in rewards_by_task.values()]
    all_rewards = [r for rewards in rewards_by_task.values() for r in rewards]
    return {
        "pass@k": sum(pass_at_k) / len(pass_at_k),
        "avg@k": sum(avg_at_k) / len(avg_at_k),
        "mean_reward": sum(all_rewards) / len(all_rewards),
        "num_tasks": len(rewards_by_task),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True, help="Directory of Harbor task subdirectories.")
    parser.add_argument("--model", required=True, help="Model name to request, e.g. 'openai/gpt-5'.")
    parser.add_argument("--base-url", default=None, help="Defaults to $OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="Defaults to $OPENAI_API_KEY.")
    parser.add_argument("-k", type=int, default=4, help="Rollouts per task for pass@k/avg@k.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N tasks.")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=8, help="Concurrent Daytona sandboxes.")
    parser.add_argument("--wandb-project", default="nano-swe-eval")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-org", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args()

    load_dotenv()
    task_dirs = sorted(p.parent for p in args.task_dir.glob("*/task.toml"))
    if not task_dirs:
        raise ValueError(f"No Harbor tasks found under {args.task_dir}")
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]

    rewards_by_task = run_pass_at_k(
        task_dirs,
        k=args.k,
        max_concurrency=args.max_concurrency,
        base_url=args.base_url or os.environ["OPENAI_BASE_URL"],
        api_key=args.api_key or os.environ["OPENAI_API_KEY"],
        model=args.model,
        max_iterations=args.max_iterations,
    )
    metrics = summarize(rewards_by_task)
    logger.info(f"pass@{args.k}={metrics['pass@k']:.3f}  avg@{args.k}={metrics['avg@k']:.3f}  ({metrics})")

    results_path = args.task_dir / f"eval_results.{args.wandb_run_name}.json"
    results_path.write_text(json.dumps(rewards_by_task, indent=2))

    wandb_args = OmegaConf.create(
        {
            "logger": {
                "wandb": {
                    "key": os.environ.get("WANDB_API_KEY", ""),
                    "org": args.wandb_org,
                    "project": args.wandb_project,
                    "group": args.wandb_group,
                    "run_name": args.wandb_run_name,
                }
            }
        }
    )
    wandb_logger = WandbLogger(wandb_args)
    wandb_logger.log_eval(
        global_step=0,
        logs_dict={f"{key}@{args.k}" if key in ("pass", "avg") else key: value for key, value in metrics.items()},
    )
    wandb_logger.close()


if __name__ == "__main__":
    main()
