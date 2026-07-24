"""Builds a repo-stratified 50-instance sample of SWE-bench Verified for evaluation.

SWE-bench Verified has 500 human-validated instances spread unevenly across ~12 repos (django
alone is roughly a quarter of it). A plain random 50 would inherit that skew, so this instead
round-robins across repos — shuffling each repo's own instances first — so no single repo
dominates the eval set. Each pick is converted to a Harbor task with the same writer
build_dataset.py uses for SWE-Gym.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from nano_swe.swe_data.build_dataset import write_task

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 42


def stratified_sample_by_repo(instances: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Round-robins across repos (each repo's instances pre-shuffled) so no repo dominates."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for instance in instances:
        by_repo[instance["repo"]].append(instance)

    rng = random.Random(seed)
    for group in by_repo.values():
        rng.shuffle(group)
    repos = list(by_repo)
    rng.shuffle(repos)

    sample = []
    while len(sample) < sample_size and any(by_repo.values()):
        for repo in repos:
            if by_repo[repo]:
                sample.append(by_repo[repo].pop())
                if len(sample) == sample_size:
                    break
    return sample


def build(output_dir: Path, sample_size: int, seed: int) -> None:
    dataset = load_dataset(DATASET_NAME, split="test")
    instances = [dict(row) for row in dataset]
    for instance in instances:
        instance["FAIL_TO_PASS"] = json.loads(instance["FAIL_TO_PASS"])
        instance["PASS_TO_PASS"] = json.loads(instance["PASS_TO_PASS"])

    sample = stratified_sample_by_repo(instances, sample_size, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for instance in sample:
        write_task(instance, DATASET_NAME, output_dir / instance["instance_id"])

    repo_counts: dict[str, int] = defaultdict(int)
    for instance in sample:
        repo_counts[instance["repo"]] += 1
    print(f"Wrote {len(sample)} tasks from {DATASET_NAME} to {output_dir}")
    for repo, count in sorted(repo_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {repo}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "data" / "verified_eval_20"
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    build(args.output_dir, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
