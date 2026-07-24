"""Pulls SWE-Gym instances from Hugging Face and converts them into Harbor-format tasks.

Each instance becomes a task directory:
    <instance_id>/
        task.toml            # repo/version/commit metadata
        instruction.md        # agent-facing problem statement
        solution/patch.diff   # golden patch (oracle solution)
        tests/
            test_patch.diff    # patch that adds/updates the failing tests
            instance.json      # FAIL_TO_PASS / PASS_TO_PASS node ids
            grade.py           # applies test_patch.diff, runs pytest, writes reward.txt
            test.sh            # entrypoint the harness runs from the tests/ dir

Docker/environment setup is intentionally out of scope here (handled by the
Daytona sandbox the tasks are later run in); grade.py only assumes the repo is
checked out at $AGENT_WORKDIR (default /app) with its base commit and test
dependencies already installed.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

DATASETS = {
    "lite": "SWE-Gym/SWE-Gym-Lite",
    "full": "SWE-Gym/SWE-Gym",
}

INSTANCE_FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "problem_statement",
    "hints_text",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
]

TEST_SH = """#!/bin/bash
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -q pytest-json-report >/dev/null 2>&1
python3 "$DIR/grade.py"
"""

GRADE_PY = '''"""Applies the test patch, runs pytest on FAIL_TO_PASS/PASS_TO_PASS, writes reward."""

import json
import os
import subprocess
from pathlib import Path

TASK_DIR = Path(__file__).parent
REPO_DIR = Path(os.environ.get("AGENT_WORKDIR", "/app"))
LOG_DIR = Path("/logs/verifier")
LOG_DIR.mkdir(parents=True, exist_ok=True)

instance = json.loads((TASK_DIR / "instance.json").read_text())
node_ids = instance["FAIL_TO_PASS"] + instance["PASS_TO_PASS"]

subprocess.run(
    ["git", "apply", "-v", str(TASK_DIR / "test_patch.diff")], cwd=REPO_DIR, check=True
)

# A handful of SWE-Gym instances have malformed FAIL_TO_PASS/PASS_TO_PASS
# node ids (truncated parametrize ids). Passing an unresolvable id straight
# to pytest aborts the whole run before anything executes, so collect first
# and only ask pytest to run ids that actually exist.
test_files = sorted({node_id.split("::")[0] for node_id in node_ids})
collected = subprocess.run(
    ["python", "-m", "pytest", "--collect-only", "-q", *test_files],
    cwd=REPO_DIR,
    capture_output=True,
    text=True,
)
collected_ids = {line.strip() for line in collected.stdout.splitlines() if "::" in line}
valid_ids = [node_id for node_id in node_ids if node_id in collected_ids]

report_path = LOG_DIR / "report.json"
if valid_ids:
    subprocess.run(
        ["python", "-m", "pytest", "-q", "--json-report", f"--json-report-file={report_path}"]
        + valid_ids,
        cwd=REPO_DIR,
    )

report = json.loads(report_path.read_text()) if report_path.exists() else {"tests": []}
outcomes = {t["nodeid"]: t["outcome"] for t in report.get("tests", [])}
for node_id in node_ids:
    outcomes.setdefault(node_id, "not_collected")
# Ids that never resolved to a real test (a known SWE-Gym data quirk with
# truncated parametrize ids) can't be checked either way, so they're excluded
# from resolution rather than counted as failures.
resolved = bool(valid_ids) and all(outcomes[node_id] == "passed" for node_id in valid_ids)

(LOG_DIR / "reward.txt").write_text("1.0" if resolved else "0.0")
(LOG_DIR / "tests_status.json").write_text(json.dumps(outcomes, indent=2))
'''


def toml_string(value: str) -> str:
    """Renders a Python string as a quoted TOML basic string."""
    return json.dumps(value)


def write_task(instance: dict, dataset_name: str, task_dir: Path) -> None:
    """Writes one SWE-Gym instance as a Harbor-format task directory."""
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(exist_ok=True)
    (task_dir / "tests").mkdir(exist_ok=True)

    task_toml = f"""[metadata]
task_id = {toml_string(instance["instance_id"])}
dataset = {toml_string(dataset_name)}
repo = {toml_string(instance["repo"])}
version = {toml_string(instance["version"])}
base_commit = {toml_string(instance["base_commit"])}
"""
    (task_dir / "task.toml").write_text(task_toml)

    instruction = f"""# {instance["repo"]} ({instance["instance_id"]})

{instance["problem_statement"]}
"""
    if instance["hints_text"]:
        instruction += f"\n## Hints\n\n{instance['hints_text']}\n"
    (task_dir / "instruction.md").write_text(instruction)

    (task_dir / "solution" / "patch.diff").write_text(instance["patch"])

    (task_dir / "tests" / "test_patch.diff").write_text(instance["test_patch"])
    (task_dir / "tests" / "instance.json").write_text(
        json.dumps({k: instance[k] for k in INSTANCE_FIELDS}, indent=2)
    )
    (task_dir / "tests" / "grade.py").write_text(GRADE_PY)
    test_sh_path = task_dir / "tests" / "test.sh"
    test_sh_path.write_text(TEST_SH)
    test_sh_path.chmod(0o755)


def build(dataset_name: str, split: str, output_dir: Path, limit: int | None) -> None:
    """Downloads a SWE-Gym split and converts every instance into a Harbor task."""
    dataset: Dataset = load_dataset(dataset_name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    output_dir.mkdir(parents=True, exist_ok=True)
    for instance in dataset:
        write_task(instance, dataset_name, output_dir / instance["instance_id"])

    print(f"Wrote {len(dataset)} tasks from {dataset_name} ({split}) to {output_dir}")


def main() -> None:
    """Parses CLI args and runs the SWE-Gym -> Harbor conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="lite")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to write <output-dir>/<dataset>/<instance_id>/ task dirs into.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only convert the first N instances."
    )
    args = parser.parse_args()

    dataset_name = DATASETS[args.dataset]
    output_dir = args.output_dir / args.dataset
    build(dataset_name, args.split, output_dir, args.limit)


if __name__ == "__main__":
    main()
