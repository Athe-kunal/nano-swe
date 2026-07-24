"""Smoke test: runs one Harbor task end-to-end in a real Daytona sandbox.

Pipeline: spin up a Daytona sandbox, clone the task's repo at its base commit,
run an OpenHands agent (backed by a Fireworks model) against the task's
instruction, then grade the result with the task's own tests/grade.py.

This is a one-off harness smoke test, not the training loop.
"""

import argparse
import os
import tomllib
from pathlib import Path

from daytona import CreateSandboxFromImageParams, Daytona, Sandbox
from dotenv import load_dotenv

DEFAULT_MODEL = "openai/accounts/fireworks/models/gpt-oss-120b"
DEFAULT_TASK_DIR = Path(__file__).parent.parent / "swe_data" / "data" / "lite" / "getmoto__moto-5752"

DRIVER_PY = '''"""Runs an OpenHands agent against the task instruction. Executed inside the sandbox."""

import argparse
import os

from pydantic import SecretStr

from openhands.sdk import LLM, Conversation, LocalWorkspace
from openhands.tools.preset.default import get_default_agent

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--max-iterations", type=int, default=30)
args = parser.parse_args()

instruction = open("/workspace/task/instruction.md").read()

llm = LLM(
    usage_id="agent",
    model=args.model,
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
)
agent = get_default_agent(llm=llm, cli_mode=True)
workspace = LocalWorkspace(working_dir="/workspace/repo")
conversation = Conversation(agent=agent, workspace=workspace, max_iteration_per_run=args.max_iterations)

try:
    conversation.send_message(instruction)
    conversation.run()
    cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
    print(f"EPISODE_COST: {cost}")
finally:
    conversation.close()
'''


def _run(
    sandbox: Sandbox,
    command: str,
    cwd: str | None = None,
    check: bool = True,
    timeout: int = 600,
    env: dict | None = None,
) -> str:
    """Runs a shell command in the sandbox, printing and returning its output."""
    result = sandbox.process.exec(command, cwd=cwd, timeout=timeout, env=env)
    print(f"$ {command}\n{result.result}")
    if check and result.exit_code != 0:
        raise RuntimeError(f"Command failed ({result.exit_code}): {command}")
    return result.result


def run(task_dir: Path, model: str, max_iterations: int) -> None:
    """Runs one Harbor task end-to-end in a fresh Daytona sandbox and prints the reward."""
    load_dotenv()
    task = tomllib.loads((task_dir / "task.toml").read_text())["metadata"]
    instruction = (task_dir / "instruction.md").read_text()
    repo_url = f"https://github.com/{task['repo']}.git"

    daytona = Daytona()
    sandbox = daytona.create(CreateSandboxFromImageParams(image="python:3.12-slim"), timeout=120)
    print(f"Created sandbox {sandbox.id} for {task['task_id']} ({task['repo']}@{task['base_commit'][:7]})")
    try:
        _run(
            sandbox,
            "apt-get update -qq && apt-get install -y -qq git build-essential",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        _run(sandbox, f"git clone --quiet {repo_url} /workspace/repo")
        _run(sandbox, f"git checkout --quiet {task['base_commit']}", cwd="/workspace/repo")
        _run(sandbox, "pip install -q -e '.[all]' || pip install -q -e .", cwd="/workspace/repo", check=False)
        _run(
            sandbox,
            "pip install -q -r requirements-tests.txt",
            cwd="/workspace/repo",
            check=False,
        )
        _run(sandbox, "pip install -q openhands-sdk openhands-tools pytest-json-report")

        sandbox.fs.create_folder("/workspace/task/tests", "755")
        sandbox.fs.upload_file(instruction.encode(), "/workspace/task/instruction.md")
        sandbox.fs.upload_file(DRIVER_PY.encode(), "/workspace/driver.py")
        for name in ("test_patch.diff", "instance.json", "grade.py"):
            sandbox.fs.upload_file(str(task_dir / "tests" / name), f"/workspace/task/tests/{name}")

        _run(
            sandbox,
            f"python /workspace/driver.py --model {model} --max-iterations {max_iterations}",
            cwd="/workspace",
            timeout=900,
            env={
                "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "OPENAI_BASE_URL": os.environ["OPENAI_BASE_URL"],
            },
        )

        _run(
            sandbox,
            "python /workspace/task/tests/grade.py",
            cwd="/workspace",
            timeout=600,
            env={"AGENT_WORKDIR": "/workspace/repo"},
        )
        reward = sandbox.fs.download_file("/logs/verifier/reward.txt").decode()
        print(f"REWARD: {reward}")
    finally:
        sandbox.delete()


def main() -> None:
    """Parses CLI args and runs the dummy episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    run(args.task_dir, args.model, args.max_iterations)


if __name__ == "__main__":
    main()
