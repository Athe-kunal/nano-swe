"""Runs one Harbor task end-to-end in a fresh Daytona sandbox: clone the task's repo at its base
commit, run an OpenHands agent against the task instruction (against whatever OpenAI-compatible
``base_url`` the caller passes — Fireworks for a smoke test, or our own chat proxy in front of the
training-time vLLM policy for a real RL rollout), then grade the result with the task's own
tests/grade.py and return the reward.
"""

import tomllib
from pathlib import Path

from daytona import CreateSandboxFromImageParams, Daytona, Sandbox

DRIVER_PY = '''"""Runs an OpenHands agent against the task instruction. Executed inside the sandbox."""

import argparse
import os

from pydantic import SecretStr

from openhands.sdk import LLM, Conversation, LocalWorkspace
from openhands.tools.preset.default import get_default_agent

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--max-iterations", type=int, default=30)
parser.add_argument("--native-tool-calling", choices=["true", "false"], default="true")
args = parser.parse_args()

instruction = open("/workspace/task/instruction.md").read()

llm = LLM(
    usage_id="agent",
    model=args.model,
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    native_tool_calling=args.native_tool_calling == "true",
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


def run_episode(
    task_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    max_iterations: int = 30,
    native_tool_calling: bool = True,
    image: str = "python:3.12-slim",
) -> dict:
    """Runs one Harbor task end-to-end in a fresh Daytona sandbox.

    Args:
        task_dir: A Harbor task directory (see nano_swe/swe_data/build_dataset.py).
        base_url: OpenAI-compatible endpoint the in-sandbox OpenHands agent talks to.
        api_key: API key/token for that endpoint (any non-empty string if unused).
        model: Model name to request from that endpoint.
        max_iterations: Cap on the agent's tool-call loop.
        native_tool_calling: Whether the endpoint understands OpenAI-style `tools=[...]` /
            `tool_calls` (True for Fireworks and most hosted APIs; False for a plain
            text-completion proxy that can't format tool_calls JSON).
        image: Docker image the sandbox boots from.

    Returns:
        {"reward": float, "sandbox_id": str}.
    """
    task = tomllib.loads((task_dir / "task.toml").read_text())["metadata"]
    instruction = (task_dir / "instruction.md").read_text()
    repo_url = f"https://github.com/{task['repo']}.git"

    daytona = Daytona()
    sandbox = daytona.create(CreateSandboxFromImageParams(image=image), timeout=120)
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
        _run(sandbox, "pip install -q -r requirements-tests.txt", cwd="/workspace/repo", check=False)
        _run(sandbox, "pip install -q openhands-sdk openhands-tools pytest-json-report", timeout=1200)

        sandbox.fs.create_folder("/workspace/task/tests", "755")
        sandbox.fs.upload_file(instruction.encode(), "/workspace/task/instruction.md")
        sandbox.fs.upload_file(DRIVER_PY.encode(), "/workspace/driver.py")
        for name in ("test_patch.diff", "instance.json", "grade.py"):
            sandbox.fs.upload_file(str(task_dir / "tests" / name), f"/workspace/task/tests/{name}")

        _run(
            sandbox,
            f"python /workspace/driver.py --model {model} --max-iterations {max_iterations} "
            f"--native-tool-calling {'true' if native_tool_calling else 'false'}",
            cwd="/workspace",
            timeout=900,
            env={"OPENAI_API_KEY": api_key, "OPENAI_BASE_URL": base_url},
        )

        _run(
            sandbox,
            "python /workspace/task/tests/grade.py",
            cwd="/workspace",
            timeout=600,
            env={"AGENT_WORKDIR": "/workspace/repo"},
        )
        reward = float(sandbox.fs.download_file("/logs/verifier/reward.txt").decode())
        print(f"REWARD: {reward}")
        return {"reward": reward, "sandbox_id": sandbox.id}
    finally:
        sandbox.delete()
