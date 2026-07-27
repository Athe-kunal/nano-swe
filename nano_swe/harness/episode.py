"""Runs one Harbor task end-to-end in a fresh Daytona sandbox: boot from the task's prebuilt
docker image (repo already cloned/installed, in a conda env named "testbed" per SWE-bench's own
Dockerfiles), run an OpenHands agent against the task instruction (against whatever
OpenAI-compatible ``base_url`` the caller passes — Fireworks for a smoke test, or our own chat
proxy in front of the training-time vLLM policy for a real RL rollout), then grade the result
with the task's own tests/grade.py and return the reward.

Every task.toml this harness runs must set ``[environment] docker_image`` — see
build_dataset.py/build_verified_eval.py, which set it for every SWE-Gym and SWE-bench
Verified/Lite instance from the images their respective authors publish on Docker Hub.
"""

import tomllib
from pathlib import Path

from daytona import CreateSandboxFromImageParams, Daytona, Sandbox

from nano_swe.harness import paths

SANDBOX_CREATE_TIMEOUT = 600  # some images (e.g. PyTorch/CUDA-heavy repos) are multi-GB pulls

DRIVER_PY = f'''"""Runs an OpenHands agent against the task instruction. Executed inside the sandbox."""

import argparse
import os

from pydantic import SecretStr

from openhands.sdk import LLM, Conversation, LocalWorkspace
from openhands.tools.preset.default import get_default_agent

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--workdir", required=True)
parser.add_argument("--max-iterations", type=int, default=30)
parser.add_argument("--native-tool-calling", choices=["true", "false"], default="true")
args = parser.parse_args()

instruction = open("{paths.INSTRUCTION_PATH}").read()

llm = LLM(
    usage_id="agent",
    model=args.model,
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    native_tool_calling=args.native_tool_calling == "true",
)
agent = get_default_agent(llm=llm, cli_mode=True)
workspace = LocalWorkspace(working_dir=args.workdir)
conversation = Conversation(
    agent=agent,
    workspace=workspace,
    max_iteration_per_run=args.max_iterations,
    persistence_dir="{paths.TRAJECTORY_DIR}",
)

try:
    conversation.send_message(instruction)
    conversation.run()
    cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
    print(f"EPISODE_COST: {{cost}}")
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
    save_trajectory: bool = False,
) -> dict:
    """Runs one Harbor task end-to-end in a fresh Daytona sandbox.

    Args:
        task_dir: A Harbor task directory (see nano_swe/swe_data/build_dataset.py) whose
            task.toml sets [environment] docker_image.
        base_url: OpenAI-compatible endpoint the in-sandbox OpenHands agent talks to.
        api_key: API key/token for that endpoint (any non-empty string if unused).
        model: Model name to request from that endpoint.
        max_iterations: Cap on the agent's tool-call loop.
        native_tool_calling: Whether the endpoint understands OpenAI-style `tools=[...]` /
            `tool_calls` (True for Fireworks and most hosted APIs; False for a plain
            text-completion proxy that can't format tool_calls JSON).
        save_trajectory: Whether to pull OpenHands' event log (every LLM call, tool action,
            observation) out of the sandbox before it's deleted. Off by default — an RL rollout
            has nothing to do with it yet and downloading/storing it for every rollout would be
            wasteful; eval runs can opt in for debugging.

    Returns:
        {"reward": float, "sandbox_id": str, "trajectory": bytes | None}. "trajectory" is a
        gzipped tar of OpenHands' persistence_dir (JSON event log), or None if not requested (or
        if the agent never wrote one, e.g. it crashed before the first event).
    """
    toml = tomllib.loads((task_dir / "task.toml").read_text())
    task = toml["metadata"]
    docker_image = toml.get("environment", {}).get("docker_image")
    if not docker_image:
        raise ValueError(f"{task_dir}/task.toml has no [environment] docker_image")
    instruction = (task_dir / "instruction.md").read_text()

    daytona = Daytona()
    sandbox = daytona.create(CreateSandboxFromImageParams(image=docker_image), timeout=SANDBOX_CREATE_TIMEOUT)
    print(f"Created sandbox {sandbox.id} for {task['task_id']} ({task['repo']}@{task['base_commit'][:7]})")
    try:
        # The image's "testbed" conda env may predate openhands-sdk's >=3.12 requirement, so the
        # agent gets its own uv-managed interpreter instead of touching the repo's env at all.
        _run(sandbox, "curl -LsSf https://astral.sh/uv/install.sh | sh")
        _run(sandbox, f"~/.local/bin/uv venv --python 3.12 {paths.OH_VENV_DIR}")
        _run(sandbox, f"~/.local/bin/uv pip install --python {paths.OH_PYTHON} openhands-sdk openhands-tools")
        _run(sandbox, f"{paths.TESTBED_PYTHON} -m pip install -q pytest-json-report")

        sandbox.fs.create_folder(paths.TASK_TESTS_DIR, "755")
        sandbox.fs.upload_file(instruction.encode(), paths.INSTRUCTION_PATH)
        sandbox.fs.upload_file(DRIVER_PY.encode(), paths.DRIVER_PATH)
        for name in paths.TASK_TEST_FILES:
            sandbox.fs.upload_file(str(task_dir / "tests" / name), f"{paths.TASK_TESTS_DIR}/{name}")

        _run(
            sandbox,
            f"{paths.OH_PYTHON} {paths.DRIVER_PATH} --model {model} --workdir {paths.REPO_DIR} "
            f"--max-iterations {max_iterations} "
            f"--native-tool-calling {'true' if native_tool_calling else 'false'}",
            cwd=paths.WORKSPACE_DIR,
            timeout=900,
            env={"OPENAI_API_KEY": api_key, "OPENAI_BASE_URL": base_url},
        )

        _run(
            sandbox,
            f"{paths.TESTBED_PYTHON} {paths.TASK_TESTS_DIR}/grade.py",
            cwd=paths.WORKSPACE_DIR,
            timeout=600,
            env={"AGENT_WORKDIR": paths.REPO_DIR},
        )
        reward = float(sandbox.fs.download_file(paths.REWARD_PATH).decode())
        print(f"REWARD: {reward}")

        trajectory = None
        if save_trajectory:
            _run(
                sandbox,
                f"tar czf {paths.TRAJECTORY_ARCHIVE_PATH} -C {paths.TRAJECTORY_DIR} .",
                check=False,
            )
            try:
                trajectory = sandbox.fs.download_file(paths.TRAJECTORY_ARCHIVE_PATH)
            except Exception as e:
                print(f"[episode] no trajectory to save: {e}")

        return {"reward": reward, "sandbox_id": sandbox.id, "trajectory": trajectory}
    finally:
        sandbox.delete()
