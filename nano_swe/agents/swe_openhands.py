"""The RL rollout AgentRunner: OpenHands + Daytona, driven by the training-time vLLM policy.

Reuses ``nano_swe.harness.episode.run_episode`` (the same Daytona sandbox + OpenHands + grading
pipeline validated against Fireworks) unmodified, just pointed at a per-rollout
``ChatProxyServer`` session instead of a hosted LLM API.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, List

from nano_swe.agents.base import AgentRunner, Trajectory, _first_scalar
from nano_swe.agents.chat_proxy import ChatProxyServer

DEFAULT_MAX_ITERATIONS = 30

# Per-runner (per Ray actor) concurrent-sandbox cap; total cluster concurrency is this times
# --rollout.num_runners. Scale out via this env var and/or --rollout.num_runners.
_MAX_CONCURRENT_SANDBOXES = int(os.environ.get("SWE_HARNESS_MAX_CONCURRENT_SANDBOXES", "8"))


class OpenHandsSWERunner(AgentRunner):
    """One rollout = one Harbor task solved by an OpenHands agent in a Daytona sandbox."""

    PRERENDER_PROMPT = False

    def __init__(self):
        self._proxy: ChatProxyServer | None = None
        self._sandbox_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SANDBOXES)

    async def _get_proxy(self, llm_engine: Any, hf_tokenizer: Any, sampling_params: Any) -> ChatProxyServer:
        if self._proxy is None:
            self._proxy = ChatProxyServer(llm_engine, hf_tokenizer, sampling_params)
            await self._proxy.start()
        return self._proxy

    async def execute(
        self,
        prompt: Any,
        label: Any,
        sampling_params: Any,
        max_length: int,
        hf_tokenizer: Any,
        llm_engine: Any,
        tools: Any = None,
    ) -> List[Trajectory]:
        """`prompt` is a Harbor task directory path (see nano_swe/swe_data/build_dataset.py);
        `label` is passed through unchanged onto the returned Trajectory."""
        proxy = await self._get_proxy(llm_engine, hf_tokenizer, sampling_params)
        session_id = proxy.new_session()
        try:
            async with self._sandbox_semaphore:
                result = await asyncio.to_thread(
                    _run_episode_sync, Path(prompt), f"{proxy.url}/session/{session_id}/v1"
                )
            session = proxy.trajectory(session_id)
            trajectory = Trajectory(
                observation_tokens=session.observation_tokens,
                action_ranges=session.action_ranges,
                rollout_log_probs=session.rollout_log_probs,
                truncated=session.truncated,
                prompt=str(prompt),
                label=label,
                reward=_first_scalar(result["reward"]),
                scores=_first_scalar(result["reward"]),
                extra_logs={"sandbox_id": result["sandbox_id"]},
            )
        finally:
            proxy.close_session(session_id)
        return [trajectory]


def _run_episode_sync(task_dir: Path, base_url: str) -> dict:
    from nano_swe.harness.episode import run_episode

    return run_episode(
        task_dir,
        base_url=base_url,
        api_key="unused",
        model="policy",
        max_iterations=DEFAULT_MAX_ITERATIONS,
        native_tool_calling=False,
    )
