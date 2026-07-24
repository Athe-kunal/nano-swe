"""OpenAI-chat-compatible server in front of the token-in/token-out training-time vLLM policy.

Bridges OpenHands' chat-completions calls to ``llm_engine`` (a ``RouterGenerateClient``) and
tracks which token spans were policy-generated (actions) vs. injected by the harness
(observations), plus their logprobs — the ``Trajectory`` shape ``SamplesGenerator`` needs.

One server instance is scoped to one ``llm_engine`` and tokenizer; it hosts many concurrent
rollout *sessions*, one per in-flight episode.
"""

import asyncio
import socket
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.requests import Request


@dataclass
class _Session:
    observation_tokens: List[int] = field(default_factory=list)
    action_ranges: List[Tuple[int, int]] = field(default_factory=list)
    rollout_log_probs: List[float] = field(default_factory=list)
    truncated: bool = False


class ChatProxyServer:
    """Session-scoped OpenAI /v1/chat/completions proxy over a token-in/token-out llm_engine."""

    def __init__(self, llm_engine: Any, hf_tokenizer: Any, sampling_params: Any):
        self.llm_engine = llm_engine
        self.tokenizer = hf_tokenizer
        self.sampling_params = sampling_params
        self._sessions: dict[str, _Session] = {}
        self._server = None
        self.url: Optional[str] = None

        app = FastAPI()
        app.post("/session/{session_id}/v1/chat/completions")(self._chat_completions)
        self._app = app

    def new_session(self) -> str:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session()
        return session_id

    def trajectory(self, session_id: str) -> _Session:
        return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def start(self, host: str = "0.0.0.0", port: int = 0) -> str:
        """Mounts this server on the current event loop and returns its base URL."""
        config = uvicorn.Config(self._app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        self._server = server
        self._serve_task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.05)
        actual_port = server.servers[0].sockets[0].getsockname()[1]
        bind_host = host if host != "0.0.0.0" else socket.gethostbyname(socket.gethostname())
        self.url = f"http://{bind_host}:{actual_port}"
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            await self._serve_task

    async def _chat_completions(self, session_id: str, request: Request) -> JSONResponse:
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
        session = self._sessions[session_id]
        body = await request.json()
        messages = body["messages"]

        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        # The chat history only ever grows by appending messages, so the previous turn's full
        # token stream (prompt + its completion) is always a PREFIX of this turn's re-rendered
        # prompt — the new suffix is exactly what the harness injected since then (tool output,
        # new user turns): an observation span, no policy gradient.
        prev_len = len(session.observation_tokens)
        if prompt_token_ids[:prev_len] != session.observation_tokens:
            # History was rewritten out from under us (e.g. a context condenser). We can't
            # attribute the old action spans to token positions that no longer exist, so start
            # this trajectory fresh from the re-rendered prompt.
            session.observation_tokens = []
            session.action_ranges = []
            session.rollout_log_probs = []
            prev_len = 0
        session.observation_tokens.extend(prompt_token_ids[prev_len:])
        session.rollout_log_probs.extend([0.0] * (len(session.observation_tokens) - prev_len))

        output, _off_policy_len = await self.llm_engine.generate(prompt_token_ids, self.sampling_params, session_id)
        gen = output.outputs[0]

        action_start = len(session.observation_tokens)
        session.observation_tokens.extend(gen.token_ids)
        session.action_ranges.append((action_start, action_start + len(gen.token_ids)))
        session.rollout_log_probs.extend(next(iter(lp.values())).logprob for lp in gen.logprobs)
        if gen.finish_reason == "length":
            session.truncated = True

        text = self.tokenizer.decode(gen.token_ids, skip_special_tokens=True)
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "model": body.get("model", "policy"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop" if gen.finish_reason != "length" else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(gen.token_ids),
                    "total_tokens": len(prompt_token_ids) + len(gen.token_ids),
                },
            }
        )
