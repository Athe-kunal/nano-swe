"""Unit tests for the chat-completions -> token-in/token-out proxy's bookkeeping.

No network, GPU, or vLLM needed: the tokenizer and llm_engine are simple fakes so these
tests isolate the part that's easy to get subtly wrong — tracking which token spans are
policy actions vs. injected observations across a multi-turn conversation.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nano_swe.agents.chat_proxy import ChatProxyServer


class FakeTokenizer:
    """Whitespace tokenizer: each message becomes `role: content` split into "tokens" (ints
    derived from a shared vocab), so apply_chat_template output grows deterministically and
    decode can invert it exactly — enough to test the proxy's bookkeeping without a real model.
    """

    def __init__(self):
        self.vocab: list[str] = []

    def _id(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab.append(word)
        return self.vocab.index(word)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        words = []
        for m in messages:
            words.append(f"{m['role']}:")
            words.extend(m["content"].split())
        if add_generation_prompt:
            words.append("assistant:")
        return [self._id(w) for w in words]

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(self.vocab[i] for i in token_ids)


class FakeLLMEngine:
    """Returns a fixed reply for each call in sequence; records every prompt it was asked to score."""

    def __init__(self, tokenizer: FakeTokenizer, replies: list[str]):
        self.tokenizer = tokenizer
        self.replies = list(replies)
        self.seen_prompts: list[list[int]] = []

    async def generate(self, prompt_token_ids, sampling_params, session_id=None):
        self.seen_prompts.append(list(prompt_token_ids))
        reply = self.replies.pop(0)
        ids = [self.tokenizer._id(w) for w in reply.split()]
        logprobs = [{tid: SimpleNamespace(logprob=-0.1 * (i + 1))} for i, tid in enumerate(ids)]
        gen = SimpleNamespace(token_ids=ids, finish_reason="stop", logprobs=logprobs)
        return SimpleNamespace(outputs=[gen]), 0


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def _client(tokenizer, replies):
    engine = FakeLLMEngine(tokenizer, replies)
    proxy = ChatProxyServer(engine, tokenizer, sampling_params=None)
    return TestClient(proxy._app), proxy, engine


def test_single_turn_action_range_matches_completion(tokenizer):
    client, proxy, engine = _client(tokenizer, ["ok done"])
    session_id = proxy.new_session()

    resp = client.post(
        f"/session/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "fix the bug"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok done"

    traj = proxy.trajectory(session_id)
    start, end = traj.action_ranges[0]
    action_tokens = traj.observation_tokens[start:end]
    assert tokenizer.decode(action_tokens) == "ok done"
    # Everything before the action is the (single) prompt span.
    assert start == len(engine.seen_prompts[0])
    assert len(traj.rollout_log_probs) == len(traj.observation_tokens)


def test_multi_turn_marks_tool_output_as_observation_not_action(tokenizer):
    client, proxy, engine = _client(tokenizer, ["run the tests", "great it passes"])
    session_id = proxy.new_session()

    client.post(
        f"/session/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "fix the bug"}]},
    )
    # Harness appends the tool's output as a new message before the next turn.
    client.post(
        f"/session/{session_id}/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "run the tests"},
                {"role": "tool", "content": "5 passed 0 failed"},
            ]
        },
    )

    traj = proxy.trajectory(session_id)
    assert len(traj.action_ranges) == 2
    (s0, e0), (s1, e1) = traj.action_ranges
    assert tokenizer.decode(traj.observation_tokens[s0:e0]) == "run the tests"
    assert tokenizer.decode(traj.observation_tokens[e0:s1]) == "tool: 5 passed 0 failed assistant:"
    assert tokenizer.decode(traj.observation_tokens[s1:e1]) == "great it passes"
    # The whole stream is exactly the two prompts' final growth plus both completions.
    assert traj.observation_tokens[:e1] == traj.observation_tokens


def test_history_rewrite_resets_trajectory_instead_of_misattributing_it(tokenizer):
    client, proxy, engine = _client(tokenizer, ["first reply", "second reply"])
    session_id = proxy.new_session()

    client.post(
        f"/session/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "long task"}]},
    )
    before = proxy.trajectory(session_id)
    assert len(before.action_ranges) == 1

    # A condenser (or anything) rewrites history instead of appending — no longer a prefix.
    resp = client.post(
        f"/session/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "summary: condensed"}]},
    )
    assert resp.status_code == 200

    after = proxy.trajectory(session_id)
    assert len(after.action_ranges) == 1  # reset, not appended to the stale 1
    start, end = after.action_ranges[0]
    assert tokenizer.decode(after.observation_tokens[start:end]) == "second reply"


def test_unknown_session_returns_404(tokenizer):
    client, proxy, engine = _client(tokenizer, [])
    resp = client.post(
        "/session/does-not-exist/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
