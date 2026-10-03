from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    acknowledge,
    answer_from_facts,
    estimate_tokens,
    extract_profile_updates,
    is_question_message,
    merge_fact_value,
)
from model_provider import build_chat_model

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI tiếng Việt. Trả lời ngắn gọn, đúng trọng tâm. "
    "Bạn chỉ biết những gì người dùng nói trong cuộc trò chuyện hiện tại."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0
    # Facts mentioned inside this thread only (within-session memory).
    facts: dict[str, str] = field(default_factory=dict)


class BaselineAgent:
    """Agent A: within-session memory only.

    - Keeps the full message list per `thread_id` and re-sends all of it each turn.
    - No persistent `User.md`, so a new thread starts with no knowledge of the user.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if self.config.live_mode and not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting for one turn."""

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline path: remember facts within the thread, nothing across threads."""

        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        # The whole thread history is the prompt context for this turn.
        prompt_tokens = estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(
            estimate_tokens(m["content"]) for m in session.messages
        )

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            session.facts[key] = merge_fact_value(key, session.facts.get(key, ""), value)

        if is_question_message(message):
            response = answer_from_facts(message, session.facts)
        else:
            response = acknowledge(updates)

        return self._record_turn(session, message, response, prompt_tokens, "offline")

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        last = result["messages"][-1]
        response = last.content if isinstance(last.content, str) else str(last.content)
        usage = getattr(last, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        return self._record_turn(session, message, response, prompt_tokens, "live")

    def _record_turn(
        self, session: SessionState, message: str, response: str, prompt_tokens: int, mode: str
    ) -> dict[str, Any]:
        session.messages.append({"role": "assistant", "content": response})
        turn_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += turn_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": turn_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
            "mode": mode,
        }

    def _maybe_build_langchain_agent(self):
        """Wire `create_agent` + `InMemorySaver` (short-term thread state only).

        Returns None when dependencies or credentials are missing so the
        agent falls back to the offline path.
        """

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            return create_agent(
                model=build_chat_model(self.config.model),
                tools=[],
                system_prompt=BASELINE_SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
            )
        except Exception:
            return None
