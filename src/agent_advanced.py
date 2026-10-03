from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    acknowledge,
    answer_from_facts,
    estimate_tokens,
    extract_profile_candidates,
    is_question_message,
)
from model_provider import build_chat_model

try:  # Optional live-mode dependencies; imported at module level so tool type hints resolve.
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
    from langchain.tools import ToolRuntime, tool
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - offline mode works without LangChain
    create_agent = None

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI tiếng Việt có bộ nhớ dài hạn. Hồ sơ người dùng (User.md) "
    "chứa các fact ổn định; luôn ưu tiên fact mới nhất trong hồ sơ, bỏ qua thông tin cũ đã được đính chính. "
    "Trả lời theo style người dùng thích."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. within-session memory (recent messages kept by CompactMemoryManager)
    2. persistent `User.md` (UserProfileStore)
    3. compact memory: older turns folded into a bounded summary
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles",
            min_confidence=self.config.min_fact_confidence,
            half_life_turns=self.config.fact_half_life_turns,
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if self.config.live_mode and not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live mode (LangChain agent available) and offline mode."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, thread_id: str, message: str) -> dict[str, str]:
        """Steps 1-3: extract candidate facts, persist the confident ones to `User.md`
        (confidence threshold + decay happen inside `observe`), append to compact memory."""

        candidates = extract_profile_candidates(message)
        self.profile_store.observe(user_id, candidates)
        self.compact_memory.append(thread_id, "user", message)
        return {
            key: value
            for key, (value, confidence) in candidates.items()
            if confidence >= self.profile_store.min_confidence
        }

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path."""

        updates = self._remember(user_id, thread_id, message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        if is_question_message(message):
            response = self._offline_response(user_id, thread_id, message)
        else:
            response = acknowledge(updates)
        return self._record_turn(thread_id, message, response, prompt_tokens, "offline")

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Deterministic extraction still runs as a guardrail; the model can also edit User.md via tools.
        self._remember(user_id, thread_id, message)
        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        last = result["messages"][-1]
        response = last.content if isinstance(last.content, str) else str(last.content)
        usage = getattr(last, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or self._estimate_prompt_context_tokens(user_id, thread_id)
        return self._record_turn(thread_id, message, response, prompt_tokens, "live")

    def _record_turn(self, thread_id: str, message: str, response: str, prompt_tokens: int, mode: str) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", response)
        turn_tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + turn_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "response": response,
            "agent_tokens": turn_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": self.compaction_count(thread_id),
            "mode": mode,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system prompt + `User.md` + compact summary + recent messages."""

        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + self.compact_memory.total_tokens(thread_id)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic answer from persisted memory (`User.md`), so it works in a brand-new thread."""

        facts = self.profile_store.facts(user_id)
        for key in self.profile_store.stale_keys(user_id):
            facts[key] = f"{facts[key]} (thông tin cũ, cần xác nhận lại)"
        return answer_from_facts(message, facts)

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + dynamic prompt + summarization.

        Returns None when dependencies or credentials are missing so the agent
        falls back to the offline path.
        """

        if create_agent is None:
            return None

        store = self.profile_store

        @tool
        def read_user_profile(runtime: ToolRuntime[AgentContext]) -> str:
            """Read the current user's User.md profile."""
            return store.read_text(runtime.context.user_id)

        @tool
        def update_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Insert or overwrite one stable fact (e.g. name, location, profession, style) in User.md.

            Only call this for stable facts the user states about themselves, never for questions or jokes.
            """
            changed = store.apply_updates(runtime.context.user_id, {key.strip().lower(): value})
            return f"Updated: {changed}" if changed else "No change."

        @dynamic_prompt
        def inject_profile(request: ModelRequest) -> str:
            profile = store.read_text(request.runtime.context.user_id)
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n<user_profile>\n{profile}\n</user_profile>"

        try:
            model = build_chat_model(self.config.model)
            return create_agent(
                model=model,
                tools=[read_user_profile, update_user_fact],
                middleware=[
                    inject_profile,
                    SummarizationMiddleware(
                        model=model,
                        trigger=("tokens", self.config.compact_threshold_tokens),
                        keep=("messages", self.config.compact_keep_messages),
                    ),
                ],
                context_schema=AgentContext,
                checkpointer=InMemorySaver(),
            )
        except Exception:
            return None
