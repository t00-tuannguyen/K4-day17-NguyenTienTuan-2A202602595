from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config

COLUMNS = (
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
)
REFUSAL_MARKERS = ("chưa có thông tin", "không biết", "không nhớ")


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _hits(answer: str, expected: list[str]) -> int:
    lowered = answer.lower()
    return sum(1 for item in expected if item.lower() in lowered)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight 0-1 quality score for offline mode.

    - 0.7: share of expected facts covered
    - 0.15: concise / structured (bullets or <= 400 chars), matching the user's preferred style
    - 0.15: actually answers instead of saying it does not know
    """

    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    concise = 1.0 if (re.search(r"^\s*[-*•]", answer, re.MULTILINE) or len(answer) <= 400) else 0.0
    answered = 0.0 if any(marker in answer.lower() for marker in REFUSAL_MARKERS) else 1.0
    return round(0.7 * coverage + 0.15 * concise + 0.15 * answered, 4)


def judge_quality(config: LabConfig, question: str, answer: str, expected: list[str]) -> float:
    """LLM-as-judge score (0-1) with the judge model; falls back to the heuristic."""

    try:
        from model_provider import build_chat_model

        judge = build_chat_model(config.judge_model)
        prompt = (
            "Chấm điểm câu trả lời của trợ lý từ 1 đến 5 (5 là tốt nhất) dựa trên: đúng các fact mong đợi, "
            "ngắn gọn, đúng trọng tâm. Chỉ trả về một con số.\n"
            f"Câu hỏi: {question}\nFact mong đợi: {', '.join(expected)}\nCâu trả lời: {answer}"
        )
        match = re.search(r"[1-5]", str(judge.invoke(prompt).content))
        if match:
            return (int(match.group(0)) - 1) / 4
    except Exception:
        pass
    return heuristic_quality(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    1. Feed all turns of each conversation into its own thread.
    2. Ask each recall question in a fresh thread (cross-session).
    3. Sum agent tokens / prompt tokens over every thread, average recall and quality.
    4. Record `User.md` growth and compaction count.
    """

    user_ids = {conv["user_id"] for conv in conversations}
    memory_size = getattr(agent, "memory_file_size", lambda _user_id: 0)
    size_before = sum(memory_size(user_id) for user_id in user_ids)

    threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    compactions = 0

    for conv in conversations:
        thread_id = conv["id"]
        threads.append(thread_id)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], thread_id, turn)
        compactions += agent.compaction_count(thread_id)

        for index, item in enumerate(conv.get("recall_questions", []), start=1):
            recall_thread = f"{thread_id}-recall-{index}"
            threads.append(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            if config.live_mode:
                quality_scores.append(judge_quality(config, item["question"], answer, expected))
            else:
                quality_scores.append(heuristic_quality(answer, expected))

    size_after = sum(memory_size(user_id) for user_id in user_ids)

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=average(recall_scores),
        response_quality=average(quality_scores),
        memory_growth_bytes=size_after - size_before,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render rows as a GitHub-flavored markdown table."""

    table = [
        [
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.0%}",
            f"{row.response_quality:.2f}",
            f"{row.memory_growth_bytes:,}",
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(cell) for cell in line) + " |" for line in table]
        return "\n".join(lines)


def _delta(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    def pct(new: int, old: int) -> str:
        return f"{(new - old) / old:+.1%}" if old else "n/a"

    return (
        f"Advanced vs Baseline: prompt tokens {pct(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"agent tokens {pct(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
        f"recall {advanced.recall_score - baseline.recall_score:+.0%}"
    )


def run_suite(title: str, dataset: Path, config: LabConfig, force_offline: bool) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    rows = []
    for name, agent_cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        # Fresh, isolated state per suite and agent so memory growth is measured from zero.
        state_dir = config.state_dir / "benchmark" / dataset.stem / name.lower()
        shutil.rmtree(state_dir, ignore_errors=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        agent_config = dataclasses.replace(config, state_dir=state_dir)
        agent = agent_cls(agent_config, force_offline=force_offline)
        rows.append(run_agent_benchmark(name, agent, conversations, agent_config))

    turns = sum(len(conv["turns"]) for conv in conversations)
    questions = sum(len(conv.get("recall_questions", [])) for conv in conversations)
    print(f"\n## {title}\n")
    print(f"Dataset: {dataset.name} — {len(conversations)} conversation(s), {turns} turns, {questions} recall questions\n")
    print(format_rows(rows))
    print(f"\n{_delta(rows[0], rows[1])}")
    return rows


def main() -> None:
    """Run the Standard and Long-Context Stress benchmarks for Baseline vs Advanced."""

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark")
    parser.add_argument("--offline", action="store_true", help="force the deterministic offline path")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    force_offline = args.offline or not config.live_mode
    mode = "offline" if force_offline else f"live ({config.model.provider}/{config.model.model_name})"
    print(
        f"# Memory benchmark — mode: {mode}, compact threshold: {config.compact_threshold_tokens} tokens, "
        f"keep {config.compact_keep_messages} messages"
    )

    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, force_offline)
    run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config, force_offline)


if __name__ == "__main__":
    main()
