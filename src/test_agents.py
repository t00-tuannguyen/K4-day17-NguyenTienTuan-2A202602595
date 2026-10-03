from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, recall_points
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_candidates, extract_profile_updates

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LONG_FILLER = (
    "Mình kể thêm một đoạn dài về tin tức và công việc để thread phình to: "
    "NASA, X-59, WMO và kế hoạch điện sạch đều là ví dụ về trade-off vận hành. " * 3
)


def make_config(tmp_path: Path):
    """Isolated config: state in tmp_path, small compact threshold, always offline."""

    return dataclasses.replace(
        load_config(),
        state_dir=tmp_path / "state",
        compact_threshold_tokens=200,
        compact_keep_messages=2,
        live_mode=False,
    )


def make_agents(tmp_path: Path) -> tuple[BaselineAgent, AdvancedAgent]:
    config = make_config(tmp_path)
    return BaselineAgent(config, force_offline=True), AdvancedAgent(config, force_offline=True)


def feed(agent, user_id: str, thread_id: str, turns: list[str]) -> None:
    for turn in turns:
        agent.reply(user_id, thread_id, turn)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Missing profile: default template, nothing on disk yet.
    assert "## Facts" in store.read_text("dungct")
    assert store.file_size("dungct") == 0

    path = store.write_text("dungct", "# User Profile: dungct\n\n## Facts\n- name: DũngCT\n")
    assert path == store.path_for("dungct") and path.name == "User.md"
    assert store.facts("dungct") == {"name": "DũngCT"}
    assert store.file_size("dungct") == path.stat().st_size > 0

    assert store.edit_text("dungct", "DũngCT", "DũngCT Nguyễn") is True
    assert store.facts("dungct")["name"] == "DũngCT Nguyễn"
    assert store.edit_text("dungct", "không tồn tại", "x") is False

    # upsert overwrites instead of duplicating; unchanged value does not rewrite.
    assert store.upsert_fact("dungct", "location", "Đà Nẵng") is True
    assert store.upsert_fact("dungct", "location", "Huế") is True
    assert store.upsert_fact("dungct", "location", "Huế") is False
    assert store.read_text("dungct").count("- location:") == 1

    # User ids cannot escape the profiles directory.
    assert store.path_for("../../etc/passwd").resolve().is_relative_to((tmp_path / "profiles").resolve())


def test_compact_trigger(tmp_path: Path) -> None:
    _, advanced = make_agents(tmp_path)

    feed(advanced, "u", "short", ["Mình tên là DũngCT.", "Mình ở Huế."])
    assert advanced.compaction_count("short") == 0

    turns = ["Chào bạn, mình tên là DũngCT và mình đang làm MLOps engineer."] + [LONG_FILLER] * 6
    feed(advanced, "u", "long", turns)

    state = advanced.compact_memory.context("long")
    assert advanced.compaction_count("long") >= 1
    assert len(state["messages"]) <= advanced.config.compact_keep_messages
    # The fact-bearing first turn survives inside the summary, not as a raw message.
    assert "DũngCT" in state["summary"]


def test_compact_summary_stays_bounded() -> None:
    manager = CompactMemoryManager(threshold_tokens=150, keep_messages=2)
    sizes = []
    for i in range(40):
        manager.append("t", "user", f"Lượt {i}: " + LONG_FILLER)
        sizes.append(manager.total_tokens("t"))
    assert manager.compaction_count("t") > 5
    # Context does not keep growing with the number of turns.
    assert max(sizes[-10:]) <= max(sizes[:10]) * 1.5


def test_cross_session_recall(tmp_path: Path) -> None:
    baseline, advanced = make_agents(tmp_path)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Đồ uống yêu thích là cà phê sữa đá.",
        "Mình muốn bạn trả lời ngắn gọn, rõ ý và có ví dụ thực tế.",
    ]
    question = "Mình tên gì và đồ uống yêu thích là gì?"
    feed(baseline, "dungct", "s1", turns)
    feed(advanced, "dungct", "s1", turns)

    # Baseline remembers within the thread only.
    assert "DũngCT" in baseline.reply("dungct", "s1", question)["response"]
    baseline_new = baseline.reply("dungct", "s2", question)["response"]
    assert "DũngCT" not in baseline_new and "cà phê sữa đá" not in baseline_new

    # Advanced remembers in a brand-new thread...
    advanced_new = advanced.reply("dungct", "s2", question)["response"]
    assert recall_points(advanced_new, ["DũngCT", "cà phê sữa đá"]) == 1.0

    # ...and even after a "restart", because User.md lives on disk.
    restarted = AdvancedAgent(advanced.config, force_offline=True)
    assert "ngắn gọn" in restarted.reply("dungct", "s3", "Mình thích style trả lời như thế nào?")["response"]
    assert restarted.memory_file_size("dungct") > 0


def test_correction_keeps_latest_fact_only(tmp_path: Path) -> None:
    _, advanced = make_agents(tmp_path)
    feed(advanced, "dungct", "c1", ["Mình ở Đà Nẵng và đang làm backend engineer cho startup AI."])
    feed(
        advanced,
        "dungct",
        "c2",
        [
            "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.",
            "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
            "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé.",
        ],
    )

    facts = advanced.profile_store.facts("dungct")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    profile = advanced.profile_store.read_text("dungct")
    assert "Đà Nẵng" not in profile and "backend" not in profile

    answer = advanced.reply("dungct", "c3", "Hiện tại mình làm nghề gì và mình còn ở Huế không?")["response"]
    assert "MLOps engineer" in answer and "backend" not in answer


def test_questions_and_noise_are_not_stored() -> None:
    # Questions / recall requests never become facts.
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Nhắc lại giúp mình: tên, món ăn yêu thích và mình nuôi con gì.") == {}
    assert extract_profile_updates("Sang thread mới rồi, nhắc lại giúp mình tên và nơi ở hiện tại.") == {}

    # Jokes, meeting places, and pet names are noise.
    joke = extract_profile_updates(
        "Có lúc mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa."
    )
    assert "profession" not in joke
    assert "location" not in extract_profile_updates("Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày.")
    pet = extract_profile_updates("Mình nuôi một bé corgi tên Bơ.")
    assert pet == {"pet": "corgi tên Bơ"}


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    conversation = json.loads((DATA_DIR / "advanced_long_context.json").read_text(encoding="utf-8"))[0]
    config = dataclasses.replace(make_config(tmp_path), compact_threshold_tokens=800, compact_keep_messages=4)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    feed(baseline, conversation["user_id"], "stress", conversation["turns"])
    feed(advanced, conversation["user_id"], "stress", conversation["turns"])

    assert advanced.compaction_count("stress") >= 2
    assert advanced.prompt_token_usage("stress") < 0.6 * baseline.prompt_token_usage("stress")


def test_advanced_costs_more_on_short_threads(tmp_path: Path) -> None:
    """Trade-off: without anything to compact, User.md is pure overhead per turn."""

    baseline, advanced = make_agents(tmp_path)
    turns = ["Mình tên là DũngCT.", "Mình ở Huế.", "Món ăn yêu thích là mì Quảng."]
    feed(baseline, "dungct", "short", turns)
    feed(advanced, "dungct", "short", turns)

    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


def test_benchmark_scoring() -> None:
    expected = ["DũngCT", "cà phê sữa đá"]
    assert recall_points("- Tên: DũngCT\n- Đồ uống yêu thích: cà phê sữa đá", expected) == 1.0
    assert recall_points("Tên bạn là DũngCT", expected) == 0.5
    assert recall_points("Mình chưa có thông tin này.", expected) == 0.0
    assert heuristic_quality("- Tên: DũngCT\n- Đồ uống: cà phê sữa đá", expected) == 1.0
    assert heuristic_quality("Mình chưa có thông tin này trong ngữ cảnh hiện tại.", expected) < 0.2


# ---------------------------------------------------------------------------
# Bonus: confidence threshold + memory decay
# ---------------------------------------------------------------------------


def test_confidence_threshold_blocks_uncertain_facts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", min_confidence=0.6)

    # Hedged, third-party, conditional and temporary statements score below the threshold.
    for message in (
        "Hình như tên mình là Tuấn.",
        "Bạn mình đang ở Hà Nội và làm data engineer.",
        "Nếu mình chuyển ra Đà Nẵng thì sẽ đi biển nhiều hơn.",
        "Tạm thời mình ở Huế.",
    ):
        candidates = extract_profile_candidates(message)
        assert candidates and all(conf < 0.6 for _, conf in candidates.values()), message
        assert extract_profile_updates(message) == {}

    store.observe("u", extract_profile_candidates("Hình như tên mình là Tuấn."))
    assert store.facts("u") == {}
    assert store.read_meta("u")["pending"]["name"]["value"] == "Tuấn"

    # A second independent mention is combined (noisy-OR) and promoted into User.md.
    store.observe("u", extract_profile_candidates("Hình như tên mình là Tuấn."))
    assert store.facts("u") == {"name": "Tuấn"}
    assert store.read_meta("u")["pending"] == {}

    # A confident correction still overwrites immediately.
    store.observe("u", extract_profile_candidates("Mình ở Huế."))
    store.observe("u", extract_profile_candidates("Giờ mình đang ở Đà Nẵng chứ không còn ở Huế."))
    assert store.facts("u")["location"] == "Đà Nẵng"


def test_memory_decay_marks_then_prunes_old_facts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", half_life_turns=10)
    store.observe("u", extract_profile_candidates("Mình tên là Tuấn. Mình ở Huế. Món ăn yêu thích là bún bò."))
    idle = extract_profile_candidates("Hôm nay mình đọc tài liệu.")

    for _ in range(10):
        store.observe("u", idle)
    # Volatile facts (location: half-life x0.5) go stale first; still answerable, but flagged.
    assert "location" in store.stale_keys("u")
    assert "food" not in store.stale_keys("u")
    assert "## Cần xác nhận lại" in store.read_text("u")

    # Re-mentioning a stale fact revives it.
    store.observe("u", extract_profile_candidates("Mình vẫn ở Huế."))
    assert "location" not in store.stale_keys("u")

    for _ in range(40):
        store.observe("u", idle)
    facts = store.facts("u")
    assert "location" not in facts and "food" not in facts  # pruned
    assert facts["name"] == "Tuấn"  # identity never decays


def test_advanced_flags_stale_fact_in_answer(tmp_path: Path) -> None:
    config = dataclasses.replace(make_config(tmp_path), fact_half_life_turns=4)
    advanced = AdvancedAgent(config, force_offline=True)
    feed(advanced, "u", "t1", ["Mình tên là Tuấn.", "Mình ở Huế."])
    feed(advanced, "u", "t2", ["Hôm nay mình đọc tài liệu."] * 3)

    answer = advanced.reply("u", "t3", "Hiện tại mình đang ở đâu?")["response"]
    assert "Huế" in answer and "cần xác nhận lại" in answer
