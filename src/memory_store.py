from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 characters per token, 0 for empty text."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return math.ceil(len(stripped) / 4)


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------

FACT_LINE = re.compile(r"^- (?P<key>[a-z_]+): (?P<value>.*)$", re.MULTILINE)
FACTS_HEADING = "## Facts"
STALE_HEADING = "## Cần xác nhận lại (thông tin cũ)"

# Facts whose new values are merged with the old ones instead of overwriting.
LIST_FACTS = ("interests", "style")

# Memory decay (bonus): half-life multipliers per fact. Identity never decays,
# volatile facts (where you live, what you do) decay faster than preferences.
NON_DECAYING_FACTS = ("name",)
HALF_LIFE_FACTOR = {"location": 0.5, "profession": 0.75}


def noisy_or(*confidences: float) -> float:
    """Combine independent mentions: 1 - prod(1 - c), capped below 1."""

    remaining = 1.0
    for confidence in confidences:
        remaining *= 1.0 - confidence
    return round(min(0.99, 1.0 - remaining), 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user.

    Facts are stored as `- key: value` lines under `## Facts` so the file stays
    human-readable and can be injected directly into a prompt. Bookkeeping for
    the bonus features (confidence, mentions, last seen, pending candidates)
    lives in a sidecar `User.meta.json` so it never costs prompt tokens.
    """

    root_dir: Path
    min_confidence: float = 0.6
    half_life_turns: float = 200.0
    stale_score: float = 0.35
    prune_score: float = 0.1

    # -- raw file API ---------------------------------------------------------

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", user_id.strip()).strip("-").lower() or "default"
        return self.root_dir / slug / "User.md"

    def _default_text(self, user_id: str) -> str:
        return f"# User Profile: {user_id}\n\n{FACTS_HEADING}\n"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return self._default_text(user_id)
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # -- structured facts -----------------------------------------------------

    def _sections(self, user_id: str) -> tuple[str, dict[str, str], dict[str, str], str]:
        """Split User.md into (header, active facts, stale facts, other trailing text)."""

        header_lines: list[str] = []
        other_lines: list[str] = []
        active: dict[str, str] = {}
        stale: dict[str, str] = {}
        section = "header"
        for line in self.read_text(user_id).splitlines():
            if line.startswith("## "):
                section = "active" if line.strip() == FACTS_HEADING else "stale" if line.strip() == STALE_HEADING else "other"
                if section == "other":
                    other_lines.append(line)
                continue
            match = FACT_LINE.match(line)
            if match and section in ("header", "active", "stale"):
                (stale if section == "stale" else active)[match.group("key")] = match.group("value").strip()
            elif section == "header":
                header_lines.append(line)
            elif section == "other":
                other_lines.append(line)
        return "\n".join(header_lines).rstrip(), active, stale, "\n".join(other_lines).strip()

    def _render(self, header: str, active: dict[str, str], stale: dict[str, str], other: str) -> str:
        parts = [header or "# User Profile", "", FACTS_HEADING, *(f"- {k}: {v}" for k, v in active.items())]
        if stale:
            parts += ["", STALE_HEADING, *(f"- {k}: {v}" for k, v in stale.items())]
        if other:
            parts += ["", other]
        return "\n".join(parts) + "\n"

    def _write_sections(self, user_id: str, header: str, active: dict[str, str], stale: dict[str, str], other: str) -> bool:
        content = self._render(header, active, stale, other)
        if content == self.read_text(user_id):
            return False
        self.write_text(user_id, content)
        return True

    def facts(self, user_id: str) -> dict[str, str]:
        """All known facts, including those marked as needing re-confirmation."""

        _, active, stale, _ = self._sections(user_id)
        return {**active, **stale}

    def stale_keys(self, user_id: str) -> set[str]:
        return set(self._sections(user_id)[2])

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or overwrite one fact. Returns True if the file changed.

        Overwriting (instead of appending) is what handles corrections: the old
        value is removed, so the profile never holds two conflicting facts. A
        re-confirmed stale fact moves back to the active section.
        """

        value = " ".join(value.split())
        header, active, stale, other = self._sections(user_id)
        if active.get(key) == value:
            return False
        stale.pop(key, None)
        active[key] = value
        return self._write_sections(user_id, header, active, stale, other)

    def remove_fact(self, user_id: str, key: str) -> bool:
        header, active, stale, other = self._sections(user_id)
        if key not in active and key not in stale:
            return False
        active.pop(key, None)
        stale.pop(key, None)
        return self._write_sections(user_id, header, active, stale, other)

    def apply_updates(self, user_id: str, updates: dict[str, str]) -> dict[str, str]:
        """Persist extracted facts, merging list-like facts. Returns the facts that changed."""

        current = self.facts(user_id)
        changed: dict[str, str] = {}
        for key, value in updates.items():
            value = merge_fact_value(key, current.get(key, ""), value)
            if self.upsert_fact(user_id, key, value):
                changed[key] = value
        return changed

    # -- bonus: confidence threshold + memory decay ------------------------------

    def meta_path_for(self, user_id: str) -> Path:
        return self.path_for(user_id).with_name("User.meta.json")

    def read_meta(self, user_id: str) -> dict:
        path = self.meta_path_for(user_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"clock": 0, "facts": {}, "pending": {}}

    def write_meta(self, user_id: str, meta: dict) -> None:
        path = self.meta_path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def observe(self, user_id: str, candidates: dict[str, tuple[str, float]]) -> dict[str, str]:
        """Process one user turn of candidate facts. Returns the facts written to User.md.

        - confidence >= `min_confidence`: written (corrections overwrite).
        - below threshold, same value as stored: only refreshes `last_seen`.
        - below threshold, new value: kept as *pending* outside User.md; repeated
          mentions are combined with noisy-OR and promoted once confident enough.
        Then decay is applied to every tracked fact.
        """

        meta = self.read_meta(user_id)
        meta["clock"] += 1
        clock = meta["clock"]
        current = self.facts(user_id)
        accepted: dict[str, float] = {}
        values: dict[str, str] = {}

        for key, (value, confidence) in candidates.items():
            info = meta["facts"].get(key)
            known = key in current and _contains_value(key, current[key], value)
            if confidence >= self.min_confidence:
                accepted[key], values[key] = confidence, value
            elif known and info:
                info["last_seen"] = clock
                info["mentions"] += 1
            else:
                pending = meta["pending"].get(key)
                if pending and pending["value"].lower() == value.lower():
                    pending["confidence"] = noisy_or(pending["confidence"], confidence)
                    pending["mentions"] += 1
                else:
                    pending = {"value": value, "confidence": confidence, "mentions": 1}
                meta["pending"][key] = pending
                if pending["confidence"] >= self.min_confidence:
                    accepted[key], values[key] = pending["confidence"], value

        changed = self.apply_updates(user_id, values)
        for key, confidence in accepted.items():
            info = meta["facts"].get(key)
            is_correction = key not in LIST_FACTS and key in changed and key in current
            if info and not is_correction:
                info["confidence"] = noisy_or(info["confidence"], confidence)
                info["mentions"] += 1
            else:
                info = meta["facts"][key] = {"confidence": confidence, "mentions": 1}
            info["last_seen"] = clock
            meta["pending"].pop(key, None)

        self.apply_decay(user_id, meta)
        return changed

    def fact_score(self, key: str, info: dict, clock: int) -> float:
        """Decayed confidence: confidence * 0.5 ** (age / half_life)."""

        if key in NON_DECAYING_FACTS:
            return info["confidence"]
        half_life = self.half_life_turns * HALF_LIFE_FACTOR.get(key, 1.0)
        age = max(0, clock - info["last_seen"])
        return round(info["confidence"] * 0.5 ** (age / half_life), 4)

    def apply_decay(self, user_id: str, meta: dict | None = None) -> dict[str, float]:
        """Move decayed facts to the re-confirm section, prune very old ones. Returns scores."""

        meta = meta if meta is not None else self.read_meta(user_id)
        header, active, stale, other = self._sections(user_id)
        scores: dict[str, float] = {}
        status: dict[str, str] = {}
        for key, info in list(meta["facts"].items()):
            if key not in active and key not in stale:
                continue
            score = scores[key] = self.fact_score(key, info, meta["clock"])
            if score < self.prune_score:
                status[key] = "pruned"
                del meta["facts"][key]
            else:
                status[key] = "stale" if score < self.stale_score else "active"
        # Untracked facts (written by hand / tools) keep their current section; order is preserved.
        everything = {**active, **stale}
        for key in everything:
            status.setdefault(key, "active" if key in active else "stale")
        active = {k: v for k, v in everything.items() if status[k] == "active"}
        stale = {k: v for k, v in everything.items() if status[k] == "stale"}
        self._write_sections(user_id, header, active, stale, other)
        self.write_meta(user_id, meta)
        return scores


def _contains_value(key: str, stored: str, value: str) -> bool:
    if key in LIST_FACTS:
        return all(item.strip().lower() in stored.lower() for item in value.split(",") if item.strip())
    return stored.lower() == value.lower()


def merge_fact_value(key: str, old: str, new: str) -> str:
    """Corrections overwrite single-valued facts; list-like facts accumulate."""

    if key == "style":
        return merge_style(old, new)
    if key in LIST_FACTS:
        return merge_list(old, new)
    return new


def merge_list(old: str, new: str, limit: int = 8) -> str:
    items: list[str] = []
    for item in [*old.split(","), *new.split(",")]:
        item = item.strip()
        if item and item.lower() not in {i.lower() for i in items}:
            items.append(item)
    return ", ".join(items[-limit:])


def _style_key(label: str) -> str:
    if "bullet" in label:
        return "bullet"
    if "ví dụ" in label:
        return "examples"
    return label


def merge_style(old: str, new: str) -> str:
    features: dict[str, str] = {}
    for label in [*old.split(","), *new.split(",")]:
        label = label.strip()
        if label:
            features[_style_key(label)] = label
    order = {key: i for i, key in enumerate(STYLE_ORDER)}
    return ", ".join(features[k] for k in sorted(features, key=lambda k: order.get(k, len(order))))


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
QUESTION_PREFIXES = (
    "nhắc lại giúp",
    "nhắc lại style",
    "tóm tắt",
    "bạn có biết",
    "bạn biết",
    "bạn có thể nhắc",
    "bạn thử nhớ lại",
)
# Recall requests that can appear mid-sentence ("Sang thread mới rồi, nhắc lại giúp mình ...").
RECALL_REQUEST_MARKERS = ("nhắc lại giúp", "bạn có biết")
NEGATION_MARKERS = ("không còn", "không phải", "chứ không", "đừng nói", "không làm")
JOKE_MARKERS = ("đùa",)

NAME_TRIGGER = re.compile(
    r"(?:mình\s+tên(?:\s+là)?|tên\s+(?:của\s+)?mình\s+là|(?:^|[:,]\s*)tên(?:\s+là)?)\s+",
    re.IGNORECASE,
)
LOCATION_TRIGGER = re.compile(
    r"(?:đang ở|hiện ở|sống ở|làm việc ở|mình ở|vẫn ở|"
    r"nơi ở hiện tại (?:vẫn )?là|chuyển (?:về|đến|tới|ra|vào))\s+",
    re.IGNORECASE,
)
JOB_TITLE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9]*)\s+(engineer|developer|scientist|manager|designer|analyst|researcher)\b",
    re.IGNORECASE,
)
DRINK_PATTERNS = (
    re.compile(r"đồ uống (?:yêu thích|ruột)(?: của mình)?(?: vẫn)? là ([^.,!?]+)", re.IGNORECASE),
    re.compile(r"(?:vẫn|hay|thường|thích) uống ([^.,!?]+?)(?:\s+(?:như cũ|nhưng|mỗi ngày)|[.,!?]|$)", re.IGNORECASE),
)
FOOD_PATTERN = re.compile(r"món (?:ăn )?(?:yêu thích|ruột)(?: của mình)?(?: vẫn)? là ([^.,!?]+)", re.IGNORECASE)
PET_PATTERN = re.compile(
    r"nuôi (?:một |1 )?(?:bé |con |chú )?([^\s.,!?]+)(?:\s+tên\s+([^\s.,!?]+))?", re.IGNORECASE
)
INTEREST_TRIGGER = re.compile(
    r"(?:thích|quan tâm(?: nhiều)?(?: đến| tới)?|đang học(?: thêm)? về)\s+([^.!?]+)", re.IGNORECASE
)
STYLE_TRIGGERS = ("trả lời", "style", "giải thích", "trình bày")
STYLE_ORDER = ("ngắn gọn", "bullet", "rõ ý", "có cấu trúc", "examples", "nhấn trade-off")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]


def is_question(sentence: str) -> bool:
    """True for questions / recall requests, which must never be stored as facts."""

    lowered = sentence.strip().lower()
    return (
        "?" in lowered
        or lowered.startswith(QUESTION_PREFIXES)
        or any(marker in lowered for marker in RECALL_REQUEST_MARKERS)
    )


def _leading_capitalized(text: str, max_words: int = 3) -> str:
    """Return the run of capitalized words at the start of `text` (e.g. `Đà Nẵng`)."""

    words: list[str] = []
    for raw in text.split()[:max_words]:
        word = raw.strip(".,!?;:()\"'")
        if not word or not word[0].isupper():
            break
        words.append(word)
        if raw[-1] in ".,!?;:":
            break
    return " ".join(words)


def _negated(sentence: str, start: int, window: int = 20) -> bool:
    prefix = sentence[max(0, start - window):start].lower()
    return any(marker in prefix for marker in NEGATION_MARKERS)


def _extract_name(sentence: str) -> str | None:
    for match in NAME_TRIGGER.finditer(sentence):
        name = _leading_capitalized(sentence[match.end():])
        if name:
            return name
    return None


def _extract_location(sentence: str) -> str | None:
    found = None
    for match in LOCATION_TRIGGER.finditer(sentence):
        place = _leading_capitalized(sentence[match.end():])
        if place and not _negated(sentence, match.start()):
            found = place  # last mention wins: "lúc đầu ở Huế, nhưng thực ra ở Đà Nẵng"
    return found


def _extract_profession(sentence: str) -> str | None:
    if any(marker in sentence.lower() for marker in JOKE_MARKERS):
        return None
    found = None
    for match in JOB_TITLE.finditer(sentence):
        if not _negated(sentence, match.start()):
            found = f"{match.group(1)} {match.group(2).lower()}"
    return found


def _extract_style(sentence: str) -> str | None:
    lowered = sentence.lower()
    if not any(trigger in lowered for trigger in STYLE_TRIGGERS):
        return None
    features: list[str] = []
    if re.search(r"ngắn|\bgọn\b|lan man", lowered):
        features.append("ngắn gọn")
    bullet = re.search(r"(\d+)\s*bullet", lowered)
    if bullet:
        features.append(f"{bullet.group(1)} bullet")
    elif "bullet" in lowered:
        features.append("bullet")
    if "rõ ý" in lowered:
        features.append("rõ ý")
    if "cấu trúc" in lowered:
        features.append("có cấu trúc")
    example = re.search(r"ví dụ (thực tế|thực chiến)", lowered)
    if example:
        features.append(f"có ví dụ {example.group(1)}")
    if "trade-off" in lowered:
        features.append("nhấn trade-off")
    return ", ".join(features) or None


def _extract_interests(sentence: str) -> list[str]:
    interests: list[str] = []
    for match in INTEREST_TRIGGER.finditer(sentence):
        for item in re.split(r",\s*|\s+và\s+", match.group(1)):
            item = item.strip()
            # Technical interests carry a Latin capitalized token (Python, AI, MLOps, RAG...).
            if item and len(item.split()) <= 4 and re.search(r"\b[A-Z][A-Za-z]*\b", item):
                interests.append(item)
    return interests


# Confidence model (bonus): every candidate fact gets a score in [0, 1].
DEFAULT_MIN_CONFIDENCE = 0.6
BASE_CONFIDENCE = {
    "name": 0.95,
    "location": 0.8,
    "profession": 0.85,
    "style": 0.8,
    "drink": 0.9,  # "đồ uống yêu thích là ..."; casual "vẫn uống ..." is lower, see below
    "food": 0.9,
    "pet": 0.85,
    "interests": 0.7,
}
CASUAL_DRINK_CONFIDENCE = 0.7
# Sentence-level modifiers. Style preferences are naturally phrased conditionally
# ("nếu bạn giải thích, hãy ..."), so the conditional penalty skips them.
HEDGE_MARKERS = ("hình như", "chắc là", "có lẽ", "có thể là", "không chắc", "đang cân nhắc", "dự định", "đang định", "tạm thời", "giả sử")
THIRD_PARTY_MARKERS = ("bạn mình", "bạn tôi", "đồng nghiệp", "anh ấy", "chị ấy", "vợ mình", "chồng mình", "sếp mình")
HEDGE_PENALTY = 0.4
CONDITIONAL_PENALTY = 0.3
THIRD_PARTY_PENALTY = 0.5


def _sentence_confidence(sentence: str, key: str, base: float) -> float:
    lowered = sentence.lower()
    score = base
    if any(marker in lowered for marker in HEDGE_MARKERS):
        score -= HEDGE_PENALTY
    if key != "style" and re.search(r"(?:^|\s)nếu\s", lowered):
        score -= CONDITIONAL_PENALTY
    if any(marker in lowered for marker in THIRD_PARTY_MARKERS):
        score -= THIRD_PARTY_PENALTY
    return round(max(0.0, min(1.0, score)), 2)


def extract_profile_candidates(message: str) -> dict[str, tuple[str, float]]:
    """Convert raw user text into candidate facts with a confidence score.

    - Works sentence by sentence and skips questions / recall requests.
    - Ignores negated mentions ("không còn làm backend engineer") and jokes.
    - Hedged, conditional or third-party sentences get a lower confidence.
    - Later sentences override earlier ones, so in-message corrections win.
    """

    candidates: dict[str, tuple[str, float]] = {}
    interests: list[str] = []
    interest_confidence = 0.0

    def put(key: str, value: str, sentence: str, base: float | None = None) -> None:
        confidence = _sentence_confidence(sentence, key, BASE_CONFIDENCE[key] if base is None else base)
        if key == "style" and key in candidates:
            old_value, old_confidence = candidates[key]
            candidates[key] = (merge_style(old_value, value), max(old_confidence, confidence))
        else:
            candidates[key] = (value, confidence)

    for sentence in split_sentences(message):
        if is_question(sentence):
            continue

        extractors = {
            "name": _extract_name,
            "location": _extract_location,
            "profession": _extract_profession,
            "style": _extract_style,
        }
        for key, extractor in extractors.items():
            value = extractor(sentence)
            if value:
                put(key, value, sentence)

        for index, pattern in enumerate(DRINK_PATTERNS):
            match = pattern.search(sentence)
            if match:
                put("drink", match.group(1).strip(), sentence, None if index == 0 else CASUAL_DRINK_CONFIDENCE)
                break
        food = FOOD_PATTERN.search(sentence)
        if food:
            put("food", food.group(1).strip(), sentence)
        pet = PET_PATTERN.search(sentence)
        if pet:
            put("pet", f"{pet.group(1)} tên {pet.group(2)}" if pet.group(2) else pet.group(1), sentence)

        found = _extract_interests(sentence)
        if found:
            interests.extend(found)
            interest_confidence = max(interest_confidence, _sentence_confidence(sentence, "interests", BASE_CONFIDENCE["interests"]))

    if interests:
        candidates["interests"] = (merge_list("", ", ".join(interests)), interest_confidence)
    return candidates


def extract_profile_updates(message: str, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict[str, str]:
    """Stable profile facts whose confidence reaches `min_confidence`."""

    return {
        key: value
        for key, (value, confidence) in extract_profile_candidates(message).items()
        if confidence >= min_confidence
    }


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------

SUMMARY_HEADER = "Tóm tắt hội thoại trước:"
BRIEF_MAX_CHARS = 160


def _brief(message: dict[str, str]) -> str:
    """One bullet per message, preferring the sentence that carries a profile fact."""

    sentences = split_sentences(message.get("content", ""))
    if not sentences:
        return ""
    chosen = next((s for s in sentences if not is_question(s) and extract_profile_updates(s)), sentences[0])
    if len(chosen) > BRIEF_MAX_CHARS:
        chosen = chosen[: BRIEF_MAX_CHARS - 3].rstrip() + "..."
    return f"- {message.get('role', 'user')}: {chosen}"


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages.

    A previous summary can be passed in as a message with role `summary`; its
    bullets are carried over so repeated compactions stay bounded. Bullets that
    contain profile facts are kept first, the rest are filled by recency.
    """

    bullets: list[str] = []
    for message in messages:
        if message.get("role") == "summary":
            bullets.extend(line for line in message.get("content", "").splitlines() if line.startswith("- "))
        else:
            brief = _brief(message)
            if brief:
                bullets.append(brief)
    if not bullets:
        return ""

    def has_fact(bullet: str) -> bool:
        return bullet.startswith("- user:") and bool(extract_profile_updates(bullet.split(":", 1)[1]))

    fact_idx = [i for i, b in enumerate(bullets) if has_fact(b)]
    other_idx = [i for i, b in enumerate(bullets) if i not in set(fact_idx)]
    keep = set(fact_idx[-max_items:])
    for i in reversed(other_idx):
        if len(keep) >= max_items:
            break
        keep.add(i)
    return "\n".join([SUMMARY_HEADER, *(bullets[i] for i in sorted(keep))])


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - Keeps the most recent `keep_messages` messages in full.
    - When summary + messages exceed `threshold_tokens`, older messages are
      folded into a bounded heuristic summary.
    - Counts compactions per thread for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.context(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self.total_tokens(thread_id) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages:]
        previous = [{"role": "summary", "content": thread["summary"]}] if thread["summary"] else []
        thread["summary"] = summarize_messages(previous + older)
        thread["messages"] = recent
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def total_tokens(self, thread_id: str) -> int:
        thread = self.context(thread_id)
        return estimate_tokens(thread["summary"]) + sum(estimate_tokens(m["content"]) for m in thread["messages"])

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]


# ---------------------------------------------------------------------------
# Deterministic offline answers (shared by both agents for a fair comparison)
# ---------------------------------------------------------------------------

# (fact key, label, question keywords)
RECALL_INTENTS = (
    ("name", "Tên", ("tên", "là ai")),
    ("profession", "Nghề nghiệp hiện tại", ("nghề", "làm gì", "công việc")),
    ("location", "Nơi ở hiện tại", ("ở đâu", "nơi ở", "còn ở")),
    ("drink", "Đồ uống yêu thích", ("đồ uống", "uống gì")),
    ("food", "Món ăn yêu thích", ("món ăn", "ăn gì")),
    ("pet", "Thú cưng", ("nuôi", "con gì")),
    ("interests", "Mối quan tâm chính", ("quan tâm", "là ai", "sở thích")),
    ("style", "Style trả lời", ("style", "kiểu trả lời", "trả lời như thế nào", "trả lời mình thích")),
)


def is_question_message(message: str) -> bool:
    return any(is_question(sentence) for sentence in split_sentences(message))


def answer_from_facts(question: str, facts: dict[str, str]) -> str:
    """Answer a recall question using only the facts the caller can see."""

    lowered = question.lower()
    requested = [(key, label) for key, label, keywords in RECALL_INTENTS if any(k in lowered for k in keywords)]
    if not requested:
        requested = [(key, label) for key, label, _ in RECALL_INTENTS if key in facts]
    if not requested or not any(key in facts for key, _ in requested):
        return "Mình chưa có thông tin này trong ngữ cảnh hiện tại."
    lines = [f"- {label}: {facts.get(key, 'chưa có thông tin')}" for key, label in requested]
    return "\n".join(lines)


def acknowledge(updates: dict[str, str]) -> str:
    """Short deterministic reply for non-question turns."""

    if updates:
        return "Đã ghi nhận: " + "; ".join(f"{key} = {value}" for key, value in updates.items()) + "."
    return "Đã ghi nhận, mình sẽ dùng ngữ cảnh này khi trả lời."
