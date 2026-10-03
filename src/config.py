from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Env var holding the API key / base URL for each provider.
API_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "custom": ("CUSTOM_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    "openrouter": ("OPENROUTER_API_KEY",),
}
BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "custom": "CUSTOM_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
}

DEFAULT_COMPACT_THRESHOLD_TOKENS = 800
DEFAULT_COMPACT_KEEP_MESSAGES = 4
DEFAULT_MIN_FACT_CONFIDENCE = 0.6
DEFAULT_FACT_HALF_LIFE_TURNS = 200.0


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths for the repo root, dataset directory, and state directory.
    - Compact-memory settings (threshold and number of messages to keep).
    - Provider settings for the main model and the judge model.
    - `live_mode`: True only when LAB_MODE=live and the provider has credentials;
      otherwise agents use the deterministic offline path.
    - Bonus memory guardrails: minimum confidence before a fact is written to
      `User.md`, and the half-life (in user turns) used for memory decay.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    live_mode: bool = False
    min_fact_confidence: float = DEFAULT_MIN_FACT_CONFIDENCE
    fact_half_life_turns: float = DEFAULT_FACT_HALF_LIFE_TURNS


def _load_dotenv(root: Path) -> None:
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Real environment variables win over `.env`.
    load_dotenv(env_file, override=False)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Build a ProviderConfig from `<prefix>_PROVIDER`, `<prefix>_MODEL`, `<prefix>_TEMPERATURE`."""

    raw_provider = os.getenv(f"{prefix}_PROVIDER", "").strip()
    if not raw_provider and fallback is not None:
        provider = fallback.provider
    else:
        provider = normalize_provider(raw_provider or "openai")

    default_model = fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider]
    model_name = os.getenv(f"{prefix}_MODEL", "").strip() or default_model
    temperature = _env_float(f"{prefix}_TEMPERATURE", 0.0)

    api_key = next((os.getenv(name) for name in API_KEY_ENV[provider] if os.getenv(name)), None)
    base_url = os.getenv(BASE_URL_ENV[provider]) if provider in BASE_URL_ENV else None
    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def _has_credentials(config: ProviderConfig) -> bool:
    if config.provider == "ollama":
        return True
    if config.provider == "custom":
        return bool(config.base_url)
    return bool(config.api_key)


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables (and `.env`) and return a LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    live_requested = os.getenv("LAB_MODE", "offline").strip().lower() == "live"

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=max(1, _env_int("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS)),
        compact_keep_messages=max(1, _env_int("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES)),
        model=model,
        judge_model=judge_model,
        live_mode=live_requested and _has_credentials(model),
        min_fact_confidence=min(1.0, max(0.0, _env_float("MEMORY_MIN_CONFIDENCE", DEFAULT_MIN_FACT_CONFIDENCE))),
        fact_half_life_turns=max(1.0, _env_float("MEMORY_HALF_LIFE_TURNS", DEFAULT_FACT_HALF_LIFE_TURNS)),
    )
