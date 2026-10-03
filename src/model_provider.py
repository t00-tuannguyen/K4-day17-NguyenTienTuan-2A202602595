from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai_compatible": "custom",
    "openai-compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open_router": "openrouter",
    "open-router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases / typos like `anthorpic` -> `anthropic`."""

    key = (value or "").strip().lower().replace(" ", "_")
    if key not in PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider {value!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}")
    return PROVIDER_ALIASES[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Provider SDKs are imported lazily so offline mode works without them.
    """

    provider = normalize_provider(config.provider)
    kwargs = {"model": config.model_name, "temperature": config.temperature}

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        if provider == "custom" and not config.base_url:
            raise ValueError("Provider 'custom' requires a base_url (OpenAI-compatible endpoint).")
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    # provider == "openrouter"
    from langchain_openrouter import ChatOpenRouter

    if config.api_key:
        kwargs["api_key"] = config.api_key
    return ChatOpenRouter(**kwargs)
