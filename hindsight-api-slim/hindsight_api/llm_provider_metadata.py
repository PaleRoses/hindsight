"""Canonical metadata for every LLM provider supported by Hindsight.

This module deliberately has no engine or configuration dependencies so both the
configuration loader and provider factory can enforce the same provider law.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class LLMProviderMetadata:
    """Static policy shared by configuration and provider construction."""

    provider_id: str
    requires_api_key: bool
    default_model: str
    default_base_url: str = ""
    openai_compatible: bool = False


_PROVIDER_METADATA = (
    LLMProviderMetadata("openai", True, "gpt-4o-mini", openai_compatible=True),
    LLMProviderMetadata("openai-responses", True, "gpt-5.6"),
    LLMProviderMetadata("anthropic", True, "claude-haiku-4-5"),
    LLMProviderMetadata("gemini", True, "gemini-3.5-flash"),
    LLMProviderMetadata(
        "groq",
        True,
        "openai/gpt-oss-120b",
        "https://api.groq.com/openai/v1",
        openai_compatible=True,
    ),
    LLMProviderMetadata("minimax", True, "MiniMax-M3", "https://api.minimax.io/v1", openai_compatible=True),
    LLMProviderMetadata("deepseek", True, "deepseek-v4-flash", "https://api.deepseek.com", openai_compatible=True),
    LLMProviderMetadata("zai", True, "glm-4.5-flash", "https://api.z.ai/api/coding/paas/v4", openai_compatible=True),
    LLMProviderMetadata(
        "opencode-go", True, "deepseek-v4-flash", "https://opencode.ai/zen/go/v1", openai_compatible=True
    ),
    LLMProviderMetadata(
        "atlas", True, "deepseek-ai/deepseek-v4-pro", "https://api.atlascloud.ai/v1", openai_compatible=True
    ),
    LLMProviderMetadata("ollama", False, "gemma3:12b", "http://localhost:11434/v1", openai_compatible=True),
    LLMProviderMetadata("ollama-cloud", True, "gemma3:12b", "https://ollama.com/v1", openai_compatible=True),
    LLMProviderMetadata("llamacpp", False, "gemma-4-e2b-it", openai_compatible=True),
    LLMProviderMetadata("lmstudio", False, "local-model", "http://localhost:1234/v1", openai_compatible=True),
    LLMProviderMetadata("vertexai", False, "google/gemini-3.1-flash-lite"),
    LLMProviderMetadata("openai-codex", False, "gpt-5.4-mini", "https://chatgpt.com/backend-api"),
    LLMProviderMetadata("claude-code", False, "claude-sonnet-4-5-20250929"),
    LLMProviderMetadata("github-copilot", False, "gpt-5.6-terra"),
    LLMProviderMetadata("mock", False, "mock-model"),
    LLMProviderMetadata("none", False, "none"),
    LLMProviderMetadata("litellm", False, "gpt-4o-mini"),
    LLMProviderMetadata("litellmrouter", False, "gpt-4o-mini"),
    LLMProviderMetadata("bedrock", False, "us.amazon.nova-2-lite-v1:0"),
    LLMProviderMetadata("volcano", True, "doubao-pro-32k", openai_compatible=True),
    LLMProviderMetadata("openrouter", True, "qwen/qwen3.5-9b", "https://openrouter.ai/api/v1", openai_compatible=True),
    LLMProviderMetadata(
        "requesty", True, "openai/gpt-4o-mini", "https://router.requesty.ai/v1", openai_compatible=True
    ),
    LLMProviderMetadata(
        "fireworks",
        True,
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "https://api.fireworks.ai/inference/v1",
        openai_compatible=True,
    ),
    LLMProviderMetadata(
        "nous",
        False,
        "deepseek/deepseek-v4-flash",
        "https://inference-api.nousresearch.com/v1",
    ),
    LLMProviderMetadata("xai-oauth", False, "grok-4.5"),
)

LLM_PROVIDER_METADATA: Mapping[str, LLMProviderMetadata] = MappingProxyType(
    {metadata.provider_id: metadata for metadata in _PROVIDER_METADATA}
)
SUPPORTED_LLM_PROVIDERS = frozenset(LLM_PROVIDER_METADATA)
OPENAI_COMPATIBLE_PROVIDERS = tuple(
    provider_id for provider_id, metadata in LLM_PROVIDER_METADATA.items() if metadata.openai_compatible
)
PROVIDER_DEFAULT_MODELS: Mapping[str, str] = MappingProxyType(
    {provider_id: metadata.default_model for provider_id, metadata in LLM_PROVIDER_METADATA.items()}
)
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = LLM_PROVIDER_METADATA[DEFAULT_LLM_PROVIDER].default_model


def get_llm_provider_metadata(provider: str) -> LLMProviderMetadata:
    """Return canonical metadata for ``provider`` or reject an unknown ID."""

    provider_id = provider.lower()
    try:
        return LLM_PROVIDER_METADATA[provider_id]
    except KeyError:
        supported = ", ".join(LLM_PROVIDER_METADATA)
        raise ValueError(f"Unknown LLM provider {provider_id!r}. Must be one of: {supported}") from None


def get_default_model_for_provider(provider: str) -> str:
    """Return the provider's canonical default model."""

    return get_llm_provider_metadata(provider).default_model


def requires_api_key(provider: str) -> bool:
    """Return whether the provider requires a top-level API key."""

    return get_llm_provider_metadata(provider).requires_api_key
