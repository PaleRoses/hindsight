"""Canonical metadata for every built-in LLM provider.

This module has no engine or configuration dependencies. Configuration, provider
construction, and direct OpenAI-compatible construction therefore share one
closed provider vocabulary and one policy table.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class LLMProviderFactory(str, Enum):
    """Provider implementation selected by ``create_llm_provider``."""

    OPENAI_COMPATIBLE = "openai-compatible"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI_CODEX = "openai-codex"
    CLAUDE_CODE = "claude-code"
    GITHUB_COPILOT = "github-copilot"
    MOCK = "mock"
    NONE = "none"
    LITELLM = "litellm"
    LITELLM_ROUTER = "litellm-router"
    BEDROCK = "bedrock"
    LLAMACPP = "llamacpp"
    FIREWORKS = "fireworks"
    NOUS = "nous"
    XAI_OAUTH = "xai-oauth"


_OPENAI_COMPATIBLE_FACTORIES = frozenset(
    {
        LLMProviderFactory.OPENAI_COMPATIBLE,
        LLMProviderFactory.LLAMACPP,
        LLMProviderFactory.FIREWORKS,
    }
)


@dataclass(frozen=True)
class LLMProviderMetadata:
    """Static policy shared by configuration and provider construction."""

    provider_id: str
    factory: LLMProviderFactory
    requires_api_key: bool
    default_model: str
    default_base_url: str = ""

    @property
    def openai_compatible(self) -> bool:
        """Whether ``OpenAICompatibleLLM`` can speak to this provider directly."""

        return self.factory in _OPENAI_COMPATIBLE_FACTORIES


_PROVIDER_METADATA = (
    LLMProviderMetadata("openai", LLMProviderFactory.OPENAI_COMPATIBLE, True, "gpt-4o-mini"),
    LLMProviderMetadata("openai-responses", LLMProviderFactory.OPENAI_RESPONSES, True, "gpt-5.6"),
    LLMProviderMetadata("anthropic", LLMProviderFactory.ANTHROPIC, True, "claude-haiku-4-5"),
    LLMProviderMetadata("gemini", LLMProviderFactory.GEMINI, True, "gemini-3.5-flash"),
    LLMProviderMetadata(
        "groq",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "openai/gpt-oss-120b",
        "https://api.groq.com/openai/v1",
    ),
    LLMProviderMetadata(
        "minimax",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "MiniMax-M3",
        "https://api.minimax.io/v1",
    ),
    LLMProviderMetadata(
        "deepseek",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "deepseek-v4-flash",
        "https://api.deepseek.com",
    ),
    LLMProviderMetadata(
        "zai",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "glm-4.5-flash",
        "https://api.z.ai/api/coding/paas/v4",
    ),
    LLMProviderMetadata(
        "opencode-go",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "deepseek-v4-flash",
        "https://opencode.ai/zen/go/v1",
    ),
    LLMProviderMetadata(
        "atlas",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "deepseek-ai/deepseek-v4-pro",
        "https://api.atlascloud.ai/v1",
    ),
    LLMProviderMetadata(
        "ollama",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        False,
        "gemma3:12b",
        "http://localhost:11434/v1",
    ),
    LLMProviderMetadata(
        "ollama-cloud",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "gemma3:12b",
        "https://ollama.com/v1",
    ),
    LLMProviderMetadata("llamacpp", LLMProviderFactory.LLAMACPP, False, "gemma-4-e2b-it"),
    LLMProviderMetadata(
        "lmstudio",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        False,
        "local-model",
        "http://localhost:1234/v1",
    ),
    LLMProviderMetadata("vertexai", LLMProviderFactory.GEMINI, False, "google/gemini-3.1-flash-lite"),
    LLMProviderMetadata(
        "openai-codex",
        LLMProviderFactory.OPENAI_CODEX,
        False,
        "gpt-5.4-mini",
        "https://chatgpt.com/backend-api",
    ),
    LLMProviderMetadata("claude-code", LLMProviderFactory.CLAUDE_CODE, False, "claude-sonnet-4-5-20250929"),
    LLMProviderMetadata("github-copilot", LLMProviderFactory.GITHUB_COPILOT, False, "gpt-5.6-terra"),
    LLMProviderMetadata("mock", LLMProviderFactory.MOCK, False, "mock-model"),
    LLMProviderMetadata("none", LLMProviderFactory.NONE, False, "none"),
    LLMProviderMetadata("litellm", LLMProviderFactory.LITELLM, False, "gpt-4o-mini"),
    LLMProviderMetadata("litellmrouter", LLMProviderFactory.LITELLM_ROUTER, False, "gpt-4o-mini"),
    LLMProviderMetadata("bedrock", LLMProviderFactory.BEDROCK, False, "us.amazon.nova-2-lite-v1:0"),
    LLMProviderMetadata("volcano", LLMProviderFactory.OPENAI_COMPATIBLE, True, "doubao-pro-32k"),
    LLMProviderMetadata(
        "openrouter",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "qwen/qwen3.5-9b",
        "https://openrouter.ai/api/v1",
    ),
    LLMProviderMetadata(
        "requesty",
        LLMProviderFactory.OPENAI_COMPATIBLE,
        True,
        "openai/gpt-4o-mini",
        "https://router.requesty.ai/v1",
    ),
    LLMProviderMetadata(
        "fireworks",
        LLMProviderFactory.FIREWORKS,
        True,
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "https://api.fireworks.ai/inference/v1",
    ),
    LLMProviderMetadata(
        "nous",
        LLMProviderFactory.NOUS,
        False,
        "deepseek/deepseek-v4-flash",
        "https://inference-api.nousresearch.com/v1",
    ),
    LLMProviderMetadata("xai-oauth", LLMProviderFactory.XAI_OAUTH, False, "grok-4.5"),
)

_PROVIDER_METADATA_BY_ID = {metadata.provider_id: metadata for metadata in _PROVIDER_METADATA}
if len(_PROVIDER_METADATA_BY_ID) != len(_PROVIDER_METADATA):
    raise ValueError("Duplicate provider ID in canonical LLM provider metadata")

LLM_PROVIDER_METADATA: Mapping[str, LLMProviderMetadata] = MappingProxyType(_PROVIDER_METADATA_BY_ID)
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
