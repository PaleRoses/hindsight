import pytest

from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.llm_provider_metadata import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDER_METADATA,
    OPENAI_COMPATIBLE_PROVIDERS,
    PROVIDER_DEFAULT_MODELS,
    SUPPORTED_LLM_PROVIDERS,
    LLMProviderFactory,
    get_default_model_for_provider,
    get_llm_provider_metadata,
    requires_api_key,
)

EXPECTED_PROVIDERS = {
    "anthropic",
    "atlas",
    "bedrock",
    "claude-code",
    "deepseek",
    "fireworks",
    "gemini",
    "github-copilot",
    "groq",
    "litellm",
    "litellmrouter",
    "llamacpp",
    "lmstudio",
    "minimax",
    "mock",
    "none",
    "nous",
    "ollama",
    "ollama-cloud",
    "openai",
    "openai-codex",
    "openai-responses",
    "opencode-go",
    "openrouter",
    "requesty",
    "vertexai",
    "volcano",
    "xai-oauth",
    "zai",
}

EXPECTED_OPENAI_COMPATIBLE = {
    "atlas",
    "deepseek",
    "fireworks",
    "groq",
    "llamacpp",
    "lmstudio",
    "minimax",
    "ollama",
    "ollama-cloud",
    "openai",
    "opencode-go",
    "openrouter",
    "requesty",
    "volcano",
    "zai",
}

EXPECTED_WITHOUT_API_KEY = {
    "bedrock",
    "claude-code",
    "github-copilot",
    "litellm",
    "litellmrouter",
    "llamacpp",
    "lmstudio",
    "mock",
    "none",
    "nous",
    "ollama",
    "openai-codex",
    "vertexai",
    "xai-oauth",
}


def test_registry_is_total_over_the_builtin_provider_vocabulary() -> None:
    assert set(LLM_PROVIDER_METADATA) == EXPECTED_PROVIDERS
    assert SUPPORTED_LLM_PROVIDERS == EXPECTED_PROVIDERS
    assert set(PROVIDER_DEFAULT_MODELS) == EXPECTED_PROVIDERS
    assert {metadata.provider_id for metadata in LLM_PROVIDER_METADATA.values()} == EXPECTED_PROVIDERS
    assert {metadata.factory for metadata in LLM_PROVIDER_METADATA.values()} == set(LLMProviderFactory)
    assert all(metadata.default_model for metadata in LLM_PROVIDER_METADATA.values())
    assert DEFAULT_LLM_PROVIDER in EXPECTED_PROVIDERS
    assert DEFAULT_LLM_MODEL == LLM_PROVIDER_METADATA[DEFAULT_LLM_PROVIDER].default_model


def test_derived_capability_and_api_key_views_match_the_provider_contract() -> None:
    assert set(OPENAI_COMPATIBLE_PROVIDERS) == EXPECTED_OPENAI_COMPATIBLE
    assert {provider for provider in EXPECTED_PROVIDERS if not requires_api_key(provider)} == EXPECTED_WITHOUT_API_KEY


def test_config_reexports_the_same_derived_default_model_view() -> None:
    from hindsight_api.config import PROVIDER_DEFAULT_MODELS as config_default_models

    assert config_default_models is PROVIDER_DEFAULT_MODELS


def test_lookup_normalizes_case_and_rejects_unknown_ids() -> None:
    assert get_llm_provider_metadata("AnThRoPiC") is LLM_PROVIDER_METADATA["anthropic"]
    assert get_default_model_for_provider("GEMINI") == "gemini-3.5-flash"

    with pytest.raises(ValueError, match="Unknown LLM provider 'missing'"):
        get_llm_provider_metadata("missing")
    with pytest.raises(ValueError, match="Unknown LLM provider 'missing'"):
        get_default_model_for_provider("missing")
    with pytest.raises(ValueError, match="Unknown LLM provider 'missing'"):
        requires_api_key("missing")
    with pytest.raises(ValueError, match="Unknown LLM provider 'missing'"):
        create_llm_provider("missing", "", "", "model", None)


@pytest.mark.parametrize("provider", sorted(EXPECTED_OPENAI_COMPATIBLE))
def test_every_compatible_descriptor_is_accepted_by_the_compatible_constructor(provider: str) -> None:
    metadata = get_llm_provider_metadata(provider)
    llm = OpenAICompatibleLLM(
        provider=provider,
        api_key="test-key" if metadata.requires_api_key else "",
        base_url="",
        model=metadata.default_model,
    )

    assert llm.provider == provider
    assert llm.base_url == metadata.default_base_url
    assert bool(llm.api_key)


@pytest.mark.parametrize("provider", sorted(EXPECTED_OPENAI_COMPATIBLE - EXPECTED_WITHOUT_API_KEY))
def test_every_compatible_cloud_provider_requires_its_key(provider: str) -> None:
    metadata = get_llm_provider_metadata(provider)

    with pytest.raises(ValueError, match=f"API key is required for {provider}"):
        OpenAICompatibleLLM(provider=provider, api_key="", base_url="", model=metadata.default_model)
