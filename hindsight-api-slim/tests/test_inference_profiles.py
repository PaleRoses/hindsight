import asyncio
import json
import logging
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from hindsight_api.banner import print_startup_info
from hindsight_api.config import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TIMEOUT,
    PROVIDER_DEFAULT_MODELS,
    HindsightConfig,
    clear_config_cache,
    load_inference_profiles,
)
from hindsight_api.config_resolver import ConfigResolver
from hindsight_api.engine import llm_wrapper
from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.llm_provider_metadata import LLM_PROVIDER_METADATA, OPENAI_COMPATIBLE_PROVIDERS
from hindsight_api.main import _startup_llm_identity
from hindsight_api.models import RequestContext

OPERATIONS = ("default", "retain", "consolidation", "reflect")


class _DummyEmbeddings(Embeddings):
    @property
    def provider_name(self) -> str:
        return "dummy"

    @property
    def dimension(self) -> int:
        return 1

    async def initialize(self) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _DummyCrossEncoder(CrossEncoderModel):
    @property
    def provider_name(self) -> str:
        return "dummy"

    async def initialize(self) -> None:
        pass

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.0] * len(pairs)


def _make_memory(*, skip_llm_verification: bool = True, **kwargs) -> MemoryEngine:
    return MemoryEngine(
        db_url="postgresql://localhost/hindsight-profile-test",
        embeddings=_DummyEmbeddings(),
        cross_encoder=_DummyCrossEncoder(),
        run_migrations=False,
        skip_llm_verification=skip_llm_verification,
        **kwargs,
    )


def _profile(model_prefix: str, **route_overrides: object) -> dict[str, dict[str, object]]:
    return {
        operation: {
            "provider": "mock",
            "model": f"{model_prefix}-{operation}",
            **route_overrides,
        }
        for operation in OPERATIONS
    }


def _write_registry(tmp_path, registry: object):
    path = tmp_path / "inference-profiles.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


class _FakeBankConfigConnection:
    def __init__(self, backend: "_FakeBankConfigBackend") -> None:
        self.backend = backend

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def fetchrow(self, query, bank_id):
        self.backend.fetchrow_count += 1
        return {"config": dict(self.backend.config_by_bank.get(bank_id, {}))}

    async def fetch(self, query, bank_ids):
        return [
            {"bank_id": bank_id, "config": dict(self.backend.config_by_bank.get(bank_id, {}))} for bank_id in bank_ids
        ]

    async def execute(self, query, updates_json, bank_id):
        self.backend.config_by_bank.setdefault(bank_id, {}).update(json.loads(updates_json))
        return "UPDATE 1"


class _FakeBankConfigBackend:
    def __init__(self, config_by_bank: dict[str, dict[str, object]] | None = None) -> None:
        self.config_by_bank = config_by_bank or {}
        self.fetchrow_count = 0

    def acquire(self):
        return _FakeBankConfigConnection(self)


@pytest.fixture(autouse=True)
def _isolated_inference_environment(monkeypatch):
    for name in (
        "HINDSIGHT_API_INFERENCE_PROFILES_FILE",
        "HINDSIGHT_API_INFERENCE_PROFILE",
        "HINDSIGHT_API_LLM_API_KEY",
        "HINDSIGHT_API_LLM_STRATEGY",
        "HINDSIGHT_API_LLM_1_PROVIDER",
        "HINDSIGHT_API_LLM_1_API_KEY",
        "HINDSIGHT_API_LLM_1_MODEL",
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER",
        "HINDSIGHT_API_RETAIN_LLM_MODEL",
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER",
        "HINDSIGHT_API_REFLECT_LLM_MODEL",
        "HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MODEL",
        "PROFILE_API_KEY",
        "HINDSIGHT_API_LLM_EXTRA_BODY",
        "HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG",
        "HINDSIGHT_API_LLM_MAX_RETRIES",
        "HINDSIGHT_API_LLM_DEFAULT_HEADERS",
        "HINDSIGHT_API_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_LLM_TIMEOUT",
        "HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES",
        "HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY",
        "HINDSIGHT_API_CONSOLIDATION_LLM_LITELLMROUTER_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "legacy-default")
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.mark.parametrize(
    ("registry", "match"),
    [
        ([], "root must be a JSON object"),
        ({"Bad-Id": _profile("bad")}, "Invalid inference profile id"),
        ({"incomplete": {"default": {"provider": "mock", "model": "one"}}}, "missing operations"),
        (
            {"unknown-operation": {**_profile("route"), "summarize": {"provider": "mock", "model": "x"}}},
            "unknown operations",
        ),
        (
            {"unknown-field": {**_profile("route"), "default": {"provider": "mock", "model": "x", "secret": "x"}}},
            "unknown fields",
        ),
        (
            {"missing-field": {**_profile("route"), "default": {"provider": "mock"}}},
            "missing required fields",
        ),
        (
            {"bad-provider": {**_profile("route"), "default": {"provider": "invented", "model": "x"}}},
            "unsupported provider",
        ),
        (
            {"bad-timeout": {**_profile("route"), "default": {"provider": "mock", "model": "x", "timeout": 0}}},
            "must be positive",
        ),
        (
            {
                "bad-effort": {
                    **_profile("route"),
                    "default": {"provider": "mock", "model": "x", "reasoning_effort": "impossible"},
                }
            },
            "unsupported reasoning_effort",
        ),
        (
            {
                "bad-tier": {
                    **_profile("route"),
                    "default": {"provider": "mock", "model": "x", "openai_service_tier": "flex"},
                }
            },
            "unsupported openai_service_tier",
        ),
    ],
)
def test_registry_rejects_malformed_profiles(tmp_path, registry, match) -> None:
    path = _write_registry(tmp_path, registry)
    with pytest.raises(ValueError, match=match):
        load_inference_profiles(str(path), environment={})


def test_registry_rejects_invalid_json_duplicate_fields_and_nonfinite_numbers(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="HINDSIGHT_API_INFERENCE_PROFILES_FILE"):
        load_inference_profiles(str(path), environment={})

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"profile": {"default": {}, "default": {}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate field"):
        load_inference_profiles(str(duplicate), environment={})

    for value in (float("nan"), float("inf"), float("-inf")):
        nonfinite = tmp_path / "nonfinite.json"
        registry = {"profile": _profile("route")}
        registry["profile"]["default"]["timeout"] = value
        nonfinite.write_text(json.dumps(registry), encoding="utf-8")
        with pytest.raises(ValueError, match="non-finite JSON number"):
            load_inference_profiles(str(nonfinite), environment={})

    for literal in ("NaN", "Infinity", "-Infinity", "1e400"):
        nested = tmp_path / "nested-nonfinite.json"
        registry = {"profile": _profile("route", extra_body={"nested": [{"value": "NONFINITE"}]})}
        nested.write_text(json.dumps(registry).replace('"NONFINITE"', literal), encoding="utf-8")
        with pytest.raises(ValueError, match="non-finite JSON number"):
            load_inference_profiles(str(nested), environment={})


def test_registry_rejects_unresolved_credential_environment_name(tmp_path) -> None:
    registry = {
        "production": {
            **_profile("route"),
            "default": {
                "provider": "openai",
                "model": "gpt-profile",
                "api_key_env": "MISSING_PROFILE_API_KEY",
            },
        }
    }
    path = _write_registry(tmp_path, registry)
    with pytest.raises(ValueError, match="unresolved credential environment variable"):
        load_inference_profiles(str(path), environment={})


def test_global_selector_rejects_an_unknown_profile(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"production": _profile("production")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "missing")

    with pytest.raises(ValueError, match="Unknown HINDSIGHT_API_INFERENCE_PROFILE selector"):
        HindsightConfig.from_env()


def test_selected_profile_constructs_all_four_operation_routes(tmp_path, monkeypatch) -> None:
    registry = {
        "production": {
            operation: {
                "provider": "mock",
                "model": f"profile-{operation}",
                "reasoning_effort": "high",
                "timeout": index + 1,
                "max_retries": index,
            }
            for index, operation in enumerate(OPERATIONS)
        }
    }
    path = _write_registry(tmp_path, registry)
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "production")
    clear_config_cache()

    memory = _make_memory()

    for index, operation in enumerate(OPERATIONS):
        route = memory._inference_llm(operation)
        assert route.provider == "mock"
        assert route.model == f"profile-{operation}"
        assert route.reasoning_effort == "high"
        assert route.timeout == index + 1
        assert route.max_retries == index


def test_selected_profile_boots_without_any_legacy_llm_identity(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"profile-only": _profile("profile-only")})
    for name in (
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_MODEL",
        "HINDSIGHT_API_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "profile-only")
    clear_config_cache()

    memory = _make_memory()

    assert memory._legacy_llm_config.provider == "none"
    assert memory._legacy_retain_llm_config.provider == "none"
    assert memory._legacy_consolidation_llm_config.provider == "none"
    assert memory._legacy_reflect_llm_config.provider == "none"
    for operation in OPERATIONS:
        route = memory._inference_llm(operation)
        assert route.provider == "mock"
        assert route.model == f"profile-only-{operation}"


@pytest.mark.asyncio
async def test_selected_profile_startup_verifies_all_four_routes(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"profile-only": _profile("profile-only")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "profile-only")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_ENABLED", "false")
    clear_config_cache()

    memory = _make_memory(skip_llm_verification=False)
    verified: list[str] = []
    for operation in OPERATIONS:

        async def verify(selected_operation: str = operation) -> None:
            verified.append(selected_operation)

        monkeypatch.setattr(memory._inference_llm(operation), "verify_connection", verify)

    await memory._verify_llm_connections()

    assert memory._skip_llm_verification is False
    assert verified == list(OPERATIONS)


@pytest.mark.asyncio
async def test_bank_selector_overrides_the_global_selector(tmp_path, monkeypatch) -> None:
    path = _write_registry(
        tmp_path,
        {
            "global": _profile("global"),
            "bank-route": _profile("bank"),
        },
    )
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "global")
    clear_config_cache()

    memory = _make_memory()
    assert memory._reflect_llm_config.model == "global-reflect"

    backend = _FakeBankConfigBackend({"selected-bank": {"inference_profile": "bank-route"}})
    resolver = ConfigResolver(backend=backend)
    resolved = await resolver.resolve_full_config("selected-bank")

    # Config-only reads are pure: they cannot change the engine's ambient route.
    assert resolved.inference_profile == "bank-route"
    assert resolved.inference_profiles is resolver._global_config.inference_profiles
    assert memory._llm_config.model == "global-default"
    assert memory._retain_llm_config.model == "global-retain"
    assert memory._consolidation_llm_config.model == "global-consolidation"
    assert memory._reflect_llm_config.model == "global-reflect"

    memory._config_resolver = resolver
    async with memory._bank_operation_scope("selected-bank"):
        assert memory._llm_config.model == "bank-default"
        assert memory._retain_llm_config.model == "bank-retain"
        assert memory._consolidation_llm_config.model == "bank-consolidation"
        assert memory._reflect_llm_config.model == "bank-reflect"

    assert memory._llm_config.model == "global-default"


def test_unknown_bound_profile_fails_closed_instead_of_using_the_global_route() -> None:
    memory = _make_memory()
    memory._global_inference_profile = "missing-profile"

    with pytest.raises(KeyError, match="missing-profile"):
        memory._inference_llm("retain")


def test_constructor_route_arguments_cannot_override_named_profile(tmp_path, monkeypatch) -> None:
    profile = _profile(
        "profile",
        api_key_env="PROFILE_API_KEY",
        base_url="https://profile.example/v1",
    )
    path = _write_registry(tmp_path, {"production": profile})
    monkeypatch.setenv("PROFILE_API_KEY", "profile-secret")
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "production")
    clear_config_cache()

    memory = _make_memory(
        memory_llm_provider="none",
        memory_llm_api_key="constructor-default-secret",
        memory_llm_model="constructor-default",
        memory_llm_base_url="https://constructor-default.example/v1",
        retain_llm_provider="none",
        retain_llm_api_key="constructor-retain-secret",
        retain_llm_model="constructor-retain",
        retain_llm_base_url="https://constructor-retain.example/v1",
    )

    for operation in OPERATIONS:
        route = memory._inference_llm(operation)
        assert route.provider == "mock"
        assert route.model == f"profile-{operation}"
        assert route.api_key == "profile-secret"
        assert route.base_url == "https://profile.example/v1"


def test_legacy_operation_configuration_remains_active_without_a_profile(monkeypatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_MODEL", "legacy-retain")
    clear_config_cache()

    memory = _make_memory()

    assert memory._llm_config.model == "legacy-default"
    assert memory._retain_llm_config.model == "legacy-retain"
    assert memory._reflect_llm_config.model == "legacy-default"
    assert memory._consolidation_llm_config.model == "legacy-default"


def test_selected_profile_does_not_inherit_legacy_multi_llm_chain(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"production": _profile("profile")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "production")
    monkeypatch.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_1_MODEL", "legacy-fallback")
    monkeypatch.setenv("HINDSIGHT_API_LLM_STRATEGY", '{"mode": "failover"}')
    clear_config_cache()

    memory = _make_memory()

    assert memory._legacy_llm_config.provider == "none"
    for operation in OPERATIONS:
        assert not isinstance(memory._inference_llm(operation), MultiLLMProvider)


def test_selected_profile_ignores_invalid_legacy_route_environment(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"production": _profile("profile")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "production")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "invented")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_PROVIDER", "invented")
    monkeypatch.setenv("HINDSIGHT_API_LLM_EXTRA_BODY", "{broken")
    monkeypatch.setenv("HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG", "{broken")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MAX_RETRIES", "not-an-integer")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES", "not-an-integer")
    monkeypatch.setenv("HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY", "{broken")
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_LLM_LITELLMROUTER_CONFIG", "{broken")
    monkeypatch.setenv("HINDSIGHT_API_LLM_DEFAULT_HEADERS", '{"X-Legacy": "must-not-leak"}')
    monkeypatch.setenv("HINDSIGHT_API_LLM_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("HINDSIGHT_API_LLM_TIMEOUT", "999")
    clear_config_cache()

    config = HindsightConfig.from_env()

    assert config.inference_profile == "production"
    assert config.llm_provider == "none"
    assert config.retain_llm_provider is None

    memory = _make_memory()
    for operation in OPERATIONS:
        route = memory._inference_llm(operation)
        assert route.default_headers is None
        assert route.reasoning_effort is None
        assert route.timeout == DEFAULT_LLM_TIMEOUT
        assert route.max_retries == DEFAULT_LLM_MAX_RETRIES


@pytest.mark.asyncio
async def test_unknown_bank_profile_is_rejected_on_update_and_resolution(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"production": _profile("production")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    clear_config_cache()

    backend = _FakeBankConfigBackend({"stored-bank": {"inference_profile": "missing"}})
    resolver = ConfigResolver(backend=backend)

    with pytest.raises(ValueError, match="Unknown bank 'stored-bank' inference_profile selector"):
        await resolver.resolve_full_config("stored-bank")
    with pytest.raises(ValueError, match="Unknown bank 'updated-bank' inference_profile selector"):
        await resolver.validate_bank_config_updates("updated-bank", {"inference_profile": "missing"})


@pytest.mark.asyncio
async def test_bank_config_exposes_only_selector_and_never_profile_routes_or_secrets(tmp_path, monkeypatch) -> None:
    secret = "profile-secret-that-must-not-leak"
    routes = {
        operation: {
            "provider": "openai",
            "model": f"secret-route-{operation}",
            "api_key_env": "PROFILE_API_KEY",
            "base_url": f"https://{operation}.private.example/v1",
            "default_headers": {"X-Private-Route": operation},
            "extra_body": {"private_route": operation},
        }
        for operation in OPERATIONS
    }
    path = _write_registry(tmp_path, {"production": routes})
    monkeypatch.setenv("PROFILE_API_KEY", secret)
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "production")
    clear_config_cache()

    resolver = ConfigResolver(backend=_FakeBankConfigBackend())
    bank_config = await resolver.get_bank_config("safe-bank")
    serialized = json.dumps(bank_config, sort_keys=True)

    assert bank_config["inference_profile"] == "production"
    assert "inference_profiles" not in bank_config
    assert secret not in serialized
    assert "private.example" not in serialized
    assert "X-Private-Route" not in serialized
    assert "private_route" not in serialized

    with pytest.raises(ValueError, match="credential fields"):
        await resolver.validate_bank_config_updates("safe-bank", {"inference_profiles": {}})


# Provider policy and registry immutability


def test_provider_metadata_drives_legacy_defaults_and_constructor_validation(monkeypatch) -> None:
    constructed: list[dict[str, object]] = []

    def fake_create_provider(**kwargs):
        constructed.append(kwargs)
        return object()

    monkeypatch.setattr(llm_wrapper, "create_llm_provider", fake_create_provider)

    assert DEFAULT_LLM_PROVIDER == "openai"
    assert DEFAULT_LLM_MODEL == LLM_PROVIDER_METADATA[DEFAULT_LLM_PROVIDER].default_model
    assert set(PROVIDER_DEFAULT_MODELS) == set(LLM_PROVIDER_METADATA)

    for provider_id, metadata in LLM_PROVIDER_METADATA.items():
        llm = llm_wrapper.LLMProvider(
            provider=provider_id,
            api_key="test-key",
            base_url="",
            model=metadata.default_model,
            vertexai_project_id="test-project" if provider_id == "vertexai" else None,
        )
        assert PROVIDER_DEFAULT_MODELS[provider_id] == metadata.default_model
        assert llm_wrapper.requires_api_key(provider_id) is metadata.requires_api_key
        assert llm.provider == provider_id
        assert llm.base_url == metadata.default_base_url

    assert len(constructed) == len(LLM_PROVIDER_METADATA)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm_wrapper.LLMProvider(provider="invented", api_key="", base_url="", model="invented")


def test_openai_compatible_constructor_uses_canonical_capability_keys_and_urls(monkeypatch) -> None:
    from hindsight_api.engine.providers import openai_compatible_llm

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(openai_compatible_llm, "AsyncOpenAI", FakeAsyncOpenAI)

    compatible_ids = {
        provider_id for provider_id, metadata in LLM_PROVIDER_METADATA.items() if metadata.openai_compatible
    }
    assert set(OPENAI_COMPATIBLE_PROVIDERS) == compatible_ids

    for provider_id in OPENAI_COMPATIBLE_PROVIDERS:
        metadata = LLM_PROVIDER_METADATA[provider_id]
        llm = openai_compatible_llm.OpenAICompatibleLLM(
            provider=provider_id,
            api_key="test-key",
            base_url="",
            model=metadata.default_model,
        )
        assert llm.base_url == metadata.default_base_url

    local = openai_compatible_llm.OpenAICompatibleLLM(
        provider="llamacpp",
        api_key="",
        base_url="",
        model=LLM_PROVIDER_METADATA["llamacpp"].default_model,
    )
    assert local.api_key == "local"

    with pytest.raises(ValueError, match="OpenAICompatibleLLM only supports"):
        openai_compatible_llm.OpenAICompatibleLLM(
            provider="anthropic",
            api_key="test-key",
            base_url="",
            model=LLM_PROVIDER_METADATA["anthropic"].default_model,
        )
    with pytest.raises(ValueError, match="API key is required for fireworks"):
        openai_compatible_llm.OpenAICompatibleLLM(
            provider="fireworks",
            api_key="",
            base_url="",
            model=LLM_PROVIDER_METADATA["fireworks"].default_model,
        )

    headers = {"X-Profile-Route": "production"}
    routed = llm_wrapper.create_llm_provider(
        provider="openai",
        api_key="test-key",
        base_url="",
        model=LLM_PROVIDER_METADATA["openai"].default_model,
        reasoning_effort="low",
        openai_service_tier="flex",
        default_headers=headers,
    )
    assert routed._client.kwargs["default_headers"] == headers
    assert routed.openai_service_tier == "flex"


def test_anthropic_constructor_receives_profile_transport_options() -> None:
    headers = {"X-Profile-Route": "production"}
    extra_body = {"temperature": 0.2}
    with patch("anthropic.AsyncAnthropic") as client:
        routed = llm_wrapper.create_llm_provider(
            provider="anthropic",
            api_key="test-key",
            base_url="https://anthropic.example",
            model=LLM_PROVIDER_METADATA["anthropic"].default_model,
            reasoning_effort="low",
            timeout=17.5,
            default_headers=headers,
            extra_body=extra_body,
        )

    assert client.call_args.kwargs["timeout"] == 17.5
    assert client.call_args.kwargs["default_headers"] == headers
    assert routed._extra_body == extra_body


def test_anthropic_constructor_preserves_its_timeout_default() -> None:
    with patch("anthropic.AsyncAnthropic") as client:
        llm_wrapper.create_llm_provider(
            provider="anthropic",
            api_key="test-key",
            base_url="",
            model=LLM_PROVIDER_METADATA["anthropic"].default_model,
            reasoning_effort="low",
        )

    assert client.call_args.kwargs["timeout"] == 300.0


def test_gemini_constructor_receives_profile_transport_options(monkeypatch) -> None:
    from hindsight_api.engine.providers import gemini_llm

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(gemini_llm.genai, "Client", FakeClient)
    headers = {"X-Profile-Route": "production"}
    routed = llm_wrapper.create_llm_provider(
        provider="gemini",
        api_key="test-key",
        base_url="https://gemini.example",
        model=LLM_PROVIDER_METADATA["gemini"].default_model,
        reasoning_effort="low",
        timeout=17.5,
        default_headers=headers,
    )

    assert routed._client.kwargs["http_options"] == {
        "base_url": "https://gemini.example",
        "timeout": 17_500,
        "headers": headers,
    }


def test_profile_registry_and_nested_payloads_are_deeply_immutable(tmp_path) -> None:
    path = _write_registry(
        tmp_path,
        {
            "immutable": _profile(
                "immutable",
                extra_body={"nested": {"items": [{"value": "original"}]}},
                default_headers={"X-Route": "original"},
            )
        },
    )
    profiles = load_inference_profiles(str(path), environment={})
    route = profiles["immutable"].default

    with pytest.raises(TypeError):
        profiles["other"] = profiles["immutable"]
    with pytest.raises(FrozenInstanceError):
        route.model = "mutated"
    with pytest.raises(TypeError):
        route.extra_body["nested"]["items"][0]["value"] = "mutated"
    with pytest.raises(TypeError):
        route.extra_body["nested"]["items"][0] = {"value": "mutated"}
    with pytest.raises(TypeError):
        route.default_headers["X-Route"] = "mutated"


def test_litellmrouter_profile_requires_route_config(tmp_path) -> None:
    routes = _profile("fallback")
    routes["default"] = {"provider": "litellmrouter", "model": "default"}
    path = _write_registry(tmp_path, {"router": routes})

    with pytest.raises(ValueError, match="requires a non-empty litellmrouter_config"):
        load_inference_profiles(str(path), environment={})


def test_selected_litellmrouter_profile_is_eagerly_constructed_once(tmp_path, monkeypatch) -> None:
    router_config = {
        "model_list": [
            {
                "model_name": "default",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "router-secret",
                },
            }
        ],
        "num_retries": 0,
    }
    routes = _profile("fallback")
    routes["default"] = {
        "provider": "litellmrouter",
        "model": "default",
        "litellmrouter_config": router_config,
    }
    path = _write_registry(tmp_path, {"router": routes})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "router")
    clear_config_cache()

    class FakeLiteLLMRouter:
        def __init__(self, **kwargs) -> None:
            self.config = kwargs["config"]
            instances.append(self)

    instances: list[FakeLiteLLMRouter] = []

    from hindsight_api.engine import providers

    monkeypatch.setattr(providers, "LiteLLMRouterLLM", FakeLiteLLMRouter)
    memory = _make_memory()

    selected = memory._inference_llm("default")
    assert selected.provider == "litellmrouter"
    assert len(instances) == 1
    assert selected._provider_impl is instances[0]
    assert instances[0].config == router_config

    canonical_config = HindsightConfig.from_env().inference_profiles["router"].default.litellmrouter_config
    with pytest.raises(TypeError):
        canonical_config["model_list"][0]["litellm_params"]["api_key"] = "mutated"
    instances[0].config["model_list"][0]["model_name"] = "mutated-client-copy"
    assert canonical_config["model_list"][0]["model_name"] == "default"
    assert len(instances) == 1


# Engine-owned profile operation context


@pytest.mark.asyncio
async def test_profiled_then_legacy_bank_calls_do_not_leak_selector_state(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"profiled": _profile("profiled")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    clear_config_cache()

    memory = _make_memory()
    backend = _FakeBankConfigBackend({"profiled-bank": {"inference_profile": "profiled"}})
    memory._config_resolver = ConfigResolver(backend=backend)

    async with memory._bank_operation_scope("profiled-bank"):
        assert memory._llm_config.model == "profiled-default"
    assert memory._llm_config.model == "legacy-default"

    async with memory._bank_operation_scope("legacy-bank"):
        assert memory._llm_config.model == "legacy-default"
    assert memory._llm_config.model == "legacy-default"
    assert backend.fetchrow_count == 2


@pytest.mark.asyncio
async def test_nested_and_exceptional_bank_scopes_restore_the_prior_profile(tmp_path, monkeypatch) -> None:
    path = _write_registry(
        tmp_path,
        {
            "outer": _profile("outer"),
            "inner": _profile("inner"),
        },
    )
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    clear_config_cache()

    memory = _make_memory()
    backend = _FakeBankConfigBackend(
        {
            "outer-bank": {"inference_profile": "outer"},
            "inner-bank": {"inference_profile": "inner"},
        }
    )
    memory._config_resolver = ConfigResolver(backend=backend)

    async with memory._bank_operation_scope("outer-bank"):
        assert memory._llm_config.model == "outer-default"
        outer_queries = backend.fetchrow_count
        async with memory._bank_operation_scope("outer-bank"):
            assert memory._llm_config.model == "outer-default"
        assert backend.fetchrow_count == outer_queries

        with pytest.raises(RuntimeError, match="inner failed"):
            async with memory._bank_operation_scope("inner-bank"):
                assert memory._llm_config.model == "inner-default"
                raise RuntimeError("inner failed")
        assert memory._llm_config.model == "outer-default"

    assert memory._llm_config.model == "legacy-default"


@pytest.mark.asyncio
async def test_concurrent_and_child_tasks_keep_profile_context_isolated(tmp_path, monkeypatch) -> None:
    path = _write_registry(
        tmp_path,
        {
            "alpha": _profile("alpha"),
            "beta": _profile("beta"),
        },
    )
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    clear_config_cache()

    memory = _make_memory()
    backend = _FakeBankConfigBackend(
        {
            "alpha-bank": {"inference_profile": "alpha"},
            "beta-bank": {"inference_profile": "beta"},
        }
    )
    memory._config_resolver = ConfigResolver(backend=backend)

    async def observe(bank_id: str) -> tuple[str, str]:
        async with memory._bank_operation_scope(bank_id):
            parent_model = memory._llm_config.model

            async def observe_child() -> str:
                await asyncio.sleep(0)
                return memory._llm_config.model

            child = asyncio.create_task(observe_child())
            await asyncio.sleep(0)
            return parent_model, await child

    alpha, beta, legacy = await asyncio.gather(
        observe("alpha-bank"),
        observe("beta-bank"),
        observe("legacy-bank"),
    )

    assert alpha == ("alpha-default", "alpha-default")
    assert beta == ("beta-default", "beta-default")
    assert legacy == ("legacy-default", "legacy-default")
    assert memory._llm_config.model == "legacy-default"


@pytest.mark.asyncio
async def test_nested_scopes_from_two_engines_cannot_cross_route(tmp_path, monkeypatch) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = _write_registry(
        first_dir,
        {
            "first-global": _profile("first-global"),
            "shared": _profile("first-shared"),
        },
    )
    second_path = _write_registry(
        second_dir,
        {
            "second-global": _profile("second-global"),
            "shared": _profile("second-shared"),
        },
    )

    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(first_path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "first-global")
    clear_config_cache()
    first = _make_memory()

    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(second_path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "second-global")
    clear_config_cache()
    second = _make_memory()

    first._config_resolver = ConfigResolver(
        backend=_FakeBankConfigBackend({"first-bank": {"inference_profile": "shared"}})
    )
    async with first._bank_operation_scope("first-bank"):
        assert first._llm_config.model == "first-shared-default"
        assert second._llm_config.model == "second-global-default"

    assert first._llm_config.model == "first-global-default"
    assert second._llm_config.model == "second-global-default"


@pytest.mark.asyncio
async def test_bank_config_resolution_failure_prevents_provider_use(monkeypatch) -> None:
    class FailingBackend:
        def acquire(self):
            raise RuntimeError("bank config unavailable")

    memory = _make_memory()
    memory._config_resolver = ConfigResolver(backend=FailingBackend())
    provider_calls: list[str] = []

    async def call_provider(_llm):
        provider_calls.append("called")

    with pytest.raises(RuntimeError, match="bank config unavailable"):
        async with memory._bank_operation_scope("bank"):
            await call_provider(memory._retain_llm_config)

    assert provider_calls == []


@pytest.mark.asyncio
async def test_tenant_config_resolution_failure_prevents_provider_use() -> None:
    class FailingTenantConfig:
        async def get_tenant_config(self, _context):
            raise RuntimeError("tenant config unavailable")

    memory = _make_memory()
    memory._config_resolver = ConfigResolver(
        backend=_FakeBankConfigBackend(),
        tenant_extension=FailingTenantConfig(),
    )
    provider_calls: list[str] = []

    with pytest.raises(RuntimeError, match="tenant config unavailable"):
        async with memory._bank_operation_scope(
            "bank",
            RequestContext(tenant_id="tenant"),
        ):
            provider_calls.append(memory._retain_llm_config.model)

    assert provider_calls == []


@pytest.mark.asyncio
async def test_bank_health_uses_one_resolved_profile_query(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"health": _profile("health")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    clear_config_cache()

    memory = _make_memory()
    backend = _FakeBankConfigBackend({"health-bank": {"inference_profile": "health"}})
    memory._config_resolver = ConfigResolver(backend=backend)
    memory._operation_validator = None
    probed_models: list[str] = []

    async def authenticate(_request_context) -> None:
        return None

    class ProbeOutcome:
        ok = True
        status = "connected"
        latency_ms = 0.0

    async def probe(llm):
        probed_models.append(llm.model)
        return ProbeOutcome()

    monkeypatch.setattr(memory, "_authenticate_tenant", authenticate)
    monkeypatch.setattr(memory, "_probe_llm", probe)

    result = await memory.check_bank_llm("health-bank", request_context=RequestContext(internal=True))

    assert result.bank_id == "health-bank"
    assert probed_models == ["health-retain", "health-consolidation", "health-reflect"]
    assert backend.fetchrow_count == 1
    assert memory._llm_config.model == "legacy-default"


@pytest.mark.asyncio
async def test_worker_boundary_binds_and_resets_the_bank_profile(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"worker": _profile("worker")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    clear_config_cache()

    memory = _make_memory()
    backend = _FakeBankConfigBackend({"worker-bank": {"inference_profile": "worker"}})
    memory._config_resolver = ConfigResolver(backend=backend)
    memory._audit_logger = None
    selected_models: list[str] = []

    async def handle_consolidation(_task_dict):
        deep_config = await memory._resolve_full_config(
            "worker-bank",
            RequestContext(internal=True),
        )
        assert deep_config.inference_profile == "worker"
        selected_models.append(memory._consolidation_llm_config.model)
        return {"memories_processed": 0}

    monkeypatch.setattr(memory, "_handle_consolidation", handle_consolidation)

    await memory.execute_task({"type": "consolidation", "bank_id": "worker-bank"})

    assert selected_models == ["worker-consolidation"]
    assert backend.fetchrow_count == 1
    assert memory._llm_config.model == "legacy-default"


@pytest.mark.asyncio
async def test_direct_consolidation_job_scopes_the_bank_profile_once(tmp_path, monkeypatch) -> None:
    path = _write_registry(tmp_path, {"direct": _profile("direct")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    clear_config_cache()

    memory = _make_memory()
    backend = _FakeBankConfigBackend({"direct-bank": {"inference_profile": "direct"}})
    memory._config_resolver = ConfigResolver(backend=backend)
    captured: dict[str, object] = {}

    async def run_inner(
        _memory_engine,
        bank_id,
        _request_context,
        config,
        llm_config,
        _operation_id,
        _observation_scopes,
        _pending_refresh_tags,
    ):
        captured["bank_id"] = bank_id
        captured["profile"] = config.inference_profile
        captured["model"] = llm_config.model
        return {"status": "completed"}

    monkeypatch.setattr(consolidator, "_run_consolidation_job", run_inner)

    result = await consolidator.run_consolidation_job(
        memory,
        "direct-bank",
        RequestContext(internal=True),
    )

    assert result == {"status": "completed"}
    assert captured == {
        "bank_id": "direct-bank",
        "profile": "direct",
        "model": "direct-consolidation",
    }
    assert backend.fetchrow_count == 1
    assert memory._llm_config.model == "legacy-default"


@pytest.mark.asyncio
async def test_bank_selected_consolidation_route_controls_direct_and_worker_execution(
    tmp_path,
    monkeypatch,
) -> None:
    active = _profile("active")
    disabled = _profile("disabled")
    disabled["consolidation"] = {"provider": "none", "model": "none"}
    path = _write_registry(tmp_path, {"active": active, "disabled": disabled})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))

    calls: list[tuple[str, str]] = []

    async def run_inner(
        _memory_engine,
        bank_id,
        _request_context,
        config,
        llm_config,
        _operation_id,
        _observation_scopes,
        _pending_refresh_tags,
    ):
        calls.append((bank_id, llm_config.model))
        assert config.enable_observations is True
        return {"status": "completed", "bank_id": bank_id}

    monkeypatch.setattr(consolidator, "_run_consolidation_job", run_inner)

    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "disabled")
    clear_config_cache()
    globally_disabled = _make_memory()
    globally_disabled._config_resolver = ConfigResolver(
        backend=_FakeBankConfigBackend({"active-bank": {"inference_profile": "active"}})
    )

    direct_active = await consolidator.run_consolidation_job(
        globally_disabled,
        "active-bank",
        RequestContext(internal=True),
    )
    worker_active = await globally_disabled._handle_consolidation({"bank_id": "active-bank"})

    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "active")
    clear_config_cache()
    globally_active = _make_memory()
    globally_active._config_resolver = ConfigResolver(
        backend=_FakeBankConfigBackend({"disabled-bank": {"inference_profile": "disabled"}})
    )

    direct_disabled = await consolidator.run_consolidation_job(
        globally_active,
        "disabled-bank",
        RequestContext(internal=True),
    )
    worker_disabled = await globally_active._handle_consolidation({"bank_id": "disabled-bank"})

    assert direct_active["status"] == "completed"
    assert worker_active["status"] == "completed"
    assert calls == [
        ("active-bank", "active-consolidation"),
        ("active-bank", "active-consolidation"),
    ]
    assert direct_disabled == {"status": "disabled", "bank_id": "disabled-bank"}
    assert worker_disabled == {"status": "disabled", "bank_id": "disabled-bank"}


def test_startup_logs_exclusively_report_selected_profile_routes(tmp_path, monkeypatch, caplog, capsys) -> None:
    path = _write_registry(tmp_path, {"startup": _profile("startup")})
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "startup")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "unreachable-legacy-default")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_MODEL", "unreachable-legacy-retain")
    monkeypatch.setenv("HINDSIGHT_API_REFLECT_LLM_MODEL", "unreachable-legacy-reflect")
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_LLM_MODEL", "unreachable-legacy-consolidation")
    clear_config_cache()

    selected = HindsightConfig.from_env()
    with caplog.at_level(logging.INFO, logger="hindsight_api.config"):
        selected.log_config()
    selected_messages = [record.getMessage() for record in caplog.records]

    inference_messages = [message for message in selected_messages if message.startswith("Inference profile:")]
    assert len(inference_messages) == 1
    assert "Inference profile: startup" in inference_messages[0]
    for operation in OPERATIONS:
        assert f"{operation}=mock/startup-{operation}" in inference_messages[0]
    assert not any(message.startswith("LLM:") or message.startswith("LLM (") for message in selected_messages)
    assert "unreachable-legacy" not in "\n".join(selected_messages)
    banner_provider, banner_model = _startup_llm_identity(selected)
    assert banner_provider == "profile=startup"
    assert banner_model == ", ".join(f"{operation}=mock/startup-{operation}" for operation in OPERATIONS)
    print_startup_info(
        host="127.0.0.1",
        port=8888,
        database_url="postgresql://localhost/hindsight",
        llm_provider=banner_provider,
        llm_model=banner_model,
        embeddings_provider="dummy",
        reranker_provider="dummy",
    )
    banner_output = capsys.readouterr().out
    assert "profile=startup" in banner_output
    for operation in OPERATIONS:
        assert f"{operation}=mock/startup-{operation}" in banner_output
    assert "unreachable-legacy" not in banner_output

    caplog.clear()
    monkeypatch.delenv("HINDSIGHT_API_INFERENCE_PROFILE")
    clear_config_cache()
    legacy = HindsightConfig.from_env()
    with caplog.at_level(logging.INFO, logger="hindsight_api.config"):
        legacy.log_config()
    legacy_messages = [record.getMessage() for record in caplog.records]

    assert "LLM: provider=mock, model=unreachable-legacy-default" in legacy_messages
    assert "LLM (retain): provider=mock, model=unreachable-legacy-retain" in legacy_messages
    assert "LLM (reflect): provider=mock, model=unreachable-legacy-reflect" in legacy_messages
    assert "LLM (consolidation): provider=mock, model=unreachable-legacy-consolidation" in legacy_messages
    assert not any(message.startswith("Inference profile:") for message in legacy_messages)
    assert _startup_llm_identity(legacy) == ("mock", "unreachable-legacy-default")


@pytest.mark.asyncio
async def test_profile_selector_precedence_is_global_then_tenant_then_bank(tmp_path, monkeypatch) -> None:
    path = _write_registry(
        tmp_path,
        {
            "global": _profile("global"),
            "tenant": _profile("tenant"),
            "bank": _profile("bank"),
        },
    )
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILES_FILE", str(path))
    monkeypatch.setenv("HINDSIGHT_API_INFERENCE_PROFILE", "global")
    clear_config_cache()

    class TenantProfiles:
        async def get_tenant_config(self, _context):
            return {"inference_profile": "tenant"}

    resolver = ConfigResolver(
        backend=_FakeBankConfigBackend({"bank-override": {"inference_profile": "bank"}}),
        tenant_extension=TenantProfiles(),
    )
    context = RequestContext(tenant_id="tenant-id")

    tenant_config = await resolver.resolve_full_config("tenant-default", context)
    bank_config = await resolver.resolve_full_config("bank-override", context)

    assert tenant_config.inference_profile == "tenant"
    assert bank_config.inference_profile == "bank"
