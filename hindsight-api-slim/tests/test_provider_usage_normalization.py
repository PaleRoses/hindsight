from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.gemini_llm import GeminiLLM
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


@pytest.mark.asyncio
async def test_openai_call_normalizes_sparse_sdk_usage_before_metrics() -> None:
    llm = OpenAICompatibleLLM(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
    )
    usage = SimpleNamespace(
        prompt_tokens=MagicMock(),
        completion_tokens=None,
        total_tokens=MagicMock(),
        prompt_tokens_details=SimpleNamespace(cached_tokens=MagicMock()),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=MagicMock()),
    )
    response = SimpleNamespace(
        usage=usage,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
            )
        ],
    )
    llm._client.chat.completions.create = AsyncMock(return_value=response)
    metrics = MagicMock()

    with patch(
        "hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector",
        return_value=metrics,
    ):
        assert await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0) == "ok"

    recorded = metrics.record_llm_call.call_args.kwargs
    assert recorded["input_tokens"] == 0
    assert recorded["output_tokens"] == 0
    assert recorded["cached_input_tokens"] == 0
    assert recorded["thoughts_tokens"] == 0


@pytest.mark.asyncio
async def test_gemini_call_normalizes_sparse_sdk_usage_before_metrics() -> None:
    llm = GeminiLLM(
        provider="gemini",
        api_key="test-key",
        base_url="",
        model="gemini-test",
    )
    response = SimpleNamespace(
        text="ok",
        usage_metadata=SimpleNamespace(
            prompt_token_count=MagicMock(),
            candidates_token_count=None,
            cached_content_token_count=MagicMock(),
            thoughts_token_count=MagicMock(),
        ),
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    llm._client = MagicMock()
    llm._client.aio.models.generate_content = AsyncMock(return_value=response)
    metrics = MagicMock()

    with patch(
        "hindsight_api.engine.providers.gemini_llm.get_metrics_collector",
        return_value=metrics,
    ):
        assert await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0) == "ok"

    recorded = metrics.record_llm_call.call_args.kwargs
    assert recorded["input_tokens"] == 0
    assert recorded["output_tokens"] == 0
    assert recorded["cached_input_tokens"] == 0
    assert recorded["thoughts_tokens"] == 0
