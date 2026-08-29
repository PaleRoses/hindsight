import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM


class StructuredReply(BaseModel):
    value: str


def _llm(reasoning_effort: str | None) -> AnthropicLLM:
    llm = AnthropicLLM(
        provider="anthropic",
        api_key="test-key",
        base_url="",
        model="claude-sonnet-4-20250514",
        reasoning_effort=reasoning_effort,
    )
    llm._client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=2, cache_read_input_tokens=0),
            stop_reason="end_turn",
        )
    )
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("effort", "budget"),
    [("low", 4096), ("medium", 8192), ("high", 16384), ("xhigh", 32768), ("max", 32768)],
)
async def test_plain_call_maps_reasoning_effort_to_native_thinking_budget(effort: str, budget: int) -> None:
    llm = _llm(effort)

    assert await llm.call(
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=512,
        max_retries=0,
    ) == "ok"

    request = llm._client.messages.create.call_args.kwargs
    assert request["thinking"] == {"type": "enabled", "budget_tokens": budget}
    assert request["max_tokens"] == 512 + budget


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", [None, "none"])
async def test_unset_or_none_effort_keeps_native_thinking_off(effort: str | None) -> None:
    llm = _llm(effort)

    await llm.call(
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=512,
        max_retries=0,
    )

    request = llm._client.messages.create.call_args.kwargs
    assert "thinking" not in request
    assert request["max_tokens"] == 512


@pytest.mark.asyncio
async def test_verification_call_stays_minimal_when_thinking_is_configured() -> None:
    llm = _llm("high")

    await llm.call(
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=64,
        scope="verification",
        max_retries=0,
    )

    request = llm._client.messages.create.call_args.kwargs
    assert "thinking" not in request
    assert request["max_tokens"] == 64


@pytest.mark.asyncio
async def test_forced_structured_output_omits_incompatible_thinking() -> None:
    llm = _llm("high")
    llm._client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="structured_response", input={"value": "ok"})],
            usage=SimpleNamespace(input_tokens=5, output_tokens=2, cache_read_input_tokens=0),
            stop_reason="tool_use",
        )
    )

    result = await llm.call(
        messages=[{"role": "user", "content": "hello"}],
        response_format=StructuredReply,
        strict_schema=True,
        max_completion_tokens=512,
        max_retries=0,
    )

    assert result == StructuredReply(value="ok")
    request = llm._client.messages.create.call_args.kwargs
    assert "thinking" not in request
    assert request["max_tokens"] == 512


def test_unknown_effort_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _llm("unbounded")

    assert any("reasoning_effort" in record.message and "ignored" in record.message for record in caplog.records)
