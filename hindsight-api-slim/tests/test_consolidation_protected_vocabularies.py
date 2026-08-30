"""Regression tests for consolidation's bank-configured protected-term boundary."""

import types
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.config import (
    ConsolidationProtectedVocabulary,
    HindsightConfig,
    parse_consolidation_protected_vocabularies,
)
from hindsight_api.engine.consolidation.consolidator import (
    ConsolidationPerfLog,
    _BatchLLMResult,
    _compile_protected_vocabularies,
    _consolidate_batch_with_protected_vocabularies,
    _CreateAction,
    _dedup_adjudicate,
    _dedup_reconcile_create,
    _dedup_reconcile_update,
    _DedupDecision,
    _find_protected_vocabulary_violations,
    _protected_vocabulary_conflicts,
    _ProtectedVocabulary,
    _TemporalBounds,
    _UpdateAction,
)
from hindsight_api.engine.response_models import MemoryFact
from hindsight_api.engine.search.retrieval import SemanticBm25Result
from hindsight_api.engine.search.types import RetrievalResult

_ENVIRONMENT_SOURCE_ID = "ef7c88ad-3726-48cc-b1a7-1a78b806d3c0"
_ENVIRONMENT_TARGET_ID = "0b9c5c4b-c6af-4b20-9881-3f40961137c0"
_ENVIRONMENT_SOURCE = "The staging deployment enables the canary scheduler for ten percent of requests."
_ENVIRONMENT_TARGET = "The production deployment keeps the stable scheduler for all requests."
_PRODUCT_SOURCE = "Atlas release 4.2 enables streaming export and records its rollout status."
_PRODUCT_TARGET = "Atlas release 4.1 supports batch export and remains the stable release."
_UNKNOWN_SOURCE = (
    "The benchmark fixture was deleted because compiler time and request latency measure different things."
)
_UNKNOWN_TARGET = "Benchmark guidance requires larger fixtures and isolated processes for comparable latency ratios."


def _vocabularies() -> tuple[_ProtectedVocabulary, ...]:
    return _compile_protected_vocabularies(
        parse_consolidation_protected_vocabularies(
            [
                {
                    "name": "environment",
                    "terms": ["staging", "production"],
                },
                {
                    "name": "product",
                    "terms": [
                        "atlas",
                        "borealis",
                    ],
                },
            ]
        )
    )


def _observation(observation_id: str, text: str) -> MemoryFact:
    return MemoryFact(id=observation_id, text=text, fact_type="observation")


def test_protected_vocabulary_is_validated_and_canonicalized() -> None:
    vocabularies = parse_consolidation_protected_vocabularies(
        [{"name": " Environment ", "terms": ["Staging", " production "]}]
    )

    assert vocabularies[0].name == "Environment"
    assert vocabularies[0].terms == ("staging", "production")


@pytest.mark.parametrize(
    "raw",
    [
        {},
        [{"name": "environment"}],
        [{"name": "environment", "terms": ["staging"]}],
        [{"name": "environment", "terms": ["same", "SAME"]}],
        [
            {"name": "environment", "terms": ["one", "two"]},
            {"name": "ENVIRONMENT", "terms": ["three", "four"]},
        ],
    ],
)
def test_protected_vocabulary_rejects_ambiguous_shapes(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_consolidation_protected_vocabularies(raw)


def test_environment_registry_is_loaded_from_server_config(monkeypatch) -> None:
    monkeypatch.setenv(
        "HINDSIGHT_API_CONSOLIDATION_PROTECTED_VOCABULARIES",
        '[{"name":"environment","terms":["staging","production"]}]',
    )

    config = HindsightConfig.from_env()

    assert config.consolidation_protected_vocabularies == (
        ConsolidationProtectedVocabulary(
            name="environment",
            terms=("staging", "production"),
        ),
    )


def test_cross_environment_twin_is_rejected() -> None:
    conflicts = _protected_vocabulary_conflicts(_ENVIRONMENT_SOURCE, _ENVIRONMENT_TARGET, _vocabularies())

    assert len(conflicts) == 1
    assert conflicts[0].vocabulary == "environment"
    assert conflicts[0].source_terms == ("staging",)
    assert conflicts[0].target_terms == ("production",)


def test_protected_term_matching_requires_identifier_boundaries() -> None:
    source = "staging-old is an unrelated identifier"

    assert _protected_vocabulary_conflicts(source, _ENVIRONMENT_TARGET, _vocabularies()) == ()


def test_same_product_update_remains_lawful() -> None:
    assert _protected_vocabulary_conflicts(_PRODUCT_SOURCE, _PRODUCT_TARGET, _vocabularies()) == ()


def test_unknown_terms_remains_lawful() -> None:
    assert _protected_vocabulary_conflicts(_UNKNOWN_SOURCE, _UNKNOWN_TARGET, _vocabularies()) == ()


def test_update_violation_preserves_source_and_target_evidence() -> None:
    result = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_ENVIRONMENT_TARGET_ID,
                source_fact_ids=[_ENVIRONMENT_SOURCE_ID],
                text="Combined text that must never be written.",
            )
        ]
    )

    violations = _find_protected_vocabulary_violations(
        result,
        [{"id": _ENVIRONMENT_SOURCE_ID, "text": _ENVIRONMENT_SOURCE}],
        [_observation(_ENVIRONMENT_TARGET_ID, _ENVIRONMENT_TARGET)],
        _vocabularies(),
    )

    assert len(violations) == 1
    assert violations[0].source_fact_id == _ENVIRONMENT_SOURCE_ID
    assert violations[0].observation_id == _ENVIRONMENT_TARGET_ID


async def test_guard_retries_without_rejected_target_and_preserves_fact() -> None:
    bad = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_ENVIRONMENT_TARGET_ID,
                source_fact_ids=[_ENVIRONMENT_SOURCE_ID],
                text="Bad cross-environment merge.",
            )
        ],
        obs_count=1,
        prompt_chars=100,
    )
    good = _BatchLLMResult(
        creates=[_CreateAction(text="Standalone staging observation.", source_fact_ids=[_ENVIRONMENT_SOURCE_ID])],
        obs_count=0,
        prompt_chars=80,
    )
    call = AsyncMock(side_effect=[bad, good])
    perf = ConsolidationPerfLog("bank")

    with patch(
        "hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm",
        call,
    ):
        result = await _consolidate_batch_with_protected_vocabularies(
            llm_config=object(),
            memories=[{"id": _ENVIRONMENT_SOURCE_ID, "text": _ENVIRONMENT_SOURCE}],
            union_observations=[_observation(_ENVIRONMENT_TARGET_ID, _ENVIRONMENT_TARGET)],
            union_source_facts={},
            config=types.SimpleNamespace(consolidation_max_attempts=3),
            remaining_observation_slots=None,
            max_observations_per_scope=-1,
            protected_vocabularies=_vocabularies(),
            perf=perf,
        )

    assert result == good
    assert call.await_count == 2
    assert call.await_args_list[1].kwargs["union_observations"] == []
    assert perf.llm_calls == 2


async def test_guard_real_worker_keeps_vocabulary_retries_at_the_outer_boundary() -> None:
    attempts = 0

    async def call(**kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient provider failure")
        response_format = kwargs["response_format"]
        return response_format(creates=[], updates=[], deletes=[])

    llm = types.SimpleNamespace(call=AsyncMock(side_effect=call))
    perf = ConsolidationPerfLog("bank")
    config = types.SimpleNamespace(
        consolidation_max_attempts=2,
        consolidation_llm_max_retries=0,
        llm_output_language=None,
        observations_mission=None,
        llm_supports_max_items=True,
        llm_temperature_consolidation=0.0,
        llm_strict_schema_consolidation=False,
        consolidation_max_completion_tokens=None,
    )

    with patch(
        "hindsight_api.engine.consolidation.consolidator.asyncio.sleep",
        new=AsyncMock(),
    ):
        result = await _consolidate_batch_with_protected_vocabularies(
            llm_config=llm,
            memories=[{"id": _ENVIRONMENT_SOURCE_ID, "text": _ENVIRONMENT_SOURCE}],
            union_observations=[],
            union_source_facts={},
            config=config,
            remaining_observation_slots=None,
            max_observations_per_scope=-1,
            protected_vocabularies=_vocabularies(),
            perf=perf,
        )

    assert result.failed is False
    assert llm.call.await_count == 2
    assert perf.llm_calls == 2


async def test_guard_exhaustion_fails_closed() -> None:
    bad = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_ENVIRONMENT_TARGET_ID,
                source_fact_ids=[_ENVIRONMENT_SOURCE_ID],
                text="Bad cross-environment merge.",
            )
        ]
    )
    call = AsyncMock(side_effect=[bad, bad])

    with patch(
        "hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm",
        call,
    ):
        result = await _consolidate_batch_with_protected_vocabularies(
            llm_config=object(),
            memories=[{"id": _ENVIRONMENT_SOURCE_ID, "text": _ENVIRONMENT_SOURCE}],
            union_observations=[_observation(_ENVIRONMENT_TARGET_ID, _ENVIRONMENT_TARGET)],
            union_source_facts={},
            config=types.SimpleNamespace(consolidation_max_attempts=2),
            remaining_observation_slots=None,
            max_observations_per_scope=-1,
            protected_vocabularies=_vocabularies(),
            perf=None,
        )

    assert result.failed is True
    assert result.creates == []
    assert result.updates == []


async def test_dedup_skips_incompatible_best_candidate_before_llm() -> None:
    candidate = RetrievalResult(
        id=_ENVIRONMENT_TARGET_ID,
        text=_ENVIRONMENT_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    llm = types.SimpleNamespace(call=AsyncMock())

    with patch(
        "hindsight_api.engine.memories.get_memories",
        return_value=types.SimpleNamespace(
            recall_unified=AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)})
        ),
    ):
        outcome = await _dedup_adjudicate(
            pool=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            anchor_text=_ENVIRONMENT_SOURCE,
            anchor_emb_str="[0.1, 0.2]",
            tags=["project:pale-meridian"],
            exclude_id=None,
            protected_vocabularies=_vocabularies(),
        )

    assert outcome.best_id is None
    llm.call.assert_not_called()


async def test_dedup_can_select_compatible_candidate_after_rejected_twin() -> None:
    incompatible = RetrievalResult(
        id=_ENVIRONMENT_TARGET_ID,
        text=_ENVIRONMENT_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    compatible_id = "11111111-1111-4111-8111-111111111111"
    compatible = RetrievalResult(
        id=compatible_id,
        text="The staging environment completed its source-root split.",
        fact_type="observation",
        similarity=0.98,
    )
    llm = types.SimpleNamespace(call=AsyncMock(return_value=_DedupDecision(action="keep")))

    with (
        patch(
            "hindsight_api.engine.memories.get_memories",
            return_value=types.SimpleNamespace(
                recall_unified=AsyncMock(
                    return_value={"observation": SemanticBm25Result([incompatible, compatible], [], None)}
                )
            ),
        ),
        patch(
            "hindsight_api.engine.consolidation.consolidator.get_config",
            return_value=types.SimpleNamespace(llm_strict_schema_consolidation=False),
        ),
    ):
        outcome = await _dedup_adjudicate(
            pool=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(
                consolidation_dedup_threshold=0.97,
                llm_temperature_consolidation=0.2,
            ),
            dedup_llm_config=llm,
            anchor_text=_ENVIRONMENT_SOURCE,
            anchor_emb_str="[0.1, 0.2]",
            tags=["project:pale-meridian"],
            exclude_id=None,
            protected_vocabularies=_vocabularies(),
        )

    assert outcome.best_id == compatible_id
    prompt = llm.call.await_args.kwargs["messages"][0]["content"]
    assert "staging environment" in prompt
    assert "production" not in prompt


def test_typed_protected_vocabulary_is_revalidated() -> None:
    malformed = ConsolidationProtectedVocabulary(name="", terms=("only-one",))

    with pytest.raises(ValueError):
        parse_consolidation_protected_vocabularies((malformed,))


def test_protected_term_matching_uses_unicode_boundaries() -> None:
    smuggled = f"{_ENVIRONMENT_SOURCE} αproductionβ"

    conflicts = _protected_vocabulary_conflicts(smuggled, _ENVIRONMENT_TARGET, _vocabularies())

    assert len(conflicts) == 1
    assert conflicts[0].source_terms == ("staging",)


def test_protected_term_matching_normalizes_multicodepoint_casefolds() -> None:
    vocabularies = _compile_protected_vocabularies(
        parse_consolidation_protected_vocabularies([{"name": "environment", "terms": ["Straße", "production"]}])
    )

    conflicts = _protected_vocabulary_conflicts("The Straße environment changed.", _ENVIRONMENT_TARGET, vocabularies)

    assert len(conflicts) == 1
    assert conflicts[0].source_terms == ("strasse",)


def test_update_proposed_text_cannot_change_compatible_target_term() -> None:
    target = "The staging environment owns the extraction pipeline."
    result = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_ENVIRONMENT_TARGET_ID,
                source_fact_ids=[_ENVIRONMENT_SOURCE_ID],
                text=_ENVIRONMENT_TARGET,
            )
        ]
    )

    violations = _find_protected_vocabulary_violations(
        result,
        [{"id": _ENVIRONMENT_SOURCE_ID, "text": _ENVIRONMENT_SOURCE}],
        [_observation(_ENVIRONMENT_TARGET_ID, target)],
        _vocabularies(),
    )

    assert violations
    assert all(violation.action == "update" for violation in violations)


def test_create_text_cannot_spoof_its_source_term() -> None:
    result = _BatchLLMResult(
        creates=[
            _CreateAction(
                text=_ENVIRONMENT_TARGET,
                source_fact_ids=[_ENVIRONMENT_SOURCE_ID],
            )
        ]
    )

    violations = _find_protected_vocabulary_violations(
        result,
        [{"id": _ENVIRONMENT_SOURCE_ID, "text": _ENVIRONMENT_SOURCE}],
        [],
        _vocabularies(),
    )

    assert len(violations) == 1
    assert violations[0].action == "create"


def test_fresh_process_median_decision_remains_lawful() -> None:
    source = (
        "Decision to measure separate fresh benchmark processes and compare medians, "
        "while retaining one higher-precision canary run. Rationale: thermal/runtime "
        "variance is real; a single attractive row is not performance evidence."
    )

    assert _protected_vocabulary_conflicts(source, _UNKNOWN_TARGET, _vocabularies()) == ()


async def test_dedup_uses_source_term_when_generated_anchor_is_unknown() -> None:
    candidate = RetrievalResult(
        id=_ENVIRONMENT_TARGET_ID,
        text=_ENVIRONMENT_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    llm = types.SimpleNamespace(call=AsyncMock())

    with patch(
        "hindsight_api.engine.memories.get_memories",
        return_value=types.SimpleNamespace(
            recall_unified=AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)})
        ),
    ):
        outcome = await _dedup_adjudicate(
            pool=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            anchor_text="A source-root split was completed.",
            anchor_emb_str="[0.1, 0.2]",
            tags=[],
            exclude_id=None,
            protected_vocabularies=_vocabularies(),
            protected_evidence=(_ENVIRONMENT_SOURCE,),
        )

    assert outcome.should_merge is False
    llm.call.assert_not_called()


async def test_dedup_rejects_protected_term_drift_in_merged_text() -> None:
    candidate = RetrievalResult(
        id="11111111-1111-4111-8111-111111111111",
        text="The staging environment completed its source-root split.",
        fact_type="observation",
        similarity=0.99,
    )
    llm = types.SimpleNamespace(call=AsyncMock(return_value=_DedupDecision(action="merge", text=_ENVIRONMENT_TARGET)))

    with (
        patch(
            "hindsight_api.engine.memories.get_memories",
            return_value=types.SimpleNamespace(
                recall_unified=AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)})
            ),
        ),
        patch(
            "hindsight_api.engine.consolidation.consolidator.get_config",
            return_value=types.SimpleNamespace(llm_strict_schema_consolidation=False),
        ),
    ):
        outcome = await _dedup_adjudicate(
            pool=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(
                consolidation_dedup_threshold=0.97,
                llm_temperature_consolidation=0.2,
            ),
            dedup_llm_config=llm,
            anchor_text=_ENVIRONMENT_SOURCE,
            anchor_emb_str="[0.1, 0.2]",
            tags=[],
            exclude_id=None,
            protected_vocabularies=_vocabularies(),
            protected_evidence=(_ENVIRONMENT_SOURCE,),
        )

    assert outcome.should_merge is False


async def test_incompatible_create_reconciliation_performs_no_write() -> None:
    candidate = RetrievalResult(
        id=_ENVIRONMENT_TARGET_ID,
        text=_ENVIRONMENT_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    conn = AsyncMock()
    llm = types.SimpleNamespace(call=AsyncMock())

    with (
        patch(
            "hindsight_api.engine.memories.get_memories",
            return_value=types.SimpleNamespace(
                recall_unified=AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)})
            ),
        ),
        patch(
            "hindsight_api.engine.consolidation.consolidator.embedding_utils.generate_embeddings_batch",
            AsyncMock(return_value=[[0.1, 0.2]]),
        ),
    ):
        merged_into = await _dedup_reconcile_create(
            pool=conn,
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            create_text="A source-root split was completed.",
            create_source_ids=[uuid.UUID(_ENVIRONMENT_SOURCE_ID)],
            tags=[],
            source_bounds=_TemporalBounds(),
            protected_vocabularies=_vocabularies(),
            source_protected_texts=(_ENVIRONMENT_SOURCE,),
        )

    assert merged_into is None
    conn.execute.assert_not_called()
    llm.call.assert_not_called()


async def test_incompatible_update_reconciliation_performs_no_write() -> None:
    candidate = RetrievalResult(
        id=_ENVIRONMENT_TARGET_ID,
        text=_ENVIRONMENT_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    conn = AsyncMock()
    llm = types.SimpleNamespace(call=AsyncMock())

    with patch(
        "hindsight_api.engine.memories.get_memories",
        return_value=types.SimpleNamespace(
            recall_unified=AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)})
        ),
    ):
        await _dedup_reconcile_update(
            pool=conn,
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            updated_id="11111111-1111-4111-8111-111111111111",
            updated_text="A source-root split was completed.",
            updated_emb_str="[0.1, 0.2]",
            tags=[],
            protected_vocabularies=_vocabularies(),
            protected_evidence=(_ENVIRONMENT_SOURCE,),
        )

    conn.execute.assert_not_called()
    llm.call.assert_not_called()


async def test_guard_fails_when_retry_omits_displaced_source_at_zero_capacity() -> None:
    bad = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_ENVIRONMENT_TARGET_ID,
                source_fact_ids=[_ENVIRONMENT_SOURCE_ID],
                text="Bad cross-environment merge.",
            )
        ]
    )
    call = AsyncMock(side_effect=[bad, _BatchLLMResult()])

    with patch(
        "hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm",
        call,
    ):
        result = await _consolidate_batch_with_protected_vocabularies(
            llm_config=object(),
            memories=[{"id": _ENVIRONMENT_SOURCE_ID, "text": _ENVIRONMENT_SOURCE}],
            union_observations=[_observation(_ENVIRONMENT_TARGET_ID, _ENVIRONMENT_TARGET)],
            union_source_facts={},
            config=types.SimpleNamespace(consolidation_max_attempts=2),
            remaining_observation_slots=0,
            max_observations_per_scope=1,
            protected_vocabularies=_vocabularies(),
            perf=None,
        )

    assert result.failed is True
    assert call.await_count == 2
    assert all(call_args.kwargs["max_attempts_override"] == 1 for call_args in call.await_args_list)
