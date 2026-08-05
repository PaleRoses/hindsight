"""Regression tests for consolidation's bank-configured semantic identity boundary."""

import types
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.config import ConsolidationIdentityAxis, parse_consolidation_identity_axes
from hindsight_api.engine.consolidation.consolidator import (
    ConsolidationPerfLog,
    _BatchLLMResult,
    _compile_identity_axes,
    _consolidate_batch_with_identity_guard,
    _CreateAction,
    _dedup_adjudicate,
    _dedup_reconcile_create,
    _dedup_reconcile_update,
    _DedupDecision,
    _find_identity_violations,
    _identity_conflicts,
    _IdentityAxis,
    _UpdateAction,
)
from hindsight_api.engine.response_models import MemoryFact
from hindsight_api.engine.search.retrieval import SemanticBm25Result
from hindsight_api.engine.search.types import RetrievalResult

_PACKAGE_SOURCE_ID = "ef7c88ad-3726-48cc-b1a7-1a78b806d3c0"
_PACKAGE_TARGET_ID = "0b9c5c4b-c6af-4b20-9881-3f40961137c0"
_PACKAGE_SOURCE = (
    "Working tree contains staged content-preserving source-root split for moonlight-analysis "
    "with 85 renames at 100% similarity plus unstaged work."
)
_PACKAGE_TARGET = (
    "All redundant child packages were deleted and converted to public named sublibraries "
    "under the single melusine-world-schober umbrella manifest."
)
_DBSP_SOURCE = (
    "Project project-dbsp-incrementalize-campaign implemented Phase 4b and directly revised its recorded phase status."
)
_DBSP_TARGET = (
    "Project project-dbsp-incrementalize-campaign is a DBSP incrementalization campaign whose phases 1-3 are complete."
)
_BENCHMARK_SOURCE = (
    "The constant-folded benchmark row was deleted because compiler time and end-to-end runtime "
    "measure different things."
)
_BENCHMARK_TARGET = (
    "Benchmark guidance requires larger fixtures and isolated processes so comparable CSV output "
    "reflects meaningful runtime ratios."
)


def _axes() -> tuple[_IdentityAxis, ...]:
    return _compile_identity_axes(
        parse_consolidation_identity_axes(
            [
                {
                    "name": "package",
                    "tokens": ["moonlight-analysis", "melusine-world-schober"],
                },
                {
                    "name": "campaign",
                    "tokens": [
                        "project-dbsp-incrementalize-campaign",
                        "project-melusine-world-schober",
                    ],
                },
            ]
        )
    )


def _observation(observation_id: str, text: str) -> MemoryFact:
    return MemoryFact(id=observation_id, text=text, fact_type="observation")


def test_identity_vocabulary_is_validated_and_canonicalized() -> None:
    axes = parse_consolidation_identity_axes(
        [{"name": " Package ", "tokens": ["Moonlight-Analysis", " melusine-world-schober "]}]
    )

    assert axes[0].name == "Package"
    assert axes[0].tokens == ("moonlight-analysis", "melusine-world-schober")


@pytest.mark.parametrize(
    "raw",
    [
        {},
        [{"name": "package"}],
        [{"name": "package", "tokens": ["moonlight-analysis"]}],
        [{"name": "package", "tokens": ["same", "SAME"]}],
        [
            {"name": "package", "tokens": ["one", "two"]},
            {"name": "PACKAGE", "tokens": ["three", "four"]},
        ],
    ],
)
def test_identity_vocabulary_rejects_ambiguous_shapes(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_consolidation_identity_axes(raw)


def test_package_textual_twin_is_rejected() -> None:
    conflicts = _identity_conflicts(_PACKAGE_SOURCE, _PACKAGE_TARGET, _axes())

    assert len(conflicts) == 1
    assert conflicts[0].axis == "package"
    assert conflicts[0].source_tokens == ("moonlight-analysis",)
    assert conflicts[0].target_tokens == ("melusine-world-schober",)


def test_identity_matching_requires_identifier_boundaries() -> None:
    source = "moonlight-analysis-old is an unrelated identifier"

    assert _identity_conflicts(source, _PACKAGE_TARGET, _axes()) == ()


def test_same_campaign_update_remains_lawful() -> None:
    assert _identity_conflicts(_DBSP_SOURCE, _DBSP_TARGET, _axes()) == ()


def test_unknown_benchmark_identity_remains_lawful() -> None:
    assert _identity_conflicts(_BENCHMARK_SOURCE, _BENCHMARK_TARGET, _axes()) == ()


def test_update_violation_preserves_source_and_target_evidence() -> None:
    result = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_PACKAGE_TARGET_ID,
                source_fact_ids=[_PACKAGE_SOURCE_ID],
                text="Combined text that must never be written.",
            )
        ]
    )

    violations = _find_identity_violations(
        result,
        [{"id": _PACKAGE_SOURCE_ID, "text": _PACKAGE_SOURCE}],
        [_observation(_PACKAGE_TARGET_ID, _PACKAGE_TARGET)],
        _axes(),
    )

    assert len(violations) == 1
    assert violations[0].source_fact_id == _PACKAGE_SOURCE_ID
    assert violations[0].observation_id == _PACKAGE_TARGET_ID


async def test_guard_retries_without_rejected_target_and_preserves_fact() -> None:
    bad = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_PACKAGE_TARGET_ID,
                source_fact_ids=[_PACKAGE_SOURCE_ID],
                text="Bad cross-package merge.",
            )
        ],
        obs_count=1,
        prompt_chars=100,
    )
    good = _BatchLLMResult(
        creates=[
            _CreateAction(text="Standalone moonlight-analysis observation.", source_fact_ids=[_PACKAGE_SOURCE_ID])
        ],
        obs_count=0,
        prompt_chars=80,
    )
    call = AsyncMock(side_effect=[bad, good])
    perf = ConsolidationPerfLog("bank")

    with patch(
        "hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm",
        call,
    ):
        result = await _consolidate_batch_with_identity_guard(
            llm_config=object(),
            memories=[{"id": _PACKAGE_SOURCE_ID, "text": _PACKAGE_SOURCE}],
            union_observations=[_observation(_PACKAGE_TARGET_ID, _PACKAGE_TARGET)],
            union_source_facts={},
            config=types.SimpleNamespace(consolidation_max_attempts=3),
            remaining_observation_slots=None,
            max_observations_per_scope=-1,
            identity_axes=_axes(),
            perf=perf,
        )

    assert result == good
    assert call.await_count == 2
    assert call.await_args_list[1].kwargs["union_observations"] == []
    assert perf.llm_calls == 2


async def test_guard_exhaustion_fails_closed() -> None:
    bad = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_PACKAGE_TARGET_ID,
                source_fact_ids=[_PACKAGE_SOURCE_ID],
                text="Bad cross-package merge.",
            )
        ]
    )
    call = AsyncMock(side_effect=[bad, bad])

    with patch(
        "hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm",
        call,
    ):
        result = await _consolidate_batch_with_identity_guard(
            llm_config=object(),
            memories=[{"id": _PACKAGE_SOURCE_ID, "text": _PACKAGE_SOURCE}],
            union_observations=[_observation(_PACKAGE_TARGET_ID, _PACKAGE_TARGET)],
            union_source_facts={},
            config=types.SimpleNamespace(consolidation_max_attempts=2),
            remaining_observation_slots=None,
            max_observations_per_scope=-1,
            identity_axes=_axes(),
            perf=None,
        )

    assert result.failed is True
    assert result.creates == []
    assert result.updates == []


async def test_dedup_skips_incompatible_best_candidate_before_llm() -> None:
    candidate = RetrievalResult(
        id=_PACKAGE_TARGET_ID,
        text=_PACKAGE_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    llm = types.SimpleNamespace(call=AsyncMock())

    with patch(
        "hindsight_api.engine.search.retrieval.retrieve_semantic_bm25_combined",
        AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)}),
    ):
        outcome = await _dedup_adjudicate(
            conn=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            anchor_text=_PACKAGE_SOURCE,
            anchor_emb_str="[0.1, 0.2]",
            tags=["project:pale-meridian"],
            exclude_id=None,
            identity_axes=_axes(),
        )

    assert outcome.best_id is None
    llm.call.assert_not_called()


async def test_dedup_can_select_compatible_candidate_after_rejected_twin() -> None:
    incompatible = RetrievalResult(
        id=_PACKAGE_TARGET_ID,
        text=_PACKAGE_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    compatible_id = "11111111-1111-4111-8111-111111111111"
    compatible = RetrievalResult(
        id=compatible_id,
        text="The moonlight-analysis package completed its source-root split.",
        fact_type="observation",
        similarity=0.98,
    )
    llm = types.SimpleNamespace(call=AsyncMock(return_value=_DedupDecision(action="keep")))

    with (
        patch(
            "hindsight_api.engine.search.retrieval.retrieve_semantic_bm25_combined",
            AsyncMock(return_value={"observation": SemanticBm25Result([incompatible, compatible], [], None)}),
        ),
        patch(
            "hindsight_api.engine.consolidation.consolidator.get_config",
            return_value=types.SimpleNamespace(llm_strict_schema_consolidation=False),
        ),
    ):
        outcome = await _dedup_adjudicate(
            conn=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            anchor_text=_PACKAGE_SOURCE,
            anchor_emb_str="[0.1, 0.2]",
            tags=["project:pale-meridian"],
            exclude_id=None,
            identity_axes=_axes(),
        )

    assert outcome.best_id == compatible_id
    prompt = llm.call.await_args.kwargs["messages"][0]["content"]
    assert "moonlight-analysis package" in prompt
    assert "melusine-world-schober" not in prompt


def test_typed_identity_vocabulary_is_revalidated() -> None:
    malformed = ConsolidationIdentityAxis(name="", tokens=("only-one",))

    with pytest.raises(ValueError):
        parse_consolidation_identity_axes((malformed,))


def test_identity_matching_uses_unicode_boundaries() -> None:
    smuggled = f"{_PACKAGE_SOURCE} αmelusine-world-schoberβ"

    conflicts = _identity_conflicts(smuggled, _PACKAGE_TARGET, _axes())

    assert len(conflicts) == 1
    assert conflicts[0].source_tokens == ("moonlight-analysis",)


def test_identity_matching_normalizes_multicodepoint_casefolds() -> None:
    axes = _compile_identity_axes(
        parse_consolidation_identity_axes([{"name": "package", "tokens": ["Straße", "melusine-world-schober"]}])
    )

    conflicts = _identity_conflicts("The Straße package changed.", _PACKAGE_TARGET, axes)

    assert len(conflicts) == 1
    assert conflicts[0].source_tokens == ("strasse",)


def test_update_proposed_text_cannot_change_compatible_target_identity() -> None:
    target = "The moonlight-analysis package owns the extraction pipeline."
    result = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_PACKAGE_TARGET_ID,
                source_fact_ids=[_PACKAGE_SOURCE_ID],
                text=_PACKAGE_TARGET,
            )
        ]
    )

    violations = _find_identity_violations(
        result,
        [{"id": _PACKAGE_SOURCE_ID, "text": _PACKAGE_SOURCE}],
        [_observation(_PACKAGE_TARGET_ID, target)],
        _axes(),
    )

    assert violations
    assert all(violation.action == "update" for violation in violations)


def test_create_text_cannot_spoof_its_source_identity() -> None:
    result = _BatchLLMResult(
        creates=[
            _CreateAction(
                text=_PACKAGE_TARGET,
                source_fact_ids=[_PACKAGE_SOURCE_ID],
            )
        ]
    )

    violations = _find_identity_violations(
        result,
        [{"id": _PACKAGE_SOURCE_ID, "text": _PACKAGE_SOURCE}],
        [],
        _axes(),
    )

    assert len(violations) == 1
    assert violations[0].action == "create"


def test_fresh_process_median_decision_remains_lawful() -> None:
    source = (
        "Decision to measure separate fresh benchmark processes and compare medians, "
        "while retaining one higher-precision canary run. Rationale: thermal/runtime "
        "variance is real; a single attractive row is not performance evidence."
    )

    assert _identity_conflicts(source, _BENCHMARK_TARGET, _axes()) == ()


async def test_dedup_uses_source_identity_when_generated_anchor_is_unknown() -> None:
    candidate = RetrievalResult(
        id=_PACKAGE_TARGET_ID,
        text=_PACKAGE_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    llm = types.SimpleNamespace(call=AsyncMock())

    with patch(
        "hindsight_api.engine.search.retrieval.retrieve_semantic_bm25_combined",
        AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)}),
    ):
        outcome = await _dedup_adjudicate(
            conn=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            anchor_text="A source-root split was completed.",
            anchor_emb_str="[0.1, 0.2]",
            tags=[],
            exclude_id=None,
            identity_axes=_axes(),
            identity_evidence=(_PACKAGE_SOURCE,),
        )

    assert outcome.should_merge is False
    llm.call.assert_not_called()


async def test_dedup_rejects_identity_drift_in_merged_text() -> None:
    candidate = RetrievalResult(
        id="11111111-1111-4111-8111-111111111111",
        text="The moonlight-analysis package completed its source-root split.",
        fact_type="observation",
        similarity=0.99,
    )
    llm = types.SimpleNamespace(call=AsyncMock(return_value=_DedupDecision(action="merge", text=_PACKAGE_TARGET)))

    with (
        patch(
            "hindsight_api.engine.search.retrieval.retrieve_semantic_bm25_combined",
            AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)}),
        ),
        patch(
            "hindsight_api.engine.consolidation.consolidator.get_config",
            return_value=types.SimpleNamespace(llm_strict_schema_consolidation=False),
        ),
    ):
        outcome = await _dedup_adjudicate(
            conn=AsyncMock(),
            memory_engine=types.SimpleNamespace(),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            anchor_text=_PACKAGE_SOURCE,
            anchor_emb_str="[0.1, 0.2]",
            tags=[],
            exclude_id=None,
            identity_axes=_axes(),
            identity_evidence=(_PACKAGE_SOURCE,),
        )

    assert outcome.should_merge is False


async def test_incompatible_create_reconciliation_performs_no_write() -> None:
    candidate = RetrievalResult(
        id=_PACKAGE_TARGET_ID,
        text=_PACKAGE_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    conn = AsyncMock()
    llm = types.SimpleNamespace(call=AsyncMock())

    with (
        patch(
            "hindsight_api.engine.search.retrieval.retrieve_semantic_bm25_combined",
            AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)}),
        ),
        patch(
            "hindsight_api.engine.consolidation.consolidator.embedding_utils.generate_embeddings_batch",
            AsyncMock(return_value=[[0.1, 0.2]]),
        ),
    ):
        merged_into = await _dedup_reconcile_create(
            conn=conn,
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            create_text="A source-root split was completed.",
            create_source_ids=[uuid.UUID(_PACKAGE_SOURCE_ID)],
            tags=[],
            identity_axes=_axes(),
            source_identity_texts=(_PACKAGE_SOURCE,),
        )

    assert merged_into is None
    conn.execute.assert_not_called()
    llm.call.assert_not_called()


async def test_incompatible_update_reconciliation_performs_no_write() -> None:
    candidate = RetrievalResult(
        id=_PACKAGE_TARGET_ID,
        text=_PACKAGE_TARGET,
        fact_type="observation",
        similarity=0.999,
    )
    conn = AsyncMock()
    llm = types.SimpleNamespace(call=AsyncMock())

    with patch(
        "hindsight_api.engine.search.retrieval.retrieve_semantic_bm25_combined",
        AsyncMock(return_value={"observation": SemanticBm25Result([candidate], [], None)}),
    ):
        await _dedup_reconcile_update(
            conn=conn,
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank",
            config=types.SimpleNamespace(consolidation_dedup_threshold=0.97),
            dedup_llm_config=llm,
            updated_id="11111111-1111-4111-8111-111111111111",
            updated_text="A source-root split was completed.",
            updated_emb_str="[0.1, 0.2]",
            tags=[],
            identity_axes=_axes(),
            identity_evidence=(_PACKAGE_SOURCE,),
        )

    conn.execute.assert_not_called()
    llm.call.assert_not_called()


async def test_guard_fails_when_retry_omits_displaced_source_at_zero_capacity() -> None:
    bad = _BatchLLMResult(
        updates=[
            _UpdateAction(
                observation_id=_PACKAGE_TARGET_ID,
                source_fact_ids=[_PACKAGE_SOURCE_ID],
                text="Bad cross-package merge.",
            )
        ]
    )
    call = AsyncMock(side_effect=[bad, _BatchLLMResult()])

    with patch(
        "hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm",
        call,
    ):
        result = await _consolidate_batch_with_identity_guard(
            llm_config=object(),
            memories=[{"id": _PACKAGE_SOURCE_ID, "text": _PACKAGE_SOURCE}],
            union_observations=[_observation(_PACKAGE_TARGET_ID, _PACKAGE_TARGET)],
            union_source_facts={},
            config=types.SimpleNamespace(consolidation_max_attempts=2),
            remaining_observation_slots=0,
            max_observations_per_scope=1,
            identity_axes=_axes(),
            perf=None,
        )

    assert result.failed is True
    assert call.await_count == 2
    assert all(call_args.kwargs["max_attempts_override"] == 1 for call_args in call.await_args_list)
