"""Regression tests for consolidation's bank-configured protected-term boundary."""

import dataclasses
import types
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.config import ConsolidationProtectedVocabulary, HindsightConfig
from hindsight_api.config import parse_consolidation_protected_vocabularies as parse_vocabularies
from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.response_models import LLMCallResult, MemoryFact, TokenUsage
from hindsight_api.engine.search.retrieval import SemanticBm25Result
from hindsight_api.engine.search.types import RetrievalResult

_SOURCE_ID = "ef7c88ad-3726-48cc-b1a7-1a78b806d3c0"
_TARGET_ID = "0b9c5c4b-c6af-4b20-9881-3f40961137c0"
_TWIN_ID = "11111111-1111-4111-8111-111111111111"
_STAGING = "The staging deployment enables the canary scheduler for ten percent of requests."
_PRODUCTION = "The production deployment keeps the stable scheduler for all requests."
_STAGING_TWIN = "The staging environment completed its source-root split."
_UNKNOWN = "A source-root split was completed."
_NEUTRAL = "The extraction pipeline was rebuilt."
_ENVIRONMENT_CONFLICT = (("environment", ("staging",), ("production",)),)

_VOCABULARIES = C._compile_protected_vocabularies(
    parse_vocabularies(
        [
            {"name": "environment", "terms": ["staging", "production"]},
            {"name": "street", "terms": ["Straße", "fast lane"]},
        ]
    )
)
_MEMORIES = [{"id": _SOURCE_ID, "text": _STAGING}]
_BAD_UPDATE = C._BatchLLMResult(
    updates=[C._UpdateAction(observation_id=_TARGET_ID, source_fact_ids=[_SOURCE_ID], text="Bad merge.")]
)


def _observation(observation_id: str, text: str) -> MemoryFact:
    return MemoryFact(id=observation_id, text=text, fact_type="observation")


def _candidate(observation_id: str, text: str, similarity: float) -> RetrievalResult:
    return RetrievalResult(id=observation_id, text=text, fact_type="observation", similarity=similarity)


def _recall(candidates: list[RetrievalResult]):
    return patch(
        "hindsight_api.engine.memories.get_memories",
        return_value=types.SimpleNamespace(
            recall_unified=AsyncMock(return_value={"observation": SemanticBm25Result(candidates, [], None)})
        ),
    )


def _lenient_schema():
    return patch(
        "hindsight_api.engine.consolidation.consolidator.get_config",
        return_value=types.SimpleNamespace(llm_strict_schema_consolidation=False),
    )


async def _adjudicate(dedup_llm_config, anchor_text, *, protected_evidence=(), pool=None):
    return await C._dedup_adjudicate(
        pool=pool if pool is not None else AsyncMock(),
        memory_engine=types.SimpleNamespace(embeddings=object()),
        bank_id="bank",
        config=types.SimpleNamespace(consolidation_dedup_threshold=0.97, llm_temperature_consolidation=0.2),
        dedup_llm_config=dedup_llm_config,
        anchor_text=anchor_text,
        anchor_emb_str="[0.1, 0.2]",
        tags=None,
        exclude_id=None,
        protected_vocabularies=_VOCABULARIES,
        protected_evidence=protected_evidence,
    )


async def _guard(config, perf, *, observations=None, llm_config=None):
    return await C._consolidate_batch_with_protected_vocabularies(
        llm_config=llm_config if llm_config is not None else object(),
        memories=_MEMORIES,
        union_observations=[_observation(_TARGET_ID, _PRODUCTION)] if observations is None else observations,
        union_source_facts={},
        config=config,
        remaining_observation_slots=None,
        max_observations_per_scope=-1,
        protected_vocabularies=_VOCABULARIES,
        perf=perf,
    )


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        pytest.param(_STAGING, _PRODUCTION, _ENVIRONMENT_CONFLICT, id="disjoint-terms-conflict"),
        pytest.param("staging-old is an unrelated identifier", _PRODUCTION, (), id="hyphen-is-an-identifier-char"),
        pytest.param(f"{_STAGING} αproductionβ", _PRODUCTION, _ENVIRONMENT_CONFLICT, id="unicode-letters-bind"),
        pytest.param(
            "The Straße crew moved.",
            "Route it via the fast\n  lane.",
            (("street", ("strasse",), ("fast lane",)),),
            id="multi-codepoint-casefold-and-multi-word-term",
        ),
        pytest.param(_STAGING, _STAGING_TWIN, (), id="same-term-on-both-sides"),
        pytest.param("Compiler time is not request latency.", "Latency needs fixtures.", (), id="unknown-terms"),
    ],
)
def test_protected_term_matching(source: str, target: str, expected: tuple) -> None:
    conflicts = C._protected_vocabulary_conflicts((source,), (target,), _VOCABULARIES)

    assert tuple((c.vocabulary, c.source_terms, c.target_terms) for c in conflicts) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({}, id="not-a-list"),
        pytest.param([{"name": "env"}], id="missing-terms"),
        pytest.param([{"name": "env", "terms": "staging"}], id="terms-not-a-list"),
        pytest.param([{"name": "env", "terms": ["staging", ""]}], id="blank-term"),
        pytest.param([{"name": "env", "terms": ["staging"]}], id="single-term"),
        pytest.param([{"name": "env", "terms": ["same", "SAME"]}], id="terms-collide-when-folded"),
        pytest.param([{"name": "env", "terms": ["a", "b"]}, {"name": "ENV", "terms": ["c", "d"]}], id="names-collide"),
        pytest.param((ConsolidationProtectedVocabulary(name="", terms=("only-one",)),), id="typed-value-revalidated"),
    ],
)
def test_protected_vocabulary_rejects_ambiguous_shapes(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_vocabularies(raw)


def test_protected_vocabulary_is_canonicalized() -> None:
    vocabularies = parse_vocabularies([{"name": " Environment ", "terms": ["Staging", " production "]}])

    assert vocabularies == (ConsolidationProtectedVocabulary(name="Environment", terms=("staging", "production")),)


def test_protected_vocabularies_load_from_server_config(monkeypatch) -> None:
    monkeypatch.setenv(
        "HINDSIGHT_API_CONSOLIDATION_PROTECTED_VOCABULARIES", '[{"name":"env","terms":["staging","production"]}]'
    )

    assert HindsightConfig.from_env().consolidation_protected_vocabularies == (
        ConsolidationProtectedVocabulary(name="env", terms=("staging", "production")),
    )


@pytest.mark.parametrize(
    ("proposed", "stored", "sources", "expected"),
    [
        pytest.param(_PRODUCTION, None, [_SOURCE_ID], {("create", None, _SOURCE_ID)}, id="create-text-vs-source"),
        pytest.param("Merged.", _PRODUCTION, [_SOURCE_ID], {("update", _TARGET_ID, _SOURCE_ID)}, id="source-vs-stored"),
        pytest.param(
            _PRODUCTION, _NEUTRAL, [_SOURCE_ID], {("update", _TARGET_ID, _SOURCE_ID)}, id="source-vs-proposed"
        ),
        pytest.param(
            _PRODUCTION, _STAGING_TWIN, [], {("update", _TARGET_ID, "<update-text>")}, id="proposed-vs-stored"
        ),
        pytest.param(_STAGING, _STAGING_TWIN, [_SOURCE_ID], set(), id="same-term-write-is-lawful"),
    ],
)
def test_write_boundary_violations(proposed: str, stored: str | None, sources: list[str], expected: set) -> None:
    # No stored observation is the CREATE case; otherwise the model proposes an UPDATE over it.
    if stored is None:
        result = C._BatchLLMResult(creates=[C._CreateAction(text=proposed, source_fact_ids=sources)])
        observations = []
    else:
        update = C._UpdateAction(observation_id=_TARGET_ID, source_fact_ids=sources, text=proposed)
        result = C._BatchLLMResult(updates=[update])
        observations = [_observation(_TARGET_ID, stored)]

    violations = C._find_protected_vocabulary_violations(result, _MEMORIES, observations, _VOCABULARIES)

    assert {(v.action, v.observation_id, v.source_fact_id) for v in violations} == expected
    # Every violation names the vocabulary and the terms on both sides, so the log says why.
    assert all(
        conflict.vocabulary == "environment"
        and {*conflict.source_terms, *conflict.target_terms} == {"staging", "production"}
        for violation in violations
        for conflict in violation.conflicts
    )


@pytest.mark.parametrize(
    ("anchor_text", "protected_evidence"),
    [
        pytest.param(_STAGING, (), id="term-in-the-anchor-text"),
        pytest.param(_UNKNOWN, (_STAGING,), id="term-only-in-the-source-fact"),
    ],
)
async def test_dedup_rejects_an_incompatible_candidate_before_the_llm(anchor_text: str, protected_evidence) -> None:
    llm = types.SimpleNamespace(call=AsyncMock())

    with _recall([_candidate(_TARGET_ID, _PRODUCTION, 0.999)]):
        outcome = await _adjudicate(llm, anchor_text, protected_evidence=protected_evidence)

    assert (outcome.best_id, outcome.should_merge) == (None, False)
    llm.call.assert_not_called()


async def test_dedup_selects_a_compatible_candidate_after_rejecting_the_twin() -> None:
    decision = LLMCallResult(content=C._DedupDecision(action="keep"), usage=TokenUsage())
    llm = types.SimpleNamespace(call=AsyncMock(return_value=decision))
    candidates = [_candidate(_TARGET_ID, _PRODUCTION, 0.999), _candidate(_TWIN_ID, _STAGING_TWIN, 0.98)]

    with _recall(candidates), _lenient_schema():
        outcome = await _adjudicate(llm, _STAGING)

    assert outcome.best_id == _TWIN_ID
    prompt = llm.call.await_args.kwargs["messages"][0]["content"]
    assert "staging environment" in prompt
    assert "production" not in prompt


async def test_dedup_rejects_merged_text_drift_and_writes_nothing() -> None:
    decision = LLMCallResult(content=C._DedupDecision(action="merge", text=_PRODUCTION), usage=TokenUsage())
    llm = types.SimpleNamespace(call=AsyncMock(return_value=decision))
    conn = AsyncMock()
    engine = types.SimpleNamespace(embeddings=object())

    with _recall([_candidate(_TWIN_ID, _STAGING_TWIN, 0.99)]), _lenient_schema():
        outcome = await _adjudicate(llm, _UNKNOWN, protected_evidence=(_STAGING,), pool=conn)
        # Adjudication is connection-free; both folds run later on the batch's connection. A merge
        # the guard refused must reach neither write half even though the twin stayed the nearest.
        merged_into = await C._apply_dedup_create_fold(
            conn, engine, "bank", object(), outcome, [uuid.UUID(_SOURCE_ID)], C._TemporalBounds()
        )
        folded = await C._apply_dedup_update_fold(conn, engine, "bank", object(), outcome, _TARGET_ID, _UNKNOWN)

    assert (outcome.best_id, outcome.should_merge) == (_TWIN_ID, False)
    assert (merged_into, folded) == (None, False)
    assert conn.mock_calls == []


async def test_guard_drops_the_conflicting_target_and_retries_without_it() -> None:
    good = C._BatchLLMResult(creates=[C._CreateAction(text="Standalone staging fact.", source_fact_ids=[_SOURCE_ID])])
    call = AsyncMock(side_effect=[_BAD_UPDATE, good])
    perf = C.ConsolidationPerfLog("bank")

    with patch("hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm", call):
        result = await _guard(types.SimpleNamespace(consolidation_max_attempts=3), perf)

    assert result == good
    assert [[str(o.id) for o in c.kwargs["union_observations"]] for c in call.await_args_list] == [[_TARGET_ID], []]
    assert {c.kwargs["max_attempts_override"] for c in call.await_args_list} == {1}
    assert perf.llm_calls == 2


@pytest.mark.parametrize(
    "retry",
    [
        pytest.param(_BAD_UPDATE, id="retry-repeats-the-conflict"),
        pytest.param(C._BatchLLMResult(), id="retry-abandons-the-displaced-source"),
        pytest.param(
            C._BatchLLMResult(
                creates=[C._CreateAction(text=_STAGING_TWIN, source_fact_ids=[_SOURCE_ID])],
                deletes=[C._DeleteAction(observation_id=_TARGET_ID)],
            ),
            id="retry-reaches-for-the-rejected-target",
        ),
    ],
)
async def test_guard_fails_closed_when_a_retry_cannot_preserve_the_constraints(retry) -> None:
    call = AsyncMock(side_effect=[_BAD_UPDATE, retry])

    with patch("hindsight_api.engine.consolidation.consolidator._consolidate_batch_with_llm", call):
        result = await _guard(types.SimpleNamespace(consolidation_max_attempts=2), None)

    assert result.failed is True
    assert (result.creates, result.updates, result.deletes) == ([], [], [])
    assert call.await_count == 2


async def test_guard_spends_the_attempt_budget_at_the_outer_boundary() -> None:
    attempts = 0

    async def call(**kwargs: object) -> LLMCallResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient provider failure")
        response_format = kwargs["response_format"]
        return LLMCallResult(content=response_format(creates=[], updates=[], deletes=[]), usage=TokenUsage())

    llm = types.SimpleNamespace(call=AsyncMock(side_effect=call))
    perf = C.ConsolidationPerfLog("bank")
    config = dataclasses.replace(
        HindsightConfig.from_env(), consolidation_max_attempts=2, consolidation_llm_max_retries=0
    )

    with patch("hindsight_api.engine.consolidation.consolidator.asyncio.sleep", new=AsyncMock()):
        result = await _guard(config, perf, observations=[], llm_config=llm)

    # The inner attempt budget is pinned to 1, so the transient failure costs one guard attempt
    # instead of being retried invisibly inside a single accounted call.
    assert result.failed is False
    assert (llm.call.await_count, perf.llm_calls) == (2, 2)
