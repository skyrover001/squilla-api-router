from __future__ import annotations

from benchmarks.benchmark_task_change import (
    _decision_digest,
    load_cases,
    run_benchmark,
)
from benchmarks.benchmark_task_change_hybrid import (
    HOLDOUT,
    TRAIN,
    _digest as hybrid_digest,
    run as run_hybrid_benchmark,
)


def test_task_change_dataset_covers_intent_boundaries() -> None:
    cases = load_cases()

    assert len(cases) >= 50
    categories = {case["category"] for case in cases}
    assert {
        "explicit_continue",
        "implicit_continue",
        "explicit_redo",
        "explicit_new_task",
        "tool_continuation",
        "task_changed_shared_domain",
        "task_changed_cross_domain",
        "terse_continue",
        "session_contract_change",
    } <= categories


def test_task_change_benchmark_is_deterministic() -> None:
    first = run_benchmark()
    second = run_benchmark()

    assert _decision_digest(first) == _decision_digest(second)
    assert first["summary"]["current_rule"]["cases"] == len(load_cases())
    assert first["summary"]["semantic_v1"]["cases"] == len(load_cases())


def test_hybrid_holdout_benchmark_is_deterministic() -> None:
    first = run_hybrid_benchmark()
    second = run_hybrid_benchmark()

    assert hybrid_digest(first) == hybrid_digest(second)
    assert first["holdout_cases"] == len(load_cases(HOLDOUT))
    assert first["ambiguous_train_cases"] >= 20


def test_expanded_train_and_holdout_are_disjoint() -> None:
    train = load_cases(TRAIN)
    holdout = load_cases(HOLDOUT)

    assert len(train) >= 100
    assert len(holdout) >= 145
    assert {case["id"] for case in train}.isdisjoint(
        {case["id"] for case in holdout}
    )
    assert any(case["id"].startswith("v2-") for case in holdout)
