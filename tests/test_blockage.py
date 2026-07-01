"""Classification de la nature d'un blocage agentique."""

from rune.agentic.blockage import (classify_blockage, strategy_for,
                                     FAILURE, BLOCKED, UNKNOWN, WAITING_INPUT)


def test_failure_on_plain_test_error():
    k = classify_blockage(has_test_failure=True, no_call_streak=0,
                          repeat_calls=0,
                          last_error="assert 659.96 == 659.95")
    assert k == FAILURE


def test_unknown_on_missing_import():
    k = classify_blockage(has_test_failure=True, no_call_streak=0,
                          repeat_calls=0,
                          last_error="ModuleNotFoundError: No module named 'x'")
    assert k == UNKNOWN


def test_unknown_on_no_tests_collected():
    k = classify_blockage(has_test_failure=False, no_call_streak=0,
                          repeat_calls=0, last_error="no tests ran in 0.00s")
    assert k == UNKNOWN


def test_blocked_on_no_progress():
    # pas d'erreur franche mais l'agent n'appelle plus d'outil / se répète
    k = classify_blockage(has_test_failure=False, no_call_streak=3,
                          repeat_calls=0, last_error="")
    assert k == BLOCKED
    k2 = classify_blockage(has_test_failure=True, no_call_streak=0,
                           repeat_calls=3, last_error="assert False")
    assert k2 == BLOCKED


def test_waiting_input_on_ambiguity():
    k = classify_blockage(has_test_failure=False, no_call_streak=0,
                          repeat_calls=0, asked_user=True)
    assert k == WAITING_INPUT


def test_priority_waiting_over_unknown():
    # ambiguïté ET import manquant → WAITING_INPUT gagne (priorité)
    k = classify_blockage(has_test_failure=True, no_call_streak=0,
                          repeat_calls=0,
                          last_error="ambiguïté", asked_user=True)
    assert k == WAITING_INPUT


def test_strategy_for_each_kind_differs():
    advices = {strategy_for(k) for k in
               (FAILURE, BLOCKED, UNKNOWN, WAITING_INPUT)}
    # quatre conseils distincts (pas un message générique unique)
    assert len(advices) == 4
    assert "approche" in strategy_for(BLOCKED).lower()
    assert "cherch" in strategy_for(UNKNOWN).lower()
