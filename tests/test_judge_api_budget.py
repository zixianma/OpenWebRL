"""The shared API ledger must stop dispatch before its exact limits."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from openwebrl.eval.judge_api_budget import JudgeBudget, JudgeBudgetExhausted, RESERVATION_USD


def test_judge_api_budget_reservations_and_attempt_limit(tmp_path: Path) -> None:
    budget = JudgeBudget(tmp_path / "budget.json", cap_usd=2 * RESERVATION_USD, max_attempts=2)
    first = budget.reserve()
    second = budget.reserve()
    with pytest.raises(JudgeBudgetExhausted):
        budget.reserve()
    budget.settle(first, SimpleNamespace(prompt_tokens=1000, completion_tokens=100))
    budget.settle(second)
    with pytest.raises(JudgeBudgetExhausted):
        budget.reserve()


def test_judge_api_budget_dollar_limit(tmp_path: Path) -> None:
    budget = JudgeBudget(tmp_path / "budget.json", cap_usd=RESERVATION_USD)
    first = budget.reserve()
    with pytest.raises(JudgeBudgetExhausted):
        budget.reserve()
    budget.settle(first)
    with pytest.raises(JudgeBudgetExhausted):
        budget.reserve()


def test_judge_api_budget_keeps_prior_charges_when_judge_price_changes(tmp_path: Path) -> None:
    """A deprecated-judge smoke charge remains in the shared dollar cap."""
    import json

    path = tmp_path / "budget.json"
    path.write_text(json.dumps({
        "cap_usd": 1.0, "spent_usd": 0.24, "reserved": {},
        "attempts": 1, "max_attempts": 3, "unknown_usage_attempts": 1,
    }))
    budget = JudgeBudget(path, 1.0, 3, input_usd_per_million=1.0, output_usd_per_million=5.0)
    ticket = budget.reserve()
    budget.settle(ticket, SimpleNamespace(prompt_tokens=1000, completion_tokens=100))
    state = json.loads(path.read_text())
    assert state["prior_attempts_at_legacy_pricing"] == 1
    assert state["prior_spent_usd_at_legacy_pricing"] == 0.24
    assert state["attempts"] == 2
    assert state["spent_usd"] == pytest.approx(0.2415)
    with pytest.raises(ValueError, match="pricing changed"):
        JudgeBudget(path, 1.0, 3).reserve()
