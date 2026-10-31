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
