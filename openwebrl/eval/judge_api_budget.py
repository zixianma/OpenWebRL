"""Shared conservative dollar cap for terminal-judge requests.

Enabled only when OPENWEBRL_JUDGE_BUDGET_PATH and
OPENWEBRL_JUDGE_BUDGET_USD are both set. Failed/unknown HTTP attempts retain
their full reservation because their provider billing is uncertain.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4


INPUT_USD_PER_TOKEN = 1.10 / 1_000_000
OUTPUT_USD_PER_TOKEN = 4.40 / 1_000_000
CONTEXT_TOKENS = 200_000
OUTPUT_TOKENS = 4_096
RESERVATION_USD = CONTEXT_TOKENS * INPUT_USD_PER_TOKEN + OUTPUT_TOKENS * OUTPUT_USD_PER_TOKEN


class JudgeBudgetExhausted(RuntimeError):
    pass


class JudgeBudget:
    def __init__(
        self, path: Path, cap_usd: float, max_attempts: int | None = None,
        *, input_usd_per_million: float = 1.10, output_usd_per_million: float = 4.40,
    ) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.cap_usd = cap_usd
        self.max_attempts = max_attempts
        self.pricing = {
            "input_usd_per_million": input_usd_per_million,
            "output_usd_per_million": output_usd_per_million,
        }
        self.input_usd_per_token = input_usd_per_million / 1_000_000
        self.output_usd_per_token = output_usd_per_million / 1_000_000
        self.reservation_usd = CONTEXT_TOKENS * self.input_usd_per_token + OUTPUT_TOKENS * self.output_usd_per_token
        if cap_usd <= 0:
            raise ValueError("judge API cap must be positive")
        if input_usd_per_million <= 0 or output_usd_per_million <= 0:
            raise ValueError("judge prices must be positive")
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("judge attempt cap must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "JudgeBudget | None":
        path = os.getenv("OPENWEBRL_JUDGE_BUDGET_PATH")
        cap = os.getenv("OPENWEBRL_JUDGE_BUDGET_USD")
        if not path and not cap:
            return None
        if not path or not cap:
            raise ValueError("judge budget requires both path and USD cap")
        attempts = os.getenv("OPENWEBRL_JUDGE_ATTEMPT_CAP")
        return cls(
            Path(path), float(cap), int(attempts) if attempts else None,
            input_usd_per_million=float(os.getenv("OPENWEBRL_JUDGE_INPUT_USD_PER_MILLION", "1.10")),
            output_usd_per_million=float(os.getenv("OPENWEBRL_JUDGE_OUTPUT_USD_PER_MILLION", "4.40")),
        )

    def _transaction(self, operation):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                "cap_usd": self.cap_usd, "spent_usd": 0.0, "reserved": {},
                "attempts": 0, "max_attempts": self.max_attempts,
                "unknown_usage_attempts": 0,
            }
            if state["cap_usd"] != self.cap_usd or state["max_attempts"] != self.max_attempts:
                raise ValueError("judge budget cap changed for a live ledger")
            if "pricing" not in state:
                if state["reserved"]:
                    raise ValueError("cannot change judge pricing with outstanding reservations")
                if state["attempts"]:
                    state["prior_attempts_at_legacy_pricing"] = state["attempts"]
                    state["prior_spent_usd_at_legacy_pricing"] = state["spent_usd"]
                state["pricing"] = self.pricing
            elif state["pricing"] != self.pricing:
                raise ValueError("judge pricing changed for a live ledger")
            result = operation(state)
            temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            with temporary.open("w") as stream:
                json.dump(state, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            return result

    def reserve(self) -> str:
        ticket = uuid4().hex

        def operation(state):
            outstanding = sum(state["reserved"].values())
            if self.max_attempts is not None and state["attempts"] >= self.max_attempts:
                raise JudgeBudgetExhausted("judge attempt cap reached")
            if state["spent_usd"] + outstanding + self.reservation_usd > self.cap_usd:
                raise JudgeBudgetExhausted("judge dollar cap reached")
            state["reserved"][ticket] = self.reservation_usd
            state["attempts"] += 1
            return ticket

        return self._transaction(operation)

    def settle(self, ticket: str, usage=None) -> None:
        def operation(state):
            reservation = state["reserved"].pop(ticket)
            if usage is None:
                cost = reservation
                state["unknown_usage_attempts"] += 1
            else:
                prompt = int(getattr(usage, "prompt_tokens", -1))
                completion = int(getattr(usage, "completion_tokens", -1))
                if prompt < 0 or completion < 0 or prompt > CONTEXT_TOKENS or completion > OUTPUT_TOKENS:
                    cost = reservation
                    state["unknown_usage_attempts"] += 1
                else:
                    cost = prompt * self.input_usd_per_token + completion * self.output_usd_per_token
            state["spent_usd"] += cost

        self._transaction(operation)
