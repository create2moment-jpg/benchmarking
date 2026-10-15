"""Cancellation and budget primitives shared by providers, harnesses and the runner."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class CancelledError(Exception):
    """Raised when a run is cancelled (Ctrl-C, SIGTERM or whole-run deadline)."""


class BudgetExceededError(Exception):
    """Raised when an attempt exhausts its turn or token budget."""


class DeadlineExceededError(Exception):
    """Raised when an attempt's wall-clock budget is exhausted."""


@dataclass
class CancelToken:
    """Thread-safe cancellation flag, optionally linked to a parent token."""

    _event: threading.Event = field(default_factory=threading.Event)
    parent: CancelToken | None = None
    reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or (self.parent is not None and self.parent.cancelled)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError(self.reason or (self.parent.reason if self.parent else None))

    def child(self) -> CancelToken:
        return CancelToken(parent=self)


@dataclass
class Budget:
    """Per-attempt budget: wall-clock deadline, turns and total tokens."""

    deadline: float  # time.monotonic() value
    max_turns: int
    max_total_tokens: int | None = None
    turns_used: int = 0
    tokens_used: int = 0

    @classmethod
    def from_timeout(
        cls, timeout_s: float, max_turns: int, max_total_tokens: int | None = None
    ) -> Budget:
        return cls(time.monotonic() + timeout_s, max_turns, max_total_tokens)

    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check_time(self) -> None:
        if self.remaining_s() <= 0:
            raise DeadlineExceededError("attempt wall-clock budget exhausted")

    def use_turn(self) -> None:
        self.check_time()
        if self.turns_used >= self.max_turns:
            raise BudgetExceededError(f"max_turns={self.max_turns} exhausted")
        self.turns_used += 1

    def add_tokens(self, tokens: int | None) -> None:
        if tokens:
            self.tokens_used += tokens
        if self.max_total_tokens is not None and self.tokens_used > self.max_total_tokens:
            raise BudgetExceededError(f"max_total_tokens={self.max_total_tokens} exhausted")
