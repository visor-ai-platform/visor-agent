"""[TODO-002] Feedback: failure summary → planner in-context learning.

Aggregates recent execution failures, summarizes via LLM, injects into the next
planner prompt as context to bias around known-bad patterns.
"""
from __future__ import annotations

from typing import Any


async def summarize_failures(records: list[dict[str, Any]]) -> str:
    _ = records
    raise NotImplementedError
