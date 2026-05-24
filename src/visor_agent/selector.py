"""[SPEC] Skill selector. Matches a DAG node's `id` to a registered skill.

Reads the visor-skills registry (via HTTP from visor-app or local file mount)
and returns the resolved skill spec (with binding to a concrete version).
"""
from __future__ import annotations

from typing import Any


def select_skill(skill_id: str, registry: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [s for s in registry if s["id"] == skill_id]
    if not matches:
        raise ValueError(f"unknown skill: {skill_id}")
    # TODO: version constraint resolution; for now pick highest.
    matches.sort(key=lambda s: s["version"], reverse=True)
    return matches[0]
