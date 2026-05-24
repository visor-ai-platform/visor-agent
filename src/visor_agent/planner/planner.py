"""[SPEC-003] Planner: user intent → Skill DAG JSON.

LLM emits a strictly-structured DAG (see DESIGN §6.1 example). Free-text outputs
are rejected. The agent does **not** emit shell commands or scripts.
"""
from __future__ import annotations

from typing import Any

from visor_agent.models.dag import PlannerInput, SkillDAG


async def plan(inp: PlannerInput) -> SkillDAG:
    """Generate a Skill DAG via LLM with structured output.

    Implementation contract:
    - LLM call MUST use structured output / JSON mode with the DAG schema.
    - Output passed through `selector.select_skill` per node to bind exact version.
    - Output passed through `validator.validate_dag` before returning.
    """
    # TODO: render prompts/planner.j2; call LLM; parse JSON; bind skill versions; validate.
    _ = inp
    raise NotImplementedError
