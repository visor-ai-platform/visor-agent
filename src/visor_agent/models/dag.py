"""Pydantic models for structured Skill DAG planning."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DagNode(BaseModel):
    id: str
    params: dict[str, Any]
    deps: list[int] = Field(default_factory=list)


class SkillDAG(BaseModel):
    dag_id: str
    skills: list[DagNode]


class PlannerInput(BaseModel):
    user_intent: str
    available_skills: list[dict[str, Any]]
    context: dict[str, Any] = Field(default_factory=dict)