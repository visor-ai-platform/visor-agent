"""[SPEC-005] Gather small metadata for planning via visor-tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetContext:
    input_paths: list[str] = field(default_factory=list)
    vsr_metadata: dict[str, Any] = field(default_factory=dict)
    storage_entries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    slurm_partitions: list[dict[str, Any]] = field(default_factory=list)


async def gather_context(input_paths: list[str]) -> DatasetContext:
    """Collect KB-scale context needed by the planner.

    Implementations must call visor-tools storage/vsr_meta/slurm tools only for
    control-plane metadata. Large VSR array chunks are never read here.
    """
    _ = input_paths
    raise NotImplementedError