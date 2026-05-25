"""Build a visualization spec for a chosen specimen.

This was previously the `visor.skills.visualize_dataset` skill. The
agent now owns this responsibility directly because no compute is
involved — only URL construction and view-mode selection — and the
agent is also where any future large-data streaming for visualization
would live. Keeping it here removes the in-process import of
`visor_skills` from the agent.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ViewType = Literal["volume", "xy", "xz", "yz"]


class _Specimen(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str
    species: str | None = None
    description: str | None = None
    image_variants: list[str] = Field(default_factory=list, alias="imageVariants")
    mesh_variants: list[str] = Field(default_factory=list, alias="meshVariants")


class _Input(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    specimen: _Specimen
    view_type: ViewType = "volume"
    variant: str | None = None
    catalog_base_url: str
    channel: int | None = None
    resolution_level: int | None = None

    @field_validator("catalog_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        if not isinstance(v, str) or not v:
            raise ValueError("catalog_base_url must be a non-empty string")
        return v.rstrip("/")


_VIEW_TO_MODE: dict[ViewType, str] = {
    "volume": "3d",
    "xy": "xy",
    "xz": "xz",
    "yz": "yz",
}


def build_visualization_spec(params: dict[str, Any]) -> dict[str, Any]:
    """Build a galavi-ready VisualizationSpec dict.

    Inputs: specimen record (cerevi-server shape, accepts both snake_case
    and camelCase aliases), `view_type`, optional `variant`, and a
    `catalog_base_url` that the browser can reach (typically the
    same-origin `/cerevi` gateway, never the LAN catalog directly).
    """
    args = _Input(**params)
    specimen = args.specimen

    if args.variant is not None:
        variant = args.variant
    elif specimen.image_variants:
        variant = specimen.image_variants[0]
    else:
        raise ValueError(
            f"specimen {specimen.id} has no imageVariants and no variant override"
        )

    mode = _VIEW_TO_MODE[args.view_type]
    zarr_url = f"{args.catalog_base_url}/ome-zarr/{specimen.id}/image/{variant}/{mode}"
    mesh_variant = specimen.mesh_variants[0] if specimen.mesh_variants else None

    return {
        "specimen_id": specimen.id,
        "specimen_name": specimen.name,
        "species": specimen.species,
        "view_type": args.view_type,
        "variant": variant,
        "zarr_url": zarr_url,
        "channel": args.channel,
        "resolution_level": args.resolution_level,
        "mesh_variant": mesh_variant,
    }
