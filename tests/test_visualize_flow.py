"""Tests for the two-turn visualize flow in visor-agent.

The agent talks to visor-tools / visor-skills exclusively over MCP, so these
tests monkey-patch the local MCP client module and never import the upstream
packages.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from visor_agent.agent import RegistryDecision, app


SPECIMEN: dict[str, Any] = {
    "id": "RM009",
    "kind": "specimen",
    "name": "Mouse brain RM009",
    "species": "Mus musculus",
    "description": "Adult mouse whole brain, light-sheet recon.",
    "imageVariants": ["recon-v2"],
    "meshVariants": ["v1"],
}

CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "RM009",
        "name": "Mouse brain RM009",
        "species": "Mus musculus",
        "description": "Adult mouse whole brain, light-sheet recon.",
        "image_variants": ["recon-v2"],
        "score": 5.0,
        "reason": "matched on name, species",
    },
    {
        "id": "ATLAS",
        "name": "Reference atlas",
        "species": "Mus musculus",
        "description": "Atlas entry without image chunks.",
        "image_variants": [],
        "score": 4.0,
        "reason": "matched on species",
    },
]


async def fake_visualize_decision(message: str, context: dict[str, Any]) -> RegistryDecision:
    _ = message, context
    return RegistryDecision(
        needs_skill_registry=False,
        source="deepseek",
        reason="User wants to visualize a specimen.",
        intent="visualize",
    )


async def fake_search_specimens(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    _ = query, max_results
    return CANDIDATES


async def fake_get_specimen(specimen_id: str) -> dict[str, Any]:
    assert specimen_id == "RM009"
    return SPECIMEN


def _patch_visualize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "visor_agent.agent._ask_deepseek_for_decision", fake_visualize_decision
    )
    monkeypatch.setattr("visor_agent.mcp_client.search_specimens", fake_search_specimens)
    monkeypatch.setattr("visor_agent.mcp_client.get_specimen", fake_get_specimen)


def test_visualize_intent_returns_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_visualize(monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "show me a mouse brain volume"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "deepseek+catalog"
    assert payload["reply"] == "I found 1 matching brain. Select it to preview:"
    assert payload["candidates"]
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["id"] == "RM009"
    assert payload["visualization"] is None


def test_selected_specimen_renders_visualization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visualize(monkeypatch)
    monkeypatch.setenv("VISOR_VISUALIZATION_BASE_URL", "https://catalog.test")
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "render it",
            "context": {"selected_specimen_id": "RM009"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "catalog+skill"
    viz = payload["visualization"]
    assert viz is not None
    assert viz["specimen_id"] == "RM009"
    assert viz["view_type"] == "volume"
    assert viz["zarr_url"] == "https://catalog.test/ome-zarr/RM009/image/recon-v2/3d"
    assert "Previewing" in payload["reply"]
    assert "Coronal" in payload["reply"]
    assert payload["view_suggestions"]
    assert [item["view_type"] for item in payload["view_suggestions"]] == ["xy", "xz", "yz"]


def test_selected_specimen_renders_coronal_slice_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visualize(monkeypatch)
    monkeypatch.setenv("VISOR_VISUALIZATION_BASE_URL", "https://catalog.test")
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "show coronal preview",
            "context": {"selected_specimen_id": "RM009", "view_type": "xy", "variant": "recon-v2"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    viz = payload["visualization"]
    assert viz is not None
    assert viz["view_type"] == "xy"
    assert viz["zarr_url"] == "https://catalog.test/ome-zarr/RM009/image/recon-v2/xy"
    assert "Coronal" in payload["reply"]
    suggestion_view_types = [item["view_type"] for item in payload["view_suggestions"]]
    assert "volume" in suggestion_view_types
    assert "xy" not in suggestion_view_types


def test_stream_emits_candidates_for_visualize_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visualize(monkeypatch)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "show me a mouse brain volume"},
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "User wants to visualize" in body
    assert "RM009" in body
    assert "deepseek+catalog" in body
