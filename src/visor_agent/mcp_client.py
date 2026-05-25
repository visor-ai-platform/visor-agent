"""Thin MCP client used by the agent to talk to visor-tools and visor-skills.

Both servers expose JSON-returning tools over streamable HTTP. We open a
fresh `ClientSession` per call: tool invocations from the agent are
infrequent (one per chat turn) and short-lived, so connection reuse is
not worth the lifecycle complexity in the single-process demo. Move to a
pooled session if you start invoking dozens of tools per turn.

Tool return values are JSON-encoded as the text content of the MCP
response (FastMCP's default for non-`Content` returns). We parse that
text into Python objects; structured-content variants are ignored on
purpose to keep the client format-stable across MCP SDK versions.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import HTTPException
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _tools_url() -> str:
    url = os.environ.get("VISOR_TOOLS_MCP_URL", "").strip()
    if not url:
        raise HTTPException(
            status_code=503,
            detail="VISOR_TOOLS_MCP_URL is not configured",
        )
    return url.rstrip("/")


def _skills_url() -> str:
    url = os.environ.get("VISOR_SKILLS_MCP_URL", "").strip()
    if not url:
        raise HTTPException(
            status_code=503,
            detail="VISOR_SKILLS_MCP_URL is not configured",
        )
    return url.rstrip("/")


@asynccontextmanager
async def _session(url: str) -> AsyncIterator[ClientSession]:
    async with streamablehttp_client(url) as (read, write, _meta):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _decode_result(result: Any, tool: str) -> Any:
    """Extract the JSON payload from an MCP CallToolResult."""
    if getattr(result, "isError", False):
        detail = _text_blocks(result) or "MCP tool reported an error"
        raise HTTPException(status_code=502, detail=f"{tool}: {detail}")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps non-dict return values in {"result": ...}.
        if set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured
    text = _text_blocks(result)
    if text is None:
        raise HTTPException(status_code=502, detail=f"{tool}: empty MCP response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{tool}: MCP response was not JSON: {exc}",
        ) from exc


def _text_blocks(result: Any) -> str | None:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts) if parts else None


async def _call_tool(url: str, tool: str, arguments: dict[str, Any]) -> Any:
    try:
        async with _session(url) as session:
            result = await session.call_tool(tool, arguments)
    except HTTPException:
        raise
    except Exception as exc:  # network / protocol failures
        raise HTTPException(
            status_code=503,
            detail=f"MCP call {tool} failed: {exc}",
        ) from exc
    return _decode_result(result, tool)


# ---------------------------------------------------------------------------
# visor-skills MCP tools
# ---------------------------------------------------------------------------


async def list_skills() -> list[dict[str, Any]]:
    data = await _call_tool(_skills_url(), "list_skills", {})
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="list_skills did not return a list")
    return data


async def get_skill_by_id(skill_id: str) -> dict[str, Any]:
    data = await _call_tool(_skills_url(), "get_skill_by_id", {"skill_id": skill_id})
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="get_skill_by_id did not return an object")
    return data


# ---------------------------------------------------------------------------
# visor-tools dataset_catalog MCP tools
# ---------------------------------------------------------------------------


async def search_specimens(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    data = await _call_tool(
        _tools_url(),
        "search_specimens",
        {"query": query, "max_results": max_results},
    )
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="search_specimens did not return a list")
    return data


async def get_specimen(specimen_id: str) -> dict[str, Any]:
    data = await _call_tool(
        _tools_url(),
        "get_specimen",
        {"specimen_id": specimen_id},
    )
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="get_specimen did not return an object")
    return data
