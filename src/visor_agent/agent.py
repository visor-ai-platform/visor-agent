"""FastAPI entrypoint for the VISoR agent runtime."""
from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


app = FastAPI(title="visor-agent")


class PlanRequest(BaseModel):
    user_intent: str
    context: dict | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    skills: list[dict[str, Any]] = Field(default_factory=list)
    source: str
    off_topic_remaining: int | None = None
    off_topic_limit: int | None = None
    scope_notice: str | None = None
    candidates: list[dict[str, Any]] | None = None
    visualization: dict[str, Any] | None = None
    view_suggestions: list[dict[str, Any]] | None = None


class RegistryDecision(BaseModel):
    needs_skill_registry: bool
    source: str
    reason: str = ""
    intent: str | None = None  # "skills" | "visualize" | "other" | None


ROUTER_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "router_system.md"
SUPPORTED_LANGUAGES = {"en", "zh"}
NON_SKILL_LIMIT = 5
# In-memory by design for the single-process local demo. Move this to Redis or
# session storage before running multiple agent replicas.
NON_SKILL_COUNTS: dict[str, int] = {}

COPY = {
    "en": {
        "visualize_intro_one": "I found 1 matching brain. Select it to preview:",
        "visualize_intro_many": "I found {count} matching brains. Select one to preview:",
        "visualize_no_match": "I could not find a matching brain in the catalog.",
        "visualize_ready": "Previewing {name} in {view_label}.",
        "visualize_suggestion": "Do you want to preview {options} next?",
        "visualize_search": "Searching brain catalog",
        "visualize_found": "Found {count} matching brains",
        "visualize_render": "Preparing preview for {name}",
        "catalog_unavailable": "Dataset catalog is unavailable: {error}",
        "non_skill_reply": (
            "I can help with VISoR skills, capabilities, and registry-backed workflows. "
            "This request looks outside that scope, so I did not fetch the skill registry."
        ),
        "non_skill_limit_reply": (
            "VISoR is staying focused on skills to keep this demo efficient. "
            "This session has reached the general-question limit."
        ),
        "remaining_one": "1 general-question check left",
        "remaining_many": "{remaining} general-question checks left",
        "scope_notice": "Skill mode: {remaining_text}. Use generic app (DeepSeek etc.) for broader chat.",
        "no_skills": "I do not see any registered skills yet.",
        "skills_intro": "I found {count} registered VISoR skills:",
        "no_required_inputs": "no required inputs",
        "skill_requires": "requires {required}.",
        "read_registry": "Read skill registry service",
        "found_skills": "Found {count} registered skills",
        "constructed_response": "Constructed final skill list response",
    },
    "zh": {
        "visualize_intro_one": "我找到了 1 个匹配脑数据，请选择预览：",
        "visualize_intro_many": "我找到了 {count} 个匹配脑数据，请选择一个预览：",
        "visualize_no_match": "在数据目录中没有找到与该描述匹配的数据集。",
        "visualize_ready": "正在以 {view_label} 预览 {name}。",
        "visualize_suggestion": "接下来要预览 {options} 吗？",
        "visualize_search": "检索数据集目录",
        "visualize_found": "找到 {count} 个候选",
        "visualize_render": "正在为 {name} 准备可视化",
        "catalog_unavailable": "数据集目录不可用：{error}",
        "non_skill_reply": (
            "我可以帮助查询 VISoR 技能、能力和基于注册表的工作流。"
            "这个请求看起来超出了当前范围，所以我没有读取技能注册表。"
        ),
        "non_skill_limit_reply": "VISoR 会继续聚焦技能演示。本会话已经达到通用问题次数上限。",
        "remaining_one": "还剩 1 次通用问题检查",
        "remaining_many": "还剩 {remaining} 次通用问题检查",
        "scope_notice": "技能模式：{remaining_text}。更广泛的聊天请使用通用应用（DeepSeek 等）。",
        "no_skills": "我还没有看到已注册的技能。",
        "skills_intro": "我找到了 {count} 个已注册的 VISoR 技能：",
        "no_required_inputs": "无必填输入",
        "skill_requires": "需要 {required}。",
        "read_registry": "读取技能注册表服务",
        "found_skills": "找到 {count} 个已注册技能",
        "constructed_response": "已生成最终技能列表回复",
    },
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _secret(name: str) -> str:
    file_path = _env(f"{name}_FILE")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return _env(name)


def _router_system_prompt() -> str:
    configured_path = _env("VISOR_ROUTER_PROMPT_FILE")
    prompt_path = Path(configured_path) if configured_path else ROUTER_SYSTEM_PROMPT_PATH
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"router prompt could not be read: {exc}") from exc


def _normalize_language(language: str | None) -> str:
    value = (language or "").strip().lower()
    if value.startswith("zh") or value in {"cn", "中文", "中"}:
        return "zh"
    return "en"


def _ui_language(context: dict[str, Any]) -> str:
    value = context.get("ui_language") or context.get("language")
    return _normalize_language(str(value) if value is not None else None)


def _copy(language: str, key: str) -> str:
    normalized = _normalize_language(language)
    return COPY.get(normalized, COPY["en"])[key]


def _deepseek_api_key() -> str:
    try:
        api_key = _secret("DEEPSEEK_API_KEY")
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"DeepSeek API key file could not be read: {exc}",
        ) from exc
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="DeepSeek API key is not configured. Set DEEPSEEK_API_KEY or DEEPSEEK_API_KEY_FILE.",
        )
    return api_key


def _registry_url() -> str:
    return _env("VISOR_SKILLS_MCP_URL").rstrip("/")


def _tools_url() -> str:
    return _env("VISOR_TOOLS_MCP_URL").rstrip("/")


async def _fetch_skills() -> list[dict[str, Any]]:
    """Fetch the skill registry via the visor-skills MCP server."""
    from visor_agent.mcp_client import list_skills

    return await list_skills()


def _deepseek_reasoning_body() -> dict[str, Any] | None:
    enabled = _env("DEEPSEEK_ENABLE_THINKING", "true").lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    return {
        "thinking": {"type": "enabled"},
        "reasoning_effort": _env("DEEPSEEK_REASONING_EFFORT", "high"),
    }


def _required_inputs(skill: dict[str, Any]) -> list[str]:
    input_schema = skill.get("interface", {}).get("input_schema", {})
    required = input_schema.get("required", [])
    return [str(item) for item in required]


def _format_skills(skills: list[dict[str, Any]], language: str = "en") -> str:
    language = _normalize_language(language)
    if not skills:
        return _copy(language, "no_skills")

    lines = [_copy(language, "skills_intro").format(count=len(skills))]
    for skill in skills:
        required = ", ".join(_required_inputs(skill)) or _copy(language, "no_required_inputs")
        lines.append(
            f"- {skill['id']} ({skill.get('type', 'unknown')} v{skill.get('version', 'unknown')}): "
            f"{_copy(language, 'skill_requires').format(required=required)}"
        )
    return "\n".join(lines)


async def _ask_deepseek_for_decision(message: str, context: dict[str, Any]) -> RegistryDecision:
    api_key = _deepseek_api_key()
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="openai package is not installed") from exc

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    context_json = json.dumps(context, sort_keys=True)
    request: dict[str, Any] = {
        "model": _env("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": [
            {
                "role": "system",
                "content": _router_system_prompt(),
            },
            {"role": "system", "content": f"Request context JSON: {context_json}"},
            {"role": "user", "content": message},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    reasoning_body = _deepseek_reasoning_body()
    if reasoning_body:
        request["extra_body"] = reasoning_body

    try:
        completion = await client.chat.completions.create(**request)
    except Exception:
        if not reasoning_body:
            raise
        request.pop("extra_body", None)
        completion = await client.chat.completions.create(**request)

    content = completion.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="DeepSeek returned invalid router JSON")
    intent_raw = payload.get("intent")
    intent = intent_raw if intent_raw in {"skills", "visualize", "other"} else None
    needs = bool(payload.get("needs_skill_registry"))
    # Backfill needs_skill_registry from intent when only intent is provided.
    if intent == "skills":
        needs = True
    elif intent in {"visualize", "other"} and "needs_skill_registry" not in payload:
        needs = False
    return RegistryDecision(
        needs_skill_registry=needs,
        source="deepseek",
        reason=str(payload.get("reason") or ""),
        intent=intent,
    )


async def _decide_registry_need(message: str, context: dict[str, Any]) -> RegistryDecision:
    return await _ask_deepseek_for_decision(message, context)


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _status(kind: str, title: str, content: str | list[str] = "") -> dict[str, Any]:
    return {"kind": kind, "title": title, "content": content, "message": title}


def _error_message(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, HTTPException):
        return exc.status_code, str(exc.detail)
    return 502, f"DeepSeek request failed: {exc}"


def _session_key(context: dict[str, Any]) -> str:
    session_id = context.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()[:128]
    return "anonymous"


def _non_skill_payload(context: dict[str, Any], source: str) -> dict[str, Any]:
    language = _ui_language(context)
    key = _session_key(context)
    current_count = NON_SKILL_COUNTS.get(key, 0)
    if current_count >= NON_SKILL_LIMIT:
        remaining = 0
        reply = _copy(language, "non_skill_limit_reply")
    else:
        current_count += 1
        NON_SKILL_COUNTS[key] = current_count
        remaining = NON_SKILL_LIMIT - current_count
        reply = _copy(language, "non_skill_reply")

    if remaining == 1:
        remaining_text = _copy(language, "remaining_one")
    else:
        remaining_text = _copy(language, "remaining_many").format(remaining=remaining)
    notice = _copy(language, "scope_notice").format(remaining_text=remaining_text)
    return {
        "reply": reply,
        "skills": [],
        "source": source,
        "off_topic_remaining": remaining,
        "off_topic_limit": NON_SKILL_LIMIT,
        "scope_notice": notice,
    }


# ---------------------------------------------------------------------------
# Visualize-dataset flow (two-turn: search candidates → render selected)
# ---------------------------------------------------------------------------

_DEFAULT_VIEW_TYPE = "volume"

_VIEW_LABELS = {
    "en": {
        "volume": "3D",
        "xy": "Coronal",
        "xz": "Horizontal",
        "yz": "Sagittal",
    },
    "zh": {
        "volume": "三维",
        "xy": "冠状",
        "xz": "水平",
        "yz": "矢状",
    },
}

_VIEW_DESCRIPTIONS = {
    "en": {
        "volume": "3D volume",
        "xy": "xy projection",
        "xz": "xz projection",
        "yz": "yz projection",
    },
    "zh": {
        "volume": "可交互三维体视图",
        "xy": "带切片选择的 xy 投影切片",
        "xz": "带切片选择的 xz 投影切片",
        "yz": "带切片选择的 yz 投影切片",
    },
}


def _view_label(view_type: str | None, language: str) -> str:
    normalized = _normalize_language(language)
    labels = _VIEW_LABELS.get(normalized, _VIEW_LABELS["en"])
    return labels.get(view_type or _DEFAULT_VIEW_TYPE, labels[_DEFAULT_VIEW_TYPE])


def _join_options(labels: list[str], language: str) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if _normalize_language(language) == "zh":
        return "、".join(labels)
    if len(labels) == 2:
        return " or ".join(labels)
    return f"{', '.join(labels[:-1])}, or {labels[-1]}"


def _view_suggestions(visualization: dict[str, Any], language: str) -> list[dict[str, Any]]:
    normalized = _normalize_language(language)
    current = str(visualization.get("view_type") or _DEFAULT_VIEW_TYPE)
    order = ["xy", "xz", "yz"] if current == "volume" else ["volume", "xy", "xz", "yz"]
    labels = _VIEW_LABELS.get(normalized, _VIEW_LABELS["en"])
    descriptions = _VIEW_DESCRIPTIONS.get(normalized, _VIEW_DESCRIPTIONS["en"])
    suggestions: list[dict[str, Any]] = []
    for view_type in order:
        if view_type == current:
            continue
        label = labels[view_type]
        suggestions.append(
            {
                "view_type": view_type,
                "label": label,
                "description": descriptions[view_type],
                "prompt": f"Preview {label} for {visualization.get('specimen_name') or visualization.get('specimen_id')}",
            }
        )
    return suggestions


def _is_visualize_intent(decision: RegistryDecision, message: str) -> bool:
    """Trust the LLM's explicit intent label; fall back to a tiny keyword
    heuristic only when the router didn't classify and didn't claim skills."""
    if decision.intent == "visualize":
        return True
    if decision.intent is not None:
        return False
    if decision.needs_skill_registry:
        return False
    lowered = (message or "").lower()
    return any(
        kw in lowered
        for kw in (
            "visualize",
            "visualise",
            "render",
            "view the",
            "可视化",
            "看一下",
            "显示",
        )
    )


def _selected_specimen_id(context: dict[str, Any]) -> str | None:
    value = context.get("selected_specimen_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _view_type_from_context(context: dict[str, Any]) -> str:
    value = context.get("view_type")
    if isinstance(value, str) and value in {"volume", "xy", "xz", "yz"}:
        return value
    return _DEFAULT_VIEW_TYPE


def _visualization_base_url() -> str:
    configured = _env("VISOR_VISUALIZATION_BASE_URL")
    if configured:
        return configured.rstrip("/")
    # Dev fallback when visor-app's same-origin /cerevi gateway is not
    # configured: trust VISOR_CATALOG_URL (the cerevi-server origin the
    # MCP tool already targets) so the browser can reach the same host.
    fallback = _env("VISOR_CATALOG_URL")
    if not fallback:
        raise HTTPException(
            status_code=503,
            detail="VISOR_VISUALIZATION_BASE_URL or VISOR_CATALOG_URL must be set",
        )
    return fallback.rstrip("/")


async def _search_visualize_candidates(message: str) -> list[dict[str, Any]]:
    from visor_agent.mcp_client import search_specimens

    candidates = await search_specimens(message, max_results=5)
    # Drop atlas-only / variant-less hits so the user cannot pick a
    # specimen the renderer has no image pyramid for.
    return [c for c in candidates if c.get("image_variants")]


async def _build_visualization(context: dict[str, Any]) -> dict[str, Any]:
    from visor_agent.mcp_client import get_specimen
    from visor_agent.visualize import build_visualization_spec

    specimen_id = _selected_specimen_id(context)
    if not specimen_id:
        raise HTTPException(
            status_code=400,
            detail="selected_specimen_id required for visualization",
        )
    specimen = await get_specimen(specimen_id)

    spec_input: dict[str, Any] = {
        "specimen": specimen,
        "view_type": _view_type_from_context(context),
        "catalog_base_url": _visualization_base_url(),
    }
    variant = context.get("variant")
    if variant:
        spec_input["variant"] = variant
    return build_visualization_spec(spec_input)


def _visualize_candidates_payload(
    candidates: list[dict[str, Any]],
    language: str,
    source: str,
) -> dict[str, Any]:
    if not candidates:
        return {
            "reply": _copy(language, "visualize_no_match"),
            "skills": [],
            "source": source,
            "candidates": [],
        }
    intro_key = "visualize_intro_one" if len(candidates) == 1 else "visualize_intro_many"
    reply = _copy(language, intro_key).format(count=len(candidates))
    return {
        "reply": reply,
        "skills": [],
        "source": source,
        "candidates": candidates,
    }


def _visualize_render_payload(
    visualization: dict[str, Any],
    language: str,
    source: str,
) -> dict[str, Any]:
    view_type = str(visualization.get("view_type") or _DEFAULT_VIEW_TYPE)
    suggestions = _view_suggestions(visualization, language)
    reply = _copy(language, "visualize_ready").format(
        name=visualization.get("specimen_name", visualization.get("specimen_id", "")),
        view_label=_view_label(view_type, language),
    )
    if suggestions:
        reply = " ".join(
            [
                reply,
                _copy(language, "visualize_suggestion").format(
                    options=_join_options([str(item["label"]) for item in suggestions], language)
                ),
            ]
        )
    return {
        "reply": reply,
        "skills": [],
        "source": source,
        "visualization": visualization,
        "view_suggestions": suggestions,
    }


async def _chat_event_stream(req: ChatRequest) -> AsyncIterator[str]:
    try:
        language = _ui_language(req.context)

        # Turn 2 short-circuit: user already picked a specimen.
        if _selected_specimen_id(req.context):
            yield _sse(
                "status",
                _status(
                    "thought",
                    _copy(language, "visualize_render").format(
                        name=_selected_specimen_id(req.context) or ""
                    ),
                ),
            )
            visualization = await _build_visualization(req.context)
            yield _sse("final", _visualize_render_payload(visualization, language, "catalog+skill"))
            return

        decision = await _decide_registry_need(req.message, req.context)
        if decision.reason:
            yield _sse("status", _status("thought", decision.reason))

        if _is_visualize_intent(decision, req.message):
            yield _sse("status", _status("read", _copy(language, "visualize_search"), _tools_url() or "catalog"))
            candidates = await _search_visualize_candidates(req.message)
            yield _sse(
                "status",
                _status(
                    "found",
                    _copy(language, "visualize_found").format(count=len(candidates)),
                    [c["id"] for c in candidates],
                ),
            )
            yield _sse(
                "final",
                _visualize_candidates_payload(candidates, language, "deepseek+catalog"),
            )
            return

        if not decision.needs_skill_registry:
            yield _sse("final", _non_skill_payload(req.context, decision.source))
            return

        yield _sse("status", _status("read", _copy(language, "read_registry"), _registry_url() or "local registry"))
        skills = await _fetch_skills()
        yield _sse(
            "status",
            _status("found", _copy(language, "found_skills").format(count=len(skills)), [skill["id"] for skill in skills]),
        )
        yield _sse("status", _status("thought", _copy(language, "constructed_response")))
        yield _sse(
            "final",
            {"reply": _format_skills(skills, language), "skills": skills, "source": "deepseek+registry"},
        )
    except Exception as exc:
        status_code, message = _error_message(exc)
        yield _sse("error", {"message": message, "status_code": status_code})


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    language = _ui_language(req.context)

    # Turn 2: user picked a specimen — render directly.
    if _selected_specimen_id(req.context):
        visualization = await _build_visualization(req.context)
        return ChatResponse(**_visualize_render_payload(visualization, language, "catalog+skill"))

    try:
        decision = await _decide_registry_need(req.message, req.context)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=f"DeepSeek request failed: {exc}") from exc

    if _is_visualize_intent(decision, req.message):
        candidates = await _search_visualize_candidates(req.message)
        return ChatResponse(**_visualize_candidates_payload(candidates, language, "deepseek+catalog"))

    if not decision.needs_skill_registry:
        return ChatResponse(**_non_skill_payload(req.context, decision.source))

    skills = await _fetch_skills()
    return ChatResponse(reply=_format_skills(skills, language), skills=skills, source="deepseek+registry")


@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _chat_event_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/skills")
async def skills_endpoint() -> list[dict[str, Any]]:
    return await _fetch_skills()


@app.get("/skills/{skill_id}")
async def skill_endpoint(skill_id: str) -> dict[str, Any]:
    from visor_agent.mcp_client import get_skill_by_id

    try:
        return await get_skill_by_id(skill_id)
    except HTTPException as exc:
        detail = str(exc.detail)
        if "unknown skill" in detail.lower():
            raise HTTPException(status_code=404, detail=detail) from exc
        raise


@app.post("/plan")
async def plan_endpoint(req: PlanRequest) -> dict:
    _ = req
    raise HTTPException(
        status_code=501,
        detail="DAG planning is not implemented yet; use /chat for the registry-backed demo.",
    )
