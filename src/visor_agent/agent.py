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


class RegistryDecision(BaseModel):
    needs_skill_registry: bool
    source: str
    reason: str = ""


ROUTER_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "router_system.md"
SUPPORTED_LANGUAGES = {"en", "zh"}
NON_SKILL_LIMIT = 5
NON_SKILL_COUNTS: dict[str, int] = {}

COPY = {
    "en": {
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
    return _env("VISOR_REGISTRY_URL").rstrip("/")


async def _fetch_skills() -> list[dict[str, Any]]:
    registry_url = _registry_url()
    if registry_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{registry_url}/v1/skills")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"registry unavailable: {exc}") from exc
        if not isinstance(data, list):
            raise HTTPException(status_code=502, detail="registry returned a non-list response")
        return data

    try:
        from visor_skills.registry import load_registry
        from visor_skills.api import skill_to_dict
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="VISOR_REGISTRY_URL is not configured and visor_skills is not installed",
        ) from exc
    return [skill_to_dict(skill) for skill in load_registry().values()]


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
    return RegistryDecision(
        needs_skill_registry=bool(payload.get("needs_skill_registry")),
        source="deepseek",
        reason=str(payload.get("reason") or ""),
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


async def _chat_event_stream(req: ChatRequest) -> AsyncIterator[str]:
    try:
        language = _ui_language(req.context)
        decision = await _decide_registry_need(req.message, req.context)
        if decision.reason:
            yield _sse("status", _status("thought", decision.reason))

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
    try:
        decision = await _decide_registry_need(req.message, req.context)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=f"DeepSeek request failed: {exc}") from exc

    if not decision.needs_skill_registry:
        return ChatResponse(**_non_skill_payload(req.context, decision.source))

    skills = await _fetch_skills()
    return ChatResponse(reply=_format_skills(skills, _ui_language(req.context)), skills=skills, source="deepseek+registry")


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


@app.post("/plan")
async def plan_endpoint(req: PlanRequest) -> dict:
    _ = req
    raise HTTPException(
        status_code=501,
        detail="DAG planning is not implemented yet; use /chat for the registry-backed demo.",
    )
