from fastapi.testclient import TestClient

from visor_agent.agent import NON_SKILL_COUNTS, RegistryDecision, _router_system_prompt, app


async def fake_fetch_skills():
    return [
        {
            "id": "visor.skills.mip",
            "version": "0.1.0",
            "type": "atomic",
            "interface": {"input_schema": {"required": ["src_zarr", "dst_zarr"]}},
            "resources": {"cpu": "8", "memory": "60Gi", "gpu": False},
        }
    ]


async def fake_deepseek_decision(message, context):
    _ = message, context
    return RegistryDecision(
        needs_skill_registry=True,
        source="deepseek",
        reason="The user is asking for available VISoR skills.",
    )


async def fake_non_skill_decision(message, context):
    _ = message, context
    return RegistryDecision(
        needs_skill_registry=False,
        source="deepseek",
        reason="User is greeting, not requesting information about skills.",
    )


def test_skill_query_returns_registry(monkeypatch):
    monkeypatch.setattr("visor_agent.agent._fetch_skills", fake_fetch_skills)
    monkeypatch.setattr("visor_agent.agent._ask_deepseek_for_decision", fake_deepseek_decision)
    client = TestClient(app)

    response = client.post("/chat", json={"message": "show me your skills"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "deepseek+registry"
    assert payload["skills"][0]["id"] == "visor.skills.mip"
    assert "visor.skills.mip" in payload["reply"]
    assert ">" not in payload["reply"]


def test_stream_starts_with_decision_reason(monkeypatch):
    monkeypatch.setattr("visor_agent.agent._fetch_skills", fake_fetch_skills)
    monkeypatch.setattr("visor_agent.agent._ask_deepseek_for_decision", fake_deepseek_decision)
    client = TestClient(app)

    with client.stream("POST", "/chat/stream", json={"message": "show me your skills"}) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "The user is asking for available VISoR skills." in body
    assert "Connecting to DeepSeek" not in body
    assert "Forwarded prompt to the LLM router" not in body
    assert "LLM routing decision" not in body


def test_router_prompt_is_externalized():
    prompt = _router_system_prompt()

    assert "VISoR agent runtime router" in prompt
    assert "same language as the user's latest input" in prompt


def test_skill_query_uses_ui_language(monkeypatch):
    monkeypatch.setattr("visor_agent.agent._fetch_skills", fake_fetch_skills)
    monkeypatch.setattr("visor_agent.agent._ask_deepseek_for_decision", fake_deepseek_decision)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "展示你的技能", "context": {"ui_language": "zh"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "我找到了 1 个已注册的 VISoR 技能" in payload["reply"]
    assert "需要 src_zarr, dst_zarr。" in payload["reply"]


def test_non_skill_reply_tracks_remaining_allowance(monkeypatch):
    NON_SKILL_COUNTS.clear()
    monkeypatch.setattr("visor_agent.agent._ask_deepseek_for_decision", fake_non_skill_decision)
    client = TestClient(app)

    response = client.post("/chat", json={"message": "hello", "context": {"session_id": "quota-test"}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["off_topic_remaining"] == 4
    assert payload["off_topic_limit"] == 5
    assert "show me your skills" not in payload["reply"]
    assert "generic app" in payload["scope_notice"]


def test_non_skill_reply_uses_ui_language(monkeypatch):
    NON_SKILL_COUNTS.clear()
    monkeypatch.setattr("visor_agent.agent._ask_deepseek_for_decision", fake_non_skill_decision)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "你好", "context": {"session_id": "zh-quota-test", "ui_language": "zh"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "我可以帮助查询 VISoR 技能" in payload["reply"]
    assert "技能模式" in payload["scope_notice"]


def test_non_skill_limit_message(monkeypatch):
    NON_SKILL_COUNTS.clear()
    monkeypatch.setattr("visor_agent.agent._ask_deepseek_for_decision", fake_non_skill_decision)
    client = TestClient(app)

    for _ in range(6):
        response = client.post("/chat", json={"message": "hello", "context": {"session_id": "limit-test"}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["off_topic_remaining"] == 0
    assert "reached the general-question limit" in payload["reply"]


def test_chat_requires_deepseek_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_FILE", raising=False)
    client = TestClient(app)

    response = client.post("/chat", json={"message": "show me your skills"})

    assert response.status_code == 503
    assert "DEEPSEEK_API_KEY" in response.json()["detail"]


def test_stream_reports_missing_deepseek_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_FILE", raising=False)
    client = TestClient(app)

    with client.stream("POST", "/chat/stream", json={"message": "show me your skills"}) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "event: error" in body
    assert "DEEPSEEK_API_KEY" in body