# ai-service/tests/test_security_and_rag.py
import pytest
from httpx import AsyncClient, ASGITransport, Response
import app.main as main_module
from app.chatbot.rag_engine import ChatbotRAGEngine, chatbot_engine


@pytest.mark.anyio
async def test_debug_endpoints_blocked_by_default(monkeypatch):
    monkeypatch.setattr(main_module, "ENABLE_DEBUG_ENDPOINTS", False)
    async with AsyncClient(transport=ASGITransport(app=main_module.app), base_url="http://test") as client:
        for ep in ("/api/v1/vision/last-capture", "/api/v1/vision/last-prediction", "/api/v1/vision/diagnose-last"):
            resp = await client.get(ep)
            assert resp.status_code == 403


@pytest.mark.anyio
async def test_segment_food_rejects_oversized_payload(monkeypatch):
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 100)
    oversized = b"\xff\xd8\xff" + b"0" * 150
    async with AsyncClient(transport=ASGITransport(app=main_module.app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/vision/segment",
            files={"image": ("big.jpg", oversized, "image/jpeg")},
        )
        assert resp.status_code == 413


@pytest.mark.anyio
async def test_chatbot_history_enforces_auth_and_anti_idor():
    conv_id = "conv-anti-idor-test-1"
    chatbot_engine.clear_conversation_history(conv_id)
    chatbot_engine.append_message_to_history(conv_id, "user", "Hôm nay ăn gì?", owner_username="alice")
    chatbot_engine.append_message_to_history(conv_id, "assistant", "Phở bò rất phù hợp!", owner_username="alice")

    async with AsyncClient(transport=ASGITransport(app=main_module.app), base_url="http://test") as client:
        # Missing user header -> 401
        r_unauth = await client.get("/api/v1/chatbot/history", params={"conversationId": conv_id})
        assert r_unauth.status_code == 401

        # Different user -> 403 on GET and DELETE
        r_bob_get = await client.get(
            "/api/v1/chatbot/history",
            params={"conversationId": conv_id},
            headers={"X-User-Username": "bob", "X-User-Role": "ROLE_USER"},
        )
        assert r_bob_get.status_code == 403

        r_bob_del = await client.delete(
            "/api/v1/chatbot/history",
            params={"conversationId": conv_id},
            headers={"X-User-Username": "bob", "X-User-Role": "ROLE_USER"},
        )
        assert r_bob_del.status_code == 403

        # Owner -> 200
        r_alice = await client.get(
            "/api/v1/chatbot/history",
            params={"conversationId": conv_id},
            headers={"X-User-Username": "alice", "X-User-Role": "ROLE_USER"},
        )
        assert r_alice.status_code == 200
        assert len(r_alice.json()["messages"]) == 2

        # Admin -> 200
        r_admin = await client.get(
            "/api/v1/chatbot/history",
            params={"conversationId": conv_id},
            headers={"X-User-Username": "admin", "X-User-Role": "ROLE_ADMIN"},
        )
        assert r_admin.status_code == 200

    chatbot_engine.clear_conversation_history(conv_id)


@pytest.mark.anyio
async def test_rag_fetch_personal_context_parses_raw_float_and_total_kcal(monkeypatch):
    engine = ChatbotRAGEngine()

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None):
            if "/api/v1/auth/internal/users/" in url:
                return Response(200, json={
                    "fitnessGoal": "LOSE_WEIGHT",
                    "weightKg": 64.0,
                    "heightCm": 170.0,
                    "dailyCalorieGoal": 1800,
                })
            if "/api/v1/meal-logs/internal/today-calories" in url:
                # meal-service returns raw JSON number (BigDecimal), not a dict
                return Response(200, json=650.5)
            if "/api/v1/nutrition/meal-logs" in url:
                return Response(200, json=[
                    {"mealType": "BREAKFAST", "foodName": "Bánh Mì Thịt Nướng", "totalKcal": 450.5},
                    {"mealType": "SNACK", "foodName": "Trứng Luộc", "totalKcal": 200.0},
                ])
            return Response(404)

    monkeypatch.setattr("app.chatbot.rag_engine.httpx.AsyncClient", MockAsyncClient)

    ctx = await engine.fetch_personal_context("baophan")
    assert ctx["fitnessGoal"] == "LOSE_WEIGHT"
    assert ctx["dailyCalorieGoal"] == 1800
    assert ctx["todayConsumedKcal"] == pytest.approx(650.5)
    assert ctx["remainingKcalToday"] == pytest.approx(1149.5)
    assert len(ctx["todayMeals"]) == 2
    assert ctx["todayMeals"][0]["dishName"] == "Bánh Mì Thịt Nướng"
    assert ctx["todayMeals"][0]["calories"] == pytest.approx(450.5)


@pytest.mark.anyio
async def test_gemini_failover_uses_header_skips_503_cooldown_and_filters_thoughts(monkeypatch):
    engine = ChatbotRAGEngine()
    engine.api_key = "test-secret-api-key-value"
    monkeypatch.setattr("app.chatbot.rag_engine._load_gemini_api_key", lambda: "test-secret-api-key-value")
    monkeypatch.setenv("GEMINI_CHAT_MODELS", "model-overloaded,model-thinking,model-healthy")

    called_urls = []
    called_headers = []

    class MockGeminiClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None, timeout=None):
            called_urls.append(url)
            called_headers.append(headers or {})
            if "model-overloaded" in url:
                return Response(503, json={"error": {"code": 503, "message": "high demand"}})
            if "model-thinking" in url:
                # Returns internal thought part + actual visible reply
                return Response(200, json={
                    "candidates": [{
                        "content": {
                            "parts": [
                                {"thought": True, "text": "Internal English reasoning..."},
                                {"text": "Bạn nên chọn Ức Gà Nướng (380 kcal)."}
                            ]
                        }
                    }]
                })
            return Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "Fallback healthy reply."}]}}]
            })

    monkeypatch.setattr("app.chatbot.rag_engine.httpx.AsyncClient", MockGeminiClient)

    # First call: model-overloaded returns 503 -> cooldown, fails over to model-thinking -> strips thought part
    reply1, used1 = await engine._call_gemini_api("sys", [], "Hôm nay ăn gì?")
    assert used1 == "model-thinking"
    assert reply1 == "Bạn nên chọn Ức Gà Nướng (380 kcal)."
    assert "Internal English reasoning" not in reply1

    # Verify API key is NEVER placed in query string and IS sent via x-goog-api-key header
    for u, h in zip(called_urls, called_headers):
        assert "?key=" not in u
        assert "test-secret-api-key-value" not in u
        assert h.get("x-goog-api-key") == "test-secret-api-key-value"

    # Second call: model-overloaded is in cooldown, so it must be skipped immediately
    called_urls.clear()
    reply2, used2 = await engine._call_gemini_api("sys", [], "Gợi ý bữa tối?")
    assert used2 == "model-thinking"
    assert all("model-overloaded" not in u for u in called_urls)
    assert "model-overloaded" in engine.get_llm_status()["cooledDownModels"]
