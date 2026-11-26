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
            if "/api/auth/internal/users/" in url:
                return Response(200, json={
                    "fitnessGoal": "LOSE_WEIGHT",
                    "weightKg": 64.0,
                    "heightCm": 170.0,
                    "dailyCalorieGoal": 1800,
                })
            if "/api/meal-logs/internal/today-calories" in url:
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
