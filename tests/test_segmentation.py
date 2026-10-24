# ai-service/tests/test_segmentation.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert data["service"] == "nutrilens-ai-segmentation"

@pytest.mark.anyio
async def test_segment_food_valid_image():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Dummy 1x1 PNG bytes
        fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        files = {"image": ("dish_plate.png", fake_png, "image/png")}
        
        response = await client.post("/api/v1/vision/segment", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["totalComponentsDetected"] >= 1
        
        for item in data["items"]:
            assert "foodCode" in item
            assert 0.0 <= item["confidence"] <= 1.0
            assert len(item["polygonNormalized"]) >= 3
            # Check normalized bounds [0, 1]
            for pt in item["polygonNormalized"]:
                assert 0.0 <= pt[0] <= 1.0
                assert 0.0 <= pt[1] <= 1.0

@pytest.mark.anyio
async def test_segment_food_invalid_mime_type():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"image": ("malicious_script.sh", b"echo 'hack'", "text/plain")}
        response = await client.post("/api/v1/vision/segment", files=files)
        assert response.status_code == 400
        assert "Unsupported image format" in response.json()["detail"]

@pytest.mark.anyio
async def test_segment_food_empty_file():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"image": ("empty.jpg", b"", "image/jpeg")}
        response = await client.post("/api/v1/vision/segment", files=files)
        assert response.status_code == 400
        assert "Uploaded image payload is empty" in response.json()["detail"]
