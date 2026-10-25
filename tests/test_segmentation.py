# ai-service/tests/test_segmentation.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app, engine
from app.schemas import SegmentedFoodItem

@pytest.mark.anyio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert data["service"] == "nutrilens-ai-segmentation"

@pytest.mark.anyio
async def test_segment_food_valid_image(monkeypatch):
    # Mock segmentation to return sample qualified and low-confidence items
    mock_qualified = [
        SegmentedFoodItem(
            foodCode="BOILED_EGG",
            confidence=0.88,
            polygonNormalized=[[0.2, 0.2], [0.5, 0.2], [0.5, 0.6], [0.2, 0.6]],
            boundingBox=[50.0, 50.0, 200.0, 200.0]
        )
    ]
    mock_all = [
        mock_qualified[0],
        SegmentedFoodItem(
            foodCode="RICE_WHITE_COOKED",
            confidence=0.25,
            polygonNormalized=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
            boundingBox=[20.0, 20.0, 150.0, 150.0]
        )
    ]
    monkeypatch.setattr(engine, "segment_image_bytes", lambda *args, **kwargs: (mock_qualified, mock_all))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        files = {"image": ("dish_plate.png", fake_png, "image/png")}
        
        response = await client.post("/api/v1/vision/segment", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["totalComponentsDetected"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["foodCode"] == "BOILED_EGG"
        assert len(data["allDetected"]) == 2
        assert data["allDetected"][1]["foodCode"] == "RICE_WHITE_COOKED"
        assert data["allDetected"][1]["confidence"] == 0.25

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
