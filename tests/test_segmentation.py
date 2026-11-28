# ai-service/tests/test_segmentation.py
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app, engine
from app.schemas import SegmentedFoodItem
from app.segmenter import FOOD_MAPPING, FoodSegmentationEngine


@pytest.mark.anyio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert data["service"] == "nutrilens-ai-segmentation"


@pytest.mark.anyio
async def test_segment_food_valid_image_compact_and_diagnostic_modes(monkeypatch):
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
    captured_kwargs = []

    def fake_segment(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return (mock_qualified, mock_all)

    monkeypatch.setattr(engine, "segment_image_bytes", fake_segment)

    fake_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00"
        b"\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Default compact mode: scan_min_conf == conf_threshold (0.40), allDetected is None
        resp_compact = await client.post(
            "/api/v1/vision/segment",
            files={"image": ("dish_plate.png", fake_png, "image/png")},
        )
        assert resp_compact.status_code == 200
        data = resp_compact.json()
        assert data["status"] == "SUCCESS"
        assert data["totalComponentsDetected"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["foodCode"] == "BOILED_EGG"
        assert data["allDetected"] is None
        assert captured_kwargs[-1]["scan_min_conf"] == pytest.approx(0.40)

        # 2. Diagnostic mode (include_all=true): lower scan_min_conf and returns allDetected
        resp_diag = await client.post(
            "/api/v1/vision/segment?include_all=true",
            files={"image": ("dish_plate.png", fake_png, "image/png")},
        )
        assert resp_diag.status_code == 200
        diag_data = resp_diag.json()
        assert diag_data["totalComponentsDetected"] == 2
        assert len(diag_data["allDetected"]) == 2
        assert diag_data["allDetected"][1]["foodCode"] == "RICE_WHITE_COOKED"
        assert captured_kwargs[-1]["scan_min_conf"] == pytest.approx(0.20)


def test_auto_orientation_early_exits_and_clean_food_mapping():
    # Ensure misleading COCO utensil/dessert mappings were removed
    for bad_key in ("BOWL", "CUP", "CAKE", "DONUT", "HOT DOG"):
        assert bad_key not in FOOD_MAPPING

    # Verify early-exit: 0° returns 0 items, rot_90 returns 1 qualified item -> total 2 calls (not 4 or 5)
    test_engine = object.__new__(FoodSegmentationEngine)
    test_engine.weights_path = "mock.pt"
    call_count = {"n": 0}

    class FakeBox:
        def __init__(self, cls_id, conf):
            self.cls = [cls_id]
            self.conf = [conf]
            self.xyxy = [np.array([10.0, 10.0, 50.0, 50.0])]

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.masks = None

    class FakeModel:
        names = {0: "PHO_BO"}

        def __call__(self, img, conf=0.4, verbose=False):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [FakeResult([])]  # 0 deg misses
            return [FakeResult([FakeBox(0, 0.91)])]  # rot_90 succeeds

    test_engine.model = FakeModel()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    qual, all_items = test_engine.segment_image_bytes(
        b"",
        conf_threshold=0.40,
        auto_orient=True,
        decoded_img=dummy_img,
    )
    assert call_count["n"] == 2
    assert len(qual) == 1
    assert qual[0].foodCode == "PHO_BO"
    assert len(all_items) == 1


def test_per_class_confidence_calibration_and_balanced_weights():
    from collections import Counter
    from rebalance_dataset import compute_class_weights
    from train_vietfood_seg import compute_dataset_class_and_sample_weights

    # 1. Per-class confidence calibration in FoodSegmentationEngine
    test_engine = object.__new__(FoodSegmentationEngine)
    test_engine.weights_path = "mock.pt"
    test_engine.atomic_conf_threshold = 0.30
    test_engine.class_thresholds = {"CHAO_LONG": 0.50}

    assert test_engine.get_class_threshold("BOILED_EGG", 0.40) == pytest.approx(0.30)
    assert test_engine.get_class_threshold("APPLE", 0.40) == pytest.approx(0.30)
    assert test_engine.get_class_threshold("PHO_BO", 0.40) == pytest.approx(0.40)
    assert test_engine.get_class_threshold("CHAO_LONG", 0.40) == pytest.approx(0.50)

    class FakeBox:
        def __init__(self, cls_id, conf):
            self.cls = [cls_id]
            self.conf = [conf]
            self.xyxy = [np.array([10.0, 10.0, 50.0, 50.0])]

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.masks = None

    class FakeModel:
        names = {0: "CHAO_LONG", 1: "BOILED_EGG"}

        def __call__(self, img, conf=0.3, verbose=False):
            # Both detected at 0.34 -> BOILED_EGG qualifies (>=0.30), CHAO_LONG rejected (<0.50)
            return [FakeResult([FakeBox(0, 0.34), FakeBox(1, 0.34)])]

    test_engine.model = FakeModel()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    qual, _ = test_engine.segment_image_bytes(b"", conf_threshold=0.40, auto_orient=False, decoded_img=dummy_img)
    assert len(qual) == 1
    assert qual[0].foodCode == "BOILED_EGG"

    # 2. Smoothed inverse-frequency class weights & WeightedRandomSampler image weights
    w = compute_class_weights(Counter({0: 20, 7: 400, 14: 100}), num_classes=30)
    assert w[0] > w[14] > w[7]
    assert w[29] == 1.0  # Inactive class defaults safely to 1.0

    class DummyDataset:
        labels = [
            {"cls": np.array([[7]])},
            {"cls": np.array([[7]])},
            {"cls": np.array([[7]])},
            {"cls": np.array([[7]])},
            {"cls": np.array([[0]])},
        ]

    cls_w, sample_w = compute_dataset_class_and_sample_weights(DummyDataset(), num_classes=30)
    assert cls_w[0] > cls_w[7]
    assert sample_w[4] > sample_w[0]


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
