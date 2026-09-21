"""API tests for the YOLO demo backend.

These tests use the REAL pretrained YOLO model end-to-end (no mocked
inference): the model loads once per test session and every /detect call
runs real CPU inference.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.main import app

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(scope="module")
def client():
    # TESTCLIENT CONTEXT TRIGGERS THE APP LIFESPAN, WHICH LOADS THE MODEL
    with TestClient(app) as testClient:
        yield testClient


def makeJpeg(width: int = 640, height: int = 480, color=(120, 90, 60)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_app_imports() -> None:
    assert main.app is app
    assert main.getDetector is not None


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"]
    assert body["device"] == "cpu"
    assert body["runtime"]


def test_index_serves_page(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<video" in resp.text
    assert "app.js" in resp.text


def test_static_files_served(client: TestClient) -> None:
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_detect_valid_image(client: TestClient) -> None:
    jpeg = makeJpeg()
    resp = client.post("/detect", files={"image": ("frame.jpg", jpeg, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 640
    assert body["height"] == 480
    assert body["inference_ms"] >= 0
    assert isinstance(body["detections"], list)


def test_detect_rejects_invalid_image(client: TestClient) -> None:
    resp = client.post(
        "/detect", files={"image": ("bad.jpg", b"this is definitely not an image", "image/jpeg")}
    )
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_detect_rejects_empty_upload(client: TestClient) -> None:
    resp = client.post("/detect", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400


def assertValidDetection(det: dict, width: int, height: int) -> None:
    assert isinstance(det["class_id"], int)
    assert det["class_id"] >= 0
    assert isinstance(det["class_name"], str) and det["class_name"]
    assert 0.0 <= det["confidence"] <= 1.0
    assert 0 <= det["x1"] < det["x2"] <= width
    assert 0 <= det["y1"] < det["y2"] <= height


def test_detect_real_photo(client: TestClient) -> None:
    # REAL PRETRAINED INFERENCE ON A PHOTO OF PEOPLE AND A BUS
    with (ASSETS / "bus.jpg").open("rb") as fh:
        resp = client.post("/detect", files={"image": ("bus.jpg", fh.read(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detections"], "expected at least one detection on bus.jpg"
    names = {det["class_name"] for det in body["detections"]}
    assert "person" in names
    for det in body["detections"]:
        assertValidDetection(det, body["width"], body["height"])


def test_detection_structure_on_plain_image(client: TestClient) -> None:
    resp = client.post("/detect", files={"image": ("plain.jpg", makeJpeg(), "image/jpeg")})
    body = resp.json()
    assert resp.status_code == 200
    for det in body["detections"]:
        assertValidDetection(det, body["width"], body["height"])
