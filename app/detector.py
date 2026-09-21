"""Pretrained YOLO detector: one-time model loading and CPU inference.

Runtime preference:
1. OpenVINO IR (yolo11n_openvino_model/) when present — fastest CPU path.
2. One-time OpenVINO export at startup when the openvino package is installed
   and the IR is not present yet (cached in the repository root).
3. Plain PyTorch CPU (yolo11n.pt) as the always-available fallback.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

# Smallest current pretrained Ultralytics detection model (verified against
# the installed ultralytics package: downloads from the official v8.4.0
# release, 80 COCO classes).
MODEL_NAME = "yolo11n.pt"
OPENVINO_DIR_NAME = "yolo11n_openvino_model"
CONF_THRESHOLD = 0.40
MAX_BOXES = 20


class DetectorError(RuntimeError):
    """Model loading or inference failed."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector:
    """Wraps a pretrained Ultralytics YOLO model.

    The model is loaded once per process and reused for every request.
    """

    def __init__(self, modelName: str = MODEL_NAME, confThreshold: float = CONF_THRESHOLD) -> None:
        self.modelName = modelName
        self.confThreshold = confThreshold
        self.model: Any = None
        self.runtime = "pytorch-cpu"
        self.weightsPath: str | None = None

    def loadModel(self) -> None:
        # LOAD THE MODEL ONCE; LATER CALLS ARE NO-OPS
        if self.model is not None:
            return
        try:
            self.weightsPath, self.runtime = self._resolveWeights()
            self.model = YOLO(self.weightsPath)
        except Exception as exc:
            raise DetectorError(f"could not load YOLO model: {exc}") from exc

    def _resolveWeights(self) -> tuple[str, str]:
        openvinoDir = REPO_ROOT / OPENVINO_DIR_NAME
        if openvinoDir.is_dir():
            return str(openvinoDir), "openvino-cpu"
        try:
            import openvino  # noqa: F401
        except ImportError:
            return self.modelName, "pytorch-cpu"
        try:
            YOLO(self.modelName).export(
                format="openvino", imgsz=640, device="cpu", verbose=False
            )
        except Exception as exc:
            print(f"WARNING: OpenVINO export failed, using PyTorch CPU: {exc}", file=sys.stderr)
        if (REPO_ROOT / OPENVINO_DIR_NAME).is_dir():
            return str(REPO_ROOT / OPENVINO_DIR_NAME), "openvino-cpu"
        return self.modelName, "pytorch-cpu"

    @property
    def isLoaded(self) -> bool:
        return self.model is not None

    def metadata(self) -> dict[str, str]:
        return {"model": self.modelName, "runtime": self.runtime, "device": "cpu"}

    def detect(self, imageData: np.ndarray) -> tuple[int, int, float, list[Detection]]:
        """Run inference on an RGB numpy image.

        Returns (width, height, inferenceMs, detections) with bounding-box
        coordinates in the submitted image's pixel space.
        """
        if self.model is None:
            raise DetectorError("detector not loaded; call loadModel() first")
        try:
            startTime = time.perf_counter()
            results = self.model.predict(
                imageData,
                conf=self.confThreshold,
                device="cpu",
                verbose=False,
            )
            inferenceMs = (time.perf_counter() - startTime) * 1000.0
        except Exception as exc:
            raise DetectorError(f"inference failed: {exc}") from exc

        result = results[0]
        names: dict[int, str] = result.names
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                classId = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_id=classId,
                        class_name=str(names.get(classId, f"class_{classId}")),
                        confidence=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        # MOST CONFIDENT DETECTIONS FIRST
        detections.sort(key=lambda det: det.confidence, reverse=True)
        width = int(result.orig_shape[1])
        height = int(result.orig_shape[0])
        return width, height, inferenceMs, detections[:MAX_BOXES]


_detector: Detector | None = None


def getDetector() -> Detector:
    """Return the process-wide detector, loading it on first use."""
    global _detector
    if _detector is None:
        _detector = Detector()
        _detector.loadModel()
    return _detector


def detectionToDict(detection: Detection) -> dict[str, Any]:
    return asdict(detection)
