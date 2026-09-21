"""Simple manual benchmark: time real YOLO CPU inference.

Usage:
    .venv/bin/python scripts/bench.py [imagePath]

Runs each warm model on the given image (default tests/assets/bus.jpg)
several times and prints median latency. No framework, no storage.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF = 0.40


def bench(model: YOLO, label: str, image: np.ndarray, warmup: int = 3, runs: int = 10) -> float:
    for _ in range(warmup):
        model.predict(image, conf=CONF, device="cpu", verbose=False)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = model.predict(image, conf=CONF, device="cpu", verbose=False)
        times.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(times)
    boxes = result[0].boxes
    labels = [
        f"{result[0].names[int(box.cls.item())]} {box.conf.item():.2f}"
        for box in boxes
    ]
    print(f"{label:>14}: median {median:7.1f} ms  (~{1000 / median:4.1f} fps)  {labels}")
    return median


def main() -> None:
    imagePath = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "tests" / "assets" / "bus.jpg"
    image = np.asarray(Image.open(imagePath).convert("RGB"))
    print(f"image: {imagePath.name} {image.shape[1]}x{image.shape[0]}  conf={CONF}")

    pytorch = YOLO(str(REPO_ROOT / "yolo11n.pt"))
    bench(pytorch, "pytorch-cpu", image)

    openvinoDir = REPO_ROOT / "yolo11n_openvino_model"
    if openvinoDir.exists():
        openvino = YOLO(str(openvinoDir))
        bench(openvino, "openvino-cpu", image)
    else:
        print("openvino IR not found; export with: model.export(format='openvino')")


if __name__ == "__main__":
    main()
