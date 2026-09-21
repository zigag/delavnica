"""FastAPI application: /, /health, /detect and the static demo page."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .detector import DetectorError, detectionToDict, getDetector

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    # LOAD THE MODEL AT STARTUP SO A BROKEN MODEL KILLS THE SERVER LOUDLY
    try:
        getDetector()
    except DetectorError as exc:
        raise RuntimeError(str(exc)) from exc
    yield


app = FastAPI(title="delavnica YOLO demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    detector = getDetector()
    if not detector.isLoaded:
        raise HTTPException(status_code=503, detail="detector not loaded")
    return {"status": "ok", **detector.metadata()}


@app.post("/detect")
def detect(image: UploadFile = File(...)) -> dict:
    # SYNC ENDPOINT: FASTAPI RUNS IT IN A WORKER THREAD, KEEPING THE
    # EVENT LOOP FREE WHILE CPU INFERENCE RUNS
    fileData = image.file.read()
    if not fileData:
        raise HTTPException(status_code=400, detail="upload is empty")
    if len(fileData) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large (max 8 MiB)")
    try:
        with Image.open(io.BytesIO(fileData)) as opened:
            imageData = np.asarray(opened.convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc
    try:
        width, height, inferenceMs, detections = getDetector().detect(imageData)
    except DetectorError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "width": width,
        "height": height,
        "inference_ms": round(inferenceMs, 1),
        "detections": [detectionToDict(det) for det in detections],
    }
