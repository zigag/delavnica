"use strict";

// CONFIGURATION: KEEP SENT FRAMES SMALL, RUN A FEW DETECTIONS PER SECOND
const SEND_LONG_EDGE = 640;   // longest edge of the JPEG sent to /detect
const JPEG_QUALITY = 0.8;
const FRAME_GAP_MS = 100;     // pause between loop iterations
const ERROR_GAP_MS = 1500;    // backoff after a failed request

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const backendText = document.getElementById("backendText");
const statusText = document.getElementById("statusText");
const latencyText = document.getElementById("latencyText");
const fpsText = document.getElementById("fpsText");
const countText = document.getElementById("countText");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let running = false;
let requestInFlight = false;
let stream = null;
let fpsWindow = []; // completion timestamps within the last 5 seconds

function setStatus(text) {
  statusText.textContent = text;
}

// SHOW MODEL/RUNTIME INFO ON PAGE LOAD
async function checkBackend() {
  try {
    const resp = await fetch("/health");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const info = await resp.json();
    backendText.textContent = `${info.model} (${info.runtime})`;
  } catch (err) {
    backendText.textContent = `unreachable (${err.message})`;
  }
}
checkBackend();

async function startCamera() {
  setStatus("Requesting camera…");
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  await new Promise((resolve) => {
    if (video.readyState >= 2) resolve();
    else video.addEventListener("loadeddata", resolve, { once: true });
  });
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  // MATCH THE STAGE TO THE VIDEO ASPECT RATIO SO THE OVERLAY STAYS ALIGNED
  document.getElementById("stage").style.aspectRatio =
    `${video.videoWidth} / ${video.videoHeight}`;
}

function stopCamera() {
  running = false;
  requestInFlight = false;
  if (stream !== null) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  fpsText.textContent = "–";
  latencyText.textContent = "–";
  countText.textContent = "–";
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// CAPTURE THE CURRENT FRAME, DOWNSCALE IT, ENCODE AS JPEG DATA URL
function captureFrame() {
  const scale = Math.min(1, SEND_LONG_EDGE / Math.max(video.videoWidth, video.videoHeight));
  const width = Math.round(video.videoWidth * scale);
  const height = Math.round(video.videoHeight * scale);
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = width;
  frameCanvas.height = height;
  frameCanvas.getContext("2d").drawImage(video, 0, 0, width, height);
  return frameCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

async function sendDetection(dataUrl) {
  const blob = await (await fetch(dataUrl)).blob();
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  const start = performance.now();
  const resp = await fetch("/detect", { method: "POST", body: form });
  const roundTripMs = performance.now() - start;
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body.detail) detail = body.detail;
    } catch (ignored) {
      // NON-JSON ERROR BODY: KEEP THE HTTP STATUS TEXT
    }
    throw new Error(detail);
  }
  return { data: await resp.json(), roundTripMs };
}

// SCALE COORDINATES FROM THE SENT IMAGE ONTO THE DISPLAYED OVERLAY
function drawDetections(result) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const scaleX = overlay.width / result.width;
  const scaleY = overlay.height / result.height;
  overlayCtx.font = "14px sans-serif";
  for (const det of result.detections) {
    const x = det.x1 * scaleX;
    const y = det.y1 * scaleY;
    const w = (det.x2 - det.x1) * scaleX;
    const h = (det.y2 - det.y1) * scaleY;
    overlayCtx.strokeStyle = "rgba(0, 255, 128, 0.95)";
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(x, y, w, h);
    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    const textWidth = overlayCtx.measureText(label).width;
    overlayCtx.fillStyle = "rgba(0, 0, 0, 0.65)";
    overlayCtx.fillRect(x, Math.max(0, y - 18), textWidth + 8, 18);
    overlayCtx.fillStyle = "#3dff8f";
    overlayCtx.fillText(label, x + 4, Math.max(13, y - 5));
  }
}

function recordFps() {
  const now = performance.now();
  fpsWindow = fpsWindow.filter((t) => now - t < 5000);
  fpsWindow.push(now);
  fpsText.textContent = (fpsWindow.length / 5).toFixed(1);
}

// SINGLE-IN-FLIGHT LOOP: CAPTURE, SEND, WAIT, DRAW, REPEAT
async function detectionLoop() {
  while (running) {
    requestInFlight = true;
    let gapMs = FRAME_GAP_MS;
    try {
      const dataUrl = captureFrame();
      const { data, roundTripMs } = await sendDetection(dataUrl);
      if (!running) return;
      drawDetections(data);
      countText.textContent = String(data.detections.length);
      latencyText.textContent =
        `${data.inference_ms.toFixed(0)} ms infer / ${roundTripMs.toFixed(0)} ms round trip`;
      recordFps();
      setStatus("detecting");
    } catch (err) {
      setStatus(`error: ${err.message}`);
      gapMs = ERROR_GAP_MS;
    } finally {
      requestInFlight = false;
    }
    if (running) {
      await new Promise((resolve) => setTimeout(resolve, gapMs));
    }
  }
}

async function onStart() {
  try {
    await startCamera();
  } catch (err) {
    let message = `camera error: ${err.message}`;
    if (err.name === "NotAllowedError") {
      message = "camera permission denied — allow camera access, then press Start";
    } else if (err.name === "NotFoundError") {
      message = "no camera found on this machine";
    }
    setStatus(message);
    return;
  }
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  detectionLoop();
}

startBtn.addEventListener("click", onStart);
stopBtn.addEventListener("click", stopCamera);
