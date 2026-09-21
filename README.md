# delavnica

*delavnica* — Slovenian for "workshop".

A small, reliable demo of **pretrained YOLO object detection on CPU**, served
to a plain browser page. Open the page, allow camera access, and watch
bounding boxes and class labels appear over your live video — no GPU, no
cloud, no training. All inference runs locally on your machine.

## How it works

```text
Browser camera
    |
    | JPEG frames over HTTP (at most one request in flight)
    v
FastAPI backend
    |
    v
Ultralytics YOLO11 nano (pretrained, COCO 80 classes)
    |
    v
CPU inference (OpenVINO, PyTorch fallback)
    |
    v
JSON detections
    |
    v
Canvas overlay in the browser
```

A single-in-flight loop in the browser samples the webcam at a low rate:
capture frame → downscale to 640 px on the long edge → JPEG encode →
`POST /detect` → draw boxes on a `<canvas>` over the live `<video>`.
The video stays smooth; detections update a few times per second.

## Requirements

- Linux or WSL2 (Ubuntu 26.04 tested)
- Python 3.11+ (tested on 3.14)
- CPU only — no GPU required
- A webcam and a modern browser (Chrome/Firefox) for the live view

## Installation

```bash
git clone https://github.com/zigag/delavnica.git
cd delavnica

python3 -m venv .venv
source .venv/bin/activate

# CPU-only torch first (skips multi-GB CUDA downloads)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`, ~5 MB) download automatically on first start
and are cached in the repository root. The OpenVINO IR model
(`yolo11n_openvino_model/`, ~10 MB) is exported automatically on first start
if the `openvino` package is installed.

## Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.

**Camera note:** browsers only allow camera access from a *secure context*.
`http://localhost` qualifies, so use the local URL. Under WSL2, a Windows
browser can reach the service through `localhost:8000` via WSL's automatic
port forwarding, which keeps the camera permission working.

## Usage

1. Open `http://localhost:8000`.
2. Press **Start** and allow the camera permission prompt.
3. Live detections — box, class name, confidence % — are drawn over the video.
4. The status panel shows backend/model/runtime, per-frame inference latency,
   processed FPS, and the current detection count.
5. Press **Stop** to end the session.

## API

### `GET /health`

Liveness check including model and active runtime:

```bash
curl -s http://localhost:8000/health
```

```json
{ "status": "ok", "model": "yolo11n.pt", "runtime": "openvino-cpu", "device": "cpu" }
```

### `POST /detect`

Send an image (JPEG or PNG); get back detections in the image's own pixel
coordinates:

```bash
curl -s -F "image=@photo.jpg" http://localhost:8000/detect | jq
```

```json
{
  "width": 640,
  "height": 480,
  "inference_ms": 34.2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.93,
      "x1": 120.2,
      "y1": 45.8,
      "x2": 330.1,
      "y2": 470.0
    }
  ]
}
```

Undecodable images return `400` with a readable error message; an oversized
upload (over 8 MiB) returns `413`.

## Model

- **`yolo11n.pt`** — Ultralytics YOLO11 *nano*, pretrained on COCO (80
  everyday classes: person, bicycle, car, bus, dog, chair, …).
- The smallest current detection model in the installed Ultralytics 8.4.x
  family; name verified against the installed package (official v8.4.0
  release asset).
- Confidence threshold **0.40**, at most 20 boxes per image, most confident
  first.

## Runtime

- **OpenVINO CPU is the preferred runtime.** On first start the detector
  exports `yolo11n.pt` to an OpenVINO IR (`yolo11n_openvino_model/`) and uses
  it; on later starts the cached IR is loaded directly.
- If `openvino` is not installed or the export fails, the detector falls back
  to plain **PyTorch CPU** automatically. `GET /health` reports the active
  runtime (`openvino-cpu` or `pytorch-cpu`).

## Performance (measured on this machine)

Environment: AMD Ryzen 7 7730U (8 cores / 16 threads), 13 GB RAM,
Ubuntu 26.04. No GPU involved.

| Input | Runtime | Inference (median) |
| --- | --- | --- |
| 810×1080 photo | pytorch-cpu | 87 ms |
| 810×1080 photo | openvino-cpu | 79 ms |
| 640×480 demo frame (what the browser sends) | openvino-cpu | 34 ms |

End-to-end stability run (3 minutes, single-flight loop, 640×480 JPEG,
~300 ms spacing): **531 requests, 0 errors, 2.95 processed FPS**,
round trip median 39 ms / p95 49 ms.

Measured with `scripts/bench.py` (simple manual benchmark, re-runnable):

```bash
.venv/bin/python scripts/bench.py [path/to/image.jpg]
```

First request after startup is slower (~1 s) while the runtime warms up.

## Project structure

```text
.
├── AGENTS.md            # project spec and working rules
├── app/
│   ├── main.py          # FastAPI app: /, /health, /detect, static files
│   ├── detector.py      # model loading (OpenVINO preferred) + inference
│   └── static/
│       ├── index.html   # page layout
│       ├── app.js       # camera, single-in-flight loop, canvas overlay
│       └── style.css
├── scripts/
│   └── bench.py         # simple manual CPU inference benchmark
├── tests/
│   ├── test_api.py      # API tests (real model, no mocked inference)
│   └── assets/bus.jpg   # COCO sample photo used by the tests
└── requirements.txt
```

## Tests

```bash
.venv/bin/python -m pytest tests -v
```

9 tests, all using the **real pretrained model** end-to-end: app import,
`/health`, page and static serving, valid image, invalid/empty upload
rejection, real-photo inference (person + bus), and detection structure
validity.

## Privacy

This is a local demo: camera frames are processed in memory, never stored,
and never sent anywhere except this backend.

## Known limitations

- Single-user local demo: one uvicorn worker, no authentication, no
  multi-client load balancing.
- First request after startup takes ~1 s (runtime warm-up).
- 640 px input is a speed/accuracy trade-off: small or distant objects can be
  missed.
- COCO classes only (80 everyday classes); no custom classes.
- Browser camera behavior (permission flow, overlay alignment on unusual
  aspect ratios) is implemented but could not be exercised in a headless
  build environment; the HTTP path and real inference are fully tested.
- Not production-ready: no persistence, monitoring, or deployment tooling.

## License

[Apache License 2.0](LICENSE)
