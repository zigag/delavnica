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
    | JPEG frames over HTTP (one request in flight)
    v
FastAPI backend
    |
    v
Ultralytics YOLO (pretrained nano model)
    |
    v
CPU inference (PyTorch, OpenVINO where available)
    |
    v
JSON detections
    |
    v
Canvas overlay in the browser
```

A simple single-in-flight loop in the browser samples a camera frame at a low
rate (a few frames per second), encodes it as JPEG, and POSTs it to
`/detect`. Detections are drawn on a `<canvas>` aligned over the live
`<video>` element, with boxes, class labels, and confidence scores.

## Features

- Live webcam object detection in the browser (HTML5 `getUserMedia`)
- Pretrained YOLO nano model — people plus common COCO classes (cars, dogs, chairs, …)
- Bounding boxes, class labels, and confidence scores
- Inference latency / processed-FPS shown in the UI
- CPU-only; designed for Linux and WSL2
- Deliberately small stack: no frontend framework, no database, no training

## Tech stack

| Layer            | Choice                                        |
| ---------------- | --------------------------------------------- |
| Frontend         | Plain HTML5, vanilla JavaScript, `<canvas>`   |
| Backend          | FastAPI + Uvicorn (Python 3.11+)              |
| Detection        | Ultralytics YOLO, pretrained nano model       |
| Inference        | PyTorch on CPU; OpenVINO when practical       |
| Tests            | pytest                                        |

## Quick start

```bash
# 1. Get the code
git clone https://github.com/zigag/delavnica.git
cd delavnica

# 2. Create an environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. Open the demo in your browser
#    http://localhost:8000
```

**Camera note:** browsers only allow camera access from a *secure context*.
`http://localhost` qualifies, so use the local URL. Under WSL2, Windows
browsers can reach the service through `localhost:8000` via WSL's automatic
port forwarding, which keeps the camera permission working.

The model is downloaded automatically on first run and cached locally.

## API

### `POST /detect`

Send an image (JPEG or PNG); get back detections in the image's own pixel
coordinates.

```bash
curl -s -F "image=@photo.jpg" http://localhost:8000/detect | jq
```

Response:

```json
{
  "width": 640,
  "height": 480,
  "inference_ms": 123.4,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "x1": 100,
      "y1": 50,
      "x2": 300,
      "y2": 450
    }
  ]
}
```

Invalid or undecodable images return `400` with a readable error message.

### `GET /health`

Liveness check, including the loaded model and runtime:

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "yolo nano (pretrained)",
  "runtime": "pytorch-cpu"
}
```

## Project structure

```text
.
├── AGENTS.md            # project spec and working rules
├── README.md
├── requirements.txt
├── app/
│   ├── main.py          # FastAPI app: /detect, /health, static files
│   ├── detector.py      # model loading and inference (model kept in memory)
│   └── static/
│       ├── index.html
│       ├── app.js       # camera capture, single-in-flight loop, canvas overlay
│       └── style.css
└── tests/               # API and inference tests (pytest)
```

## Privacy

This is a local demo: camera frames are processed in memory, never stored,
and never sent anywhere except this backend.

## Status

🚧 **In progress.** The repository currently holds the project specification
([AGENTS.md](AGENTS.md)) and scaffolding. Implementation follows the plan in
AGENTS.md; measured performance (inference latency and processed FPS on a
real CPU) will be recorded in this README once the demo runs end-to-end.

## License

[Apache License 2.0](LICENSE)
