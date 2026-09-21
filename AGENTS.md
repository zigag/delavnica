# AGENTS.md

## Mission

Build a small, reliable demonstration system that performs pretrained Ultralytics YOLO object detection on CPU and exposes the detector through a browser-accessible web service.

The primary goal is a working demo for tomorrow, not a production platform.

The demo should allow a user to open a web page, use the browser camera, send images or low-rate video frames to the backend, run YOLO inference on CPU under Linux/WSL2, and display detected objects with bounding boxes and labels.

## Project priorities

In order of importance:

1. Working end-to-end demo.
2. Simple and understandable architecture.
3. Stable CPU inference at a few frames per second.
4. Easy startup and reproducibility.
5. Clear test evidence.
6. Reasonable performance.
7. Clean code.
8. Future extensibility.

Do not sacrifice a working demo for unnecessary abstraction or production infrastructure.

## Strategic baseline

This is a greenfield demonstration project.

Chosen product shape:

- browser frontend;
- Python backend;
- pretrained Ultralytics YOLO model;
- CPU-only inference;
- HTTP API first;
- optional WebSocket only if it clearly improves the demo after HTTP works;
- Linux or WSL2 runtime;
- no model training.

Expected first architecture:

```text
Browser camera
    |
    | JPEG/image frames over HTTP
    v
FastAPI backend
    |
    v
Ultralytics YOLO
    |
    v
CPU inference
    |
    v
JSON detections
    |
    v
Browser canvas overlay
```

Prefer OpenVINO for CPU inference when it is compatible with the selected Ultralytics model and improves performance without making the setup fragile.

## First-release scope

The first demo must provide:

- a browser page;
- camera access through HTML5;
- live or periodically sampled camera frames;
- backend inference endpoint;
- pretrained YOLO object detection;
- detection of standard pretrained classes such as people and common objects;
- confidence scores;
- bounding boxes;
- class labels;
- browser visualization of detections;
- CPU-only operation;
- clear startup instructions.

A target of roughly 2-5 processed frames per second is sufficient for the demo.

Actual achievable rate depends on CPU, input resolution, model size, runtime, and browser/backend overhead.

## Non-goals

Do NOT add these unless explicitly requested:

- YOLO training;
- fine-tuning;
- custom datasets;
- GPU/CUDA support;
- Kubernetes;
- cloud deployment;
- authentication;
- user accounts;
- database persistence;
- message queues;
- Redis;
- Docker orchestration beyond what is genuinely useful for local setup;
- production observability infrastructure;
- complex frontend frameworks;
- React/Vue/Angular unless there is a strong demonstrated reason;
- microservices;
- distributed inference;
- video recording;
- permanent image storage;
- production security certification;
- mobile-native applications.

Do not widen scope because a neighboring feature looks useful.

## Technology preferences

Prefer:

- Python 3.11+;
- Ultralytics package;
- FastAPI;
- Uvicorn;
- plain HTML5;
- plain JavaScript;
- HTML canvas for drawing detections;
- OpenCV or Pillow only where useful;
- OpenVINO for optimized CPU inference when practical;
- pytest for backend tests.

Keep dependencies minimal.

Before adding a dependency, check whether the existing stack already solves the problem.

## YOLO model choice

Start with the smallest current pretrained Ultralytics detection model appropriate for CPU inference.

Prefer the nano-sized model first.

Only move to a larger model if:

- the current model is demonstrably too inaccurate for the demo; and
- CPU performance remains acceptable.

Do not train or modify model weights.

The model should use standard pretrained object classes.

The exact current Ultralytics model name and supported export/runtime combinations must be verified against installed package capabilities or current official documentation rather than guessed.

## CPU performance rules

CPU performance matters.

Prefer these optimizations before introducing architectural complexity:

1. Use a small YOLO model.
2. Resize inference input appropriately.
3. Process only a few frames per second.
4. Avoid sending full-resolution camera frames unnecessarily.
5. Reuse a loaded model; never reload the model per request.
6. Avoid unnecessary image copies and conversions.
7. Test OpenVINO inference if available.
8. Measure actual inference latency.

Record approximate inference latency and achieved processed FPS during verification.

Do not claim a specific FPS until measured on the actual machine.

## Backend API

Start with a simple HTTP API.

Suggested endpoint:

```text
POST /detect
```

Input:

- JPEG or other simple image upload.

Output:

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

Exact schema may evolve slightly if there is a good reason, but keep it simple and documented.

Add:

```text
GET /health
```

that confirms the service is running.

If practical, include model/runtime information in a separate status field or endpoint.

## Frontend

The frontend should be deliberately simple.

Use:

- `navigator.mediaDevices.getUserMedia()`;
- `<video>` for the live camera;
- `<canvas>` for bounding-box overlay;
- periodic frame capture;
- JPEG encoding;
- `fetch()` to send frames to `/detect`.

The browser should avoid sending another frame while too many previous requests are pending.

A simple single-in-flight request loop is preferred over uncontrolled concurrent requests.

The UI should show at least:

- live camera;
- boxes and labels;
- backend/inference status;
- approximate inference latency or FPS if easy to expose.

Do not spend significant time on visual polish before the complete pipeline works.

## HTTP versus WebSocket

HTTP is the default implementation.

Get the complete HTTP path working first.

Only implement WebSocket streaming if:

- the HTTP version is already complete and tested; and
- WebSocket provides an obvious demo improvement without threatening delivery.

Do not replace a working HTTP implementation with a more complex protocol shortly before the demo.

## WSL2 / Linux assumptions

The primary runtime is Ubuntu/Linux, including WSL2.

Keep the project inside the Linux filesystem when running under WSL2.

The backend should bind to a configurable host and port.

For local development, typical startup may use:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Do not hard-code machine-specific IP addresses.

Document any WSL2 networking consideration that is actually encountered.

## Repository structure

Prefer a small structure similar to:

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt or pyproject.toml
├── app/
│   ├── main.py
│   ├── detector.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── tests/
    └── ...
```

This is guidance, not an absolute requirement.

Avoid unnecessary architecture layers.

## Coding rules

Keep code straightforward.

Prefer readable functions and small modules.

Avoid speculative abstractions.

Avoid premature class hierarchies.

Avoid generic framework code whose only justification is possible future expansion.

Use type hints where useful.

Handle errors explicitly.

Do not silently swallow exceptions.

For comments:

- comments describing a coherent important code block should be in English and uppercase, placed before the block;
- short comments for an individual line should be on the same line only when genuinely useful.

Use 4 spaces for indentation.

Use medium-short variable names such as:

```text
numFrames
imageData
modelPath
inferenceMs
```

## Error handling

The application must fail clearly when:

- the YOLO model cannot load;
- an uploaded image cannot be decoded;
- inference fails;
- the browser has no camera permission;
- the backend is unreachable.

Return appropriate HTTP error codes and readable error messages.

Do not hide failures merely to keep the demo appearing alive.

## Security and privacy

This is a local demonstration, but still follow basic safety rules.

Do not:

- commit secrets;
- add credentials;
- upload camera frames to external services;
- store camera images permanently unless explicitly requested;
- add telemetry that sends image data externally.

All inference should remain local.

If any package or runtime unexpectedly requires external image upload for inference, stop and report it.

## Agent autonomy

You are the execution agent.

You may:

- inspect the repository;
- create files;
- edit files;
- install ordinary project dependencies inside the isolated development environment;
- run the application;
- run tests;
- perform local benchmarks;
- fix issues within scope;
- update documentation.

Do not ask the human to perform routine setup work that you can safely perform yourself.

Stop and report if an action would require:

- production credentials;
- access outside the intended project environment;
- destructive changes to unrelated files;
- weakening host security;
- a major architecture decision not covered here.

## Work discipline

Before editing:

1. Read this `AGENTS.md`.
2. Inspect the repository.
3. Check current working-tree state.
4. Understand existing code before replacing it.
5. Preserve useful existing work.

During implementation:

1. Build the smallest working vertical slice first.
2. Test it.
3. Add the next piece.
4. Test again.
5. Avoid broad refactoring unrelated to the demo.

Preferred implementation order:

1. environment and dependencies;
2. standalone YOLO CPU inference on one test image;
3. `/health`;
4. `/detect`;
5. static web page;
6. webcam capture;
7. browser overlay;
8. throttled repeated detection;
9. CPU/OpenVINO optimization;
10. final verification and README.

## Testing

Tests are evidence.

At minimum verify:

- application imports successfully;
- model loads successfully;
- `/health` works;
- `/detect` accepts a valid image;
- `/detect` rejects invalid input cleanly;
- returned detections have valid coordinates and confidence values;
- browser page is served;
- end-to-end manual camera detection works.

Where practical, include automated API tests.

Do not fake model inference in the final end-to-end verification.

Mocks are acceptable for focused unit tests, but at least one verification path must use the real pretrained model.

A skipped test is not a passed test.

A test that was not run is not evidence.

## Performance verification

Measure at least:

- model used;
- inference backend/runtime;
- input resolution;
- approximate inference latency;
- approximate processed FPS;
- CPU environment if easy to identify.

A simple manual benchmark is sufficient.

Do not spend excessive time building a benchmarking framework.

## Documentation

Maintain a concise `README.md`.

It should contain:

- what the demo does;
- supported environment;
- installation commands;
- startup command;
- browser URL;
- camera permission note;
- selected YOLO model;
- whether PyTorch or OpenVINO is used;
- basic API example;
- known limitations;
- measured performance if available.

Do not claim production readiness.

## Git workflow

Keep changes scoped.

If Git is configured and repository workflow permits:

- work on a feature branch;
- commit related files only;
- keep commits understandable.

Do not merge your own pull request unless explicitly instructed by the human.

Do not modify unrelated repositories or directories.

## Definition of done for tomorrow's demo

The demo is ready when all of the following are true:

- backend starts with a documented command;
- YOLO model loads on CPU;
- browser page opens;
- browser can access the camera;
- frames reach the backend;
- backend returns actual YOLO detections;
- browser draws boxes and labels;
- repeated detection runs stably for several minutes;
- no GPU is required;
- setup is reproducible from README;
- basic tests pass;
- measured performance is reported honestly;
- known limitations are documented.

## If time becomes limited

Prioritize in this order:

1. Real YOLO inference.
2. Working backend endpoint.
3. Browser camera.
4. Bounding-box display.
5. Stable repeated requests.
6. README/startup instructions.
7. Performance optimization.
8. Automated tests beyond the critical path.
9. UI polish.
10. WebSocket or additional features.

Never sacrifice the working end-to-end demo for optional infrastructure.

## Required final report

At the end of a work session, report:

```markdown
## Agent Report

### Summary
- What was implemented.

### Files changed
- File list with purpose.

### Model/runtime
- YOLO model:
- Runtime:
- CPU/OpenVINO status:

### Tests run
- Exact command:
- Result:

### Manual verification
- Browser tested:
- Camera tested:
- Real inference tested:

### Performance
- Input size:
- Approx inference latency:
- Approx processed FPS:

### Dependencies installed
- Packages/tools added.

### Known limitations
- Anything incomplete, fragile, skipped, or untested.

### Recommended next step
- One concise next action.
```

Be precise.

Do not say "all tests passed" unless all relevant tests were actually run and passed.

Do not describe a feature as complete if only a partial path works.
