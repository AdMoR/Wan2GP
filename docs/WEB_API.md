# WanGP HTTP API

`wangp_server.py` exposes the WanGP generation engine as a single-worker HTTP API built on FastAPI.
It wraps `shared/api.py` (see [API.md](API.md)) and adds a persistent job queue, file upload
handling, Server-Sent Events streaming, and optional API-key authentication.

## Running the server

```bash
uvicorn wangp_server:app --host 0.0.0.0 --port 8082 --workers 1
```

> **`--workers 1` is required.** The WanGP runtime is not safe to share across processes.

Or run directly:

```bash
python wangp_server.py
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WANGP_ROOT` | directory of `wangp_server.py` | Path to the WanGP installation folder |
| `WANGP_CONFIG` | *(none)* | Path to `wgp_config.json` (uses WanGP default when omitted) |
| `WANGP_OUTPUT_DIR` | `<root>/outputs` | Where generated files are written |
| `WANGP_UPLOAD_DIR` | `<root>/uploads` | Where uploaded media files are stored |
| `WANGP_CLI_ARGS` | *(none)* | Space-separated startup flags forwarded to WanGP (e.g. `--profile 4`) |
| `WANGP_MAX_QUEUE` | `10` | Maximum number of pending jobs allowed in the queue |
| `WANGP_JOB_TTL` | `3600` | Seconds to keep completed job records in memory before eviction |
| `WANGP_API_KEY` | *(none)* | When set, all non-health requests must supply this key |

## Authentication

When `WANGP_API_KEY` is set every protected endpoint requires the key either as a request header or a query parameter:

```
X-API-Key: <key>
```

or

```
GET /jobs/job_123?api_key=<key>
```

The `/health` endpoint is always public.

Requests with a missing or wrong key receive `401 Unauthorized`.

## Job lifecycle

```
POST /jobs  →  queued  →  running  →  completed
                                   ↘  failed
            ↓
         cancelled  (via DELETE /jobs/{job_id})
```

Jobs are processed one at a time in submission order.  While a job is queued
its `queue_position` counts down to 0 as earlier jobs complete.

Completed and failed job records are kept in memory for `WANGP_JOB_TTL` seconds,
then evicted.  Accessing an evicted job returns `404`.

## Endpoints

### `GET /health`

Returns server liveness and queue state.  No authentication required.

**Response `200`**

```json
{
  "status": "ok",
  "runtime_loaded": true,
  "generation_in_progress": false,
  "queue_depth": 2
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when the server is up |
| `runtime_loaded` | boolean | `true` once the WanGP model runtime has finished loading |
| `generation_in_progress` | boolean | `true` while a job is actively running |
| `queue_depth` | integer | Number of jobs currently waiting in the pending queue |

---

### `POST /files/upload`

Upload an image, video, audio, or mask file for use as a generation input.

**Request** — `multipart/form-data`

| Field | Description |
|---|---|
| `file` | The file to upload |

**Response `200`**

```json
{
  "file_id": "upload_1714500000_a3f7c2b1",
  "filename": "upload_1714500000_a3f7c2b1.mp4",
  "size": 2097152
}
```

| Field | Type | Description |
|---|---|---|
| `file_id` | string | Opaque identifier used to reference this file in job settings |
| `filename` | string | Actual filename stored on disk |
| `size` | integer | Uploaded size in bytes |

To reference an uploaded file in job settings, use the `file:<file_id>` syntax:

```json
{
  "settings": {
    "model_type": "wan",
    "image_start": "file:upload_1714500000_a3f7c2b1"
  }
}
```

Supported attachment keys: `image_start`, `image_end`, `image_refs`, `image_guide`,
`image_mask`, `video_guide`, `video_mask`, `video_source`, `audio_guide`,
`audio_guide2`, `audio_source`, `custom_guide`.

To target a specific frame range within an uploaded video, append a virtual-media suffix:

```json
"video_guide": "file:upload_1714500000_a3f7c2b1|start_frame=100,end_frame=200"
```

---

### `POST /jobs`

Enqueue a new generation job.  Returns immediately with HTTP `202 Accepted`.

**Request body** — `application/json`

```json
{
  "settings": {
    "model_type": "wan",
    "prompt": "Cinematic shot of a neon train entering a rainy station",
    "resolution": "832x480",
    "num_inference_steps": 30,
    "video_length": 81
  }
}
```

The `settings` object is passed directly to the WanGP runtime.  `model_type` is
the only required field.  All other generation parameters (`prompt`, `resolution`,
`num_inference_steps`, LoRA weights, attachment keys, etc.) depend on the chosen
model.  Use WanGP's **Export Settings** button in the web UI to get the exact
parameter names for any model.

#### Video duration

Use **`video_length`** (not `num_frames` or `duration`) to control the number of
frames generated:

```json
"video_length": 97
```

A few things to keep in mind:

- **Frame-count alignment** — many models require frame counts of the form `n·k + 1`
  (e.g. LTX-2 requires `8n + 1`: 17, 25, 33, … 97, 105 …).  Values that don't
  satisfy this constraint are silently rounded **down** to the nearest valid count.
  For example, `video_length: 100` produces 97 frames on an LTX-2 model.
- **Model defaults** — if you omit `video_length` the runtime uses the model's own
  default (e.g. 241 frames for LTX-2 22B, 81 for most Wan models).
- **Sliding-window models** — for models that support long video generation via a
  sliding window (LTX-2, Wan 5B …), the `sliding_window_size` parameter controls
  the window length.  When omitted the model's default window size is used.
  Setting `video_length` larger than `sliding_window_size` triggers multi-window
  generation automatically.

**Response `202`**

```json
{
  "job_id": "job_1714500000_b9e2d4f0",
  "status": "queued",
  "queue_position": 0,
  "poll_url": "/jobs/job_1714500000_b9e2d4f0"
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string | Unique job identifier |
| `status` | string | `"queued"` |
| `queue_position` | integer | Zero-based position in the pending queue |
| `poll_url` | string | Relative URL to poll for status |

#### Video-to-video

Upload the source video first with `POST /files/upload`, then pass the returned
`file_id` as `video_guide`.  The server automatically applies the correct WanGP
wiring (`video_prompt_type: "VG"`) when `video_guide` is present, so
`denoising_strength` is respected and the source video is not discarded.

> **Do not set `video_prompt_type: "G"` manually.** That combination silently
> nullifies `video_guide` and forces `denoising_strength` to `1.0`, making the
> output identical to text-to-video.  The server corrects this automatically when
> `video_guide` is present, but an explicit `"G"` overrides the fix.

| Setting | Typical value | Description |
|---|---|---|
| `video_guide` | `"file:<file_id>"` | Uploaded source video |
| `denoising_strength` | `0.4`–`0.9` | Main creative lever — lower = closer to source |
| `input_video_strength` | `0.85` | Latent conditioning strength of the source video |

```json
{
  "settings": {
    "model_type": "ltx2.3_22B_distilled",
    "prompt": "same scene but at night, neon lights reflecting on wet pavement",
    "video_guide": "file:upload_1714500000_a3f7c2b1",
    "denoising_strength": 0.7,
    "input_video_strength": 0.85,
    "resolution": "1280x720",
    "video_length": 97,
    "num_inference_steps": 8
  }
}
```

**Errors**

| Status | `error` key | Description |
|---|---|---|
| `400` | `validation_error` | `model_type` missing from settings |
| `503` | `queue_full` | Pending queue is at capacity (`WANGP_MAX_QUEUE`) |

---

### `GET /jobs/{job_id}`

Poll the current status and result of a job.

**Response `200`** — fields vary by status:

**While queued**

```json
{
  "job_id": "job_1714500000_b9e2d4f0",
  "status": "queued",
  "queue_position": 2
}
```

**While running**

```json
{
  "job_id": "job_1714500000_b9e2d4f0",
  "status": "running",
  "queue_position": 0,
  "progress": 0.54,
  "phase": "inference"
}
```

**When completed**

```json
{
  "job_id": "job_1714500000_b9e2d4f0",
  "status": "completed",
  "success": true,
  "generated_files": [
    "http://localhost:8082/files/output_b9e2d4f0.mp4"
  ],
  "errors": [],
  "total_tasks": 1,
  "successful_tasks": 1,
  "failed_tasks": 0
}
```

**When failed**

```json
{
  "job_id": "job_1714500000_b9e2d4f0",
  "status": "failed",
  "success": false,
  "generated_files": [],
  "errors": [
    {"message": "CUDA out of memory", "stage": "generation", "task_index": 1}
  ],
  "total_tasks": 1,
  "successful_tasks": 0,
  "failed_tasks": 1
}
```

| Field | Present when | Description |
|---|---|---|
| `job_id` | always | Job identifier |
| `status` | always | `queued \| running \| completed \| failed \| cancelled` |
| `queue_position` | queued or running | Zero-based position; `0` while running |
| `progress` | running (if available) | Completion fraction `[0.0, 1.0]` |
| `phase` | running (if available) | Current generation phase (e.g. `"inference"`, `"decoding"`) |
| `success` | done | `true` only when all tasks completed without error |
| `generated_files` | done | Absolute download URLs — only populated once the job is done |
| `errors` | done | List of `{message, stage, task_index}` objects |
| `total_tasks` | done | Number of tasks submitted |
| `successful_tasks` | done | Number of tasks that succeeded |
| `failed_tasks` | done | Number of tasks that failed |

`generated_files` contains complete, ready-to-use URLs (e.g.
`http://host:8082/files/output.mp4`).  These URLs are absent while the job is
still queued or running.  Use them directly with `GET /files/{filename}` — do not
construct the path manually.

**Errors**

| Status | Description |
|---|---|
| `404` | Job not found or evicted after TTL |

---

### `DELETE /jobs/{job_id}`

Request cancellation of a queued or running job.

- **Queued jobs** are cancelled immediately.
- **Running jobs** receive a best-effort cancellation signal; the status transitions
  to `cancelled` or `failed` once the generation thread stops.
- **Already-finished jobs** return their current status unchanged.

**Response `200`**

```json
{"job_id": "job_1714500000_b9e2d4f0", "status": "cancelled"}
```

or, for a job that was still running when the request arrived:

```json
{"job_id": "job_1714500000_b9e2d4f0", "status": "cancelling"}
```

**Errors**

| Status | Description |
|---|---|
| `404` | Job not found |

---

### `GET /jobs/{job_id}/events`

Stream real-time generation events as [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (SSE).

Events emitted before the client connects are replayed immediately, making it safe
to connect at any point during or after a job's lifetime.  The stream closes after
the terminal `completed` or `error` event.  A keep-alive comment (`: keep-alive`)
is sent every 15 seconds while waiting.

**Response** — `text/event-stream`

Each SSE message has the form:

```
data: {"kind": "progress", "data": {...}, "timestamp": 1714500042.3}

```

#### Event kinds

**`progress`**

```json
{
  "kind": "progress",
  "data": {
    "phase": "inference",
    "status": "Prompt 1/1 | Denoising | 7.2s",
    "progress": 54,
    "current_step": 4,
    "total_steps": 8
  },
  "timestamp": 1714500042.3
}
```

**`preview`**

Same fields as `progress`, plus:

```json
"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."
```

`image` is a base64-encoded JPEG data URL of the current latent preview frame.
It may be `null` if no preview image was available for this update.

**`completed`** *(terminal)*

```json
{
  "kind": "completed",
  "data": {
    "success": true,
    "generated_files": ["output_b9e2d4f0.mp4"],
    "errors": [],
    "total_tasks": 1,
    "successful_tasks": 1,
    "failed_tasks": 0
  },
  "timestamp": 1714500120.0
}
```

Note: `generated_files` in the SSE `completed` event contains bare filenames.
Use `GET /jobs/{job_id}` to get the fully-qualified download URLs.

**`error`**

```json
{
  "kind": "error",
  "data": {
    "message": "CUDA out of memory",
    "stage": "generation",
    "task_index": 1
  },
  "timestamp": 1714500080.0
}
```

**`stream`**

```json
{
  "kind": "stream",
  "data": {"stream": "stdout", "text": "Model loaded in 4.2 s"},
  "timestamp": 1714500010.0
}
```

**Errors**

| Status | Description |
|---|---|
| `404` | Job not found |

---

### `GET /files/{filename}`

Download a file produced by a completed generation job.

`filename` must be a bare filename with no path separators.  Use the URLs returned
by `GET /jobs/{job_id}` in `generated_files` — they already point here.

**Response** — file content with inferred `Content-Type` and
`Content-Disposition: attachment`.

**Errors**

| Status | Description |
|---|---|
| `400` | Filename contains `/`, `\`, or `..` |
| `403` | Resolved path escapes the output directory |
| `404` | File does not exist |

---

## Complete workflow example

### Python (`httpx`)

```python
import time
import httpx

BASE = "http://localhost:8082"
HEADERS = {"X-API-Key": "my-secret"}  # omit if no API key is set

client = httpx.Client(base_url=BASE, headers=HEADERS)

# 1. (Optional) Upload a reference image
with open("reference.png", "rb") as f:
    upload = client.post("/files/upload", files={"file": f}).json()
image_ref = f"file:{upload['file_id']}"

# 2. Submit a job
job = client.post("/jobs", json={
    "settings": {
        "model_type": "wan",
        "prompt": "A red fox running through a snowy forest",
        "resolution": "832x480",
        "num_inference_steps": 30,
        "video_length": 81,
        "image_start": image_ref,
    }
}).json()

job_id = job["job_id"]
print("Queued:", job_id, "position", job["queue_position"])

# 3. Poll until done
while True:
    status = client.get(f"/jobs/{job_id}").json()
    print(status["status"], status.get("progress", ""))
    if status["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(2)

# 4. Download generated files
if status.get("success"):
    for url in status["generated_files"]:
        filename = url.split("/")[-1]
        content = client.get(f"/files/{filename}").content
        with open(filename, "wb") as f:
            f.write(content)
        print("Saved:", filename)
else:
    for err in status.get("errors", []):
        print("Error:", err["message"])
```

### Video-to-video (Python)

```python
import time
import httpx

BASE = "http://localhost:8082"
HEADERS = {"X-API-Key": "my-secret"}

client = httpx.Client(base_url=BASE, headers=HEADERS)

# 1. Upload the source video
with open("source.mp4", "rb") as f:
    upload = client.post("/files/upload", files={"file": f}).json()

# 2. Submit the v2v job
job = client.post("/jobs", json={
    "settings": {
        "model_type": "ltx2.3_22B_distilled",
        "prompt": "same scene but at night, neon lights reflecting on wet pavement",
        "video_guide": f"file:{upload['file_id']}",
        "denoising_strength": 0.7,
        "resolution": "1280x720",
        "video_length": 97,
        "num_inference_steps": 8,
    }
}).json()

job_id = job["job_id"]
print("Queued:", job_id)

# 3. Poll until done
while True:
    status = client.get(f"/jobs/{job_id}").json()
    print(status["status"], status.get("progress", ""))
    if status["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(2)

# 4. Download
if status.get("success"):
    for url in status["generated_files"]:
        filename = url.split("/")[-1]
        with open(filename, "wb") as f:
            f.write(client.get(f"/files/{filename}").content)
        print("Saved:", filename)
```

### SSE streaming (Python)

```python
import json
import httpx

BASE = "http://localhost:8082"
HEADERS = {"X-API-Key": "my-secret"}

with httpx.Client(base_url=BASE, headers=HEADERS) as client:
    job = client.post("/jobs", json={
        "settings": {
            "model_type": "wan",
            "prompt": "Time-lapse of a blooming flower",
            "resolution": "832x480",
            "num_inference_steps": 30,
            "video_length": 81,
        }
    }).json()
    job_id = job["job_id"]

with httpx.stream("GET", f"{BASE}/jobs/{job_id}/events", headers=HEADERS) as r:
    for line in r.iter_lines():
        if not line.startswith("data:"):
            continue
        event = json.loads(line[len("data:"):])
        kind = event["kind"]
        if kind == "progress":
            d = event["data"]
            print(f"[{d['phase']}] {d['progress']}% — {d['status']}")
        elif kind == "completed":
            print("Done:", event["data"]["generated_files"])
        elif kind == "error":
            print("Error:", event["data"]["message"])
```

### curl

```bash
# Health check
curl http://localhost:8082/health

# Submit a text-to-video job
curl -s -X POST http://localhost:8082/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret" \
  -d '{"settings": {"model_type": "wan", "prompt": "Ocean waves at sunset", "resolution": "832x480", "num_inference_steps": 30, "video_length": 81}}'

# Submit a video-to-video job
FILE_ID=$(curl -s -X POST http://localhost:8082/files/upload \
  -H "X-API-Key: my-secret" \
  -F "file=@source.mp4" | jq -r .file_id)

curl -s -X POST http://localhost:8082/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret" \
  -d "{\"settings\": {\"model_type\": \"ltx2.3_22B_distilled\", \"prompt\": \"same scene but at night\", \"video_guide\": \"file:$FILE_ID\", \"denoising_strength\": 0.7, \"video_length\": 97, \"num_inference_steps\": 8}}"

# Poll status
curl -s http://localhost:8082/jobs/job_1714500000_b9e2d4f0 \
  -H "X-API-Key: my-secret"

# Stream events
curl -sN http://localhost:8082/jobs/job_1714500000_b9e2d4f0/events \
  -H "X-API-Key: my-secret"

# Download result
curl -OJ http://localhost:8082/files/output_b9e2d4f0.mp4 \
  -H "X-API-Key: my-secret"

# Cancel a job
curl -X DELETE http://localhost:8082/jobs/job_1714500000_b9e2d4f0 \
  -H "X-API-Key: my-secret"
```

## Interactive API docs

When the server is running, FastAPI exposes auto-generated documentation at:

- **Swagger UI** — `http://localhost:8082/docs`
- **ReDoc** — `http://localhost:8082/redoc`
- **OpenAPI schema** — `http://localhost:8082/openapi.json`

## See also

- [API.md](API.md) — the underlying Python API (`shared/api.py`) used by this server
- [CLI.md](CLI.md) — command-line usage
- [GETTING_STARTED.md](GETTING_STARTED.md) — installation and first run
