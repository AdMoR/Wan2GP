"""
WanGP FastAPI Server

Single-worker HTTP API wrapping the WanGP in-process Python API (shared/api.py).

Usage:
    uvicorn wangp_server:app --host 0.0.0.0 --port 8000 --workers 1

Environment variables:
    WANGP_ROOT         Path to the WanGP root directory (default: directory of this file)
    WANGP_CONFIG       Path to wgp_config.json (optional)
    WANGP_OUTPUT_DIR   Where generated files are written (default: <root>/outputs)
    WANGP_UPLOAD_DIR   Where uploaded media is stored (default: <root>/uploads)
    WANGP_CLI_ARGS     Space-separated extra CLI flags forwarded to WanGP (e.g. "--profile 4")
    WANGP_MAX_QUEUE    Maximum pending-job queue depth (default: 10)
    WANGP_JOB_TTL      Seconds to keep completed jobs in memory (default: 3600)
    WANGP_API_KEY      If set, all requests must carry X-API-Key: <value> header
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from shared.api import GenerationError, GenerationResult, SessionEvent, SessionJob, init

# ── Configuration ─────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

WANGP_ROOT = Path(os.environ.get("WANGP_ROOT", _HERE))
WANGP_CONFIG: Optional[str] = os.environ.get("WANGP_CONFIG") or None
WANGP_OUTPUT_DIR = Path(os.environ.get("WANGP_OUTPUT_DIR", WANGP_ROOT / "outputs"))
WANGP_UPLOAD_DIR = Path(os.environ.get("WANGP_UPLOAD_DIR", WANGP_ROOT / "uploads"))
WANGP_CLI_ARGS: list[str] = os.environ.get("WANGP_CLI_ARGS", "").split() if os.environ.get("WANGP_CLI_ARGS") else []
WANGP_MAX_QUEUE = int(os.environ.get("WANGP_MAX_QUEUE", "10"))
WANGP_JOB_TTL = int(os.environ.get("WANGP_JOB_TTL", "3600"))
API_KEY: Optional[str] = os.environ.get("WANGP_API_KEY") or None

WANGP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WANGP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("wangp_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ATTACHMENT_KEYS = [
    "image_start", "image_end", "image_refs", "image_guide", "image_mask",
    "video_guide", "video_mask", "video_source", "audio_guide", "audio_guide2",
    "audio_source", "custom_guide",
]

_SENTINEL = object()

# ── Upload Store ──────────────────────────────────────────────────────────────


class UploadStore:
    """Maps file_id → absolute path on disk."""

    def __init__(self, upload_dir: Path) -> None:
        self._dir = upload_dir
        self._map: dict[str, Path] = {}
        self._lock = threading.Lock()

    def save(self, filename: str, data: bytes) -> tuple[str, Path]:
        file_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        suffix = Path(filename).suffix or ""
        dest = self._dir / f"{file_id}{suffix}"
        dest.write_bytes(data)
        with self._lock:
            self._map[file_id] = dest
        return file_id, dest

    def resolve(self, file_id: str) -> Optional[Path]:
        with self._lock:
            return self._map.get(file_id)


# ── File-reference resolution ─────────────────────────────────────────────────


def _resolve_file_ref(value: Any, upload_store: UploadStore) -> Any:
    """
    Expand "file:<file_id>[|virtual_suffix]" references to local absolute paths.
    Handles list values (e.g. image_refs) recursively.
    """
    if isinstance(value, list):
        return [_resolve_file_ref(v, upload_store) for v in value]
    if not isinstance(value, str) or not value.startswith("file:"):
        return value

    rest = value[len("file:"):]
    file_id, suffix = (rest.split("|", 1) + [""])[:2]
    suffix = ("|" + suffix) if suffix else ""

    path = upload_store.resolve(file_id)
    if path is None:
        raise ValueError(f"Unknown file_id: {file_id!r}")
    return str(path) + suffix


def resolve_settings(settings: dict, upload_store: UploadStore) -> dict:
    """Return a shallow copy of settings with all file: refs resolved to local paths."""
    resolved = dict(settings)
    for key in ATTACHMENT_KEYS:
        if key in resolved:
            resolved[key] = _resolve_file_ref(resolved[key], upload_store)
    return resolved


_MAX_VALUE_LEN = 200


def _format_setting_value(v: Any) -> str:
    """Return a loggable representation of a settings value, skipping binary/large data."""
    if isinstance(v, bytes):
        return f"<bytes {len(v):,}>"
    if isinstance(v, str):
        if v.startswith("data:") or len(v) > _MAX_VALUE_LEN:
            return f"<str {len(v):,}>"
        return repr(v)
    if isinstance(v, list):
        parts = [_format_setting_value(item) for item in v]
        return "[" + ", ".join(parts) + "]"
    return repr(v)


def _log_settings(settings: dict) -> None:
    lines = ["Received job settings:"]
    for k, v in settings.items():
        lines.append(f"  {k}: {_format_setting_value(v)}")
    log.info("\n".join(lines))


# ── Event serialisation ───────────────────────────────────────────────────────


def _encode_preview(image: Any) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def serialize_event(event: SessionEvent) -> dict:
    kind = event.kind
    data = event.data

    if kind == "preview" and data is not None:
        img = getattr(data, "image", None)
        sd: Any = {
            "phase": getattr(data, "phase", None),
            "status": getattr(data, "status", None),
            "progress": getattr(data, "progress", None),
            "current_step": getattr(data, "current_step", None),
            "total_steps": getattr(data, "total_steps", None),
            "image": _encode_preview(img) if img is not None else None,
        }
    elif kind == "progress" and data is not None:
        sd = {
            "phase": getattr(data, "phase", None),
            "status": getattr(data, "status", None),
            "progress": getattr(data, "progress", None),
            "current_step": getattr(data, "current_step", None),
            "total_steps": getattr(data, "total_steps", None),
        }
    elif kind == "completed" and data is not None:
        sd = {
            "success": data.success,
            "generated_files": [Path(f).name for f in getattr(data, "generated_files", [])],
            "errors": [
                {"message": e.message, "stage": e.stage, "task_index": e.task_index}
                for e in getattr(data, "errors", [])
            ],
            "total_tasks": data.total_tasks,
            "successful_tasks": data.successful_tasks,
            "failed_tasks": data.failed_tasks,
        }
    elif kind == "error" and data is not None:
        sd = {
            "message": getattr(data, "message", str(data)),
            "stage": getattr(data, "stage", None),
            "task_index": getattr(data, "task_index", None),
        }
    elif kind == "stream" and data is not None:
        sd = {"stream": getattr(data, "stream", None), "text": getattr(data, "text", str(data))}
    elif isinstance(data, str):
        sd = data
    else:
        sd = str(data) if data is not None else None

    return {"kind": kind, "data": sd, "timestamp": event.timestamp}


def _format_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Job State & Store ─────────────────────────────────────────────────────────


@dataclass
class JobState:
    job_id: str
    settings: dict
    status: str  # queued | running | completed | cancelled | failed
    queue_position: int
    created_at: float = field(default_factory=time.time)
    result: Optional[GenerationResult] = None
    last_progress: Any = None
    events: list[SessionEvent] = field(default_factory=list)
    cancel_requested: bool = False
    wangp_job: Optional[SessionJob] = None
    _live_queues: list[asyncio.Queue] = field(default_factory=list)
    _live_queues_lock: threading.Lock = field(default_factory=threading.Lock)
    _done_event: threading.Event = field(default_factory=threading.Event)
    # asyncio event loop reference for thread-safe queue writes
    _loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def done(self) -> bool:
        return self.status in ("completed", "cancelled", "failed")

    def add_live_queue(self, q: asyncio.Queue) -> None:
        with self._live_queues_lock:
            self._live_queues.append(q)

    def remove_live_queue(self, q: asyncio.Queue) -> None:
        with self._live_queues_lock:
            try:
                self._live_queues.remove(q)
            except ValueError:
                pass

    def fan_out(self, event: Any) -> None:
        """Push event to all live SSE queues from any thread."""
        if self._loop is None:
            return
        with self._live_queues_lock:
            for q in list(self._live_queues):
                self._loop.call_soon_threadsafe(_put_nowait_safe, q, event)

    def close_live_queues(self) -> None:
        if self._loop is None:
            return
        with self._live_queues_lock:
            for q in list(self._live_queues):
                self._loop.call_soon_threadsafe(_put_nowait_safe, q, _SENTINEL)


def _put_nowait_safe(q: asyncio.Queue, item: Any) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


class JobStore:
    def __init__(self, ttl: int = WANGP_JOB_TTL) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        threading.Thread(target=self._evict_loop, daemon=True, name="wangp-ttl-evictor").start()

    def add(self, job: JobState) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[JobState]:
        with self._lock:
            return list(self._jobs.values())

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "queued")

    def recalc_positions(self) -> None:
        with self._lock:
            pos = 0
            for j in self._jobs.values():
                if j.status == "queued":
                    j.queue_position = pos
                    pos += 1

    def _evict_loop(self) -> None:
        while True:
            time.sleep(60)
            cutoff = time.time() - self._ttl
            with self._lock:
                stale = [jid for jid, j in self._jobs.items() if j.done and j.created_at < cutoff]
                for jid in stale:
                    del self._jobs[jid]


# ── Queue Worker ──────────────────────────────────────────────────────────────


class QueueWorker:
    """
    Background thread that processes jobs one at a time.
    Bridges SessionStream (blocking queue.Queue) events into per-job asyncio queues
    for SSE fan-out.
    """

    def __init__(self, session: Any, job_store: JobStore, upload_store: UploadStore) -> None:
        self._session = session
        self._job_store = job_store
        self._upload_store = upload_store
        self._queue: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="wangp-queue-worker").start()

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self._job_store.get(job_id)
            if job is None:
                continue

            if job.cancel_requested:
                job.status = "cancelled"
                job._done_event.set()
                job.close_live_queues()
                self._job_store.recalc_positions()
                continue

            job.status = "running"
            job.queue_position = 0
            self._job_store.recalc_positions()

            try:
                resolved = resolve_settings(job.settings, self._upload_store)
            except ValueError as exc:
                _fail_job(job, str(exc), "validation")
                self._job_store.recalc_positions()
                continue

            try:
                wangp_job = self._session.submit_task(resolved)
                job.wangp_job = wangp_job

                for event in wangp_job.events.iter(timeout=0.5):
                    job.events.append(event)
                    job.fan_out(event)
                    if event.kind == "progress" and event.data is not None:
                        job.last_progress = event.data
                    elif event.kind == "completed":
                        result = event.data
                        job.result = result
                        job.status = "completed" if (result and result.success) else "failed"
                        break

            except Exception:
                log.exception("Unexpected error running job %s", job_id)
                _fail_job(job, "Internal server error during generation", "runtime")
            finally:
                if not job.done:
                    job.status = "failed"
                job._done_event.set()
                job.close_live_queues()
                self._job_store.recalc_positions()


def _fail_job(job: JobState, message: str, stage: str = "runtime") -> None:
    job.status = "failed"
    job.result = GenerationResult(
        success=False,
        generated_files=[],
        errors=[GenerationError(message=message, stage=stage)],
        total_tasks=1,
        successful_tasks=0,
        failed_tasks=1,
    )
    job._done_event.set()
    job.close_live_queues()


# ── Global state ──────────────────────────────────────────────────────────────

_session: Any = None
_job_store: Optional[JobStore] = None
_upload_store: Optional[UploadStore] = None
_queue_worker: Optional[QueueWorker] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session, _job_store, _upload_store, _queue_worker, _event_loop

    _event_loop = asyncio.get_running_loop()
    _job_store = JobStore()
    _upload_store = UploadStore(WANGP_UPLOAD_DIR)

    log.info("Loading WanGP runtime from %s (this may take a while)…", WANGP_ROOT)
    _session = init(
        root=WANGP_ROOT,
        config_path=WANGP_CONFIG,
        output_dir=WANGP_OUTPUT_DIR,
        cli_args=WANGP_CLI_ARGS,
    )
    _session.ensure_ready()
    _queue_worker = QueueWorker(_session, _job_store, _upload_store)
    log.info("WanGP runtime ready.")

    yield

    log.info("Shutting down WanGP runtime…")
    if _session is not None:
        _session.close()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="WanGP API", version="1.0.0", lifespan=lifespan)

# ── Auth ──────────────────────────────────────────────────────────────────────


def _check_api_key(request: Request) -> None:
    if API_KEY is None:
        return
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Request models ────────────────────────────────────────────────────────────


class JobSubmitRequest(BaseModel):
    settings: dict[str, Any]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", summary="Health check")
async def health() -> dict:
    """Return server liveness and high-level queue state.

    No authentication required.

    **Response fields**
    - `status` – always `"ok"` when the server is up
    - `runtime_loaded` – whether the WanGP model runtime has finished initialising
    - `generation_in_progress` – `true` while a job is actively running
    - `queue_depth` – number of jobs currently waiting in the queue
    """
    in_progress = (
        any(j.status == "running" for j in _job_store.all_jobs()) if _job_store else False
    )
    return {
        "status": "ok",
        "runtime_loaded": _session is not None,
        "generation_in_progress": in_progress,
        "queue_depth": _job_store.queue_depth() if _job_store else 0,
    }


@app.post("/files/upload", summary="Upload a media file")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload an image, video, audio, or mask file for use in a generation job.

    The returned `file_id` can be referenced in job settings using the
    `file:<file_id>` syntax for any attachment key (`image_start`, `image_end`,
    `image_refs`, `video_guide`, `audio_guide`, etc.).

    **Response fields**
    - `file_id` – opaque identifier to pass as `file:<file_id>` in job settings
    - `filename` – actual filename stored on disk
    - `size` – uploaded size in bytes
    """
    data = await file.read()
    file_id, dest = _upload_store.save(file.filename or "upload", data)
    return {"file_id": file_id, "filename": dest.name, "size": len(data)}


def _enqueue_settings(settings: dict[str, Any]) -> dict:
    """Validate queue capacity, create a JobState, and enqueue it. Returns the 202 response dict."""
    depth = _job_store.queue_depth()
    if depth >= WANGP_MAX_QUEUE:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "queue_full",
                "message": f"Queue depth limit ({WANGP_MAX_QUEUE}) reached",
                "queue_depth": depth,
            },
        )

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    job = JobState(
        job_id=job_id,
        settings=settings,
        status="queued",
        queue_position=depth,
        _loop=_event_loop,
    )
    _job_store.add(job)
    _queue_worker.enqueue(job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": depth,
        "poll_url": f"/jobs/{job_id}",
    }


def _apply_v2v_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Normalise settings for video-to-video when video_guide is present.

    WanGP's validate_settings silently breaks v2v when the caller uses the
    intuitive-but-wrong combination of video_source + video_prompt_type="G":
      - video_source is nullified unless image_prompt_type contains "V"
      - denoising_strength is forced to 1.0 unless video_prompt_type contains "V"

    The correct wiring is video_guide + video_prompt_type containing "VG":
      - "V" keeps video_guide alive through validate_settings
      - "G" preserves the caller's denoising_strength value (Control Video Strength)

    Default preprocessing mode is "DVG" (depth map extraction) because:
      - "VG" (raw) is designed for use with an IC-LoRA; without one, conditioning
        tokens exist but the base model has no trained attention to exploit them,
        so the video guide has negligible effect on the output.
      - "DVG" extracts a depth map from every guide frame, which the distilled
        model's ControlNet pathway is trained to condition on directly — no LoRA
        required and produces strong structural guidance.

    Other caller-supplied preprocessing modes ("PVG" pose, "EVG" canny, "OVG"
    aligned-pose) are respected via setdefault — pass video_prompt_type explicitly
    to override.

    denoising_strength note: for LTX2 this is "Control Video Strength" where
    higher values mean the output stays CLOSER to the guide video (0.9 = very
    close, 0.3 = lightly guided). This is the opposite of traditional img2img.

    image_start note: validate_settings nullifies image_start when "S" is absent
    from image_prompt_type. When image_start is provided alongside video_guide,
    "S" is auto-prepended so the start frame is respected.

    transition_frames note: when image_start is present, pass transition_frames=N
    to blank the first N frames of the guide video. The model freely generates
    those frames from image_start context, then follows the guide from frame N+1,
    producing a smooth handoff instead of a hard cut at frame 1. Implemented via
    keep_frames_video_guide ("{N+1}:-1"). Not supported for model_type="ltxv_13B".
    """
    if not settings.get("video_guide"):
        return settings
    s = dict(settings)
    s.setdefault("video_prompt_type", "DVG")
    s.setdefault("image_prompt_type", "")
    s["video_source"] = None  # unused and silently discarded in VG mode anyway
    # validate_settings nullifies image_start when "S" is absent from image_prompt_type
    if s.get("image_start") and "S" not in s["image_prompt_type"]:
        s["image_prompt_type"] = "S" + s["image_prompt_type"]
    transition_frames = int(s.pop("transition_frames", 0) or 0)
    if s.get("image_start") and transition_frames > 0:
        s.setdefault("keep_frames_video_guide", f"{transition_frames + 1}:-1")
    return s


@app.post("/jobs", status_code=202, summary="Submit a generation job")
async def submit_job(body: JobSubmitRequest, _: None = Depends(_check_api_key)) -> dict:
    """Enqueue a new generation job and return immediately (HTTP 202).

    The request body must contain a `settings` object with at minimum:
    - `model_type` *(required)* – WanGP model identifier (e.g. `"wan"`, `"ltx"`)

    All other generation parameters (`prompt`, `video_length`, `resolution`, lora
    weights, attachment keys, etc.) are passed through to the WanGP runtime.
    File attachments must first be uploaded via `POST /files/upload` and referenced
    as `"file:<file_id>"` strings.

    **Video-to-video**

    Pass `video_guide` with the uploaded source video.  `video_prompt_type`
    defaults to `"DVG"` (depth-map conditioning) when `video_guide` is present.
    Use `"PVG"` for human motion, `"EVG"` for edge/structure guidance.
    Do not override it with `"G"` alone — that silently discards the source
    video and forces full regeneration.

    All preprocessing modes (DVG, PVG, OVG, EVG) require the union-control
    IC-LoRA.  Pass it explicitly via `activated_loras` — auto-loading via
    WanGP's preload_URLs is unreliable through the HTTP API:

        "activated_loras": ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"],
        "loras_multipliers": "1"

    For LTX2, `denoising_strength` is **Control Video Strength**: higher values
    mean the output stays *closer* to the guide video (0.9 = very close,
    0.3 = lightly guided).  This is the opposite of traditional img2img.

    **Response fields** (202 Accepted)
    - `job_id` – unique job identifier used for status polling and SSE streaming
    - `status` – `"queued"`
    - `queue_position` – zero-based position in the pending queue
    - `poll_url` – relative URL to poll for status (`GET /jobs/{job_id}`)

    **Errors**
    - `400` – `model_type` missing from settings
    - `503` – queue is full (see `WANGP_MAX_QUEUE`)
    """
    if "model_type" not in body.settings:
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": "model_type is required"},
        )

    _log_settings(body.settings)
    return _enqueue_settings(_apply_v2v_settings(body.settings))


@app.get("/jobs/{job_id}", summary="Poll job status")
async def get_job(job_id: str, request: Request, _: None = Depends(_check_api_key)) -> dict:
    """Return the current status and result of a generation job.

    **Always present**
    - `job_id`, `status` – one of `queued | running | completed | failed | cancelled`

    **While queued**
    - `queue_position` – zero-based position in the pending queue

    **While running**
    - `queue_position` – `0`
    - `progress` – completion fraction `[0.0, 1.0]` (if available)
    - `phase` – current generation phase label (if available)

    **When completed or failed**
    - `success` – `true` if all tasks succeeded
    - `generated_files` – list of absolute download URLs (`GET /files/{filename}`);
      empty list on failure. URLs are only present once the job is done.
    - `errors` – list of `{message, stage, task_index}` objects
    - `total_tasks`, `successful_tasks`, `failed_tasks` – task counts

    **Errors**
    - `404` – job not found (unknown or evicted after TTL)
    """
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    resp: dict[str, Any] = {"job_id": job_id, "status": job.status}

    if job.status == "running":
        resp["queue_position"] = 0
        if job.last_progress is not None:
            resp["progress"] = getattr(job.last_progress, "progress", None)
            resp["phase"] = getattr(job.last_progress, "phase", None)
    elif job.status == "queued":
        resp["queue_position"] = job.queue_position

    if job.done and job.result is not None:
        r = job.result
        resp["success"] = r.success
        resp["generated_files"] = [
            str(request.url_for("download_file", filename=Path(f).name))
            for f in r.generated_files
        ]
        resp["errors"] = [
            {"message": e.message, "stage": e.stage, "task_index": e.task_index}
            for e in r.errors
        ]
        resp["total_tasks"] = r.total_tasks
        resp["successful_tasks"] = r.successful_tasks
        resp["failed_tasks"] = r.failed_tasks

    return resp


@app.delete("/jobs/{job_id}", summary="Cancel a job")
async def cancel_job(job_id: str) -> dict:
    """Request cancellation of a queued or running job.

    - **Queued jobs** are cancelled immediately.
    - **Running jobs** receive a best-effort cancellation signal; the final status
      transitions to `cancelled` or `failed` once the generation thread stops.
    - **Already-finished jobs** return their current status unchanged.

    **Response fields**
    - `job_id`, `status` – `cancelled` (queued jobs) or `cancelling` (running jobs)

    **Errors**
    - `404` – job not found
    """
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "queued":
        job.cancel_requested = True
        job.status = "cancelled"
        job._done_event.set()
        _job_store.recalc_positions()
        return {"job_id": job_id, "status": "cancelled"}

    if job.done:
        return {"job_id": job_id, "status": job.status}

    job.cancel_requested = True
    if job.wangp_job is not None:
        job.wangp_job.cancel()
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/jobs/{job_id}/events", summary="Stream job events (SSE)")
async def job_events(job_id: str, request: Request, _: None = Depends(_check_api_key)):
    """Stream real-time generation events as Server-Sent Events (SSE).

    Events already emitted before the client connects are replayed immediately,
    making it safe to connect at any point during or after a job's lifetime.
    The stream closes automatically after a `completed` or `error` terminal event.
    A keep-alive comment (`: keep-alive`) is sent every 15 s while waiting.

    **Event kinds** (`data` field shape varies by kind)
    - `progress` – `{phase, status, progress, current_step, total_steps}`
    - `preview` – same as `progress` plus `image` (base64 JPEG data-URL)
    - `completed` – `{success, generated_files, errors, total_tasks, successful_tasks, failed_tasks}`
    - `error` – `{message, stage, task_index}`
    - `stream` – `{stream, text}` for log/text output

    **Errors**
    - `404` – job not found
    """
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        live_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=512)

        # Replay already-received events for late joiners
        for event in list(job.events):
            yield _format_sse(serialize_event(event))

        if job.done:
            return

        job.add_live_queue(live_queue)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(live_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is _SENTINEL:
                    break
                yield _format_sse(serialize_event(item))
                if item.kind == "completed":
                    break
        finally:
            job.remove_live_queue(live_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/files/{filename}", summary="Download a generated file")
async def download_file(filename: str) -> FileResponse:
    """Download a file produced by a completed generation job.

    `filename` must be a bare filename with no path separators.
    The `Content-Type` is inferred from the file extension.
    The response uses `Content-Disposition: attachment` to trigger a browser download.

    URLs for generated files are returned directly by `GET /jobs/{job_id}` in the
    `generated_files` list once the job is done — clients should use those URLs
    rather than constructing this path manually.

    **Errors**
    - `400` – filename contains path separators or `..`
    - `403` – resolved path escapes the output directory
    - `404` – file does not exist
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = (WANGP_OUTPUT_DIR / filename).resolve()
    output_root = WANGP_OUTPUT_DIR.resolve()

    try:
        path.relative_to(output_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type, _ = mimetypes.guess_type(filename)
    return FileResponse(
        path=str(path),
        media_type=media_type or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    uvicorn.run("wangp_server:app", host="0.0.0.0", port=8082, workers=1)
