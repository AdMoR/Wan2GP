# WanGP Server Mode — Model Flow

This document traces the complete execution path of a generation request in **server mode**:
from the moment a job arrives at the HTTP API through model loading, pipeline dispatch, and
output delivery.  It is intended for contributors who need to understand the architecture
deeply — e.g. to add a new pipeline type, integrate a new model family, or debug generation
failures.

> **See also**
> - [WEB_API.md](WEB_API.md) — HTTP endpoint reference and usage examples
> - [API.md](API.md) — Python `WanGPSession` reference and callback API

---

## Architecture Overview

Server mode is composed of six distinct layers, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────┐
│  1. HTTP Server          wangp_server.py                │
│     FastAPI endpoints, job store, file upload, SSE      │
├─────────────────────────────────────────────────────────┤
│  2. Queue Worker         wangp_server.QueueWorker       │
│     One job at a time; bridges SSE to session events    │
├─────────────────────────────────────────────────────────┤
│  3. Python API           shared/api.py                  │
│     WanGPSession, SessionJob, SessionStream             │
├─────────────────────────────────────────────────────────┤
│  4. CLI Runner           shared/api_cli.py              │
│     run_cli_job(), _run_tasks_worker(), _handle_command()│
├─────────────────────────────────────────────────────────┤
│  5. WanGP Runtime        wgp.py                         │
│     validate_task(), load_models(), generate_media()    │
├─────────────────────────────────────────────────────────┤
│  6. Model Handler & Pipeline                            │
│     models/<family>/<handler>.py  +  <pipeline>.py      │
└─────────────────────────────────────────────────────────┘
```

Layers 1–2 are HTTP-server-specific.  Layers 3–6 are shared with the Python API and the
Gradio UI (which uses its own async queue instead of `run_cli_job`).

---

## Concurrency Model

**Only one generation runs at a time** — enforced by a module-level reentrant lock in
`shared/api.py`:

```python
_GENERATION_LOCK = threading.RLock()   # shared/api.py
```

`run_cli_job()` acquires this lock before calling into the WanGP runtime and holds it until
the generation thread exits.  Concurrent `POST /jobs` requests are serialised by the
`QueueWorker`, which dequeues and executes jobs one at a time from a `queue.Queue[str]`.

Within a single generation, there are **two threads**:

- **Main thread** (`run_cli_job`) — holds `_GENERATION_LOCK`, drains the `AsyncStream`
  output queue, dispatches events, handles cancellation checks.
- **Worker thread** (`_run_tasks_worker`) — calls `wgp.validate_task()` and
  `wgp.generate_media()` for each task; pushes progress commands onto the `AsyncStream`.

stdout and stderr are redirected inside the worker thread via `_OutputCapture` wrappers so
all console output is captured and forwarded as `stream` events.

---

## Detailed Execution Flow

### Step 1 — File upload (optional)

```
POST /files/upload  multipart/form-data
  → UploadStore.save(filename, data)
      file saved to WANGP_UPLOAD_DIR/<file_id><ext>
      file_id registered in UploadStore._map
  ← {"file_id": "upload_<timestamp>_<hex>", "filename": "...", "size": N}
```

Uploaded files are referenced later with `"file:<file_id>"` in any
`ATTACHMENT_KEYS` field of job settings.  The `UploadStore` holds them in memory
(path mapping) for the server's lifetime.

---

### Step 2 — Job submission

```
POST /jobs  {"settings": {...}}
  → validate model_type present
  → _log_settings(settings)
  → settings = _apply_h3_settings(settings)    MiniMax H3 validation + alias expansion
  → settings = _apply_v2v_settings(settings)   LTX-2 video-to-video flag repair
  → check queue depth < WANGP_MAX_QUEUE
  → JobState created: status="queued", queue_position=N
  → JobStore.add(job)
  → QueueWorker.enqueue(job_id)   (pushes job_id onto queue.Queue)
  ← HTTP 202  {"job_id": "...", "status": "queued", "queue_position": N}
```

The HTTP response returns immediately.  The job is not yet running.

#### The settings-normalisation layer

Both normalisers are pure `dict → dict` functions in `wangp_server.py`.  They exist
because `wgp.validate_settings()` runs *inside the worker*, minutes into a run and long
after the HTTP response has been sent — so a bad flag surfaces as a failed job rather
than as a `400`.  Each normaliser keys off `settings["model_type"]` and returns the
input untouched for families it does not own.

`POST /jobs/raw` deliberately skips **both**, passing settings through verbatim for
callers who want full manual flag control.

**`_apply_h3_settings()`** — MiniMax H3 (`minimax_h3_fl2va[_pruned][_turbo]`,
`minimax_h3_ref2va[_pruned]`; the full list is `H3_MODEL_TYPES` in `wangp_server.py`,
which every new H3 model type must be added to or it silently skips this validation).  Validates the v12.434 accelerators against constants
mirrored from `models/minimax_h3/minimax_h3_handler.py`, and expands API-friendly
aliases into the keys `wgp.generate_media()` actually consumes:

| Caller passes | Becomes | Validated against |
|---|---|---|
| `skip_steps_cache_type` | *(unchanged)* | `("", "first_block", "spectrum")` |
| `skip_steps_multiplier` | *(unchanged)* | `FIRST_BLOCK_CACHE_THRESHOLDS` = `(0.06, 0.08, 0.10, 0.12, 0.14)` |
| `override_attention` | *(unchanged)* | `("", "sol")` |
| `text_encoder`, `video_vae`, `dit_priority` | merged into `config` | the model's `system_configs` / `system_configs2` / `system_configs3` group ids |
| `control_video_mode` | `video_prompt_type` | `guide_custom_choices` letters (`""`, `"V-"`, `"V+-"`, `"DV"`, `"V"`) |
| `image_refs_relative_size` | *(unchanged, defaults to 100)* | `50`–`400`, ref2va only |

The `config` merge is worth spelling out: WanGP encodes the three system-config groups
plus a finetune slot as one comma-joined string (see `shared/config_groups.py`,
`split_config_selection` / `serialize_config_selection`):

```
config = "<text_encoder>,<video_vae>,<dit_priority>,<finetune>"
# e.g. video_vae="fp8mix" + text_encoder="int8"  →  config = "int8,fp8mix"
```

Callers may still set `config` directly; the alias fields are merged on top of whatever
is already there, and trailing empty slots are stripped.

**`_apply_v2v_settings()`** — LTX-2 video-to-video, triggered by the presence of
`video_guide`.  Repairs the intuitive-but-wrong `video_source` + `video_prompt_type="G"`
combination, defaults `video_prompt_type` to `"DVG"` (depth-map conditioning), auto-prepends
`"S"` when `image_start` is supplied, and converts `transition_frames=N` into
`keep_frames_video_guide="{N+1}:-1"`.

> **Ordering matters.** `_apply_h3_settings()` runs first, and `_apply_v2v_settings()`
> returns early for H3 model types.  H3 uses a different control-video alphabet
> (`"DV"`, `"V-"`, `"V+-"`, `"V"`) and has no `"G"` denoising-strength pathway, so the
> LTX-2 `"DVG"` default would silently corrupt every H3 control-video job.

---

### Step 3 — Queue worker dispatch

```
QueueWorker._run()   (daemon thread)
  job_id = self._queue.get()          blocking pop
  job = job_store.get(job_id)
  job.status = "running"
  job_store.recalc_positions()        updates queue_position for waiting jobs

  resolved = resolve_settings(job.settings, upload_store)
  │   iterates ATTACHMENT_KEYS
  │   calls _resolve_file_ref() recursively on each value
  │   expands "file:<id>" → absolute path on disk
  │   "file:upload_abc|start_frame=10,end_frame=50" → "/uploads/upload_abc.mp4|start_frame=10,end_frame=50"
  └───────────────────────────────────────────────────────

  wangp_job = session.submit_task(resolved)
  │   → WanGPSession._normalize_task(settings)
  │   → WanGPSession._absolutize_task_paths(task, cwd)
  │   → WanGPSession._submit_tasks([task])
  └───────────────────────────────────────────────────────

  for event in wangp_job.events.iter(timeout=0.5):
      job.events.append(event)        replay buffer for late SSE subscribers
      job.fan_out(event)              push to all live SSE asyncio queues
      if event.kind == "completed":
          job.result = event.data
          job.status = "completed" | "failed"
          break
```

`fan_out()` uses `loop.call_soon_threadsafe()` to push events from the worker thread into
each subscriber's `asyncio.Queue` without blocking.

---

### Step 4 — WanGPSession task submission

```
WanGPSession._submit_tasks([task])
  → SessionJob created (threading.Event for done, queue.Queue for events)
  → _active_job = job
  → threading.Thread(target=run_cli_job, args=(session, job, tasks)).start()
  ← SessionJob returned immediately to QueueWorker
```

The API returns a `SessionJob` whose `events` stream can be iterated before generation
starts.

---

### Step 5 — CLI runner setup

```
run_cli_job(session, job, tasks)   (new thread)
  runtime = session._ensure_runtime()
  │   imports wgp module (lazy, first call only)
  │   runs wgp startup: loads config, args, server_config
  └────────────────────────────────────────────────────────

  with _GENERATION_LOCK, _pushd(runtime.root):
  │   session._configure_runtime(runtime)   applies output_dir, etc.
  │   session._prepare_state_for_run(tasks) resets gen state dict
  │   job.events.put("started", ...)
  │
  │   worker_thread = Thread(target=worker)
  │   worker_thread.start()
  │
  │   while True:                           main loop
  │       if job.cancel_requested:
  │           session._request_cancel_unlocked(runtime.module)
  │       item = stream.output_queue.pop()  drains AsyncStream
  │       _handle_command(session, job, wgp, tasks, command, data)
  │       if command == "worker_exit": break
  │
  │   outputs = session._collect_outputs(...)
  │   artifacts = session._consume_output_artifacts(tasks)
  │   result = GenerationResult(success=..., generated_files=outputs, ...)
  │   job.events.put("completed", result)
  └────────────────────────────────────────────────────────
```

The **main loop** in `run_cli_job` is the event pump: it continuously pops items from the
`AsyncStream` pushed by the worker thread and dispatches them to `_handle_command`, which
converts raw commands into typed `SessionEvent` objects and emits them on `job.events`.

---

### Step 6 — Worker: task validation

```
_run_tasks_worker(session, wgp, tasks, stream, job, task_summary)
  for task_index, task in enumerate(tasks):
    validated_settings, error = wgp.validate_task(task, session._state)
```

**`wgp.validate_task(task, state)`** (`wgp.py:8391`):

```
task["params"] → inputs dict
  model_type = inputs["model_type"]   required; error if absent
  model_defaults = get_default_settings(model_type)
  │   reads defaults/<model_type>.json
  │   merges inherited defaults (preload_URLs, architecture, etc.)
  └────────────────────────────────────────────────────────
  inputs.setdefault(k, v) for k,v in model_defaults     user wins on conflicts
  clean_settings(model_type, inputs)                     normalises / strips invalid keys
  validate_settings(state, model_type, inputs, silent=True)
  │   checks resolution, video_length, guidance_scale ranges
  │   verifies attachment files exist on disk
  │   applies model-specific constraints (e.g. LTX-2 frame alignment)
  │   returns override_inputs on success, None on failure
  └────────────────────────────────────────────────────────
  inputs.update(override_inputs)
  return inputs, ""
```

A validation failure emits a `GenerationError` with `stage="validation"` and skips the task
(does not abort subsequent tasks).

---

### Step 7 — Worker: `generate_media()` call

```
task_settings = validated_settings.copy()
task_settings["state"] = session._state

expected_args = set(inspect.signature(wgp.generate_media).parameters.keys())
filtered_params = {k: v for k, v in task_settings.items() if k in expected_args}

success = wgp.generate_media(task, send_cmd, plugin_data=plugin_data, **filtered_params)
```

`filtered_params` contains only the keys that `generate_media()` explicitly declares,
so unknown settings from the client are silently dropped rather than raising `TypeError`.

**`send_cmd(command, data)`** is a closure created per task that pushes items onto the
`AsyncStream` so the main loop can relay them:

| command | meaning |
|---|---|
| `"progress"` | step counter / phase update |
| `"preview"` | latent preview frame (JPEG base64) |
| `"status"` | human-readable status string |
| `"info"` | informational message |
| `"output"` | gallery/file-list refresh |
| `"error"` | task-level generation error |
| `"worker_exit"` | worker thread is exiting |

---

### Step 8 — `generate_media()`: model loading

**`wgp.generate_media()`** (`wgp.py:6418`) — the central ~90-parameter generation function,
shared by both the Gradio UI and server mode.

The first thing it does for each request is check whether the right model is already loaded:

```
generate_media(task, send_cmd, model_type, ..., state, ...)
  if transformer_type != base_model_type or reload_needed:
      wan_model, offloadobj = load_models(model_type, ...)
```

**`load_models(model_type, ...)`** (`wgp.py:3909`):

```
base_model_type = get_base_model_type(model_type)
│   strips variant suffix: "ltx2_22B_keyframe" → "ltx2_22B"
│   result is the key into model_types_handlers dict

model_def = get_model_def(model_type)
│   reads defaults/<model_type>.json → "model" sub-object
│   e.g. {"name": "...", "architecture": "ltx2_22B", "ltx2_pipeline": "two_stage", ...}

model_filename = get_model_filename(model_type, quantization, dtype_policy)
│   selects the right URL/filename given current quantization policy

download_models(filename, model_type, ...)
│   downloads from HuggingFace if not already on disk
│   cached in the local models directory

profile = compute_profile(override_profile, output_type)
│   selects mmgp memory profile (1–6) based on available VRAM and config

model_type_handler = model_types_handlers[base_model_type]
│   model_types_handlers is populated at startup by importing each handler:
│     "models.ltx2.ltx2_handler"    → ltx2_handler.family_handler
│     "models.wan.wan_handler"      → wan_handler.family_handler
│     "models.flux.flux_handler"    → flux_handler.family_handler
│     ... (20+ families)

wan_model, pipe = model_type_handler.load_model(
    local_model_file_list, model_type, base_model_type, model_def,
    quantizeTransformer=..., dtype=..., VAE_dtype=..., ...)
│   each family handler implements this interface
│   returns (model_object, pipe_dict)

init_pipe(pipe, kwargs, profile)
│   registers VAE, text encoder, spatial upsampler etc. with mmgp

offloadobj = offload.profile(pipe, profile_no=mmgp_profile, ...)
│   mmgp sets up CPU↔GPU offloading strategy for the selected profile
│   controls which modules live in VRAM vs RAM at each inference step

return wan_model, offloadobj
```

The loaded model is cached in module-level globals (`wan_model`, `offloadobj`,
`transformer_type`).  On the next request, if `model_type` is unchanged, loading is
skipped entirely.

---

### Step 9 — Model handler: `load_model()`

Each model family implements a `family_handler` class with a static `load_model()` method.

**Example: LTX-2 handler** (`models/ltx2/ltx2_handler.py:869`):

```
ltx2_handler.family_handler.load_model(
    model_filename, model_type, base_model_type, model_def, ...)

  checkpoint_paths = _resolve_multi_file_paths(model_def, base_model_type)
  │   maps component names (transformer, vae, text_encoder, ...) to file paths

  ltx2_model = LTX2(
      model_filename, model_type, base_model_type, model_def,
      dtype=..., VAE_dtype=..., checkpoint_paths=...)
  │   → LTX2.__init__()  (ltx2.py:799)
  │       pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
  │       pipeline_models = self._init_models(...)   loads all weights
  │       if pipeline_kind == "distilled":
  │           self.pipeline = DistilledPipeline(device, models)
  │       elif pipeline_kind == "keyframe_interpolation":
  │           self.pipeline = KeyframeInterpolationPipeline(device, stage_1_models, stage_2_models)
  │       else:  # "two_stage" (default)
  │           self.pipeline = TI2VidTwoStagesPipeline(device, stage_1_models, stage_2_models)

  pipe = {
      "transformer": ltx2_model.model,
      "text_encoder": ltx2_model.text_encoder,
      "vae": ltx2_model.video_decoder,
      "video_encoder": ltx2_model.video_encoder,
      "audio_encoder": ..., "audio_decoder": ..., "vocoder": ...,
      "spatial_upsampler": ...,
  }
  return ltx2_model, pipe
```

The returned `pipe` dict is passed to `init_pipe()` and then to `mmgp.offload.profile()`
which registers every component for CPU↔GPU lifecycle management.

---

### Step 10 — `generate_media()`: conditioning & pipeline dispatch

After model loading, `generate_media()` processes all input attachments:

```
generate_media(...)
  # Image / video / audio preprocessing
  image_start_tensor = load_image(image_start)  if present
  src_video = load_video(video_guide, ...)       if present
  src_mask  = load_mask(video_mask, ...)         if present
  input_waveform = load_audio(audio_guide, ...)  if present
  # ... (many more conditionals for each attachment key)

  samples = wan_model.generate(
      input_prompt=prompt,
      image_start=image_start_tensor,
      frame_num=video_length,
      height=H, width=W,
      sampling_steps=num_inference_steps,
      guide_scale=guidance_scale,
      seed=seed,
      callback=callback,    # step-level progress callback
      keyframes=keyframes,  # passed through **kwargs for keyframe pipeline
      ...
  )
```

**`LTX2.generate()`** (`ltx2.py:1144`) — receives the fully-typed inputs and dispatches
to the selected pipeline:

```
LTX2.generate(input_prompt, image_start, frame_num, ...)
  # Build conditioning structures
  guiding_images = []                        list of (path, latent_idx, strength)
  if image_start:
      entry = (image_start, latent_idx_0, strength, "lanczos")
      guiding_images.append(entry)

  tiling_config = TilingConfig(...)          VAE tiling for large resolutions
  audio_conditionings = [...]                if audio guide present

  # Pipeline dispatch
  if isinstance(self.pipeline, TI2VidTwoStagesPipeline):
      pipeline_output = self.pipeline(
          prompt, negative_prompt, seed, height, width, num_frames,
          images=images, guiding_images=guiding_images,
          video_conditioning=video_conditioning,
          num_inference_steps=sampling_steps,
          cfg_guidance_scale=guide_scale,
          sample_solver=sample_solver,       # "euler" | "res2s" | "distilled_8_steps"
          audio_conditionings=audio_conditionings,
          callback=callback,
          ...)

  elif isinstance(self.pipeline, KeyframeInterpolationPipeline):
      kf_images = [(path, int(frame_idx), float(strength))
                   for path, frame_idx, strength in kwargs.get("keyframes") or []]
      pipeline_output = self.pipeline(
          prompt, negative_prompt, seed, height, width, num_frames,
          images=kf_images,
          num_inference_steps=sampling_steps,
          cfg_guidance_scale=guide_scale,
          callback=callback,
          ...)

  else:  # DistilledPipeline (and future variants)
      pipeline_output = self.pipeline(
          prompt, negative_prompt, seed, height, width, num_frames,
          images=images,
          ...)

  video, audio = pipeline_output   (Iterator[Tensor], Tensor)
```

---

### Step 11 — Pipeline: diffusion inference

Each pipeline class is self-contained and manages its own model component lifecycle
through the `_get_stage_model(stage, name)` helper (for LTX-2 pipelines) or equivalent.

**Two-stage example** (`TI2VidTwoStagesPipeline.__call__`):

```
Stage 1 — text encoding
  text_encoder = _get_stage_model(1, "text_encoder")
  contexts = encode_text(text_encoder, [prompt, negative_prompt])
  del text_encoder; cleanup_memory()

Stage 1 — denoising (at target resolution, e.g. 1280×720)
  video_encoder = _get_stage_model(1, "video_encoder")
  transformer   = _get_stage_model(1, "transformer")
  sigmas = LTX2Scheduler().execute(steps=num_inference_steps)
  conditionings = image_conditionings_by_replacing_latent(images, ...)
                | image_conditionings_by_adding_guiding_latent(guiding_images, ...)
  video_state, audio_state = denoise_audio_video(
      output_shape, conditionings, noiser, sigmas,
      denoising_loop_fn=first_stage_loop,    euler or res2s
      ...)
  del transformer; cleanup_memory()

Stage 2 — temporal upsampling (2× via distilled LoRA)
  upscaled_latent = upsample_video(video_state.latent, video_encoder, spatial_upsampler)
  transformer = _get_stage_model(2, "transformer")   same weights + distilled LoRA
  distilled_sigmas = [0.909, 0.725, 0.422, 0.0]     4-step refinement
  video_state, audio_state = denoise_audio_video(
      output_shape_2x, conditionings_2x, noiser, distilled_sigmas,
      initial_video_latent=upscaled_latent,
      denoising_loop_fn=second_stage_loop,
      ...)
  del transformer; del video_encoder; cleanup_memory()

Decode
  decoded_video = vae_decode_video_to_tensor(video_state.latent, video_decoder, tiling_config)
  decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
  return decoded_video, decoded_audio    (Iterator[Tensor], Tensor)
```

The `callback` passed in is called after each denoising step, which triggers a `send_cmd("progress", ...)` and occasionally `send_cmd("preview", ...)` back through the event chain.

#### Where the accelerator settings take effect

The fields normalised in Step 2 are consumed at four different points in the flow, which
is why an invalid value can fail so late without the submission-time validation:

**Attention override** — `generate_media()` (`wgp.py:6678`), just before generation:

```
overridden_attention = override_attention or get_overridden_attention(model_type)
attn = overridden_attention if ... else attention_mode
if attn not in override_attention_modes_supported:      → send_cmd("info"), send_cmd("exit")
if attn == "sol" and not model_def["sol_attention"]:    → send_cmd("info"), send_cmd("exit")
```

Note this path *aborts the job with an info message* rather than raising — an
unsupported `"sol"` request on a non-Sol GPU ends as a job that produced nothing.
Sol requires BF16, Triton 3.6+ and RTX 40/50-series, H100/H200 or B100/B200.

**Step-skipping cache** — `generate_media()` (`wgp.py:6914`), built per request and
attached to the transformer:

```
skip_steps_cache = DynamicClass(cache_type=skip_steps_cache_type)
skip_steps_cache.update({
    "multiplier": skip_steps_multiplier,
    "start_step": int(skip_steps_start_step_perc * num_inference_steps / 100),
})
model_handler.set_cache_parameters(cache_type, base_model_type, model_def, locals(), cache)
│   minimax_h3_handler: "first_block" → cache.threshold = float(cache.multiplier)
│                       anything but "spectrum" → raises ValueError
trans.cache = skip_steps_cache
```

`start_step` is where `skip_steps_start_step_perc` becomes a concrete step index — it
selects *when skipping may begin*, and is not an acceleration factor.

**First Block Cache in the H3 pipeline** — `models/minimax_h3/pipeline.py:363`:

```
first_block_cache = MiniMaxH3FirstBlockCache(cache) if cache.cache_type == "first_block" else None
for step in steps:
    first_block_cache.begin_step(step)
    transformer(video, audio, ..., spectrum=spectrum, first_block_cache=first_block_cache)
    │   transformer.py:611 — runs the first block, builds a residual signature
    │   should_compute(signature) → run the remaining blocks + store_tail_residual()
    │                            → else apply_tail_residual()  (blocks skipped)
first_block_cache.reset()
```

The cache holds one tail residual, which is why the VRAM overhead is small.

**System configs (`config`)** — resolved during model loading (Step 8), not at generation
time.  `load_models()` (`wgp.py:3917`) copies `model_def` and merges each selected group's
keys into it before the handler sees it:

```
config_groups = get_model_config_groups(model_type, model_def)
model_def = model_def.copy()
for _, _, current_config in selected_model_configs(config_groups, config_id):
    model_def.update(current_config)
```

So `video_vae="fp8mix"` rewrites `model_def["video_vae_file"]` to the FP8-mixed
checkpoint (halving the RAM held by H3's Video VAE weights), and
`dit_priority="lower_ram"` sets `model_def["qkv_splitting"] = False`.  Because the
merged `model_def` decides `model_filename`, `config` is part of the reload check in
`generate_media()` (`wgp.py:6640`):

```
if model_type != transformer_type or reload_needed or profile != loaded_profile or config != loaded_config:
    → load_models(...)
```

Changing any of `text_encoder` / `video_vae` / `dit_priority` between jobs therefore
forces a full model reload — worth batching jobs by config when driving the queue.

---

### Step 12 — Post-processing and output

Back in `generate_media()`, after `wan_model.generate()` returns:

```
video_tensor = _collect_video_chunks(video, ...)   concat frame chunks
video_path = save_video(video_tensor, audio, fps, output_dir, ...)
│   writes .mp4 to WANGP_OUTPUT_DIR

send_cmd("output", {file_list: [video_path], ...})
│   triggers gallery refresh event in UI mode
│   ignored in server mode (events are captured by _handle_command)

# Optional in-memory artifact for API callers requesting return_media=True
payload = build_api_output_artifact_payload(
    client_id, video_path, "video",
    output_video_frames=video_tensor,
    output_audio_data=audio_tensor, ...)
store_api_output_artifact(gen, client_id, payload)
│   stored in state["gen"]["api_output_artifacts"][client_id]
│   consumed by _consume_output_artifacts() in run_cli_job

send_cmd("progress", {progress: 100, phase: "done"})
return True   # success
```

---

### Step 13 — Result assembly and SSE delivery

Back in `run_cli_job()`:

```
outputs = session._collect_outputs(base_file_count, base_audio_count)
│   reads new entries from state["gen"]["file_list"] since run started

artifacts = session._consume_output_artifacts(tasks)
│   retrieves in-memory tensors from state["gen"]["api_output_artifacts"]

result = GenerationResult(
    success=True,
    generated_files=outputs,        # absolute paths on disk
    errors=[],
    total_tasks=1, successful_tasks=1, failed_tasks=0,
    artifacts=(GeneratedArtifact(...),),
)
job.events.put("completed", result)
job._set_result(result)             # unblocks job.result()
job.events.close()
```

In the `QueueWorker`, the `for event in wangp_job.events.iter()` loop receives the
`"completed"` event:

```
job.result = result
job.status = "completed"            # or "failed"
# loop exits, job._done_event.set()
# close_live_queues() signals all SSE subscribers
```

Each SSE subscriber (`GET /jobs/{job_id}/events`) receives the full event replay plus the
terminal `completed` event, then the generator exits.

---

## State Object

`session._state` is a mutable dict that threads through every layer.  The key sub-object
is `state["gen"]`:

| Key | Type | Purpose |
|---|---|---|
| `state["gen"]["queue"]` | `list[dict]` | Active task list for this run |
| `state["gen"]["file_list"]` | `list[str]` | Paths of generated files, appended per task |
| `state["gen"]["audio_file_list"]` | `list[str]` | Paths of generated audio files |
| `state["gen"]["abort"]` | `bool` | Set to `True` to request early stop |
| `state["gen"]["progress_phase"]` | `str` | Current phase label (e.g. `"Denoising"`) |
| `state["gen"]["progress_status"]` | `str` | Full progress status string |
| `state["gen"]["prompt_no"]` | `int` | 1-based index of the current task |
| `state["gen"]["prompts_max"]` | `int` | Total tasks in the current run |
| `state["gen"]["api_output_artifacts"]` | `dict` | In-memory media payloads keyed by `client_id` |

The state dict is pre-initialised by `session._prepare_state_for_run(tasks)` before each
run and reset by `session._reset_state_after_run()` after completion.

---

## Cancellation

Cancellation is cooperative and propagates through the same state object:

```
QueueWorker:    job.cancel_requested is True
                → calls wangp_job.cancel()
                → sets session._active_job.cancel_requested

run_cli_job():  main loop detects job.cancel_requested
                → session._request_cancel_unlocked(runtime.module)
                → sets state["gen"]["abort"] = True

generate_media(): checks state["gen"]["abort"] periodically
                  checks interrupt_check() callback from pipeline

Pipeline:       interrupt_check() returns True
                → denoising loop exits early
                → pipeline returns (None, None)

generate_media(): wan_model.generate() returns None
                  → exits without saving file

_run_tasks_worker(): detects abort flag
                     → appends GenerationError(stage="cancelled")
                     → breaks task loop

run_cli_job():  result built with success=False, errors=[cancelled]
```

---

## Model Handler Interface

Every model family must implement a `family_handler` class with these static methods:

```python
class family_handler:
    model_types: list[str]          # ["ltx2_22B", "ltx2_19B", ...]

    @staticmethod
    def load_model(
        model_filename,             # list of resolved local file paths
        model_type,                 # e.g. "ltx2_22B_keyframe"
        base_model_type,            # e.g. "ltx2_22B"
        model_def,                  # dict from defaults/<model_type>.json "model" key
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        **kwargs,
    ) -> tuple[model_object, pipe_dict]:
        ...

    @staticmethod
    def validate_inputs(base_model_type, model_def, inputs):
        """Optional: validate/normalise UI inputs before generate_media() is called."""
        ...

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        """Optional: migrate saved settings across version upgrades."""
        ...
```

The handler is registered automatically at startup by adding its module path to the
`family_handlers` list in `wgp.py`:

```python
family_handlers = [
    "models.wan.wan_handler",
    "models.ltx2.ltx2_handler",
    "models.flux.flux_handler",
    # ...
]
model_types_handlers = map_family_handlers(family_handlers)
# → {"ltx2_22B": ltx2_handler.family_handler, "wan": wan_handler.family_handler, ...}
```

`get_base_model_type(model_type)` strips variant suffixes so `"ltx2_22B_keyframe"` maps to
the same handler as `"ltx2_22B"`.  The variant is communicated to the handler via
`model_def["ltx2_pipeline"]` read from the model's defaults JSON.

---

## Adding a New Pipeline Type

To wire a new pipeline class into the server flow:

1. **Implement the pipeline** in `models/<family>/ltx_pipelines/<name>.py`.
   - Accept `stage_1_models=None, stage_2_models=None` for pre-initialised models
     (or `checkpoint_path` / `gemma_root` for standalone use).
   - Use `_get_stage_model(stage, name)` to load components from either source.
   - Implement `__call__(prompt, ...) -> tuple[Iterator[Tensor], Tensor]`.

2. **Dispatch in `<family>.py`** (e.g. `ltx2.py`):
   - Import the class.
   - Add `elif pipeline_kind == "<name>":` in `__init__`.
   - Add `elif isinstance(self.pipeline, NewPipeline):` in `generate()`.

3. **Create a model definition** in `defaults/<model_type>.json`:
   - Set `"architecture"` to the appropriate base type.
   - Set the pipeline selector field (e.g. `"ltx2_pipeline": "<name>"`).

4. **Add any new input keys** to `ATTACHMENT_KEYS` in `wangp_server.py` so that
   `"file:<id>"` references in nested structures are resolved before the job runs.

5. **Thread the parameter** through `generate_media()` in `wgp.py` if it is not
   already passed via `**kwargs` to `wan_model.generate()`.

---

## See Also

- `shared/api.py` — `WanGPSession`, `SessionJob`, `SessionStream`, `GenerationResult`
- `shared/api_cli.py` — `run_cli_job()`, `_run_tasks_worker()`, `_handle_command()`
- `wangp_server.py` — `QueueWorker`, `JobStore`, `UploadStore`, FastAPI endpoints
- `wangp_server.py` — `_apply_h3_settings()`, `_apply_v2v_settings()` (settings normalisation)
- `shared/config_groups.py` — comma-joined `config` selection encode/decode
- `wgp.py:3909` — `load_models()`
- `wgp.py:6418` — `generate_media()`
- `wgp.py:8391` — `validate_task()`
- `models/ltx2/ltx2.py:1144` — `LTX2.generate()`
- `models/ltx2/ltx_pipelines/` — pipeline implementations
- `models/minimax_h3/minimax_h3_handler.py` — `set_cache_parameters()`, `query_model_def()`
- `models/minimax_h3/first_block_cache.py` — `MiniMaxH3FirstBlockCache`
