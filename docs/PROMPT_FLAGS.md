# Prompt-Type Flag Reference

`video_prompt_type`, `image_prompt_type`, and `audio_prompt_type` are string fields
in the `settings` object passed to `POST /jobs` (or `POST /jobs/raw`).  Each is a
**concatenated sequence of single-letter codes**; order within the string does not
matter — the runtime checks for membership.  Flags marked ✅ are set automatically
by the server when the corresponding input file is present; all others must be
supplied explicitly.

> **Swagger UI**: the same reference is available as the **Flag Reference** section
> in the sidebar at `/docs` while the server is running.

---

## `video_prompt_type`

### Core structural flags

| Code | Name | Triggered by | Auto? | LTX pipeline effect |
|------|------|-------------|-------|---------------------|
| `V` | Video guide | `video_guide` | ✅ | Activates the video-conditioning path. Without `V` the guide is silently discarded and `denoising_strength` is clamped to `1.0` (`ltx2.py:1048, 1073–1075`) |
| `A` | Mask | `video_mask` / `image_mask` | ✅ | Builds `masking_source = {video, mask, start_frame}` injected at every denoising step. Controls which pixels are regenerated vs. kept from the guide (`ltx2.py:1128–1140`) |
| `G` | Guide conditioning | — (auto-included in compound codes) | auto | Allows `denoising_strength` < 1. Without `G` the strength is clamped to `1.0` and masking is zeroed (`ltx2.py:1073–1075`) |
| `I` | Identity reference | `image_refs` | ❌ | Encodes `image_refs[0]` into latents and appends it as conditioning attended by **all** frames across both pipeline stages. Preserves subject identity throughout the video. Requires union-control IC-LoRA. Exactly one image unless `K` or `F` is also present (`ltx2.py:1142–1148`) |
| `K` | Keyframe mode | `keyframes` array | ❌ | Switches to the `ltx2_22B_keyframe` two-stage pipeline. The `keyframes` JSON array `[[file_ref, frame_idx, strength], …]` places images as latent anchors at exact frame positions; diffusion smoothly interpolates all intermediate frames (`wgp.py:956, 2836`) |
| `F` | Injected frames | `image_refs` + `frames_positions` | ❌ | Injects reference images at arbitrary frame positions given in `frames_positions` (e.g. `"1 5 10 L"`). Values are 1-indexed; `L` = last frame of the current sliding window. The injected-frame latents pin diffusion at those positions (`ltx2.py:1170–1175`) |
| `U` | Mask suppressor / identity passthrough | — | context | **Dual meaning** — see the dedicated section below |
| `&` | HDR output | HDR `video_guide` | ❌ | Applies LogC3 compression to HDR input latents, auto-loads `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`, forces `NAG_scale=1.0`, and disables audio conditioning. LTX-2 22B only; incompatible with O/P/D/E and `F` (`ltx2.py:1017, 1076–1077`) |

> ⚠️ **Never set `"G"` alone.** It strips the `V` flag, discards `video_guide`, and
> forces `denoising_strength=1.0` — output becomes identical to text-to-video.

---

### Preprocessing codes — form `XVG` compound strings

These letters select a **neural-network preprocessing pass** applied to guide frames
before they are encoded into latent space.  The server defaults to `"DVG"` when
`video_guide` is present and `video_prompt_type` is not set explicitly.

All of `D`, `P`, `O`, `E` auto-load the union-control IC-LoRA
(`ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`).

| Code | Compound | Preprocessor | What it produces | Typical use case |
|------|----------|-------------|-----------------|-----------------|
| `D` | `DVG` *(server default)* | `DepthV3VideoAnnotator` | Depth maps (near=white, far=black) | Preserve scene geometry while changing lighting or style |
| `P` | `PVG` | `PoseBodyFaceVideoAnnotator` | DWPose skeleton keypoint frames (OpenPose format) | Transfer human motion from a guide performer to a generated subject |
| `O` | `OVG` | `PoseBodyFaceVideoAnnotator` + alignment | Skeleton frames aligned to the reference character's body proportions (taken from `image_refs` last frame via `REF_IMAGE`, `wgp.py:4892`) | Transfer motion while keeping the reference character's body scale and proportions. Note: disables outside-mask processing |
| `E` | `EVG` | `CannyVideoAnnotator` | Canny edge maps | Preserve structural outlines while changing texture or colour |
| `S` | `SVG` | Scribble annotator | Rough shape / scribble maps | Loose structural guidance from hand-drawn shapes |
| `L` | `LVG` | Optical flow | Per-frame motion-vector maps | Motion-consistent generation following guide motion |
| `C` | `CVG` | Grayscale | Luminance frames | Brightness-guided generation |
| `M` | `MVG` | Inpaint mask | Binary inpaint mask (white = regenerate, black = keep) | Masked region-specific regeneration driven by the guide |
| `U` | `UVG` | Identity passthrough | Raw guide frames unchanged | Task-specific IC-LoRA that expects unprocessed video (refocus, uncompress, ungrade…) |
| *(none)* | `VG` | None | Raw guide frames unchanged | Same as `UVG`; requires a task-specific LoRA |

---

### Outside-mask preprocessing codes

These flags apply a **second preprocessing pass to the region outside the mask**
while the primary code (P, D, E, …) applies inside the mask.  Only effective when
`A` is also present in `video_prompt_type`.

Source: `wgp.py:4240` (`process_map_outside_mask`).

| Code | Preprocessing applied outside the mask |
|------|---------------------------------------|
| `Y` | Depth map |
| `W` | Scribble / shapes |
| `X` | Inpaint fill |
| `Z` | Optical flow |

**Example:** `"APDY"` → inside mask: pose preprocessing; outside mask: depth preprocessing.

> When `O` (pose_align) is combined with a mask, outside-mask processing is
> **disabled** to prevent double-masking (`wgp.py:5046–5047`).

---

### Face and bounding-box codes

Source: `wgp.py:4242` (`all_process_map_video_guide`).

| Code | Preprocessing |
|------|--------------|
| `B` | Face movements tracking annotation |
| `H` | Bounding-box overlay |

---

### Flag `U` — two distinct meanings

**As a preprocessing selector** (in `process_map_video_guide`, `wgp.py:4241`):
`U` = "identity" — guide frames pass through unchanged without any transformation.
Use with a task-specific IC-LoRA that expects raw video frames.

**As a mask-behaviour modifier** (when `A` is also present):
- `A` alone → masking is active; `masking_strength` controls region reinjection each step.
- `A` + `U` → mask is provided but masking is **suppressed**; the full frame is denoised freely. Used for identity-mask modes (e.g. MultiTalk `UVG`).

Detection: `any_mask = "A" in video_prompt_type and not "U" in video_prompt_type` (`wgp.py:1057`).

---

### `F` vs `K` — frame injection vs keyframe interpolation

| | `F` — Injected frames | `K` — Keyframe mode |
|-|----------------------|---------------------|
| **HTTP parameter** | `frames_positions` string + `image_refs` | `keyframes` array `[[file, frame_idx, strength], …]` |
| **Model** | Standard LTX-2 pipeline | `ltx2_22B_keyframe` two-stage interpolation pipeline |
| **Indexing** | 1-indexed; `L` = last frame of window | 0-indexed absolute frame number |
| **Combines with `I`?** | ✅ `FI` — inject with identity reference | ✅ `KI` — `image_refs` become the keyframe images |

---

### `&` — HDR output in detail

Requires `video_guide` to be an HDR video (auto-detected from file metadata or FFprobe
stream probe for HDR transfer functions).  The pipeline:

1. Keeps frames as `float32` tensors — values are **not** clamped to 0–255.
2. Applies **LogC3 compression** (`hdr_linear_to_vae_range(transform="logc3")`) to map HDR linear RGB into the VAE's expected input range.
3. Auto-loads `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`.
4. Saves output via `save_hdr_video()` using the CRF from `server_config["hdr_video_crf"]`.

Constraints (enforced in `ltx2_handler.py:584–590`):
- Requires `ltx2_22B` exactly — not distilled variants.
- Incompatible with preprocessing codes O/P/D/E and with outpainting.
- Incompatible with `F` (frame injection).

---

## `image_prompt_type`

Controls how still images are fused into the generation.

Source: `wgp.py:4485` (`map_image_prompt`).

| Code | Name | HTTP parameter | Auto? | LTX pipeline effect |
|------|------|---------------|-------|---------------------|
| `S` | Start image | `image_start` | ✅ | The frame-0 latent is **replaced** by the encoded `image_start`; output pixel-matches the provided image exactly at frame 0 (`wgp.py:1102–1123`) |
| `E` | End image | `image_end` | ❌ | The last-frame latent is set as a guiding constraint; output approximates `image_end` at the final frame (`wgp.py:1125–1134`) |
| `V` | Source video | `video_source` | context | A source video separate from `video_guide`, used for continuity or outpainting stitching |
| `L` | Last video | — | ❌ | Continues generation from the final frame of a previously generated video (looping / long-video extension) |

---

## `audio_prompt_type`

Controls how audio drives or is produced by generation.

Source: `wgp.py:4486` (`map_audio_prompt`).

| Code | Name | HTTP parameter | LTX pipeline effect |
|------|------|---------------|---------------------|
| `A` | Audio source | `audio_guide` / `audio_source` | Waveform → mel-spectrogram → `AudioEncoder` → audio latent passed as cross-attention conditioning on all transformer frames (`ltx2.py:1238–1284`) |
| `B` | Audio source #2 | `audio_guide2` | Second audio track; used with multi-speaker / MultiTalk models |
| `K` | Control video audio | extracted from `video_guide` | Audio extracted from the guide video and processed identically to `A` |
| `N` | Normalized volumes | — | Volume levels normalized across all audio sources before encoding |
| `O` | Force output audio | — | Always writes an audio track in the generated video output |
| `X` | Multi-talk speaker 2 | second speaker file | Second speaker slot for MultiTalk models |
| `1` | ID-LoRA reference voice | reference audio | Audio latent used as **speaker identity** reference (not sync). Auto-loads `id-lora-celebvhq-ltx2.safetensors`, sets `audio_identity_guidance_scale=3.0` (`ltx2.py:1259–1262`) |
| `2` | Generate audio from video | control video | Freezes control video frames through denoising; audio is generated to match. Distilled mode only; requires `VG`; incompatible with O/P/D/E/&/A/F/K/I (`ltx2.py:1024, 1090–1091`) |
| `V` | Post-process audio | — | Internal migration flag — not intended for direct use |

---

## LoRA auto-loading triggered by flags

The LTX pipeline auto-loads LoRA weights when certain flags are detected.  User-supplied
`activated_loras` / `loras_multipliers` are **additive** to this auto-loaded stack.
The distilled LoRA multiplier is managed internally and is not affected by your
`loras_multipliers` value.

Source: `models/ltx2/ltx2.py:877–948` (`get_loras_transformer`).

| Flag(s) | LoRA auto-loaded | Multiplier |
|---------|-----------------|-----------|
| `D`, `P`, `O`, or `E` in `video_prompt_type` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | `1.0` |
| `I` in `video_prompt_type` | same union-control LoRA | `1.0` |
| `&` in `video_prompt_type` | `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors` | `1.0` |
| Outpainting active | `ltx-2.3-22b-ic-lora-outpaint-*.safetensors` | `1.0` |
| `1` in `audio_prompt_type` | `id-lora-celebvhq-ltx2.safetensors` | `1.0` or `"1;0"` |
| `guidance_phases > 1` or `distilled_8_steps` | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | phase-dependent |

---

## Incompatibility rules

Source: `models/ltx2/ltx2_handler.py:533–610`.

| Constraint | Reason |
|-----------|--------|
| `"2"` requires `"VG"`, forbids `"OPDE&AFKI"` | Audio-gen-from-video needs a control guide but cannot accept preprocessing or frame constraints |
| `"&"` requires `ltx2_22B` exactly | HDR LoRA trained only on the 22B checkpoint |
| `"&"` forbids `"OPDE"` and outpainting | HDR and preprocessing modes use conflicting LoRA slots |
| `"&"` forbids `"F"` | HDR + frame injection not yet implemented |
| Outpainting + `"VG"` forbids `"OPDE"`, `"F"`, `"A"` | Outpainting uses a dedicated LoRA path that conflicts |
| `"I"` requires exactly one `image_ref` unless `"K"` or `"F"` is also present | LTX-2 identity path supports only one reference in non-keyframe/non-inject modes |
| `"O"` (pose_align) + mask | Outside-mask processing is disabled to prevent double-masking |

---

## Note on `A10`

`A10` is **not a valid flag** in any of the three `*_prompt_type` fields and does not
appear anywhere in the WanGP codebase.  The two codes it most resembles are:

- `A` — mask present in `video_prompt_type` (auto-set by the server when `video_mask` or `image_mask` is supplied)
- `I` — identity reference in `video_prompt_type` (must be set explicitly, requires `image_refs`)

---

## See also

- [WEB_API.md](WEB_API.md) — full HTTP API reference including `POST /jobs/raw`
- [API.md](API.md) — the underlying Python API (`shared/api.py`)
- [CLI.md](CLI.md) — `WANGP_CLI_ARGS` flags forwarded to the WanGP runtime
