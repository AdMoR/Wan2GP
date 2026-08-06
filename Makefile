# ╔══════════════════════════════════════════════════════════════╗
# ║                  WanGP Docker Makefile                       ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Image ─────────────────────────────────────────────────────
IMAGE_NAME  := wangp

# CUDA compute capability for the target GPU.
# Override on the command line: make build CUDA_ARCH=8.9
# Common values: 7.5 (RTX 20xx) · 8.6 (RTX 30xx) · 8.9 (RTX 40xx) · 9.0 (H100) · 12.0 (RTX 50xx)
CUDA_ARCH   ?= 12.0

# ── Ports ─────────────────────────────────────────────────────
# FastAPI/uvicorn server (wangp_server.py)
PORT_SERVER := 8082
# Gradio UI (wgp.py)
PORT_GRADIO := 7860

# ── Cache / volume mounts ─────────────────────────────────────
HOST_CACHE := $(HOME)/.cache
CACHE_MOUNTS := \
	-v $(HOST_CACHE)/huggingface:/home/user/.cache/huggingface \
	-v $(HOST_CACHE)/torch:/home/user/.cache/torch \
	-v $(HOST_CACHE)/numba:/home/user/.cache/numba \
	-v $(HOST_CACHE)/matplotlib:/home/user/.cache/matplotlib

# ── GPU flags ─────────────────────────────────────────────────
# Probed once by scripts/gpu-flags.sh and cached in scripts/.gpu-flags, because
# the right incantation is host-dependent (CDI vs legacy hook, and this box's
# "nvidia" runtime is vast.ai's shim, which panics). Override on the command
# line if needed:  make server GPU_FLAGS="--gpus all"
GPU_FLAGS ?= $(shell $(CURDIR)/scripts/gpu-flags.sh)

# ── TTY ───────────────────────────────────────────────────────
# "docker run -it" aborts with "the input device is not a TTY" whenever stdin
# isn't a terminal (CI, nohup, make invoked from a script). Only ask for a TTY
# when we actually have one.
DOCKER_TTY := $(shell test -t 0 && echo -it || echo -i)

# ──────────────────────────────────────────────────────────────

.PHONY: help build server ui gpu-probe

# Default target
help:
	@echo ""
	@echo "  WanGP Docker commands"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make build  [CUDA_ARCH=<cap>]   Build the '$(IMAGE_NAME)' Docker image"
	@echo "  make server                      Run in API server mode  (port $(PORT_SERVER))"
	@echo "  make ui                          Run with Gradio UI       (port $(PORT_GRADIO))"
	@echo "  make gpu-probe                   Re-detect the docker GPU flags"
	@echo ""
	@echo "  GPU flags: $(GPU_FLAGS)"
	@echo "  CUDA_ARCH default: $(CUDA_ARCH)"
	@echo "  Example: make build CUDA_ARCH=8.9   # RTX 40xx"
	@echo ""

# ── Build ──────────────────────────────────────────────────────
## Build the Docker image.
## Pass CUDA_ARCH=<cap> to target a specific GPU architecture.
build:
	@echo "🏗  Building image '$(IMAGE_NAME)' for CUDA arch $(CUDA_ARCH)…"
	docker build \
		--build-arg CUDA_ARCHITECTURES="$(CUDA_ARCH)" \
		-t $(IMAGE_NAME) \
		.

# ── Server mode ────────────────────────────────────────────────
## Launch the container using the default entrypoint (entrypoint.sh → wangp_server.py).
## The FastAPI/uvicorn server will be available on http://localhost:$(PORT_SERVER).
server: _ensure_cache_dirs
	@echo "🚀 Starting WanGP API server on port $(PORT_SERVER)…"
	docker run --rm $(DOCKER_TTY) \
		--name $(IMAGE_NAME)-server \
		$(GPU_FLAGS) \
		-v  /home/amor/Documents/code_dw/Wan2GP:/workspace \
		-e ORT_TELEMETRY_DISABLED=1 \
		-e HF_XET_HIGH_PERFORMANCE=1 \
		-p $(PORT_SERVER):$(PORT_SERVER) \
		$(CACHE_MOUNTS) \
		$(IMAGE_NAME)

# ── Gradio UI mode ─────────────────────────────────────────────
## Launch the container with the Gradio web UI, overriding the entrypoint.
## The interface will be available on http://localhost:$(PORT_GRADIO).
ui: _ensure_cache_dirs
	@echo "🎨 Starting WanGP Gradio UI on port $(PORT_GRADIO)…"
	docker run --rm $(DOCKER_TTY) \
		--name $(IMAGE_NAME)-ui \
		$(GPU_FLAGS) \
		-e HF_XET_HIGH_PERFORMANCE=1 \
		-p $(PORT_GRADIO):$(PORT_GRADIO) \
		$(CACHE_MOUNTS) \
		--entrypoint python3 \
		$(IMAGE_NAME) /workspace/wgp.py

# ── GPU flag detection ─────────────────────────────────────────
## Re-run the GPU flag probe, showing each combination that was tried.
gpu-probe:
	@rm -f scripts/.gpu-flags
	@$(CURDIR)/scripts/gpu-flags.sh --probe >/dev/null
	@echo "✅ GPU flags: $$(cat scripts/.gpu-flags)"

# ── Internal helpers ───────────────────────────────────────────
_ensure_cache_dirs:
	@mkdir -p \
		$(HOST_CACHE)/huggingface \
		$(HOST_CACHE)/torch \
		$(HOST_CACHE)/numba \
		$(HOST_CACHE)/matplotlib
