"""
Example: submit an LTX-2 22B generation job to the WanGP server.

Usage:
    python3 test_server_ltx2_22b.py
    python3 test_server_ltx2_22b.py --download
    python3 test_server_ltx2_22b.py --host http://1.2.3.4:8082
"""

import argparse
import json
import sys
import time

import requests

DEFAULT_HOST = "http://localhost:8082"

SETTINGS = {
    "model_type": "ltx2_22B",
    "prompt": (
        "A serene mountain lake at sunrise. The camera slowly pushes forward over "
        "the glassy water surface reflecting golden light. Pine trees line the shores. "
        "A lone eagle glides silently across the sky. Cinematic, photorealistic."
    ),
    "num_inference_steps": 30,
    "video_length": 97,           # ~4s at 24fps — reduce for faster testing
    "resolution": "1280x720",
    "sample_solver": "euler",
    "guidance_scale": 3.0,
    "audio_guidance_scale": 7.0,
    "alt_guidance_scale": 3.0,
    "alt_scale": 0.7,
    "perturbation_switch": 2,
    "perturbation_layers": [28],
    "perturbation_start_perc": 0,
    "perturbation_end_perc": 100,
    "apg_switch": 0,
    "cfg_star_switch": 0,
}


def check_health(host: str) -> None:
    r = requests.get(f"{host}/health", timeout=10)
    print(f"Health: {r.json()}")
    if r.status_code != 200:
        print("Server not ready — aborting.")
        sys.exit(1)


def submit(host: str) -> str:
    r = requests.post(f"{host}/jobs", json={"settings": SETTINGS}, timeout=30)
    body = r.json()
    if r.status_code != 202:
        print(f"Submission failed: {body}")
        sys.exit(1)
    job_id = body["job_id"]
    print(f"Job submitted: {job_id}  (queue position {body['queue_position']})")
    return job_id


def stream(host: str, job_id: str) -> list[str]:
    print(f"Streaming events for {job_id} …\n")
    url = f"{host}/jobs/{job_id}/events"
    with requests.get(url, stream=True, timeout=1800) as r:
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or raw.startswith(":"):
                continue
            if not raw.startswith("data: "):
                continue
            payload = json.loads(raw[6:])
            kind = payload.get("kind")
            data = payload.get("data") or {}

            if kind == "progress":
                step   = data.get("current_step", "?")
                total  = data.get("total_steps", "?")
                pct    = data.get("progress", 0)
                phase  = data.get("phase", "")
                bar    = "#" * int(pct * 30) + "-" * (30 - int(pct * 30))
                print(f"\r  [{bar}] {pct:5.1%}  step {step}/{total}  {phase}    ", end="", flush=True)
            elif kind == "preview":
                print(f"\n  [preview] frame available")
            elif kind == "completed":
                print()
                files  = data.get("generated_files", [])
                errors = data.get("errors", [])
                print(f"  Done — success={data.get('success')}  files={files}  errors={errors}")
                return files
            elif kind == "error":
                print(f"\n  [error] {data}")
                return []

    return []


def download(host: str, filenames: list[str]) -> None:
    for name in filenames:
        print(f"Downloading {name} …")
        r = requests.get(f"{host}/files/{name}", timeout=120, stream=True)
        if r.status_code != 200:
            print(f"  Failed: {r.status_code}")
            continue
        with open(name, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"  Saved → {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    check_health(args.host)
    job_id = submit(args.host)
    files  = stream(args.host, job_id)

    if args.download and files:
        download(args.host, files)


if __name__ == "__main__":
    main()
