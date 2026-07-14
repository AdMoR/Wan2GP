"""
Quick smoke-test for the WanGP server running on port 8082.

Usage:
    python3 test_server.py                          # health check only
    python3 test_server.py --submit                 # submit a job and stream events
    python3 test_server.py --submit --download      # also download the generated file
    python3 test_server.py --host http://1.2.3.4:8082 --api-key secret --submit
"""

import argparse
import json
import sys
import time

import requests

DEFAULT_HOST = "http://localhost:8082"

# Minimal generation settings — adjust model_type and prompt to match what you have installed.
DEFAULT_SETTINGS = {
    "model_type": "t2v",
    "prompt": "A serene mountain lake at sunrise, photorealistic",
    "num_inference_steps": 20,
    "video_length": 17,
}


def headers(api_key: str | None) -> dict:
    return {"X-API-Key": api_key} if api_key else {}


# ── Health ─────────────────────────────────────────────────────────────────────

def check_health(host: str, api_key: str | None) -> bool:
    url = f"{host}/health"
    print(f"GET {url}")
    try:
        r = requests.get(url, headers=headers(api_key), timeout=10)
    except requests.ConnectionError:
        print("  ERROR: could not connect to server")
        return False
    print(f"  {r.status_code} {r.json()}")
    return r.status_code == 200


# ── Submit ─────────────────────────────────────────────────────────────────────

def submit_job(host: str, api_key: str | None, settings: dict) -> str | None:
    url = f"{host}/jobs"
    print(f"\nPOST {url}")
    print(f"  settings: {json.dumps(settings, indent=2)}")
    r = requests.post(url, json={"settings": settings}, headers=headers(api_key), timeout=30)
    print(f"  {r.status_code} {r.json()}")
    if r.status_code != 202:
        print("  ERROR: job submission failed")
        return None
    return r.json()["job_id"]


# ── Stream events ──────────────────────────────────────────────────────────────

def stream_events(host: str, api_key: str | None, job_id: str) -> list[str]:
    """Stream SSE events until the job completes. Returns generated filenames."""
    url = f"{host}/jobs/{job_id}/events"
    print(f"\nGET {url}  (streaming SSE)")
    generated_files: list[str] = []

    with requests.get(url, headers=headers(api_key), stream=True, timeout=600) as r:
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or raw.startswith(":"):
                continue  # keep-alive or blank
            if raw.startswith("data: "):
                payload = json.loads(raw[6:])
                kind = payload.get("kind")
                data = payload.get("data")

                if kind == "progress":
                    pct = data.get("progress", 0) if data else 0
                    step = data.get("current_step", "?") if data else "?"
                    total = data.get("total_steps", "?") if data else "?"
                    phase = data.get("phase", "") if data else ""
                    print(f"  [{kind}] {phase}  step {step}/{total}  ({pct:.0%})")
                elif kind == "preview":
                    print(f"  [{kind}] preview frame received")
                elif kind == "completed":
                    success = data.get("success") if data else False
                    files = data.get("generated_files", []) if data else []
                    errors = data.get("errors", []) if data else []
                    print(f"  [{kind}] success={success}  files={files}  errors={errors}")
                    generated_files = files
                    break
                elif kind == "error":
                    print(f"  [{kind}] {data}")
                    break
                else:
                    print(f"  [{kind}] {data}")

    return generated_files


# ── Poll (fallback) ────────────────────────────────────────────────────────────

def poll_job(host: str, api_key: str | None, job_id: str, interval: float = 2.0) -> list[str]:
    """Simple polling loop — use instead of stream_events if SSE isn't convenient."""
    url = f"{host}/jobs/{job_id}"
    print(f"\nPolling {url} every {interval}s")
    while True:
        r = requests.get(url, headers=headers(api_key), timeout=10)
        body = r.json()
        status = body.get("status")
        print(f"  status={status}  progress={body.get('progress', '-')}  phase={body.get('phase', '-')}")
        if status in ("completed", "failed", "cancelled"):
            print(f"  generated_files={body.get('generated_files', [])}")
            print(f"  errors={body.get('errors', [])}")
            return body.get("generated_files", [])
        time.sleep(interval)


# ── Download ───────────────────────────────────────────────────────────────────

def download_files(host: str, api_key: str | None, filenames: list[str]) -> None:
    for filename in filenames:
        url = f"{host}/files/{filename}"
        print(f"\nGET {url}")
        r = requests.get(url, headers=headers(api_key), timeout=60, stream=True)
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}")
            continue
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Saved → {filename}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="WanGP server smoke-test")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--submit", action="store_true", help="Submit a test generation job")
    parser.add_argument("--poll", action="store_true", help="Use polling instead of SSE streaming")
    parser.add_argument("--download", action="store_true", help="Download generated files when done")
    parser.add_argument("--model-type", default=DEFAULT_SETTINGS["model_type"])
    parser.add_argument("--prompt", default=DEFAULT_SETTINGS["prompt"])
    args = parser.parse_args()

    ok = check_health(args.host, args.api_key)
    if not ok:
        sys.exit(1)

    if not args.submit:
        print("\nServer is healthy. Pass --submit to run a generation job.")
        return

    settings = {**DEFAULT_SETTINGS, "model_type": args.model_type, "prompt": args.prompt}
    job_id = submit_job(args.host, args.api_key, settings)
    if job_id is None:
        sys.exit(1)

    if args.poll:
        files = poll_job(args.host, args.api_key, job_id)
    else:
        files = stream_events(args.host, args.api_key, job_id)

    if args.download and files:
        download_files(args.host, args.api_key, files)


if __name__ == "__main__":
    main()
