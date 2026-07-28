"""MAX_INFLIGHT experiment + sustained overload test for P2-b."""

import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SERVICE_URL = "http://127.0.0.1:8004"
SERVICE_LOG = Path("/tmp/p2b_overload.log")
SERVICE_DIR = Path(__file__).parent
NORMAL_TEXT = (
    "Hi, I've been having trouble with my account for the past few days. "
    "I can't log in and I keep getting an error message."
)
PROC = None


def log(msg):
    print(msg, flush=True)


def start_service(max_workers, max_inflight, req_timeout=30.0):
    global PROC
    env = os.environ.copy()
    env["INTENT_MODEL_VERSION"] = "intent-v4-en"
    env["ESCALATION_MODEL_VERSION"] = "escalation-v4-en"
    env["ESCALATION_THRESHOLD"] = "0.65"
    env["CLASSIFICATION_MAX_WORKERS"] = str(max_workers)
    env["CLASSIFICATION_MAX_INFLIGHT"] = str(max_inflight)
    env["CLASSIFICATION_REQUEST_TIMEOUT"] = str(req_timeout)
    PROC = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ai_service.app.main:app",
         "--host", "0.0.0.0", "--port", "8004"],
        cwd=str(SERVICE_DIR),
        env=env,
        stdout=open(SERVICE_LOG, "w"),
        stderr=subprocess.STDOUT,
    )


def wait_for_service(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{SERVICE_URL}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("models_loaded"):
                    return True
        except Exception:
            time.sleep(1)
    return False


def stop_service():
    global PROC
    if PROC:
        PROC.terminate()
        PROC.wait()
        PROC = None


def classify_sync(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{SERVICE_URL}/v1/classify",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def warmup(n=5):
    for _ in range(n):
        try:
            classify_sync(NORMAL_TEXT)
        except Exception:
            pass


def sustained_load_test(duration, target_rps, label):
    """Send requests at target_rps for 'duration' seconds.
    Returns stats on accepted, rejected, completed, etc."""
    import threading

    accepted = 0
    rejected = 0
    errors = 0
    latencies = []
    lock = threading.Lock()

    deadline = time.time() + duration
    interval = 1.0 / target_rps
    total_sent = 0

    def worker():
        nonlocal accepted, rejected, errors
        try:
            status, _ = classify_sync(NORMAL_TEXT)
            lat = 0  # We'll measure from outside
            with lock:
                if status == 200:
                    accepted += 1
                elif status == 503:
                    rejected += 1
                else:
                    errors += 1
        except Exception:
            with lock:
                errors += 1

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
        while time.time() < deadline:
            t0 = time.perf_counter()
            futures.append(pool.submit(worker))
            total_sent += 1
            elapsed = time.perf_counter() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    concurrent.futures.wait(futures)

    return {
        "target_rps": target_rps,
        "duration": duration,
        "total_sent": total_sent,
        "accepted": accepted,
        "rejected": rejected,
        "errors": errors,
        "accept_rate": round(accepted / total_sent * 100, 1) if total_sent else 0,
        "reject_rate": round(rejected / total_sent * 100, 1) if total_sent else 0,
    }


def run_experiment(max_inflight, arrival_rates):
    log(f"\n{'='*60}")
    log(f"  EXPERIMENT: MAX_INFLIGHT={max_inflight}")
    log(f"{'='*60}")
    start_service(max_workers=2, max_inflight=max_inflight)
    if not wait_for_service():
        log("FATAL: Service not started")
        return None
    warmup()

    results = []
    for rps in arrival_rates:
        log(f"\n  Arrival rate: {rps} req/s for 15s")
        r = sustained_load_test(duration=15, target_rps=rps, label=f"{rps} req/s")
        log(f"    sent={r['total_sent']}, accepted={r['accepted']}, "
            f"rejected={r['rejected']}, errors={r['errors']}")
        log(f"    accept={r['accept_rate']}%, reject={r['reject_rate']}%")
        results.append(r)

    # Also test /health responsiveness under load
    import threading
    health_lats = []
    lock = threading.Lock()
    def health_poller():
        deadline = time.time() + 15
        while time.time() < deadline:
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(f"{SERVICE_URL}/health")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp.read()
                with lock:
                    health_lats.append((time.perf_counter() - t0) * 1000)
            except Exception:
                pass
            time.sleep(0.1)
    log(f"\n  /health responsiveness under load (15s):")
    health_futs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for _ in range(50):
            health_futs.append(pool.submit(lambda: classify_sync(NORMAL_TEXT)))
        health_futs.append(pool.submit(health_poller))
        concurrent.futures.wait(health_futs)

    hs = sorted(health_lats)
    if hs:
        log(f"    health p50={statistics.median(hs):.1f}ms  p95={hs[int(len(hs)*0.95)]:.1f}ms  p99={hs[int(len(hs)*0.99)]:.1f}ms")

    stop_service()
    return {"max_inflight": max_inflight, "arrival_results": results, "health_latency_ms": {"count": len(hs), "p50": round(statistics.median(hs), 1) if hs else None}}


if __name__ == "__main__":
    ARRIVAL_RATES = [5, 7, 10, 20]
    all_results = []

    for inflight in [4, 6, 10]:
        r = run_experiment(inflight, ARRIVAL_RATES)
        if r:
            all_results.append(r)
        time.sleep(2)

    out_path = SERVICE_DIR / "p2_benchmark_results" / "max_inflight_experiments.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {out_path}")