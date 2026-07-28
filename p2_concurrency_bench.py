"""P2-a concurrency benchmark for specific max_workers config."""

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

SERVICE_URL = "http://127.0.0.1:8003"
SERVICE_LOG = Path("/tmp/p2a_bench.log")
SERVICE_DIR = Path(__file__).parent  # models/
REQUESTS_PER_LEVEL = 100
CONCURRENCY_LEVELS = [1, 5, 10, 25, 50]
NORMAL_TEXT = (
    "Hi, I've been having trouble with my account for the past few days. "
    "I can't log in and I keep getting an error message."
)
PROC = None


def log(msg):
    print(msg, flush=True)


def start_service(max_workers):
    global PROC
    env = os.environ.copy()
    env["INTENT_MODEL_VERSION"] = "intent-v4-en"
    env["ESCALATION_MODEL_VERSION"] = "escalation-v4-en"
    env["ESCALATION_THRESHOLD"] = "0.65"
    env["CLASSIFICATION_MAX_WORKERS"] = str(max_workers)
    PROC = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ai_service.app.main:app",
         "--host", "0.0.0.0", "--port", "8003"],
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def get_rss_mb():
    import psutil
    try:
        proc = psutil.Process(PROC.pid)
        return proc.memory_info().rss / 1024 / 1024
    except Exception:
        return None


def benchmark_concurrency(max_workers):
    log(f"\n{'='*60}")
    log(f"  CONCURRENCY BENCHMARK — max_workers={max_workers}")
    log(f"{'='*60}")

    results = {}

    for conc in CONCURRENCY_LEVELS:
        log(f"\n  Concurrency={conc} ({REQUESTS_PER_LEVEL} requests)")
        latencies = []
        errors = 0
        timeouts = 0

        import threading
        lock = threading.Lock()

        def worker():
            nonlocal errors, timeouts
            t0 = time.perf_counter()
            try:
                classify_sync(NORMAL_TEXT)
                lat = (time.perf_counter() - t0) * 1000
                with lock:
                    latencies.append(lat)
            except urllib.error.HTTPError as e:
                with lock:
                    errors += 1
            except Exception:
                with lock:
                    timeouts += 1

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [pool.submit(lambda: worker()) for _ in range(REQUESTS_PER_LEVEL)]
            concurrent.futures.wait(futures)
        duration = time.perf_counter() - t_start

        successful = len(latencies)
        rps = successful / duration

        s = sorted(latencies)
        stats = {}
        if s:
            stats = {
                "p50": round(statistics.median(s), 2),
                "p95": round(s[int(len(s) * 0.95)], 2),
                "p99": round(s[int(len(s) * 0.99)], 2),
                "mean": round(statistics.mean(s), 2),
                "min": round(min(s), 2),
                "max": round(max(s), 2),
            }

        rss = get_rss_mb()

        log(f"    duration: {duration:.1f}s")
        log(f"    requests/sec: {rps:.1f}")
        log(f"    successful: {successful}, errors: {errors}, timeouts: {timeouts}")
        if stats:
            log(f"    p50={stats['p50']:>8.1f}ms  p95={stats['p95']:>8.1f}ms  p99={stats['p99']:>8.1f}ms")
            log(f"    mean={stats['mean']:>7.1f}ms  min={stats['min']:>8.1f}ms  max={stats['max']:>8.1f}ms")
        if rss:
            log(f"    RSS: {rss:.0f} MB")

        results[conc] = {
            "duration_sec": round(duration, 1),
            "requests_per_sec": round(rps, 1),
            "successful": successful,
            "errors": errors,
            "timeouts": timeouts,
            "latency": stats,
            "rss_mb": round(rss, 0) if rss else None,
        }

    return results


def warmup(n=5):
    log("Warming up...")
    for _ in range(n):
        classify_sync(NORMAL_TEXT)


if __name__ == "__main__":
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    stop_service()
    time.sleep(1)

    log(f"Starting service with CLASSIFICATION_MAX_WORKERS={max_workers}...")
    start_service(max_workers)
    if not wait_for_service():
        log("FATAL: Service not started")
        stop_service()
        sys.exit(1)

    warmup()
    results = benchmark_concurrency(max_workers)

    stop_service()

    out = {"max_workers": max_workers, "concurrency_benchmark": results}
    path = SERVICE_DIR / "p2_benchmark_results" / f"concurrency_mw{max_workers}.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nResults saved to {path}")