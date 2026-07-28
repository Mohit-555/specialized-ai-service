"""P2: Performance benchmarking and production observability audit.
AUDIT ONLY — no optimizations implemented."""

import json
import os
import subprocess
import sys
import time
import statistics
import inspect
import urllib.request
import urllib.error
from pathlib import Path

import threading
import psutil

BASE_DIR = Path(__file__).parent
SERVICE_DIR = BASE_DIR  # ai_service package is at BASE_DIR/ai_service/
SERVICE_URL = "http://127.0.0.1:8002"
SERVICE_LOG = Path("/tmp/p2_audit_service.log")
RESULTS_DIR = BASE_DIR / "p2_benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

PROC = None

# ── Test messages of different sizes ──
SHORT_TEXT = "What is the price?"
NORMAL_TEXT = (
    "Hi, I've been having trouble with my account for the past few days. "
    "I can't log in and I keep getting an error message saying 'invalid credentials'. "
    "I've already tried resetting my password twice but it didn't help. "
    "Can you please look into this and let me know what's going on?"
)
LONG_TEXT = " ".join(["word"] * 500)
NEAR_MAX_TEXT = "A" * 4900  # just under 5000 limit

ALL_SIZES = [
    ("short (5 words)", SHORT_TEXT),
    ("normal (~60 words)", NORMAL_TEXT),
    ("long (500 words)", LONG_TEXT),
    ("near-max (4900 chars)", NEAR_MAX_TEXT),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def start_service():
    global PROC
    env = os.environ.copy()
    env["INTENT_MODEL_VERSION"] = "intent-v4-en"
    env["ESCALATION_MODEL_VERSION"] = "escalation-v4-en"
    env["ESCALATION_THRESHOLD"] = "0.65"
    PROC = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ai_service.app.main:app",
         "--host", "0.0.0.0", "--port", "8002"],
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_rss_mb():
    try:
        proc = psutil.Process(PROC.pid)
        return proc.memory_info().rss / 1024 / 1024
    except Exception:
        return None


def warmup(n=10):
    log(f"Warming up with {n} requests...")
    for _ in range(n):
        classify_sync(NORMAL_TEXT)


# ═══════════════════════════════════════════
# 1. LATENCY BASELINE
# ═══════════════════════════════════════════

def measure_latency_breakdown(text, n_runs=100):
    """Measure per-component latency using direct model calls."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    intent_dir = BASE_DIR / "models" / "intent" / "intent-v4-en"
    esc_dir = BASE_DIR / "models" / "escalation" / "escalation-v4-en"

    intent_tokenizer = AutoTokenizer.from_pretrained(str(intent_dir))
    intent_model = AutoModelForSequenceClassification.from_pretrained(str(intent_dir))
    intent_model.eval()

    esc_tokenizer = AutoTokenizer.from_pretrained(str(esc_dir))
    esc_model = AutoModelForSequenceClassification.from_pretrained(str(esc_dir))
    esc_model.eval()

    accepts_tti = "token_type_ids" in inspect.signature(intent_model.forward).parameters

    results = {"intent_tokenize": [], "intent_infer": [], "esc_tokenize": [], "esc_infer": []}

    for _ in range(n_runs):
        # Intent tokenize
        t0 = time.perf_counter()
        inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        t1 = time.perf_counter()
        results["intent_tokenize"].append((t1 - t0) * 1000)

        if "token_type_ids" in inputs and not accepts_tti:
            inputs.pop("token_type_ids")

        # Intent infer
        with torch.inference_mode():
            outputs = intent_model(**inputs)
        t2 = time.perf_counter()
        results["intent_infer"].append((t2 - t1) * 1000)

        # Escalation tokenize
        esc_inputs = esc_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        t3 = time.perf_counter()
        results["esc_tokenize"].append((t3 - t2) * 1000)

        if "token_type_ids" in esc_inputs and not accepts_tti:
            esc_inputs.pop("token_type_ids")

        # Escalation infer
        with torch.inference_mode():
            esc_outputs = esc_model(**esc_inputs)
        t4 = time.perf_counter()
        results["esc_infer"].append((t4 - t3) * 1000)

    return results


def compute_stats(values):
    if not values:
        return {}
    return {
        "p50": round(statistics.median(values), 3),
        "p95": round(sorted(values)[int(len(values) * 0.95)], 3),
        "p99": round(sorted(values)[int(len(values) * 0.99)], 3),
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def benchmark_latency():
    log("=" * 60)
    log("1. LATENCY BASELINE")
    log("=" * 60)

    all_results = {}

    for label, text in ALL_SIZES:
        log(f"\n  Measuring: {label}")
        results = measure_latency_breakdown(text, n_runs=50)
        stats = {}
        for component, vals in results.items():
            stats[component] = compute_stats(vals)
            p = stats[component]
            log(f"    {component:20s}  p50={p['p50']:>7.2f}ms  p95={p['p95']:>7.2f}ms  p99={p['p99']:>7.2f}ms  mean={p['mean']:>7.2f}ms")
        all_results[label] = stats

    # Also measure total HTTP latency from actual service
    log("\n  HTTP request latency (100 runs, normal text):")
    http_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        classify_sync(NORMAL_TEXT)
        t1 = time.perf_counter()
        http_times.append((t1 - t0) * 1000)
    http_stats = compute_stats(http_times)
    log(f"    HTTP total: p50={http_stats['p50']:.2f}ms  p95={http_stats['p95']:.2f}ms  p99={http_stats['p99']:.2f}ms  mean={http_stats['mean']:.2f}ms  min={http_stats['min']:.2f}ms  max={http_stats['max']:.2f}ms")

    return all_results, http_stats


# ═══════════════════════════════════════════
# 2. THROUGHPUT / CONCURRENCY
# ═══════════════════════════════════════════

def benchmark_concurrency():
    log("\n" + "=" * 60)
    log("2. THROUGHPUT / CONCURRENCY")
    log("=" * 60)

    import concurrent.futures

    concurrency_levels = [1, 5, 10, 25, 50]
    NUM_REQUESTS = 100
    results = {}

    for conc in concurrency_levels:
        log(f"\n  Concurrency: {conc} ({NUM_REQUESTS} requests)")
        latencies = []
        errors = 0
        timeouts = 0
        lock = threading.Lock()

        def worker(text):
            nonlocal errors, timeouts
            t0 = time.perf_counter()
            try:
                classify_sync(text)
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
            futures = [pool.submit(worker, NORMAL_TEXT) for _ in range(NUM_REQUESTS)]
            concurrent.futures.wait(futures)
        duration = time.perf_counter() - t_start

        successful = len(latencies)
        rps = successful / duration

        stats = {}
        if latencies:
            stats = compute_stats(latencies)

        rss = get_rss_mb()

        log(f"    duration: {duration:.1f}s")
        log(f"    requests/sec: {rps:.1f}")
        log(f"    successful: {successful}, errors: {errors}, timeouts: {timeouts}")
        if stats:
            log(f"    p50: {stats.get('p50', 'N/A'):>8}ms  p95: {stats.get('p95', 'N/A'):>8}ms  p99: {stats.get('p99', 'N/A'):>8}ms")
        if rss:
            log(f"    RSS: {rss:.0f} MB")

        # Stop if clearly saturated (throughput drops or errors spike)
        results[conc] = {
            "duration_sec": round(duration, 1),
            "requests_per_sec": round(rps, 1),
            "successful": successful,
            "errors": errors,
            "timeouts": timeouts,
            "latency": stats,
            "rss_mb": rss,
        }

        if errors > NUM_REQUESTS * 0.1:
            log(f"    ⚠ Error rate > 10%, stopping concurrency tests.")
            break

    return results


# ═══════════════════════════════════════════
# 4. MEMORY
# ═══════════════════════════════════════════

def benchmark_memory():
    log("\n" + "=" * 60)
    log("4. MEMORY MEASUREMENT")
    log("=" * 60)

    proc = psutil.Process(PROC.pid)

    # Measure before loading (already loaded by now, but measure after warmup)
    rss_after_warmup = proc.memory_info().rss / 1024 / 1024
    log(f"  RSS after warm-up: {rss_after_warmup:.0f} MB")

    # During concurrency
    rss_during_load = []
    for _ in range(20):
        classify_sync(NORMAL_TEXT)
        rss_during_load.append(proc.memory_info().rss / 1024 / 1024)
    log(f"  RSS during sequential load: mean={statistics.mean(rss_during_load):.0f} MB, max={max(rss_during_load):.0f} MB")

    # Peak from status file
    peak = get_rss_mb()
    log(f"  Current RSS: {peak:.0f} MB")

    # Memory cost estimate: intent model size
    import torch
    intent_dir = BASE_DIR / "models" / "intent" / "intent-v4-en"
    esc_dir = BASE_DIR / "models" / "escalation" / "escalation-v4-en"
    model = torch.load(str(intent_dir / "model.safetensors"), map_location="cpu", weights_only=True)
    intent_params_mb = sum(p.numel() * p.element_size() for p in model.values() if hasattr(p, 'numel')) / 1024 / 1024
    esc_model_tensors = torch.load(str(esc_dir / "model.safetensors"), map_location="cpu", weights_only=True)
    esc_params_mb = sum(p.numel() * p.element_size() for p in esc_model_tensors.values() if hasattr(p, 'numel')) / 1024 / 1024
    log(f"  Intent model params size (on disk in memory): ~{intent_params_mb:.0f} MB")
    log(f"  Escalation model params size (on disk in memory): ~{esc_params_mb:.0f} MB")
    log(f"  Combined params: ~{intent_params_mb + esc_params_mb:.0f} MB")

    return {
        "rss_after_warmup_mb": round(rss_after_warmup, 0),
        "rss_mean_during_load_mb": round(statistics.mean(rss_during_load), 0),
        "rss_max_during_load_mb": round(max(rss_during_load), 0),
        "current_rss_mb": round(peak, 0) if peak else None,
        "intent_params_mb": round(intent_params_mb, 0),
        "esc_params_mb": round(esc_params_mb, 0),
    }


# ═══════════════════════════════════════════
# 6. ERROR HANDLING
# ═══════════════════════════════════════════

def test_error_handling():
    log("\n" + "=" * 60)
    log("6. ERROR HANDLING AUDIT")
    log("=" * 60)

    test_cases = [
        ("empty text", {"text": ""}),
        ("whitespace-only", {"text": "   "}),
        ("5000-char input", {"text": "A" * 5000}),
        (">5000-char input", {"text": "B" * 5001}),
        ("unicode (Hindi)", {"text": "नमस्ते, मेरा नाम मोहित है"}),
        ("emoji", {"text": "I am very happy today! 😊🎉🙌"}),
        ("newline-heavy", {"text": "\n\n\n\n\nhello\n\n\n\n\nworld\n\n\n\n\n"}),
        ("text=null", {"text": None}),
        ("missing text", {}),
        ("wrong content type", None),
    ]

    results = []
    for label, payload in test_cases:
        try:
            data = json.dumps(payload).encode() if payload else b""
            req = urllib.request.Request(
                f"{SERVICE_URL}/v1/classify",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                results.append({"test": label, "status": resp.status, "body": body, "passed": True})
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                body = json.loads(body)
            except Exception:
                body = str(body)
            results.append({"test": label, "status": e.code, "body": body, "passed": True})
        except Exception as e:
            results.append({"test": label, "status": "EXCEPTION", "body": str(e), "passed": False})

    for r in results:
        status = "OK" if r["passed"] else "UNEXPECTED"
        if isinstance(r["body"], dict):
            detail = r["body"].get("detail", r["body"].get("error", str(r["body"])))
        else:
            detail = str(r["body"])[:80]
        log(f"    [{status:>10}] {r['test']:25s} -> HTTP {str(r['status']):>4s}  {detail[:80]}")

    # Also test malformed JSON
    try:
        req = urllib.request.Request(
            f"{SERVICE_URL}/v1/classify",
            data=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        log(f"    [{'OK':>10}] {'malformed JSON':25s} -> HTTP {e.code}  {body.get('detail', '')[:80]}")
    except Exception as e:
        log(f"    [{'OK':>10}] {'malformed JSON':25s} -> {str(e)[:80]}")

    return results


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    log("P2 AUDIT: Starting V4 service...")
    start_service()
    if not wait_for_service():
        log("FATAL: Service did not start")
        stop_service()
        sys.exit(1)
    log("Service is up.")

    warmup(10)

    # 1. Latency
    latency_breakdown, http_stats = benchmark_latency()

    # 2. Concurrency
    concurrency_results = benchmark_concurrency()

    # 4. Memory
    memory_results = benchmark_memory()

    # 6. Error handling
    error_results = test_error_handling()

    stop_service()
    log("\nService stopped. Audit complete.")

    # Save all results
    report = {
        "hardware": {
            "cpu": "Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz",
            "cores": 2,
            "threads": 4,
            "ram_gb": 7.6,
            "gpu": None,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": __import__('torch').__version__,
            "transformers": __import__('transformers').__version__,
        },
        "latency_breakdown": latency_breakdown,
        "http_latency": http_stats,
        "concurrency": concurrency_results,
        "memory": memory_results,
        "error_handling": error_results,
    }

    report_path = RESULTS_DIR / "audit_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()