"""V4 behavioral regression tests.

Runs 50 production-style cases against the real V4 service.
Component-level pass/fail/xfail semantics for intent and escalation separately.

Usage:
    pytest tests/regression/test_v4_behavior.py -v
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

REGRESSION_DIR = Path(__file__).parent
CASES_PATH = REGRESSION_DIR / "v4_behavior_cases.json"
SERVICE_URL = "http://127.0.0.1:8001"
SERVICE_DIR = REGRESSION_DIR.parent.parent
SERVICE_LOG = Path("/tmp/v4_regression_service.log")


def load_cases():
    with open(CASES_PATH) as f:
        return json.load(f)


def classify(text):
    resp = requests.post(
        f"{SERVICE_URL}/v1/classify",
        json={"text": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def start_service():
    env = os.environ.copy()
    env["INTENT_MODEL_VERSION"] = "intent-v4-en"
    env["ESCALATION_MODEL_VERSION"] = "escalation-v4-en"
    env["ESCALATION_THRESHOLD"] = "0.65"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ai_service.app.main:app",
         "--host", "0.0.0.0", "--port", "8001"],
        cwd=str(SERVICE_DIR),
        env=env,
        stdout=open(SERVICE_LOG, "w"),
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_service(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("models_loaded"):
                return True
        except requests.ConnectionError:
            time.sleep(1)
    return False


@pytest.fixture(scope="session", autouse=True)
def v4_service(request):
    proc = start_service()
    assert wait_for_service(), "V4 service failed to start within 60s"
    yield
    proc.terminate()
    proc.wait()


def check_intent(result, case):
    predicted = result["intent"]
    expected = case["expected_intent"]
    acceptable = case.get("acceptable_intents") or []
    if predicted == expected or predicted in acceptable:
        return "PASS"
    if case.get("known_failure"):
        return "XFAIL"
    return "FAIL"


def check_escalation(result, case):
    predicted = result["escalation_required"]
    expected = case["expected_escalation"]
    if predicted == expected:
        return "PASS"
    if case.get("known_failure"):
        return "XFAIL"
    return "FAIL"


# ── Collect all results for summary ──

_results = []


@pytest.mark.parametrize(
    "case",
    load_cases(),
    ids=lambda c: f"id{c['id']:03d}-{c.get('tags', ['unknown'])[0]}",
)
def test_v4_behavior(case, request):
    result = classify(case["text"])
    intent_status = check_intent(result, case)
    esc_status = check_escalation(result, case)

    record = {
        "id": case["id"],
        "text": case["text"][:80],
        "tags": case.get("tags", []),
        "intent_status": intent_status,
        "esc_status": esc_status,
        "predicted_intent": result["intent"],
        "intent_confidence": result["intent_confidence"],
        "predicted_escalation": result["escalation_required"],
        "esc_confidence": result["escalation_confidence"],
    }
    _results.append(record)

    errors = []
    if intent_status == "FAIL":
        acceptable = case.get("acceptable_intents") or []
        errors.append(
            f"intent: expected='{case['expected_intent']}'"
            + (f" or {acceptable}" if acceptable else "")
            + f", got '{result['intent']}' (conf={result['intent_confidence']})"
        )
    elif intent_status == "XFAIL":
        acceptable = case.get("acceptable_intents") or []
        errors.append(
            f"intent: expected='{case['expected_intent']}'"
            + (f" or {acceptable}" if acceptable else "")
            + f", got '{result['intent']}' (conf={result['intent_confidence']})"
        )
    if esc_status == "FAIL":
        errors.append(
            f"escalation: expected={case['expected_escalation']}"
            + f", got {result['escalation_required']} (conf={result['escalation_confidence']})"
        )
    elif esc_status == "XFAIL":
        errors.append(
            f"escalation: expected={case['expected_escalation']}"
            + f", got {result['escalation_required']} (conf={result['escalation_confidence']})"
        )
    if errors:
        reason = case.get("failure_reason", "unexpected failure")
        if case.get("known_failure"):
            pytest.xfail(f"{'; '.join(errors)} [{reason}]")
        else:
            pytest.fail(f"{'; '.join(errors)}")


# ── Summary test (runs last) ──

def test_summary():
    """Print aggregated results."""
    if not _results:
        pytest.skip("No results collected")

    intent_pass = sum(1 for r in _results if r["intent_status"] == "PASS")
    intent_xfail = sum(1 for r in _results if r["intent_status"] == "XFAIL")
    intent_fail = sum(1 for r in _results if r["intent_status"] == "FAIL")
    esc_pass = sum(1 for r in _results if r["esc_status"] == "PASS")
    esc_xfail = sum(1 for r in _results if r["esc_status"] == "XFAIL")
    esc_fail = sum(1 for r in _results if r["esc_status"] == "FAIL")

    print(f"\n{'='*60}")
    print(f"  Behavioral Regression Summary (50 cases)")
    print(f"{'='*60}")
    print(f"\n  INTENT:")
    print(f"    PASS:  {intent_pass}")
    print(f"    XFAIL: {intent_xfail}")
    print(f"    FAIL:  {intent_fail}")
    print(f"\n  ESCALATION:")
    print(f"    PASS:  {esc_pass}")
    print(f"    XFAIL: {esc_xfail}")
    print(f"    FAIL:  {esc_fail}")

    tag_results = {}
    for r in _results:
        for t in r["tags"]:
            tag_results.setdefault(t, {"total": 0, "intent_ok": 0, "esc_ok": 0})
            tag_results[t]["total"] += 1
            if r["intent_status"] in ("PASS", "XFAIL"):
                tag_results[t]["intent_ok"] += 1
            if r["esc_status"] in ("PASS", "XFAIL"):
                tag_results[t]["esc_ok"] += 1

    print(f"\n  BY TAG:")
    print(f"    {'Tag':<30} {'Cases':>6} {'Intent OK':>10} {'Esc OK':>8}")
    print(f"    {'-'*56}")
    for tag in sorted(tag_results.keys()):
        tr = tag_results[tag]
        print(f"    {tag:<30} {tr['total']:>6} {tr['intent_ok']:>8}/{tr['total']:>3}  {tr['esc_ok']:>6}/{tr['total']:>3}")

    if intent_fail or esc_fail:
        print(f"\n  UNEXPECTED FAILURES:")
        for r in _results:
            if r["intent_status"] == "FAIL" or r["esc_status"] == "FAIL":
                print(f"    ID={r['id']}: {r['text'][:60]}")
                if r["intent_status"] == "FAIL":
                    print(f"      INTENT FAIL: {r['predicted_intent']} ({r['intent_confidence']})")
                if r["esc_status"] == "FAIL":
                    print(f"      ESC FAIL: {r['predicted_escalation']} ({r['esc_confidence']})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])