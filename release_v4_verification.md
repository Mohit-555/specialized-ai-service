V4 Release Verification
=======================

This document defines the canonical V4 release artifacts and provides
a verification process to detect accidental checkpoint replacement,
corruption, or version drift.

RELEASE ARTIFACTS
-----------------

  intent-v4-en/
    model.safetensors   (SHA256: 2efc17fb92a28d68ad5776bf648e04d14c8a72697da9e96067fe725882deb52b)
    config.json         (model_type: distilbert, num_labels: 10)
    metadata.json       (training_date: 2026-07-27T08:44:45, best_val_accuracy: 0.7635)

  escalation-v4-en/
    model.safetensors   (SHA256: 50d38eae8ddc73def7902c6554609e2637eae5f81e34f557c31d7fab17d7f972)
    config.json         (model_type: distilbert, num_labels: 1)
    metadata.json       (training_date: 2026-07-27T08:46:58, best_threshold: 0.65)

PRODUCTION CONFIGURATION
------------------------

  INTENT_MODEL_VERSION   = intent-v4-en
  ESCALATION_MODEL_VERSION = escalation-v4-en
  ESCALATION_THRESHOLD   = 0.65

VERIFICATION COMMANDS
---------------------

1. SHA256 check:

   sha256sum models/intent/intent-v4-en/model.safetensors
   # Expected: 2efc17fb92a28d68ad5776bf648e04d14c8a72697da9e96067fe725882deb52b

   sha256sum models/escalation/escalation-v4-en/model.safetensors
   # Expected: 50d38eae8ddc73def7902c6554609e2637eae5f81e34f557c31d7fab17d7f972

2. Training args present:

   ls models/intent/intent-v4-en/training_args.bin
   ls models/escalation/escalation-v4-en/training_args.bin

3. Service health:

   INTENT_MODEL_VERSION=intent-v4-en \
   ESCALATION_MODEL_VERSION=escalation-v4-en \
   ESCALATION_THRESHOLD=0.65 \
   uvicorn ai_service.app.main:app --host 0.0.0.0 --port 8001

   curl -s http://localhost:8001/health
   # Expected: {"status":"healthy","models_loaded":true,
   #            "intent_model":"intent-v4-en",
   #            "escalation_model":"escalation-v4-en"}

4. Quick inference smoke test:

   curl -s -X POST http://localhost:8001/v1/classify \
     -H "Content-Type: application/json" \
     -d '{"text":"What is the price?"}'
   # Expected: intent="pricing", escalation_required=false

5. Deterministic inference check (run 3 times, must match):

   for i in 1 2 3; do
     curl -s -X POST http://localhost:8001/v1/classify \
       -H "Content-Type: application/json" \
       -d '{"text":"I want a refund."}'
     echo
   done
   # All 3 responses must be identical

6. Artifact tests:

   pytest tests/test_models.py -v -m "not slow"
   # Expected: all pass, including:
   #   test_v4_intent_model_safetensors_present
   #   test_v4_escalation_model_safetensors_present
   #   test_v4_training_args_present

7. Regression tests:

   pytest tests/regression/test_v4_behavior.py -v
   # Expected: no FAIL results (XFAIL are expected known limitations)

ARCHIVE RESTORATION
-------------------

If artifacts are lost or corrupted, restore from:

  models/specialized-ai-v4-en.tar.gz

  tar -xzf specialized-ai-v4-en.tar.gz \
    -C models/intent/intent-v4-en/ \
    --strip-components=2 \
    v4_release/intent-v4-en/model.safetensors

  tar -xzf specialized-ai-v4-en.tar.gz \
    -C models/escalation/escalation-v4-en/ \
    --strip-components=2 \
    v4_release/escalation-v4-en/model.safetensors

Then re-verify SHA256.