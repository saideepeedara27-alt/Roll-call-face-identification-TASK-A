# AI Collaboration Log — RollCall (Task A: Who Is in the Photo?)

This document records the genuine pair-programming decisions, trade-offs, and evidence between the project author and the AI coding assistant throughout the development of **RollCall**.

---

## Key Decisions

### Decision 1: Technology Stack & Recognition Engine (Checkpoint 1)
- **AI Suggested:** Build a pluggable `FaceEngine` supporting both OpenCV YuNet + SFace and InsightFace (SCRFD + ArcFace via `onnxruntime`), served via FastAPI + Uvicorn, Vanilla HTML/CSS/ES6 JS frontend with `webkitdirectory` (and optional native picker), and SQLite + NumPy caching.
- **What I Decided:** Approved **OpenCV YuNet + SFace** as the single initial implementation behind a modular `FaceEngine` interface, with **FastAPI + Uvicorn** (bound strictly to `127.0.0.1`, CPU inference offloaded to background workers), **Vanilla HTML/CSS/ES6 JS** (`webkitdirectory` only, no native picker fallback unless needed), **SQLite + NumPy** (single unified binary embedding cache keyed by image hash + model version, persistent caching optional), and strict pre-download of models in `setup_models.py` (never at runtime). Kept InsightFace/SCRFD as a researched alternative only.
- **Why:**
  1. Implementing two engines upfront adds unnecessary debugging before a baseline exists.
  2. InsightFace's pretrained weights are restricted to non-commercial research, whereas OpenCV Zoo's YuNet/SFace weights carry permissive licenses (Apache 2.0 / MIT).
  3. A single unified SQLite embedding cache avoids format duplication and simplifies privacy-first data wiping.
- **Evidence:** *Pending* (implementation, model checksum verification, and M2 Mac benchmarking in upcoming commits).

---

### Decision 2: Practice, Validation & Evaluation Data Strategy (Checkpoint 2)
- **AI Suggested:** A hybrid 3-partition strategy combining LFW (for threshold calibration), Gemini-generated office group photos, and permissively licensed stock photos (e.g., Pexels/Unsplash) or consented volunteers.
- **What I Decided:** Approved the **modified hybrid 3-partition strategy** (`practice` + `dev_50` + `validation` + `eval_holdout`) with strict identity-disjoint separation, **rejecting** Pexels/Unsplash as biometric identity sources, **rejecting** LFW as the main threshold-calibration dataset, and **prohibiting** AI-generated photos from calibrating recognition thresholds. Consented volunteer photos (kept strictly local and `.gitignore`d) will be used for genuine validation wherever available; AI-generated photos will be used strictly for UI, 50-person workflow, and edge-case testing.
- **Why:**
  1. Stock photo licenses (Unsplash, Pexels) restrict or do not clear biometric identification of natural persons.
  2. LFW consists of single-person portraits with unclear image-use licensing, not realistic workplace group photos.
  3. Generative AI introduces identity drift between reference portraits and group scenes, making it invalid for calibrating real-world similarity thresholds.
  4. A dedicated 50-person dev test is needed alongside the small 10–15 person practice set to test full-scale enrollment.
- **Evidence:** *Pending* (data directory structure, evaluation harness, and validation metrics in Checkpoint 5).

---

### Decision 3: Recognition, Quality Gating & Open-Set Matching Strategy (Checkpoint 3)
- **AI Suggested:** All-exemplar max-match for multi-photo identities; quality gate before matching; 3-tier open-set (`Confirmed` / `Likely` / `unknown`) using similarity score + runner-up margin; optional threshold-gated Hungarian assignment with dummy `unknown` nodes.
- **What I Decided:** Approved all four components with four strict safeguards:
  1. Do not hardcode a 28–32 px minimum face-size cutoff—establish the usable quality floor empirically on validation data alongside blur and lighting.
  2. Keep the three states strictly separated (`Confirmed`, `Likely`, `unknown`); `Likely` never silently upgrades to `Confirmed` and requires human review/confirmation.
  3. Keep **independent per-face matching as the default**; enable Hungarian assignment only if validation proves measurable improvement, and never allow it to upgrade a weak or `Likely` match or bypass dummy `unknown` nodes.
  4. Track detection failures (missed faces) separately from recognition failures (low-quality / `unknown` / misidentification).
- **Why:** Prevents forced false positives, preserves multi-appearance variation (e.g., glasses), and ensures honest, explainable evaluation metrics.
- **Evidence:** *Pending* (validation curves, FAR/FRR benchmarks, and ablation results in Checkpoint 5).

---

## Pasted Chat Excerpts
*(To be added from genuine session transcripts as development progresses.)*

## Full Exported Chat History
*(Full exported conversation history will be added to the repository before final submission.)*
