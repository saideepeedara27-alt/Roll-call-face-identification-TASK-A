# PhotoProof — Offline, Privacy-First Group Photo Face Identification

> **Take-Home Task A: Who Is in the Photo?**

PhotoProof is a fully offline, human-in-the-loop web application that helps office staff identify who is in a group photograph. It matches detected faces against a folder of named reference portraits, classifies each result as **High-confidence match**, **Needs review**, or **Unknown**, and surfaces a per-face **Evidence Card** so a reviewer can confirm, correct, or override every suggestion before any result is treated as final.

All processing runs entirely on the local machine — no internet connection, cloud API, telemetry, or CDN is used at runtime.

---

## Repository Structure

```
PhotoProof/
├── data/practice/          # Bundled synthetic demo dataset (6 portraits + 1 group photo)
├── docs/                   # Architecture diagram, chat history, Archify HTML/SVG
├── models/                 # ONNX weights (downloaded once via setup_models.py, gitignored)
├── rollcall/               # Python backend package (FastAPI + CV pipeline)
│   ├── app.py              # API server, session state, review actions
│   ├── engine/             # Abstract FaceEngine + OpenCV YuNet/SFace implementation
│   ├── matcher.py          # Open-set three-tier matching, lookalike guard, 1-to-1 conflict
│   ├── calibration.py      # 99.5th-percentile folder threshold calibration
│   ├── enrollment.py       # Reference folder ingest, health checks, multi-face holdout
│   ├── exporter.py         # Annotated PNG, CSV, HTML print sheet export
│   └── ...                 # filename_parser, image_utils, second_look, db, self_check
├── scripts/
│   ├── setup_models.py     # One-time model download & SHA-256 manifest writer
│   └── evaluate.py         # CLI evaluation helper
├── static/                 # Vanilla HTML/CSS/ES6 frontend (zero CDN dependencies)
│   ├── index.html          # Single-page UI with Evidence Cards and Demo Readiness panel
│   ├── app.js              # Canvas drawing, Evidence Card controller, keyboard review queue
│   └── styles.css          # Warm, calm, accessible editorial palette
├── tests/                  # 42 pytest tests covering all matching, review, and export rules
├── README.md               # ← You are here
├── RESEARCH.md             # Technology comparison — 5 options evaluated, 1 chosen
├── ARCHITECTURE.md         # System diagram (Archify), module map, 8 trade-offs
├── EVAL.md                 # Honest evaluation report — synthetic only, no inflated claims
└── AI_COLLAB.md            # 12 documented decisions with verbatim chat excerpts
```

---

## 1. Prerequisites

| Requirement | Detail |
|-------------|--------|
| **OS** | macOS (tested on Apple Silicon, macOS `27.2.1`) |
| **Python** | Python 3.11+ (tested on `Python 3.11.14`) |
| **Browser** | Chromium-based browser tested end-to-end; use **Paste Folder Path** fallback if `webkitdirectory` is unavailable |
| **Internet** | Needed **once only** to install packages and download model weights — runtime is 100% offline |

---

## 2. Installation & Offline Model Setup

### Step 1 — Clone and install

```bash
git clone https://github.com/saideepeedara27-alt/Roll-call-face-identification-TASK-A.git photoproof
cd photoproof
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To also run the automated test suite:

```bash
pip install -r requirements-dev.txt
```

> **Note on `opentelemetry-api`:** `fastapi==0.165.0` pulls this in as a transitive dependency. PhotoProof never imports or emits any telemetry.

### Step 2 — Download model weights (run once, while online)

PhotoProof **never** downloads anything at startup or during runtime. Run this once to fetch the two OpenCV Zoo ONNX files and write their SHA-256 checksums to `models/manifest.json`:

```bash
python scripts/setup_models.py
```

Expected model files:

| File | Size | SHA-256 |
|------|------|---------|
| `models/face_detection_yunet_2023mar.onnx` | 232,589 bytes | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `models/face_recognition_sface_2021dec.onnx` | 38,696,353 bytes | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

Verify at any time:

```bash
shasum -a 256 models/*.onnx
```

> On every server startup, `rollcall/self_check.py` re-verifies checksums against `models/manifest.json`. If the files are missing, the server surfaces a `503` with a clear message — it never attempts a download.

**What happens if models are missing:**

```text
Missing required offline model file(s): face_detection_yunet_2023mar.onnx,
face_recognition_sface_2021dec.onnx in <repo>/models.
Run `python scripts/setup_models.py` while connected to the internet first.
```

### Step 3 — Disconnect Wi-Fi (optional for demo)

After completing Steps 1 and 2, the application runs with no network access required.

---

## 3. Running PhotoProof

```bash
source .venv/bin/activate
python -m rollcall.app
```

Or with Uvicorn directly:

```bash
uvicorn rollcall.app:app --host 127.0.0.1 --port 8000
```

Then open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

The server binds strictly to `127.0.0.1` (loopback only) and disables all FastAPI docs/redoc endpoints. A strict `Content-Security-Policy` header blocks any external resource loading.

---

## 4. Using PhotoProof

### Start Screen

The start screen shows two large primary actions and a **Demo Readiness** panel that tracks six live metrics — references loaded, references rejected, faces detected, high-confidence matches, needs-review faces, and unknown faces. A privacy notice confirms that all processing stays on this device.

### Step 1 — Load Reference Folder

Click **"Load reference folder"** and select a folder of named portrait images. Filenames encode identity, e.g. `Priya Sharma.jpg`, `Rahul Verma.png`.

PhotoProof performs a **reference health check** on every image:

| Check | Outcome |
|-------|---------|
| No face detected | Flagged — excluded from matching |
| Multiple faces detected | Held in "Needs attention" — you pick the intended person |
| Face too small (< 28 px) | Flagged |
| Image too blurry (Laplacian variance < 35.0) | Flagged |
| Duplicate name in folder | Flagged |
| System file (`.DS_Store`, `Thumbs.db`, etc.) | Silently ignored |
| Extra photo with same name but low similarity | Held for Keep/Reject review |

A **contact sheet** shows all successfully enrolled people. Rejected images are listed with their reason.

**Paste-path fallback:** If the browser folder picker is unavailable, click "Or enter folder path" and paste an absolute local path, then click **Load Folder Path**.

### Step 2 — Analyse Group Photo

Click **"Analyse group photo"** and select the group photograph (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).

PhotoProof detects every usable face, assigns each a stable label (Face 1, Face 2, …), and classifies each result:

| Status | Meaning |
|--------|---------|
| **High-confidence match** | Similarity ≥ T_high (calibrated), runner-up margin ≥ 0.05 (or 0.10 for lookalike pairs). A name is proposed. |
| **Needs review** | Borderline similarity, narrow margin, lookalike guard triggered, or low detector score. Two best candidates shown. Human decision required. |
| **Unknown** | Similarity below T_low, or face rejected by quality gate. No name is forced. |

> Thresholds are calibrated per folder: `T_low` = 99.5th percentile of all distinct enrolled identity-pair similarities, clamped to [0.363, 0.50]. `T_high = max(T_low + 0.06, 0.44)`.

### Step 3 — Review Evidence Cards

Click any face label or bounding box to open its **Evidence Card**:

- Cropped face from group photo
- Proposed reference portrait
- Second-best candidate (when relevant)
- Similarity score and runner-up margin
- Quality warnings (blur, small face, profile angle, occlusion)
- Plain-English decision explanation, e.g.:
  - *"Marked Unknown because no reference was sufficiently similar."*
  - *"Needs review because the top two candidates were too close (margin 0.03)."*

**Reviewer actions on the card:**

| Action | Effect |
|--------|--------|
| **Confirm** | Accepts proposed name → status becomes `Confirmed by reviewer` |
| **Choose another person** | Type-to-filter searchable list of all enrolled names |
| **Mark Unknown** | Clears any proposed name |

Human edits update the canvas labels and results list immediately. `Confirmed by reviewer` is visually distinct from model-suggested `High-confidence match`, so evaluators can tell them apart at a glance.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `J` / `K` | Navigate to next / previous face |
| `R` | Toggle the Keyboard Review Queue (steps through Needs review + Unknown faces) |
| `S` | Skip current face in Review Queue |
| `C` | Confirm current face → `Confirmed by reviewer` |
| `A` | Open searchable assign dropdown |
| `U` | Mark current face as Unknown |
| `Z` | Undo last reviewer action |
| `Esc` | Close Evidence Card / modal / dropdown |

### Exports

| Export | Description |
|--------|-------------|
| **Export PNG** | Annotated group photo with readable, non-overlapping face labels |
| **Export CSV** | Results table: face number, final status, name, similarity score, decision source (model vs. reviewer) |
| **Print Sheet** | Browser-printable HTML report — Page 1 summary, Page 2+ append-only Decision Trail. No external dependencies. |

> The CSV escapes formula-injection characters (`=`, `+`, `-`, `@`) to prevent spreadsheet injection.

### Clear Session

Click **"Clear session"** to wipe all in-memory identities, embeddings, and session state. Model files and original images on disk are never touched.

---

## 5. Demo Mode (Bundled Practice Dataset)

Click **"Demo Mode"** (shown on the empty canvas) to run the full YuNet + SFace pipeline on the bundled practice dataset in `data/practice/` — 6 synthetic reference portraits plus `demo_office_group.jpg`.

This lets an evaluator verify the end-to-end pipeline immediately without supplying their own images.

---

## 6. Offline Verification

### Automated tests

```bash
pytest -v
# → 42 passed
```

Tests cover: filename parsing, invalid reference images, unknown handling, ambiguity/lookalike guard, stable face labels, 1-to-1 conflict demotion, CSV formula-injection escaping, manual decision tracking, zero external URLs in `static/`, strict CSP headers, startup model checksum verification, and no-network startup.

### Health endpoint

```
GET http://127.0.0.1:8000/api/health
```

Confirms: `"offline_mode": true`, `"bind_host": "127.0.0.1"`, `"model_ready": true`, `"self_check": {"ok": true}`.

### Wi-Fi-off check

Turn off Wi-Fi on macOS, start the server, open DevTools → Network in the browser, run Demo Mode, trigger Second Look, and open Print Sheet. Every request must stay on `127.0.0.1:8000` with no external calls.

---

## 7. Observed Performance & Honest Limits

| Metric | Value | Scope |
|--------|-------|-------|
| Demo Mode server latency | ~55.5–67.1 ms | n=1, synthetic, M2 |
| Demo Mode process RSS | ~368–445 MB | n=1, synthetic, M2 |
| 50-portrait cold enrollment | 7.9–8.6 s | n=3, synthetic 3000×4000, M2 |
| 10-person group photo analysis | 355–379 ms | n=3, synthetic 4000×3000, M2 |

> **All numbers are synthetic-data benchmarks on one machine.** No real-world accuracy, precision, recall, or false-accept rate claims are made. See `EVAL.md` for the full honest evaluation report.

**Threshold and gate values** (`28 px` min face size, `35.0` Laplacian blur, `0.65` detector score, `0.363` similarity floor) are **initial experimental values from published OpenCV baselines** — not tuned or validated on real office photographs.

---

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `/api/health` shows `"status": "degraded"` | Model files missing or corrupted | Run `python scripts/setup_models.py` and verify `shasum -a 256 models/*.onnx` |
| `venv` uses wrong Python version | macOS `python3` points to a different version | Use `python3.11 -m venv .venv` or `/opt/homebrew/bin/python3.11 -m venv .venv` |
| Folder picker selects only one file | Browser lacks `webkitdirectory` support | Use **"Or enter folder path"** and paste the absolute folder path |
| `[ERROR] Access to system directories is not allowed` | Pasted path is a system root | Move images to a user folder (e.g., `~/Pictures/photoproof_roster`) |
| `[Errno 48] Address already in use` | Port 8000 already in use | Run `uvicorn rollcall.app:app --host 127.0.0.1 --port 8001` |

---

## 9. Documentation Index

| File | Contents |
|------|----------|
| `README.md` | Setup, usage, and demo instructions (this file) |
| `RESEARCH.md` | Comparison of 5 face detection/recognition approaches; final choice rationale |
| `ARCHITECTURE.md` | Archify-generated system diagram, module map, 8 documented trade-offs |
| `EVAL.md` | Synthetic-only evaluation report — performance numbers, pipeline behaviour, honest limits |
| `AI_COLLAB.md` | 12 author decisions with AI suggestions, what was decided, why, and evidence links |
| `docs/chat_history_prompts.md` | Full curated AI session prompt history (Steps 0–1020) |
| `docs/photoproof-architecture.html` | Interactive Archify architecture diagram (3 guided views) |
| `docs/photoproof-architecture.svg` | Static SVG export of architecture diagram |
