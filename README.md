# RollCall — Offline Face Identification & Attendance Intelligence

**Take-Home Task A: Who Is in the Photo?**

RollCall is a local-only web application that matches faces in a group photograph against a reference folder of named portraits (`Priya Sharma.jpg`, etc.). It detects faces with **OpenCV YuNet**, extracts 128-d face embeddings with **OpenCV SFace**, separates confident matches (**Confirmed**), borderline/ambiguous matches (**Likely**), and unrecognized faces (**unknown**), and tracks attendance with an explainable human-in-the-loop review workflow.

---

## 1. Prerequisites

- **OS:** macOS (tested on Apple Silicon, macOS `27.2.1`)
- **Python:** **Python 3.11+** (tested on `Python 3.11.14`; if your default `python3` points to another version, invoke `python3.11` explicitly)
- **Browser Tested:** End-to-end UI verification was run in the built-in Chromium webview browser. Other browsers (such as Safari or Firefox) have **not** been tested yet; if a browser environment cannot use the `webkitdirectory` folder picker, use the built-in **Paste Folder Path** fallback.
- **One-time internet access:** Needed **only once before runtime** to install Python packages and download the two OpenCV Zoo ONNX model files via `scripts/setup_models.py`. After that, RollCall runs **100% offline** on `127.0.0.1`.

---

## 2. Installation & Offline Model Setup

Clone the repository, create a Python 3.11 virtual environment, and install the pinned runtime dependencies:

```bash
git clone <your-repo-url> rollcall
cd rollcall
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the automated test suite as well, install the pinned development dependencies:

```bash
pip install -r requirements-dev.txt
```

> **Dependency note (`opentelemetry-api`):** Installing `fastapi==0.165.0` pulls in `opentelemetry-api` as a transitive dependency of FastAPI itself. RollCall never imports `opentelemetry` and sends no telemetry or network traffic.

### Download & Verify Models (Run Once Online, Before Disconnecting)

RollCall **never** downloads model weights at startup or runtime. Run `scripts/setup_models.py` once while online to download the OpenCV Zoo ONNX weights into `models/` and write `models/manifest.json`:

```bash
python scripts/setup_models.py
```

Expected files and SHA-256 digests recorded in `models/manifest.json` (model licensing and official source links are documented separately in `RESEARCH.md`):

| File | Size | SHA-256 |
| --- | --- | --- |
| `models/face_detection_yunet_2023mar.onnx` | `232,589` bytes | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `models/face_recognition_sface_2021dec.onnx` | `38,696,353` bytes | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

You can verify the weights manually at any time with:

```bash
shasum -a 256 models/*.onnx
```

> **Note:** On every server startup, `rollcall/self_check.py` verifies `models/*.onnx` against `models/manifest.json` and exposes the result at `GET /api/health`. Re-running `python scripts/setup_models.py` when both files are already cached performs only local SHA-256 hashing and needs no internet access.

### What Happens If Models Are Missing?

If `models/` is empty or missing when RollCall starts, it refuses to download anything and surfaces a `503` / degraded state (`ModelMissingError`):

```text
Missing required offline model file(s): face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx in <repo>/models. Run `python scripts/setup_models.py` while connected to the internet first.
```

---

## 3. Running RollCall (Offline)

With models prepared, you can turn Wi-Fi off and start the local server bound strictly to loopback (`127.0.0.1`):

```bash
source .venv/bin/activate
python -m rollcall.app
```

Or invoke Uvicorn directly:

```bash
uvicorn rollcall.app:app --host 127.0.0.1 --port 8000
```

Then open **`http://127.0.0.1:8000/`** in your browser.

---

## 4. Using the App

### Reference Folder Enrollment & Paste-Path Fallback
1. Click the roster button (**`0 Enrolled`**) or **`1. Select Reference Folder`** to open **Reference Folder Enrollment**.
2. Choose one of two offline ways to load your reference folder of individual portraits (named `First Last.jpg`, e.g., `Priya Sharma.jpg`):
   - **Choose Reference Folder** (`webkitdirectory` folder picker), or
   - **Paste-Folder-Path Fallback:** paste an absolute local folder path (e.g., `/Users/you/Photos/office_roster`) and click **`Load Folder Path`** (`POST /api/enroll/from-path`). System roots (`/`, `/System`, `/etc`, etc.) are blocked, symlinks escaping the folder are skipped, and system junk (`.DS_Store`, `Thumbs.db`, `__MACOSX`, hidden dotfiles) is ignored cleanly.
3. **Privacy default:** Embedding caching across runs is **ephemeral by default** (kept in memory for the active session). Leave **`Save embedding cache across runs (opt-in)`** unchecked unless you explicitly want SQLite caching (`runtime_data/rollcall.db`).
4. **Multi-face reference photos:** If a reference image contains multiple faces, RollCall does not silently pick one. It keeps the photo in **`Needs attention`** with boxed candidates so you can click the intended person. Before group-photo upload, a non-blocking banner warns if any pending people are not enrolled yet (**`Continue without them`** lists them as `Not enrolled`).
5. **Extra photos per person:** Additional photos sharing a parsed name (e.g., `Priya Sharma_2.jpg`) merge into that identity when cosine similarity passes `0.363`, or raise a warning (`May be a different person`) otherwise.

### Demo Mode (Bundled Practice Dataset)
Click **`Demo Mode`** in the top bar (or on the empty canvas) to run the real YuNet + SFace pipeline on the bundled practice set in `data/practice/` (`6` reference portraits + `demo_office_group.jpg`). Because two practice portraits have background bystanders, Demo Mode explicitly notes:

> `Demo: dominant face auto-selected for 2 photos`

### Group Photo Analysis & Second Look
1. Click **`Upload Group Photo`** (or drag & drop a `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` image onto the canvas stage).
2. Inspect each boxed face on the canvas or in the right-hand **Results & Explainability Inspector**:
   - **Confirmed:** Meets calibrated low threshold `T_low` (floored at `0.363`, capped at `0.50` when calibrated across `>= 15` identities), adaptive high threshold **`T_high = max(T_low + 0.06, 0.44)`** (so when `T_low = 0.50`, `T_high = 0.56` and is never below `T_low`), and runner-up margin (`0.05` standard, or `0.10` lookalike margin for pairs `>= 0.42`).
   - **Likely:** Borderline similarity (`T_low <= sim < T_high`), narrow margin, lookalike ambiguity, or low detector score. **Not counted as Present** until a human confirms.
   - **unknown:** Below `T_low` or rejected by the quality gate (`< 28 px` face size or `< 35.0` Laplacian blur variance — **initial experimental values, not validated** on a real benchmark).
3. Click **`About this run`** to inspect the `99.5th`-percentile folder calibration details, lookalike/possible-duplicate pairs (`>= 0.55`), and explicit limitations.
4. Click **`Second Look`** to scan overlapping tiles (`2×2` or `3×2` with 20% overlap) for missed faces while keeping existing face indices stable, or click **`Draw Box`** to outline a missed face manually (requires detector score `>= 0.35`, capped at `Likely`).
5. Use the **Mini Coverage Map** (`Map` button) to view your zoom/pan viewport and jump directly to detected face markers.

### Keyboard Review Queue & Shortcuts
Click **`Review Queue`** (or press `R`) to step through uncertain/unresolved faces (`Likely` and `unknown`). Reviewer decisions update attendance live in session state, **never mutate reference embeddings**, and record rows in the **Append-Only Decision Trail**:

| Key | Action |
| --- | --- |
| `J` / `K` | Next / previous face |
| `R` | Toggle Keyboard Review Queue |
| `S` | Skip current face in Review Queue |
| `C` | Confirm a `Likely` match -> `Confirmed by reviewer` (no-op on `unknown`) |
| `A` | Focus searchable **`Assign (A)`** name filter |
| `U` | Mark selected face as `unknown` |
| `Z` | Undo last reviewer correction |
| `Esc` | Close active modal / assign dropdown / exit Review Queue |

### Exports & Print-Ready Attendance Sheet
- **`Export PNG`** (`GET /api/sessions/{session_id}/export/png`): Annotated group photo with boxes and labels.
- **`Export CSV`** (`GET /api/sessions/{session_id}/export/csv`): Full attendance + match table.
- **`Print Sheet`** (`GET /api/sessions/{session_id}/export/print`): Opens an HTML print-ready report designed for browser **`Print / Save as PDF`** (no external dependencies, zero face thumbnails by default, HTML-escaped values, `DEMO DATA` banner in Demo Mode, Page 1 summary + absent list, Page 2+ append-only Decision Trail appendix).

### Wipe All Data
Click **`Wipe All Data`** -> **`Permanently Wipe Data`** (`POST /api/wipe`) to delete all local identities, reference crops, opted-in SQLite cached embeddings, active session state, and the decision trail. Model files in `models/` and your original images on disk are never touched.

---

## 5. Offline Verification

You can verify RollCall's offline guarantees in three ways:

1. **Automated test suite (requires `requirements-dev.txt`):**
   ```bash
   pytest -v
   ```
   Tests verify zero `http://`/`https://` URLs in `static/`, strict `Content-Security-Policy` headers (`default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'`), startup model checksum verification, XSS-escaped print sheets, and all matching/review rules.
2. **Startup self-check endpoint:**
   Visit `http://127.0.0.1:8000/api/health` and confirm `"offline_mode": true`, `"bind_host": "127.0.0.1"`, `"model_ready": true`, and `"self_check": {"ok": true, ...}`.
3. **Wi-Fi-off check:**
   Turn off Wi-Fi on macOS, start `python -m rollcall.app`, open DevTools -> Network in `http://127.0.0.1:8000/`, click **`Demo Mode`**, run **`Second Look`**, and open **`Print Sheet`**. Every request stays on `127.0.0.1:8000`.

---

## 6. Observed Measurements & Honest Limits

- **Measured Demo Mode run** (`data/practice/group/demo_office_group.jpg`, `1400×820 px`, Apple Silicon M2 macOS): `~55.5–67.1 ms` server processing latency and `~367.7–444.6 MB` process RSS memory across local runs. No throughput or batch benchmarks beyond this single local setup are claimed here.
- **Experimental threshold & gate values:** Quality gates (`28 px` minimum face size, `35.0` minimum Laplacian blur variance, `0.65` normal detector score) and default matching thresholds are **initial experimental values, not validated** on a real dataset.
- **Synthetic practice data only:** `data/practice/` contains 6 synthetic reference portraits and 1 composited group photo. **No real-world recognition accuracy, precision, or recall claims are made** on unseen faces.
- **Cosine similarity is not a probability:** Scores (`0.363`, `0.44`, etc.) are raw SFace cosine similarities, not calibrated probabilities. Folder calibration uses one photo per enrolled identity, so only impostor-pair behavior across enrolled photos is estimated—group photos with real blur, compression, occlusion, or small faces still require human review of `Likely` and `unknown` rows.
- **No unseen 50-person evaluation dataset is bundled:** Full evaluation on unseen real-world photographs takes place at demo time and is tracked separately in `EVAL.md`.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `GET /api/health` shows `"status": "degraded"` or `/api/demo` returns `503` | `models/*.onnx` missing or incomplete | Reconnect briefly and run `python scripts/setup_models.py`, then verify `shasum -a 256 models/*.onnx`. |
| `python3 -m venv` uses the wrong Python version | macOS `python3` may point to a newer/older interpreter | Run `/opt/homebrew/bin/python3.11 -m venv .venv` (or `python3.11 -m venv .venv`). |
| Folder upload dialog does not appear or selects single files | Browser environment lacks `webkitdirectory` folder support | Use **Paste Folder Path** inside **Reference Folder Enrollment** and click **`Load Folder Path`**. |
| `[ERROR] Access to system directories is not allowed.` | Pasted path is under `/`, `/System`, `/Library`, `/etc`, etc. | Move reference images into a user directory (e.g., `~/Pictures/rollcall_roster`). |
| `[Errno 48] Address already in use` on port `8000` | Another local server is already listening on `127.0.0.1:8000` | Stop the existing process or run `uvicorn rollcall.app:app --host 127.0.0.1 --port 8001`. |
