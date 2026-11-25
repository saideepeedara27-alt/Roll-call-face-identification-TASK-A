# AI Collaboration Log — PhotoProof (Task A: Who Is in the Photo?)

This document records how the author (**Saideep**, final-year B.Tech CSE Data Science student) steered the AI coding assistant through architecture, matching logic, UX, testing, and evaluation for **PhotoProof**, with verbatim evidence and a complete chat-history export inventory.

> **Attribution Transparency Note:** Decisions **1–4** originated from structured Checkpoint 1–4 reviews pasted into session Turns 0–36 — those are explicitly flagged below so you can double-check any nuance against your earlier browser brainstorming chat before submission. Decisions **5–10** come 100% directly from your prompts and code reviews in this recorded Antigravity session.

---

## Key Decisions (1–10)

### Decision 1: Technology Stack & Recognition Engine *(Checkpoint 1)*
> *[Verify origin: distilled from structured Checkpoint 1 review text pasted in Turn 16]*
- **AI Suggested:** Dual `FaceEngine` implementation upfront (YuNet + SFace alongside InsightFace SCRFD + ArcFace via `onnxruntime`), served by FastAPI + Vanilla JS, SQLite + NumPy caching, and `setup_models.py`.
- **What I Decided:** Lock **OpenCV YuNet + SFace** as the single v1 engine behind a modular `FaceEngine` interface; enforce strictly **pre-runtime** downloading in `scripts/setup_models.py` (never at startup or inference); unify binary embedding caching by `(image_sha256, model_version)` in SQLite and keep persistent caching optional.
- **Why:** InsightFace pre-trained weights restrict commercial use; two engines upfront create unnecessary debugging overhead; face embeddings are sensitive biometric data.
- **Evidence:** [`rollcall/engine/opencv_engine.py`](rollcall/engine/opencv_engine.py), [`scripts/setup_models.py`](scripts/setup_models.py), [`rollcall/db.py`](rollcall/db.py).

### Decision 2: Practice, Validation & Evaluation Data Honesty *(Checkpoint 2)*
> *[Verify origin: distilled from structured Checkpoint 2 review text pasted in Turn 18]*
- **AI Suggested:** Hybrid mix using LFW, Gemini-generated group photos, and stock photos (Pexels/Unsplash).
- **What I Decided:** Reject Unsplash/Pexels as biometric IDs, avoid LFW as main calibration truth, keep consented volunteer images private (`.gitignore`d), and **never use synthetic / AI-generated photos to calibrate thresholds or claim accuracy**.
- **Why:** Stock licenses restrict biometric person identification, LFW lacks workplace group scenes, and synthetic images create identity drift between portrait and scene.
- **Evidence:** [`.gitignore`](.gitignore), [`EVAL.md`](EVAL.md).

### Decision 3: Open-Set Three-Tier Matching & Safe Hungarian Assignment *(Checkpoint 3)*
> *[Verify origin: distilled from structured Checkpoint 3 review text pasted in Turn 29]*
- **AI Suggested:** Multi-exemplar max-match, hardcoded 28–32 px quality cutoff, three-tier open-set (`Confirmed` / `Likely` / `unknown`), and Hungarian assignment.
- **What I Decided:** Treat `28–32 px` strictly as unvalidated initial hypothesis; keep independent matching as default and **never** allow Hungarian assignment to upgrade weak or `Likely` matches or remove dummy `unknown` nodes; track detection misses separately from recognition failures.
- **Why:** `"PhotoProof should prefer an honest unknown result over a confidently incorrect identity."` (Turn 2)
- **Evidence:** [`rollcall/matcher.py`](rollcall/matcher.py), [`tests/test_engine_and_matcher.py`](tests/test_engine_and_matcher.py).

### Decision 4: Concept B Studio Inspector UI & Non-Technical UX *(Checkpoint 4)*
> *[Verify origin: distilled from structured Checkpoint 4 review text pasted in Turn 36]*
- **AI Suggested:** Studio Inspector workbench layout with canvas, results list, and explainability card.
- **What I Decided:** Label `Likely` clearly as **Needs Review** (never counted Present automatically), separate **Absent** from `unknown`, add **Demo Mode** running the real pipeline, add **Wipe All Data** confirmation, keep 100% local assets, and keep enrollment non-blocking.
- **Why:** Office staff need one uncluttered visual screen that surfaces uncertainty cleanly without external network assets.
- **Evidence:** [`static/index.html`](static/index.html), [`static/app.js`](static/app.js).

---

### Decision 5: Incremental Build Order, Ephemeral Default, Strict CSP & `99.5th`-Percentile Calibration
- **AI Suggested:** Persistent SQLite cache enabled by default and `99th` percentile folder threshold without an upward cap (calling it `"1% false accept"` in early discussion).
- **What I Decided:**
  1. Switch SQLite cache to **opt-in (ephemeral by default)** and add paste-folder-path fallback (Turn 194).
  2. Enforce Strict `Content-Security-Policy` + startup self-check (Turn 194).
  3. Require **`99.5th` percentile** (`0.363` floor, **`0.50` cap** with warning, `N < 15` fallback labeled `"Percentile: not applied (N < 15)"`), ban `"1% false accept"`, state all three calibration limits honestly, and ensure `T_high = max(T_low + 0.06, 0.44)` never sits below `T_low` (Turns 271, 315, 725).
- **Verbatim Steering Quotes:**
  > `"1. Use the 99.5th percentile (not 99th). Keep the 0.363 floor and the N < 15 fallback. 2. Cap the upward adjustment: T_low must not exceed 0.50, so one odd folder cannot cause mass false rejection... Do NOT write '1% false accept'."` *(Turn 271)*
  > `"Check how T_high is computed when T_low is calibrated up to 0.50. It must never be below T_low (e.g. T_high = max(T_low + 0.06, 0.44))."` *(Turn 725)*
- **Evidence:** [`rollcall/calibration.py`](rollcall/calibration.py), [`rollcall/self_check.py`](rollcall/self_check.py), [`tests/test_calibration.py`](tests/test_calibration.py), Excerpt #1 below.

### Decision 6: Lookalike Guard & Possible-Duplicate Separation (`>= 0.55`)
- **AI Suggested:** A single flagged lookalike bucket in summary output.
- **What I Decided:** State exact margins (`0.05` base / `0.10` lookalike), separate `>= 0.55` pairs in UI as `"possible duplicate identity"` while ensuring the `0.10` guard still protects duplicate-class pairs too, inspect what `Rahul Verma` / `Vikram Patel` (`0.5789`) actually were in practice data, and keep them as a documented synthetic lookalike test case.
- **Verbatim Steering Quotes:**
  > `"Pairs with similarity >= 0.55 are shown in the Enrollment Health Studio as 'possible duplicate identity', separate from lookalikes."` *(Turn 315)*
  > `"a probe equally close to two identities in a >= 0.55 'possible duplicate' pair must land in Likely (the guard must apply to duplicate-class pairs too, never be looser)."` *(Turn 357)*
- **Evidence:** [`rollcall/calibration.py`](rollcall/calibration.py), [`tests/test_lookalike.py`](tests/test_lookalike.py).

### Decision 7: Enrollment Health Rules — Never Auto-Pick Multi-Face Photos & `T_high` Extra-Photo Guard
- **AI Suggested:** Earlier implementation auto-picked dominant faces in multi-face portraits and silently merged extra photos sharing a person's name down to `0.363` (`Akash Verma_2.jpg` at `0.4908` merged silently during smoke testing).
- **What I Decided:**
  1. Never silently auto-pick multi-face reference photos in normal enrollment — box candidates and hold in **`Needs attention`** (`Not enrolled` in group results with non-blocking banner if skipped).
  2. Ignore `.DS_Store`, `Thumbs.db`, `__MACOSX`, and dotfiles without failing them; make counts strictly additive.
  3. Tightened extra-photo rule after smoke test: **merge silently only if `similarity >= T_high`**, otherwise show both thumbnails side-by-side with `"may be a different person"` and **keep that photo OUT of matching until Keep or Reject** (Turn 782).
- **Verbatim Steering Quotes:**
  > `"Do not silently auto-pick a face in a multi-face reference photo. Show the chosen face boxed on the thumbnail, let me switch faces, and keep the file in 'Needs attention' until chosen."` *(Turn 357)*
  > `"Akash Verma_2.jpg (actually Harsha Reddy's photo, similarity 0.4908) was merged silently. Change the extra-photo rule: merge silently only if similarity >= T_high. Between 0.363 and T_high, show both thumbnails side by side with 'may be a different person', and keep that photo OUT of matching until I keep or reject it."` *(Turn 782)*
- **Evidence:** [`rollcall/app.py`](rollcall/app.py), [`rollcall/enrollment.py`](rollcall/enrollment.py), [`tests/test_enrollment_health.py`](tests/test_enrollment_health.py), Excerpt #2 below.

### Decision 8: Non-Mutating Keyboard Review Queue (`J/K/R/S/C/A/U/Z`) & Append-Only Decision Trail
- **AI Suggested:** Include `enroll_new` from group review and plain identity dropdown.
- **What I Decided:** Remove `enroll_new` (**corrections never touch reference embeddings**); isolate `"Confirmed by reviewer"` from automatic `"Confirmed"`; restrict `"C"` from acting on `unknown`; use `J/K` cleanly if arrows conflict with canvas pan; require searchable `"A"` filter, `Z` Undo, live attendance recomputation, append-only SQLite `decision_trail` (`session_id`, image hash, bbox, actor), and `Wipe All Data` trail deletion.
- **Verbatim Steering Quote:**
  > `"1. Remove enroll_new for now. Corrections must never touch reference embeddings. 2. 'C' produces status 'Confirmed by reviewer', shown separately from automatic Confirmed... 6. Trail fields: add session_id, group-photo hash, face box, and actor (automated or reviewer). The trail is append-only"` *(Turn 408)*
- **Evidence:** [`rollcall/app.py`](rollcall/app.py), [`rollcall/db.py`](rollcall/db.py), [`static/app.js`](static/app.js), [`tests/test_review_and_trail.py`](tests/test_review_and_trail.py).

### Decision 9: Second Look Tiling, Mini Coverage Map, `1200 px` Portrait Fix & Runner-Up Bug Fix
- **AI Suggested:** Earlier proposals cited synthetic `"10–15 px"` detection claims, left `unknown` storing `top1` in `runner_up_name`, and processed close-up `3000×4000` portraits at `2400 px` (producing false multi-face splits).
- **What I Decided:**
  1. Keep original-pixel quality gating across tiling and scaling, dedup by `IoU > 0.40` or center-inside + area ratio, cap low-score (`0.35 <= score < 0.65`) detections at `Likely`, preserve face indices, and log Second Look / manual boxes in trail (Turn 490).
  2. Aspect-preserving Mini Coverage Map with non-color shape/number markers, permanent disclaimer caption, `aria-label`, click selection, and narrow-window collapse (Turn 567).
  3. Reject `"10–15 px"` claim on synthetic scenes cleanly (`"on two synthetic scenes, tiling added 0 faces"`), fix the **Face #9 runner-up calculation bug**, highlight `Likely` runner-ups in list rows & evidence cards, and mandate **`1200 px` reference portrait pre-scaling + artifact overlap/score suppression** (Turns 603, 884).
- **Verbatim Steering Quotes:**
  > `"I do not accept 'YuNet catches faces down to ~10-15 px' as a finding: both scenes were synthetic. For EVAL.md the only valid claim is 'on two synthetic scenes, tiling added 0 faces'."` *(Turn 603)*
  > `"Before ARCHITECTURE.md, fix the reference-portrait double-detection problem. It is a demo risk, not an optional improvement... Face #9 in the per-face table shows the runner-up with the same name and similarity as the top candidate (0.2413) but margin 0.0839. Tell me whether this is a display bug or a calculation bug, fix it, and add a test."` *(Turn 884)*
- **Evidence:** [`rollcall/second_look.py`](rollcall/second_look.py), [`rollcall/image_utils.py`](rollcall/image_utils.py), [`rollcall/matcher.py`](rollcall/matcher.py), [`tests/test_second_look.py`](tests/test_second_look.py), [`tests/test_coverage_map.py`](tests/test_coverage_map.py), Excerpt #3 below.

### Decision 10: Print Sheet Safety, Honest `README.md` & Evidence-Only `EVAL.md`
- **AI Suggested:** Early README included unpinned dev dependencies together with runtime packages, omitted the `T_high` tracking explanation, and included untested Safari/Firefox mentions.
- **What I Decided:** Escape every name/note cleanly (`<script>alert(1)</script>.jpg` test), hide thumbnails by default on Print Sheet, add `DEMO DATA` banner, keep zero external PDF deps, split `pytest`/`httpx` into `requirements-dev.txt`, mark `28 px` / `35.0` gates as unvalidated initial hypotheses, remove Safari/Firefox claims and unlinked license guesses from README, and run realistic 3-trial `3000×4000` / `4000×3000` M2 benchmarks labeled `[Synthetic]` with an explicit `"Not measured"` section in `EVAL.md`.
- **Verbatim Steering Quotes:**
  > `"1. Say the quality-gate values (28 px, blur 35.0) are initial experimental values, not validated... 3. State which browser was tested; remove claims about Safari/Firefox that were not tested... 5. Explain why opentelemetry-api is installed. Split pytest/httpx into requirements-dev.txt"` *(Turn 725)*
  > `"For EVAL.md (after the above): label every number as synthetic; do not quote the 76 / 194 pair counts or the 0.8849 raw value as findings (they are artifacts of blending 6 base images); include a 'Not measured' section... No accuracy claims."` *(Turn 782)*
- **Evidence:** [`rollcall/exporter.py`](rollcall/exporter.py), [`tests/test_print_sheet.py`](tests/test_print_sheet.py), [`requirements.txt`](requirements.txt), [`requirements-dev.txt`](requirements-dev.txt), [`README.md`](README.md), [`EVAL.md`](EVAL.md).

---

## Pasted Chat Excerpts (Verbatim Proof)

### Chat Excerpt #1 — Rejecting Optimistic Calibration Language (Turn 271, `2026-11-25T12:46:34Z`)
```text
> AI: [proposed Option A at 99th percentile across distinct pairs and summarized it with "1% false accept" framing]
> Saideep:
Step 1 and 2 approved. For Step 3 I approve Option A with these changes:

1. Use the 99.5th percentile (not 99th). Keep the 0.363 floor and the N < 15
   fallback.
2. Cap the upward adjustment: T_low must not exceed 0.50, so one odd folder
   cannot cause mass false rejection. If the cap is hit, show a warning.
3. Be honest in the "About this run" drawer. Show: N, number of pairs, the
   percentile used, the raw value, the final value after floor/cap, and these
   limits: (a) one photo per person means only impostor behaviour is
   calibrated, (b) the rate is per pair, not per face, (c) folder pairs are
   cleaner than group-photo faces. Do NOT write "1% false accept".
4. Add a test for: N=50 synthetic similarity matrix, N<15 fallback, floor
   binding, cap binding.
```

### Chat Excerpt #2 — Catching the `0.4908` Silent Merge in Smoke Testing (Turn 782, `2026-11-25T13:43:22Z`)
```text
> AI: [reported 50-person smoke test where Akash Verma_2.jpg (Harsha Reddy's image, sim 0.4908) merged into Akash Verma because 0.4908 >= 0.363]
> Saideep:
Smoke test: two behaviour problems and one measurement problem to fix before
EVAL.md.

1. Akash Verma_2.jpg (actually Harsha Reddy's photo, similarity 0.4908) was
   merged silently. Change the extra-photo rule: merge silently only if
   similarity >= T_high. Between 0.363 and T_high, show both thumbnails side by
   side with "may be a different person", and keep that photo OUT of matching
   until I keep or reject it. Add tests: genuine same-person at high similarity
   merges; the 0.49 case is held out of matching and flagged; rejecting it
   removes it; keeping it adds it as an exemplar.
```

### Chat Excerpt #3 — Enforcing Pre-Scale Fix, Bug Audit & Honest `EVAL.md` Scope (Turn 884, `2026-11-25T13:55:32Z`)
```text
> AI: [noted close-up reference split at 1800x2400 as an engineering observation and reported Face #9 table row]
> Saideep:
Before ARCHITECTURE.md, fix the reference-portrait double-detection problem. It
is a demo risk, not an optional improvement:
1. Pre-scale reference portraits to about 1200 px on the longest side before
   detection (EXIF transpose first; group photos keep the 2400 px cap). The
   quality gate must still measure face size in original-image pixels.
...
Face #9 in the per-face table shows the runner-up with the same name and
similarity as the top candidate (0.2413) but margin 0.0839. Tell me whether this
is a display bug or a calculation bug, fix it, and add a test. For face #1, the
runner-up (Lakshmi Narayan) must be clearly visible in the list row and the
evidence card, since the top name was wrong.
```

### Decision 11: Rename to PhotoProof + Evidence Card as Signature Innovation

- **AI Suggested:** Keep the existing RollCall brand and Studio Inspector layout.
- **What I Decided:** Rename the product to **PhotoProof** ("Who Is in the Photo?") and redesign the UI around a warm, calm, non-surveillance aesthetic. The signature innovation is a click-to-open **Evidence Card** per face that shows the cropped face, the proposed reference portrait, the second-best candidate, similarity score, runner-up margin, quality warnings (blur, small face, profile angle, occlusion), and a plain-English decision explanation. Added three reviewer actions on each card — Confirm, Choose Another Person, and Mark Unknown — with keyboard shortcuts `j/k` (navigate), `z` (undo).
- **Why (Author's Decision):** The original brief explicitly required the Evidence Card as the signature innovation. The warm editorial palette avoids the surveillance-dashboard look that office staff would find uncomfortable and reinforces the product's human-in-the-loop framing.
- **Evidence:** [`static/index.html`](static/index.html), [`static/app.js`](static/app.js), [`static/styles.css`](static/styles.css).

### Decision 12: Two-Layer Status Naming for Transparency

- **AI Suggested:** Use one uniform status field across the matcher and the API response.
- **What I Decided:** Keep the matcher's internal `Confirmed`/`Likely`/`unknown` states exactly as they are (tested and stable), but translate them to evaluator-readable labels in `_build_session_response`: `Confirmed` → `High-confidence match`, `Likely` → `Needs review`, `unknown` → `Unknown`. Human reviewer overrides produce `Confirmed by reviewer`, which is distinct from the model's auto-`Confirmed`. The demo readiness strip reports `high_confidence_matches`, `needs_review_faces`, and `unknown_faces` counts explicitly.
- **Why (Author's Decision):** Raw internal status strings like `Confirmed` and `Likely` have specific technical meanings that differ from what a non-technical evaluator or office reviewer needs to see. Separating the two layers prevents confusion between "the model is confident" and "a person confirmed it".
- **Evidence:** [`rollcall/app.py`](rollcall/app.py) (`_build_session_response`), [`tests/test_api.py`](tests/test_api.py).

---

## Full AI Chat History & Export Inventory

The Task A brief requires putting your **full AI chat history in the repo, starting from your very first prompt**.

### Included in Repository
- **[`docs/chat_history_prompts.md`](docs/chat_history_prompts.md):** Curated chronological log of all **22 engineering and design steering turns** (`Step 0` through `Step 1020`, `2026-11-25T11:47:15Z` onward), exported from the session transcript with routine pings, accidental output pastes, and broken external preview links cleaned for readability.

### Checklist of External / Prior Chats to Attach Before Submission
If you used any separate browser/IDE chats *before* pasting the Step 0 master prompt and Checkpoint 1–4 review blocks into this session, export/save those chats into `docs/` (or add their public share links below) before your final push so no earlier prompt is missing:

- [ ] **Prior Planning Chat (if external):** Earlier chat where the Checkpoints 1–4 reviews pasted in Steps 2, 16, 18, 29, and 36 were drafted -> export link / markdown into `docs/`.
- [ ] **Initial Practice Image Generation Prompts (if created externally):** Prompts used to create the 6 practice portraits in `data/practice/reference/` (turns 142/146 generated via in-session subagent; add any external Gemini image-gen share links if used separately).
- [x] **Antigravity Main Build & Verification Session (Conversation `a827afa4-b846-444d-9cd4-e7c5a9177d6c`):** Exported at [`docs/chat_history_prompts.md`](docs/chat_history_prompts.md) covering the 22 substantive steering turns (Checkpoints 1–4 + Steps 1–9 + Smoke fixes + `README.md`, `EVAL.md`, `ARCHITECTURE.md`, `RESEARCH.md`, `AI_COLLAB.md` + PhotoProof transformation + Evidence Cards + final docs update).
