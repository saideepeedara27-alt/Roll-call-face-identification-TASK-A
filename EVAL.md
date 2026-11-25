# RollCall Evaluation Report (`EVAL.md`)

> **Important Scope Statement:** Every measurement in this document was collected using **synthetic imagery only** (`data/practice/` or algorithmically generated test images) on **one machine** (Apple Silicon MacBook Air M2, 8 GB RAM). **No real-world recognition accuracy, precision, recall, or false-accept claims are made.** Quantitative validation on unseen real photographs happens at live demo time.

---

## 1. Test Environment & Dataset Status

- **Hardware / OS:** Apple Silicon MacBook Air M2 (8 GB RAM), macOS `27.2.1`
- **Runtime:** Python `3.11.14`, OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) + SFace (`face_recognition_sface_2021dec.onnx`), strict local execution (`127.0.0.1`)
- **Available dataset in repository:**
  - `data/practice/reference/`: `6` synthetic `1024×1024` portraits (`Ananya Rao.jpg`, `David Brooks.jpg`, `Mei Lin.jpg`, `Priya Sharma.jpg`, `Rahul Verma.jpg`, `Vikram Patel.jpg`).
  - `data/practice/group/demo_office_group.jpg`: `1` composited synthetic group image (`1400×820 px`) containing 3 foreground faces and 1 tiny `21×29 px` face.
- **Unseen 50-person evaluation dataset:** Not bundled in this repository; supplied at live demo time. To measure realistic file-decoding latency, memory footprint, EXIF handling, and pipeline control flow on M2, a temporary **57-file synthetic smoke folder** (`50` unique identities + duplicate extra photos, multi-face, no-face, and system files) and a **10-person synthetic group photo** were generated and benchmarked.

---

## 2. Measured M2 Performance & Resource Footprint `[Synthetic, One Machine]`

### A. Realistic-Resolution 50-Identity Smoke Benchmark (`3000×4000` Refs + `4000×3000` Group, 3 Isolated Cold Runs)

Dataset (`[Synthetic]`):
- **Reference folder (`22.1 MB`, `57` files):** `50` `3000×4000` portraits (pre-scaled to `1200 px` max side after EXIF transpose, including `5` EXIF-rotated JPEGs with `Orientation` tags `3`, `6`, `8` and messy filenames such as `rohan__mehta (1).jpg`, `ANITA_DESAI.png`, `sara.kapoor-final.jpg`), `3` extra duplicate-person `3000×4000` photos, `1` multi-face `3000×4000` file, `1` no-face `3000×4000` file, and `2` system metadata files (`.DS_Store`, `Thumbs.db`).
- **Group photo (`0.33 MB`):** `4000×3000` JPEG (capped at `2400 px` max side -> `2400×1800`) containing `10` placed synthetic faces (`9` from roster + `1` color-channel-shifted outsider).

Measured across **`n = 3`** isolated cold subprocess runs on one Apple Silicon M2 (`persist_cache=False`, synthetic images):

| Metric | Min | Median | Max | Scope |
| --- | --- | --- | --- | --- |
| Cold enrollment wall time (`57` files) | `7,880.8 ms` | `7,989.1 ms` | `8,581.2 ms` | `n = 3`, synthetic, M2 |
| Peak process memory (`ru_maxrss`) | `764.2 MB` | `769.6 MB` | `877.9 MB` | `n = 3`, synthetic, M2 |
| `4000×3000` group-photo processing time | `355.5 ms` | `367.4 ms` | `379.0 ms` | `n = 3`, synthetic, M2 |
| Reference single-face enrollment without false splits | `50 / 50` | `50 / 50` | `50 / 50` | `n = 3`, synthetic, M2 |
| Group-photo faces detected / EXIF portraits upright | `10 / 10` faces, `5 / 5` EXIF | `10 / 10`, `5 / 5` | `10 / 10`, `5 / 5` | `n = 3`, synthetic, M2 |

### B. Bundled Demo Mode Run (`data/practice/`, `1400×820`, Synthetic)

| Metric | Measured Value | Scope |
| --- | --- | --- |
| Group-photo processing latency (`demo_office_group.jpg`) | `65.7 ms` (live browser run) / `55.5 ms` (`TestClient` run) | `n = 1` each, synthetic, M2 |
| Process RSS during Demo Mode | `371.8 MB` (live browser server) / `444.6 MB` (`TestClient`) | `n = 1` each, synthetic, M2 |
| Reference auto-selection note | `"Demo: dominant face auto-selected for 1 photo"` (`Priya Sharma.jpg` >= `0.65` bystander; weak `< 0.65` bystander in `Mei Lin.jpg` suppressed) | synthetic practice set |
| Detected faces outcome | `4` detected (`3 Confirmed`, `1 unknown` at `21×29 px` rejected by `< 28 px` size gate) | `n = 1`, synthetic |

---

## 3. Observed Pipeline Behaviour `[Synthetic]`

1. **Close-Up Portrait Double-Detection Fix (`3000×4000` Refs & `4000×3000` Group `[Synthetic]`):**
   - **Before fix (`1800×2400` reference cap, no artifact suppression):** Close-up `3000×4000` portraits could cause YuNet (`2023mar`) to emit overlapping/spurious second boxes; `1 / 50` hi-res single-face synthetic portraits landed in `Needs attention` (`needs_attention = 3`), and upscaled `Ananya Rao.jpg` / `David Brooks.jpg` emitted duplicate boxes.
   - **After fix (`1200 px` reference pre-scale + original-pixel size gate + overlap/containment suppression `IoU > 0.25` or `containment > 0.40` and `< 0.65` secondary score drop):** **`0 / 50`** hi-res synthetic portraits had false splits (`needs_attention = 2` for only the intentional multi-face file and mismatched extra photo), while a genuine two-person portrait (`Rahul Verma` + `Vikram Patel`) stayed in `Needs attention` and a `4000×3000` group photo containing a large close-up face produced `1` box (tested in `tests/test_enrollment_health.py`).
2. **Enrollment Health Accounting (`57` high-res files `[Synthetic]`):**
   - Headline counts summed cleanly (`57 selected = 52 enrolled, 0 enrolled with warnings, 2 needs attention, 1 failed, 2 ignored`).
   - `.DS_Store` and `Thumbs.db` were ignored; `Conference Room.jpg` failed (`No face detected in reference photo.`); messy filenames normalized cleanly.
3. **Extra-Photo `T_high` Guard (`[Synthetic]`):**
   - An intentional mismatched extra photo (`Akash Verma_2.jpg`, containing another synthetic person's face at cosine similarity `0.4958 < T_high 0.5600`) was **held out of matching** in `Needs attention` with `"may be a different person"` and side-by-side thumbnails until Keep/Reject review (`tests/test_enrollment_health.py` verifies the `0.49` case for both `reject` and `keep`).
4. **Group-Photo Decision Behaviour (`smoke_group_10.jpg` `[Synthetic]`):**
   - Because the group faces were cut directly from the same synthetic images as the references, observed same-identity similarities (**`0.83–0.96`**) are **far above real same-person photographs**, and the **`5 / 5 Confirmed` result is strictly a control-flow check, not evidence of real recognition accuracy**.
   - Additionally, synthesizing 50 people by blending only 6 base portraits is degenerate for cosine similarity distribution modeling because cross-blends collide heavily.
   - On Face `#1` (`Lakshmi Narayan` placed), the top candidate was actually the wrong blended person (`Bhavna Choudhary`, `0.8796`) ahead of the true identity as runner-up (`Lakshmi Narayan`, `0.8733`, margin `0.0062`), and the Lookalike Guard held it at `Likely` (`3` of the `4` `Likely` faces had the correct top name; Face `#1` had the true person as runner-up, surfaced visibly in the list row and evidence card).
   - The non-enrolled synthetic stranger (`Face #9`) landed in `unknown` (`0.2413 < 0.500`), with runner-up distinctly tracking the second candidate rather than repeating the top candidate.
5. **Second Look Overlapping Tiling (`[Synthetic]`):**
   - On two synthetic scenes (`2000×1100` and `2400×1350`), tiling added `0` faces (`n = 1` per scene, one machine M2, synthetic images).
   - Idempotence, index stability, low-score `< 0.65` capping at `Likely`, and `< 28 px` quality-gate rejection are unit-tested in `tests/test_second_look.py`.

---

## 4. Initial Experimental Settings (Unvalidated)

- Quality gates (`min_face_size_px = 28`, `min_blur_laplacian = 35.0`, `min_detector_score = 0.65`), default thresholds (`similarity_low = 0.363`, `similarity_high = 0.44`), margins (`0.05` base, `0.10` lookalike), and artifact suppression (`IoU > 0.25`, `containment > 0.40`) are **initial experimental values / published baselines** — none are tuned or validated against real office photographs.

---

## 5. Not Measured

The following real-world properties were **not measured** in this repository and remain open until tested on real photographs and your local machine:

1. **Real-world recognition accuracy:** True precision, recall, F1, and rank-1 identification rate across 50 real people and real group photos.
2. **False-accept rate (FAR) on real strangers:** Whether non-enrolled real people in real lighting/occlusion ever cross calibrated `T_low` / `T_high`.
3. **Small real back-row faces:** Detection recall on real blurred, backlit, or distant faces in real `4000×3000` camera group photos, and whether Second Look tiling recovers missed real faces.
4. **Real-photo double-detection behaviour:** Whether real-world high-resolution reference portraits or close-up faces in real group photos ever still produce duplicate detections or get over-suppressed by `IoU > 0.25` / `containment > 0.40` / `< 0.65`.
5. **Other browsers:** Only the built-in Chromium webview browser was tested end-to-end; Safari, Firefox, and mobile browsers have not been tested.
6. **Physical Wi-Fi-off check by you:** Automated tests and `TestClient` checks verified zero outbound URLs, CSP enforcement, and offline execution inside an OS no-network sandbox, while a manual browser pass with macOS hardware Wi-Fi switched off still rests with you.
