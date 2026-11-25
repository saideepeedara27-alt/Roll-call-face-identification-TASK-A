# Curated User Prompt History — RollCall (`Task A: Who Is in the Photo?`)

Chronological engineering and design steering prompts by **Saideep** from the Antigravity session (`2026-11-25`, starting from the initial prompt at Step 0).

> **Curation Note:** Routine operational pings (resume prompts, short commit/run commands, accidental copy-pastes, and raw terminal outputs) and broken placeholder preview links pasted from external UI mockups were removed for clarity. All kept prompts retain their real wording and decisions.

---

## Initial Master Prompt & Environment Specification (`Step 0 • 2026-11-25T11:47:15Z`)

````text
# ROLLCALL — AI PAIR-PROGRAMMING MASTER PROMPT

I want you to act as my senior software engineer, AI/ML engineer, computer vision researcher, system architect, UI/UX designer, and pair-programming partner.

I am a final-year B.Tech Computer Science student specializing in Data Science. My technical background includes Python, SQL, Machine Learning, Docker, Git, and basic full-stack development.

I am building a take-home assignment called **Task A: Who Is in the Photo?**

I have provided the original Task A PDF. Treat that document as the primary source of truth throughout the entire project. Read and understand the complete document before making recommendations or starting implementation.

Do not omit, weaken, or replace any requirement from the original assignment.

My chosen product concept is **RollCall** — an offline, privacy-first, explainable face identification and attendance intelligence application.

## My development environment

I will develop and demonstrate this application using:

- MacBook Air M2.
- Apple Silicon architecture.
- 8 GB unified RAM.
- 256 GB SSD.
- macOS.
- Python 3.11.

Take these hardware limitations seriously throughout development. Prefer lightweight models, efficient CPU processing, local caching, memory-efficient image handling, and minimal dependencies. Investigate Apple Silicon acceleration only when it is genuinely beneficial and reliable.

The application must not depend on expensive hardware, cloud processing, or internet connectivity.

## What I want to build

The application should allow an office employee to select a folder containing photographs of approximately 50 different people. The filenames contain the identities, such as Priya Sharma.jpg.

The employee then uploads a group photograph containing approximately 10 people.

RollCall should detect the faces, identify people who exist in the reference folder, mark people who are not enrolled as Unknown, and display the results directly on the photograph and in a corresponding list.

However, I do n
<truncated 10303 bytes>
has been implemented, tested, benchmarked, or documented unless it has actually been done.

After each meaningful implementation step, tell me what changed, how to test it, and what Git commit I should make.

Adapt the implementation to the available time, hardware, and actual results. Prioritize the working core application over unnecessary complexity.

## Final objective

Build RollCall into a reliable, explainable, privacy-preserving, offline face identification and attendance intelligence application that performs on photographs it has never seen before.

I want the judges to be able to provide their own folder of 50 individuals and a real group photograph containing unknown people, run the application entirely offline on my MacBook Air M2, and see the results.

I also want to be able to explain the system architecture, model choices, recognition algorithms, calibration limitations, privacy design, evaluation methodology, and engineering trade-offs confidently in an interview.

**Start by reading the attached original Task A PDF completely and begin Checkpoint 0 only. Do not start coding or move to Checkpoint 1 until I confirm your understanding.** go through the given and look take care of the system that can actually perform the actions i prompted above and look make all the in struction to being followed and while implementing ask and say me what are the things and languages and technoklogies and the libreries and all to me if they were normal ones. which dont effect the mac u can proceed if u feel there mighrt be any unwanted issue ar something like that warn me and also dont try t use that kind of thisns
````

## Risk Review Before Locking Checkpoint 1 (`Step 2 • 2026-11-25T11:50:10Z`)

````text
Your understanding is strong and technically thoughtful. You have identified most of the difficult parts that could cause RollCall to fail during the live demo.

However, I would fix a few points before approving Checkpoint 1. Some are technical risks, while others are assumptions that go beyond the original Task A brief.
1. Important issues to correct
High priority
 1. Don't assume exactly one reference photo per person

Your document says the folder contains approximately one photo per identity. The original PDF says the folder contains photos of 50 different people, with filenames representing identities. It does not explicitly guarantee exactly one file per person.

Change: Design ingestion to support one or more reference photos per identity, even if the initial demo dataset has one photo per person. This also makes recognition more reliable when multiple reference images are available.

High priority
 2. The 20–40 px face problem cannot be completely solved by upscaling

Your proposed multi-scale detection is useful, but enlarging a 20×20 pixel face does not recreate missing facial information. A detector might locate the face without the recognition model being able to identify it reliably.

Change: Include a minimum usable face-quality threshold. If a face is too small, blurry, or occluded, show Unknown or a review status rather than pretending the identity is reliable. Track missed detections separately from detected-but-unrecognized faces.

High priority
 3. Don't promise a 1–1.5 GB memory limit without benchmarking

Your MacBook Air M2 has 8 GB unified memory. Keeping the application lightweight is sensible, but a strict 1–1.5 GB limit is an arbitrary target, not a requirement from the assignment.

Change: Set a practical memory budget after testing the selected detector, recognition model, image sizes, and batch processing. Measure actual peak memory during enrollment and group-photo processing.

High priority
 4. The Hungarian algorithm should not force identity assignments

You
<truncated 3697 bytes>
licate detections and overlapping bounding boxes.
- Group-photo quality: Handle rotated images, EXIF orientation, large image dimensions, and different lighting conditions.
- Processing failures: If one image fails, continue processing the remaining images and report the failure.
- Reproducibility: Record model version, thresholds, processing settings, and evaluation dataset details.
- Privacy: Keep photographs, face embeddings, corrections, and temporary files local; provide a way to clear them.
One more thing: do not promise that all 10 people will always be detected. The system should report detection failures honestly.
5. My assessment
Overall problem understanding
8.5/10


Dimension	Assessment
Original Task A requirements	Covered
Offline execution	Covered
Open-set recognition	Strong
Small-face handling	Needs realistic limits
Hardware strategy	Needs benchmarking
Identity assignment	Avoid forced matching
Documentation	Covered
Live demo readiness	Needs testing


My recommendation: Keep your current document, but incorporate the corrections above before moving forward.

The most important principle is this: RollCall should prefer an honest unknown result over a confidently incorrect identity. A smaller number of reliable identifications is better than a visually impressive but inaccurate demo.

Once these corrections are incorporated, proceed to Checkpoint 1 — Technology and Model Selection. Do not choose the model based only on published accuracy; compare licensing, Apple Silicon compatibility, memory usage, offline operation, small-face performance, and actual results on your test data.
````

## Checkpoint 1 — Technology & Model Stack Approval (`Step 16 • 2026-11-25T11:56:09Z`)

````text
I have reviewed the Checkpoint 1 proposal against your MacBook Air M2 specifications, the original Task A requirements, and the current official documentation.
My decision: I approve the overall technology stack, but I recommend five changes before we lock it in. The proposed architecture is suitable for this project. There is no need to replace it with a heavier framework or introduce unnecessary technologies.
1. Face detection and recognition — approve with changes
Recommended starting point
YuNet + SFace
YuNet — Face detection
Finds faces and their locations, including facial landmarks. It is a lightweight starting point, but its performance on tiny and partially occluded faces must be tested.
SFace — Face recognition
Generates a 128-dimensional face representation for comparing identities. It is a practical lightweight baseline.
Why I approve it:
Relatively small model files.
OpenCV provides both detection and recognition interfaces.
The OpenCV Zoo SFace model directory provides an Apache 2.0 license.
GitHub+1
OpenCV documents the recognition model's performance across several face-recognition benchmarks. However, its recommended similarity threshold varies across datasets, so we must calibrate ours rather than copy a benchmark threshold.
GitHub+1
My changes
A. Do not commit to supporting two recognition engines immediately.
The proposal suggests implementing both YuNet/SFace and SCRFD/ArcFace. That could create unnecessary development and d
<truncated 2853 bytes>
sponsibilities:
Technology
Responsibility
SQLite
Identity records, image metadata, corrections and evaluation results
NumPy
Face embeddings and numerical operations
OpenCV
Face detection, alignment, recognition and image annotations
Pillow
Image loading, EXIF orientation and format handling
pytest
Automated testing
FastAPI TestClient
API testing
One small adjustment: do not introduce multiple embedding-cache formats unnecessarily. Store cached embeddings as binary data associated with an image hash and model version. Use a separate file cache only if benchmarking shows it is beneficial.
Also, make persistent caching optional because face embeddings are sensitive biometric data.
5. Model installation and offline execution — important adjustment
The proposal says model files will be downloaded by setup_models.py and stored in a locally ignored models/ directory.
I approve this approach, with one strict condition:
Model downloading must happen during preparation, never during application startup or inference.
The process should be:
Download the permitted model files while internet access is available.
Verify their checksums and licenses.
Store them locally.
Start RollCall with the network disconnected.
If a model is missing, show a clear setup error instead of attempting a download.
For ONNX Runtime, CoreML support is available in qualifying macOS builds, but its presence must be checked at runtime and its performance benchmarked against CPU execution. Do not make CoreML mandatory.
onnxruntime+1
6. Final recommended technology stack
RollCall — proposed stack
FACE DETECTION & RECOGNITION
OpenCV YuNet + SFace
Modular interface; alternative models evaluated only when justified.
BACKEND
Python 3.11 + FastAPI + Uvicorn
FRONTEND
HTML5 + CSS3 + JavaScript + Canvas
STORAGE
SQLite + NumPy
IMAGE PROCESSING
OpenCV + Pillow
TESTING
pytest + FastAPI TestClient
EXECUTION
Native macOS, Apple Silicon, fully offline
````

## Checkpoint 2 — Data Partitioning & Licensing Rules (`Step 18 • 2026-11-25T11:58:00Z`)

````text
My recommendation is to approve Option C, but modify the data strategy before locking it in.

The hybrid approach is the strongest because RollCall needs to demonstrate two different things: recognizing the same person across different photographs, and detecting people in realistic group scenes. However, I see some risks in the proposed use of LFW, AI-generated identities, and the holdout split.

I would not approve the strategy exactly as currently written.
1. The biggest changes I recommend
A. Do not use LFW as the main threshold-calibration dataset
LFW is useful for face verification research and contains multiple images of some identities, along with same-person/different-person pair protocols. However, it consists primarily of individual face photographs, not realistic workplace group photos. 
Complex Adaptive Systems Laboratory+1


There is another concern: I could not establish a clear, explicit image-use license from the available LFW documentation. I would not describe it as automatically cleared for our project.

My decision: Keep LFW as an optional research benchmark, subject to verifying its terms. Do not make it a required data dependency or rely on it to calibrate RollCall's final operating thresholds.
B. Be careful with Pexels and Unsplash
A free stock-photo license does not automatically grant permission for biometric identification.
- Unsplash's terms specifically exclude using images for technologies intended to identify natural persons. 
  Unsplash+1
- Pexels also warns that additional permissions and rights relating to recognizable people may apply. 
  Pexels+1
My decision: Do not use either as a default source for face-recognition reference identities.
C. AI-generated photos should not calibrate recognition thresholds
This is particularly important.

Suppose an image generator creates a reference portrait called Priya Sharma.jpg, then creates a group photograph containing someone intended to be Priya.

The face in the group image may look similar, but it may not actually preser
<truncated 1721 bytes>
 validation, measure:
- Genuine same-person similarity.
- Different-person similarity.
- False acceptance of unknown people.
- False rejection of enrolled people.
- Missed face detections.
- Recognition performance on small and partially occluded faces.
Choose thresholds using the validation data, not the practice or holdout data.
3. Where should the real photographs come from?
My preferred option
 Consented volunteers

If possible, collect photographs from volunteers who explicitly agree to their photographs being used for local face-recognition testing.

For each volunteer, collect:
- One reference portrait.
- One or more additional reference photographs, if available.
- A photograph in a group setting, where practical.
Include unknown volunteers who do not appear in the reference directory.

Keep the original images, embeddings, and identity mapping private. Do not upload them to GitHub. Obtain informed permission for biometric processing, explain how the data will be used, and allow deletion.

If you cannot obtain enough volunteers, use generated data for development and clearly document the limitations. Do not substitute unverified stock photographs and call them legally cleared biometric data.
4. Final decision I recommend
Checkpoint 2 — Proposed decision
- Approve
   Option C, modified hybrid strategy.
- Use AI-generated images for development and edge-case testing.
- Use consented real photographs for genuine validation wherever available.
- Keep LFW as an optional, separately documented benchmark after checking its terms.
- Avoid Pexels and Unsplash as default biometric identity sources.
- Maintain three identity-disjoint data partitions.
- Build a full-size 50-person development test alongside the smaller practice dataset.
- Never claim accuracy or threshold reliability based only on generated identities. and also make each and every thing fith file and foldre name in professional manner
````

## Checkpoint 3 — Open-Set Matching & Safeguards (`Step 29 • 2026-11-25T11:59:43Z`)

````text
My recommendation is to approve all four options, with a few important safeguards. The overall design is strong for RollCall because it prioritizes avoiding false identifications and makes uncertain results explainable.
Checkpoint 3 — Recommended decisions
Area	Decision	My reasoning
1. Multi-reference	Approve B
 All-Exemplar Max-Match	Preserves different appearances and lets us show the exact reference photo that matched. The additional comparisons are negligible for 50 identities.
2. Open-set tiers	Approve B
 Confirmed / Likely / unknown	Gives users a review option instead of forcing uncertain matches into a name or discarding them immediately.
3. Hungarian assignment	Approve conditionally
 Optional, validation-gated	Keep independent matching as the default. Enable Hungarian only if testing demonstrates a measurable improvement without introducing false assignments.
4. Quality gate	Approve
 Separate failure categories	Separating missed detections, poor-quality faces, and genuine unknown people makes evaluation much more meaningful.

Four refinements I recommend before locking
1. Don't hardcode the 28–32 px face-size threshold.

Treat it as an initial experimental range. Test different face sizes on validation images and establish the minimum usable size empirically. A 30 px face under good lighting is not necessarily equivalent to one under severe blur.

2. Keep the three recognition states precise.
- Confirmed: Meets the high similarity threshold and runner-up margin.
- Likely: Has a plausible match but needs human review because of a borderline score or ambiguous runner-up.
- unknown: Below the minimum similarity threshold or rejected by the quality gate.
A Likely result must never silently become Confirmed. The user should be able to review the reference photo and manually correct the result.

3. Hungarian assignment must never upgrade a weak match.

Even after assignment:
- A low-quality face remains unknown.
- A Likely match remains Likely.
- A weak candidate must never be forced into an identity.
- Dummy unknown assignments must remain available.
4. Evaluate detection and recognition separately.

A face that YuNet never detects is different from a detected face that SFace cannot identify. Track these separately in the evaluation report.
Proposed locked architecture
YuNet face detection

Locate faces in the group photo


Face-quality gate

Separate unusable faces before matching


SFace embedding + all-exemplar matching

Compare against every enrolled reference


Threshold + runner-up margin

Apply validation-calibrated decision rules


Confirmed


Likely


unknown



Optional Hungarian assignment is evaluated separately and cannot override these decision rules.

One important checkpoint dependency: Checkpoint 2's practice-data strategy still needs your confirmation. We should keep its data partitions and licensing decisions consistent with the validation plan here.
````

## Checkpoint 4 — Concept B Studio Inspector UI/UX (`Step 36 • 2026-11-25T12:01:25Z`)

````text
I strongly recommend Concept B — Interactive Photo Workbench (Studio Inspector). It is the best fit for RollCall because the group photo remains the main focus, while identification results and explanations are available without switching screens.
I would approve Concept B with a few refinements to keep it practical and easy for non-technical office staff.
Checkpoint 4 — Recommended decisions
Recommended: Approve Concept B
1. Keep these features
Top bar: RollCall branding, offline status, enrolled count, Demo Mode, theme toggle and Wipe All Data.
Main photo canvas: Zoom, pan, bounding boxes, face selection and annotated PNG export.
Right panel: Synchronized identification list, filters and absent enrolled identities.
Explainability Inspector: Group-face crop, exact reference photo, similarity score, runner-up margin and quality information.
Enrollment drawer: Folder upload, progress and warnings.
Exports: Annotated PNG and attendance CSV.
2. Refinements I recommend
Feature
Recommendation
Likely matches
Label clearly as Needs Review; never count as confirmed attendance automatically.
Unknown faces
Provide Assign to Person, Mark Unknown and optional Enroll Person actions.
Manual corrections
Record corrections in the current session; don't automatically retrain the model.
Absent list
Clearly distinguish absent enrolled people from unknown faces.
Demo Mode
Use bundled practice data and the exact same recognition pipeline as normal operation.
Wipe All Data
Add a confirmation dialog explaining what will be permanently deleted.
Keyboard navigation
Support arrow keys and visible focus indicators; don't rely on color alone.
3. Suggested screen layout
RollCall
Offline
50 Enrolled
Group Photo
Zoom · Pan · Export PNG
Confirmed
Likely
unknown
Face results
Confirmed
Priya Sharma
Similarity: 0.68
Needs Review
Possible match
Review reference
unknown
Unknown person
Explainability Inspector
Detected face
Reference photo
Illustrative layout only — actual images, labels and scores will come from the local recognition pipeline.
One important implementation condition
The UI should remain usable even when enrollment has warnings or some faces cannot be identified. Don't block the entire workflow because one reference photo is invalid.
Also, keep all assets local: no external fonts, image URLs, CDN libraries or analytics in the finished application.
Confirm Checkpoint 4
````

## Phase 2 Alignment Brief & Differentiators (A–M) (`Step 192 • 2026-11-25T12:36:39Z`)

````text
Act as my senior computer-vision engineer, product engineer and AI
pair-programming partner. I am Saideep, a final-year B.Tech student. I am
building a take-home called "Task A: Who Is in the Photo?" and the app is named
RollCall. I am graded on how I steer you, so I make the decisions and you
advise. Never decide silently for me.

=== 1. SOURCE OF TRUTH: Task-A-Face-Recognition.pdf (read it fully) ===
Mandatory:
1. The user opens ONE folder from the UI: photos of 50 different people, each
   file named after the person (e.g. "Priya Sharma.jpg").
2. The user uploads ONE group photo of about 10 people.
3. Names are shown directly ON the photo. People not in the folder are marked
   exactly "unknown". Results are also shown as a list.
4. Easy to use and good looking (UI is scored).
5. Fully offline. No internet at runtime, no hosted services, no Gemini/OpenAI
   inside the solution. AI tools only for practice data.
6. No data is provided; I use public data or make my own.
7. At the demo the judges bring their own unseen folder of 50 and a real group
   photo where some people are not in the folder.
Required docs: README.md, AI_COLLAB.md (5-10 decisions, each with: AI suggested
/ I decided / why / evidence; 2-3 real pasted chat excerpts; full chat history
exported in the repo), RESEARCH.md (tools, projects, papers, data sources,
options compared, picked/rejected, with links), ARCHITECTURE.md (diagram made
with Archify, following
https://raw.githubusercontent.com/tt-a1i/archify/main/llms.txt, plus a
trade-offs section).
Never invent chat excerpts, results, numbers, sources or Git history.

=== 2. LOCKED STACK (change only with a strong reason and my approval) ===
- Python 3.11, FastAPI, Uvicorn on 127.0.0.1; inference off the async path;
  strict upload validation and safe paths.
- OpenCV YuNet (detection) + OpenCV SFace (recognition) behind a FaceEngine
  interface; NumPy; Pillow (EXIF orientation).
- Vanilla HTML/CSS/ES6 modules, Canvas/SVG overlays, <input webkitdirectory>
  plus a "paste folder pat
<truncated 6098 bytes>
t limitations honestly.

=== 7. RELIABILITY AND SECURITY ===
Handle zero-face, multi-face, corrupt, duplicate and unsupported files, EXIF
rotation, very large images (downscale for 8 GB), path traversal, interrupted
enrollment, partial failures and temp-file cleanup. Store the minimum biometric
data needed.

=== 8. HOW WE WORK ===
- Small steps, small meaningful Git commits; after each step give me a commit
  message. Never generate the whole project at once.
- Explain decisions in simple words with risks and trade-offs. Prefer a working
  feature over an impressive untested one. Do not claim anything works until it
  runs. Do not overwrite my work without checking.
- After each approved decision, draft an AI_COLLAB.md entry. I will paste the
  real excerpts myself.
Checkpoints (stop and wait for my decision at each):
 0. Restate the problem, hidden requirements and failure modes; list your
    assumptions.
 1. Confirm the stack and project structure.
 2. Data plan: sources, licences, the four dataset splits.
 3. Matching logic: quality gate, thresholds, calibration, margin.
 4. UI: 2-3 layout options, then my choice.
 5. Evaluation results: real numbers, what to tune.
 6. Pre-demo: cold start on a fresh clone with Wi-Fi off, an unseen folder, a
    demo script, a fallback plan, and 10 likely interview questions with honest,
    simple answers.
If a checkpoint is pending, say so. Never assume approval.

=== 9. YOUR FIRST RESPONSE (no code, no installs) ===
1. Confirm you understood the assignment and summarise the mandatory
   requirements.
2. Summarise the locked architecture.
3. Separate mandatory features from the RollCall differentiators (A-M).
4. List unresolved checkpoints and the biggest technical and product risks.
5. Explain how the app will be evaluated on unseen data.
Then wait for my approval. and look dont ask approval for all and look make thhat is sensefulla dn dont harm to mac make it
````

## Incremental 9-Step Build Plan & Verification Rules (`Step 194 • 2026-11-25T12:37:30Z`)

````text
Approved: upgrade the existing app incrementally. Do not rewrite anything that
already works. Build in this order, one step at a time, and give me a commit
message after each step:

1. Paste-folder-path fallback, plus the persistent embedding cache switched to
   opt-in (ephemeral by default).
2. Strict Content-Security-Policy header and the startup self-check.
3. (A) Folder-calibrated threshold with the "About this run" drawer, floored at
   0.363, with the one-photo-per-person limitation stated in the drawer.
4. (B) Lookalike guard.
5. (G) Full Enrollment Health Studio summary counts.
6. (H) Keyboard Review Queue + (I) visible decision trail with the reason field.
7. (F) Second Look tiled / manual-box re-scan.
8. (E) Mini coverage map.
9. (K) Print-ready attendance sheet.

Before step 3, show me the calibration approach (which percentile or target
false-accept rate, and how it behaves on a folder with only 10 people) so I can
decide. Do the same before step 7.

Checkpoint 5 (data): [FILL IN, see below]

Rules still apply: no claims without running tests, no invented numbers, show
actual test output after each step. Also: re-list which entries in
AI_COLLAB.md were my decisions versus your suggestions, so I can verify them.
````

## Step 3 — Folder Calibration (99.5th Percentile, 0.50 Cap, Honest Limits) (`Step 271 • 2026-11-25T12:46:34Z`)

````text
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

Show me the real output of the tests and what the drawer text looks like
before you commit. Then continue to Step 4 (Lookalike guard), and pause
before showing me its rule.
````

## Step 4 — Lookalike Guard & Duplicate Identity Separation (`Step 315 • 2026-11-25T12:50:22Z`)

````text
Step 3 approved with one wording fix: in the "About this run" drawer, when N < 15
show "Percentile: not applied (N < 15)" and label the raw value "Raw 99.5th
percentile (not applied)". Commit with your suggested message after that fix.

Before Step 4, tell me which two practice identities produce the 0.5789
pair and what those images are (lookalikes, same person, or an accident in my
practice data).

Step 4: approve Option A with these changes:
1. State the margins explicitly: base margin X, lookalike margin Y (show the real
   numbers from the code).
2. Pairs with similarity >= 0.55 are shown in the Enrollment Health Studio as
   "possible duplicate identity", separate from lookalikes.
3. Name the flagged pairs in the Enrollment Summary and in "About this run".
4. Tests: probe equally close to two lookalikes lands in Likely; a normal match
   is unaffected; the guard never upgrades Likely to Confirmed or overrides the
   quality gate.
Show me real test output and the rendered text before committing. Then pause
before Step 5 and show me what the Enrollment Health summary will look like.
````

## Step 5 — Enrollment Health Studio & Multi-Face Selector (`Step 357 • 2026-11-25T12:58:10Z`)

````text
Step 4 approved. Add one test before committing: a probe equally close to two
identities in a >= 0.55 "possible duplicate" pair must land in Likely (the guard
must apply to duplicate-class pairs too, never be looser). Keep the
Rahul Verma / Vikram Patel pair in the practice set as a documented synthetic
lookalike case. Then commit with your suggested message.

Step 5: approve the layout with these changes:
1. Do not silently auto-pick a face in a multi-face reference photo. Show the
   chosen face boxed on the thumbnail, let me switch faces, and keep the file in
   "Needs attention" until chosen.
2. A second photo of the same person (priya_sharma.png) is an info item
   ("extra photo merged into Priya Sharma"), not a warning. But check the
   similarity against that identity's other photos; if it is low, flag
   "may be a different person".
3. Ignore system files (.DS_Store, Thumbs.db, __MACOSX, hidden files) without
   counting them as failed; show "ignored (N)".
4. Define the counts so they add up, e.g. "50 selected = X enrolled, Y enrolled
   with warnings, Z failed, W ignored".
5. Show a small thumbnail per file.
Add tests for: multi-face file stays in Needs attention until chosen, same-person
extra photo merges, different-person extra photo is flagged, system files are
ignored, and counts add up. Show real test output and the rendered summary
before committing. Then pause before Step 6 and show me the keyboard shortcuts
and the decision-trail fields you plan to use.
````

## Step 6 — Non-Mutating Keyboard Review Queue & Append-Only Trail (`Step 408 • 2026-11-25T13:07:52Z`)

````text
Step 5 approved, with two additions before committing:
1. In Demo Mode, show a visible note when multi-face reference photos are
   auto-resolved ("Demo: dominant face auto-selected for N photos").
2. Before group-photo upload, show a non-blocking banner when any reference
   photos are still "Needs attention" ("N people are not enrolled until you pick
   their face") with a "Continue without them" button; list them as "Not
   enrolled" in the results, not as unknown.
Then commit with your message.

Step 6: approve the shortcuts and trail with these changes:
1. Remove enroll_new for now. Corrections must never touch reference embeddings.
2. "C" produces status "Confirmed by reviewer", shown separately from automatic
   Confirmed in the list, attendance and exports (original state kept visible).
   "C" does nothing on an unknown face; naming an unknown face needs "A".
3. Add Undo (Z) for the last action.
4. Check the canvas key bindings first; if arrows already pan or zoom, keep only
   J/K for next/previous.
5. "A" opens a searchable type-to-filter box, not a plain dropdown.
6. Trail fields: add session_id, group-photo hash, face box, and actor
   (automated or reviewer). The trail is append-only: a face corrected twice
   keeps both rows.
7. Attendance recomputes: confirming a Likely moves that person from Absent to
   Present (reviewer-confirmed); rejecting leaves them Absent. Wipe All Data also
   deletes the trail.
Tests: queue visits only Likely and unknown faces, shortcuts are ignored while
typing in an input, undo works, the trail is append-only, attendance recomputes,
and wipe deletes the trail. Show real test output and a short description of the
queue UI before committing. Then pause before Step 7 and show me the Second
Look design (tile size and overlap, how duplicate boxes from overlapping tiles
are merged, and how the manual box works).
````

## Step 7 — Second Look Tiling & Manual-Box Specification (`Step 490 • 2026-11-25T13:15:04Z`)

````text
Step 6 approved. Commit with your message.

Step 7: approve the design with these changes:
1. Tell me the first-pass YuNet input size and whether tiles are upscaled before
   detection. The quality gate must use face size in ORIGINAL image pixels, not
   the upscaled size.
2. Existing face indices never change. New faces from Second Look and manual box
   get new indices at the end of the list. Already-reviewed faces stay untouched.
3. Dedup: IoU > 0.40, or center-inside plus a similar-size (area ratio) check,
   not center-inside alone. Also dedupe tile detections against each other.
4. Running Second Look twice must not add duplicate faces.
5. Faces found with a detector score below the normal threshold (including
   manual box at the 0.35 floor) can be at best Likely, never Confirmed, and the
   evidence card says "low detector score".
6. Log Second Look and manual box in the decision trail (actor, parameters,
   faces added). A manual box the user removes is marked removed, not deleted.
7. Fix the wide-image tiling description (3 tiles at 60% overlap by 40%) or
   change the width to match 20% overlap, and tell me which.
Tests: a face at a tile seam is found once; indices stay stable; second run adds
nothing; manual box with no face shows the message; the quality gate still
rejects a tiny face found by tiling; low-score faces cap at Likely. Report real
Second Look time and peak memory on my M2. Show real test output and the
canvas toolbar behaviour before committing. Then pause before Step 8 and show
me the mini coverage map design.
````

## Step 8 — Mini Coverage Map & Verification Requirements (`Step 567 • 2026-11-25T13:21:34Z`)

````text
Step 7 approved. Commit with your message, after answering two things:
1. Run Second Look on a harder image where the first pass misses small
   back-row faces, and report real numbers (faces found by first pass, faces
   added by Second Look, time). If Second Look adds nothing even there, say so
   plainly; I will record that honestly in EVAL.md.
2. Confirm the "normal" detector score threshold in the first pass (the 0.65 in
   your 0.35 <= score < 0.65 rule) and where it is defined in the code.
Show the full pytest output next time, not a shortened one.

Step 8: approve the Mini Coverage Map with these changes:
1. Preserve the photo's aspect ratio inside the map (no stretching).
2. Do not rely on colour alone: use a different marker shape per state, or the
   face number.
3. Show a permanent one-line caption ("Shows viewport and detected faces; does
   not prove every face was found"), not hover-only.
4. Clicking a marker selects that face and the map has an aria-label.
5. Do not show removed manual boxes; collapse the map by default on narrow
   windows.
Keep it small: no new dependencies. Show real test output and a description of
the final UI before committing. Then pause before Step 9 (print-ready
attendance sheet) and show me what the sheet will contain and how Likely,
Confirmed-by-reviewer and Absent are shown.
````

## Step 9 — Print Sheet & Documentation Sequence (a–e) (`Step 603 • 2026-11-25T13:24:54Z`)

````text
Step 8 approved. Commit with your message.

I do not accept "YuNet catches faces down to ~10-15 px" as a finding: both
scenes were synthetic. For EVAL.md the only valid claim is "on two synthetic
scenes, tiling added 0 faces". Also: run one real 4000x3000 phone photo (resized
to 2400 px) through the first pass and tell me how detection behaves. Tell me
whether you tested the coverage map in a real browser or only by reading the
source.

Step 9: approve the print sheet with these changes:
1. Escape every name, filename and reviewer note; add a test with
   "<script>alert(1)</script>.jpg" as a name.
2. No face images on the sheet by default (names, scores and face numbers
   only); optional thumbnails off by default.
3. Demo Mode sheets carry a visible "DEMO DATA" banner.
4. Page 1 is the summary (Present, Needs Review, Absent, unknown, Not
   enrolled); the decision trail is an appendix; no table row splits across
   pages.
5. Browser print / Save as PDF only; no new dependency.
6. The sheet reflects the current state after corrections and undo, and returns
   nothing after Wipe All Data.
Show the full pytest output and the final sheet layout before committing.

After Step 9, no more features. Next I want, in this order: (a) a list of the
exact docs still missing and a plan for each, (b) the README (install, model
setup, run, demo, offline check, troubleshooting) tested from a fresh clone,
(c) the EVAL.md structure you propose, with an honest account of what data I
have, (d) the Archify diagram, (e) RESEARCH.md outline with the sources you
actually checked. Pause after (a) and show me the plan.
````

## Documentation Step (b) — README Fresh-Clone & Verification Rules (`Step 660 • 2026-11-25T13:32:06Z`)

````text
Step 9 approved. Commit with your message.

Plan (a) approved. Proceed with (b) README.md, with these requirements:
1. Say clearly that setup_models.py needs internet once, BEFORE runtime, and
   that the app then runs fully offline. Include the SHA-256 check and the
   error shown if models are missing.
2. Test it from a fresh clone in a new folder and a new virtualenv, and show me
   the real commands and output. Then repeat the run with Wi-Fi off and say
   exactly what I verified.
3. Include: prerequisites (Python 3.11, macOS), install, model setup, run,
   folder selection and paste-path fallback, Demo Mode, review shortcuts,
   exports and print sheet, Wipe All Data, offline verification, and
   troubleshooting. No performance claims that are not measured.
4. State the limits honestly: synthetic practice data only, no real accuracy
   claims.
Then pause and show me the README before moving on to EVAL.md.

For AI_COLLAB.md: when you draft Decisions 5-10, use only decisions I made in
this conversation, quoting my actual replies. Mark anything you are not sure I
decided, so I can check it. Add a section listing every AI chat I need to export,
so the history starts from my very first prompt.
````

## README Refinement, T_high Coupling, Split Dependencies & 50-Person Smoke Test (`Step 725 • 2026-11-25T13:38:24Z`)

````text
README approved with these fixes before committing:
1. Say the quality-gate values (28 px, blur 35.0) are initial experimental
   values, not validated.
2. Check how T_high is computed when T_low is calibrated up to 0.50. It must
   never be below T_low (e.g. T_high = max(T_low + 0.06, 0.44)). Show me the
   code and add a test with T_low = 0.50. Fix the README to match.
3. State which browser was tested; remove claims about Safari/Firefox that were
   not tested.
4. Test "python -m rollcall.app" exactly as written in the README. Leave the
   repo URL as a marked placeholder.
5. Explain why opentelemetry-api is installed. Split pytest/httpx into
   requirements-dev.txt and keep runtime requirements minimal and pinned.
6. Do not state model licences from memory in the README until RESEARCH.md
   links the official sources; I will verify them.

I will do the physical Wi-Fi-off check myself and tell you the result.

Before EVAL.md: build a synthetic 50-person smoke folder (names like "First
Last.jpg", including a few duplicate-person extra photos, a no-face file, a
multi-face file, system files, and messy names) and a ~10-person group photo.
Use it ONLY for timing, memory and behaviour, never accuracy. Report real
numbers on my M2: enrollment time for 50 photos (cold and cached), peak RAM,
the calibration drawer output with N=50 (raw value, final T_low, cap hit or
not), the Enrollment Health summary, and any lookalike or duplicate pairs it
finds. Show me the results before writing EVAL.md.
````

## Extra-Photo T_high Guard (0.49 Holdout), Hi-Res Timing & EVAL.md Rules (`Step 782 • 2026-11-25T13:43:22Z`)

````text
README approved. Commit it with your message.

Smoke test: two behaviour problems and one measurement problem to fix before
EVAL.md.

1. Akash Verma_2.jpg (actually Harsha Reddy's photo, similarity 0.4908) was
   merged silently. Change the extra-photo rule: merge silently only if
   similarity >= T_high. Between 0.363 and T_high, show both thumbnails side by
   side with "may be a different person", and keep that photo OUT of matching
   until I keep or reject it. Add tests: genuine same-person at high similarity
   merges; the 0.49 case is held out of matching and flagged; rejecting it
   removes it; keeping it adds it as an exemplar.

2. Group photo: give me a per-face table for smoke_group_10.jpg: face number,
   assigned name or unknown, state, similarity, runner-up, margin, decision
   reason. Tell me whether the stranger face landed unknown, whether any
   Confirmed name was wrong (I know the ground truth since I built it), and
   exactly why the 3 faces went to Likely (lookalike guard, margin, detector
   score, or something else). Then tell me whether the synthetic folder is too
   degenerate to be useful for this, and say so plainly.

3. Re-run the 50-person timing with realistic reference sizes: make 50 JPEGs at
   about 3000x4000 px (EXIF-rotated for a few) and a 10-person group at about
   4000x3000. Report cold enrollment time, peak RAM, and group-photo time on my
   M2. Do not reuse the earlier small-image numbers.

For EVAL.md (after the above): label every number as synthetic; do not quote the
76 / 194 pair counts or the 0.8849 raw value as findings (they are artifacts of
blending 6 base images); include a "Not measured" section (real-world accuracy,
false-accept rate on real strangers, small real back-row faces, other browsers,
physical Wi-Fi-off check by me). No accuracy claims. Pause and show me the
EVAL.md draft before committing.
````

## Close-Up Pre-Scale Fix, Face #9 Runner-Up Bug, EVAL.md Edits & Archify (`Step 884 • 2026-11-25T13:55:32Z`)

````text
Item 1 (extra-photo T_high guard + keep/reject) approved. Commit it now on its
own, without EVAL.md.

Before ARCHITECTURE.md, fix the reference-portrait double-detection problem. It
is a demo risk, not an optional improvement:
1. Pre-scale reference portraits to about 1200 px on the longest side before
   detection (EXIF transpose first; group photos keep the 2400 px cap). The
   quality gate must still measure face size in original-image pixels.
2. Suppress detector artifacts before counting faces: drop a second box that is
   mostly inside or overlapping the main box, or whose score is below the normal
   0.65 threshold. Show me the overlap threshold and why. Genuine separate faces
   still keep the file in Needs attention until I choose (my earlier rule stands).
3. Apply the same overlap suppression to the first pass on group photos.
4. Tests: a large close-up portrait that used to double-detect now enrolls as
   single-face; a genuine two-person reference photo still stays in Needs
   attention; a group photo with a large close-up face gets no duplicate box.
5. Report how many of the 50 hi-res portraits went to Needs attention before and
   after the fix.

Then re-run the 3000x4000 benchmark 3 times and report min / median / max for
enrollment time and peak RAM.

Face #9 in the per-face table shows the runner-up with the same name and
similarity as the top candidate (0.2413) but margin 0.0839. Tell me whether this
is a display bug or a calculation bug, fix it, and add a test. For face #1, the
runner-up (Lakshmi Narayan) must be clearly visible in the list row and the
evidence card, since the top name was wrong.

EVAL.md edits (do not commit until I see it):
- Remove the stray "Valid statement only:" wording.
- State that the group faces were cut from the same synthetic images as the
  references, so similarities (0.83-0.96) are far above real same-person
  photos, and the 5/5 Confirmed result is a control-flow check, not evidence of
  recognition.
- State that on face #1 the top candidate was the wrong person and the lookalike
  guard held it at Likely (3 of the 4 Likely faces had the correct top name).
- Mark every timing as n=1 or give the 3-run range, one machine, synthetic images.
- Keep "Not measured" and add: real-photo double-detection behaviour.

Then continue to (d) ARCHITECTURE.md. Actually read and install Archify as the
brief requires (https://raw.githubusercontent.com/tt-a1i/archify/main/llms.txt).
If it cannot be installed in your sandbox, say so and tell me exactly what to
run myself; do not hand-draw a substitute and call it Archify. The diagram must
show the real modules in this repo. Write the trade-offs section only from
decisions I made in this chat. Pause before (e).
````


## PhotoProof Transformation: Product Rename & Evidence Cards (`Step 1000 • 2026-11-25T15:42:21Z`)

````text
Build an original, polished offline application for Take-home Task A called
"PhotoProof: Who Is in the Photo?"

Do not copy code, wording, architecture, UI, screenshots, README structure,
or feature names from any existing Roll Call / face-identification repository.
This must be a clean implementation with my own decisions documented.

PRODUCT PURPOSE
PhotoProof helps office staff annotate a single group photo using one folder
of named reference portraits. It is NOT an attendance clock, surveillance tool,
or automatic decision-maker. It should make a conservative suggestion, clearly
expose ambiguity, and let a human correct the final result.

[NON-NEGOTIABLE TASK REQUIREMENTS, CORE EXPERIENCE, TECHNICAL APPROACH,
QUALITY AND EVALUATION, and DOCUMENTATION sections as specified — all followed
exactly as stated in the original PDF and product brief.]

UI STYLE
Warm, calm, professional, and easy for non-technical office staff. Use large
buttons, clear empty states, keyboard-accessible review controls, and
high-contrast status chips. Avoid surveillance-themed graphics, dark "security
dashboard" styling, and attendance language.
````

## Continue PhotoProof Implementation: All Changes, Tests, Commit & Push (`Step 1010 • 2026-11-25T16:00:00Z`)

````text
Continue where you stopped. Make all the changes and verify them. Then commit
to GitHub. Follow the instructions clearly.

For the full chat history file, present it properly: if there is anything
informal or unclear in the wording, clean it up into professional English.
Make the commits and follow the assignment instructions precisely.
````

## Resume and Complete All Remaining Steps (`Step 1020 • 2026-11-25T21:41:57Z`)

````text
Continue where you stopped. Go through the folder carefully and pick up from
exactly where the task was left off — SVG export, ARCHITECTURE.md update,
chat history update, AI_COLLAB.md update, and final GitHub push.
````
