# Research & Technology Comparison (`RESEARCH.md`)

**Take-Home Task A: Who Is in the Photo?**

This document summarizes the tools, libraries, papers, open-set recognition strategies, and datasets investigated before finalizing PhotoProof's implementation, including what was selected, what was rejected, and why.

---

## 1. Face Detection & Recognition Backends

| Option Investigated | Architecture & Pre-Trained Source | License / Restrictions | Decision & Reasoning |
| --- | --- | --- | --- |
| **OpenCV Zoo YuNet + SFace** *(Chosen)* | YuNet (`face_detection_yunet_2023mar.onnx`) + SFace (`face_recognition_sface_2021dec.onnx`, 128-d embeddings via `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`) | OpenCV Zoo repository uses **Apache-2.0**, and the YuNet directory specifies **MIT** (`opencv/opencv_zoo`) | **Picked.** Tiny footprint (~39 MB combined), permissive weights, native CPU execution via `opencv-python-headless` with zero extra C++ build or `onnxruntime` friction on macOS Python 3.11. |
| **InsightFace (`SCRFD` + `ArcFace`)** | SCRFD detector (ICLR 2022) + ArcFace additive angular margin embeddings (CVPR 2019), e.g., `buffalo_l` packs | Code is **MIT**, **but pre-trained models & datasets are Non-Commercial Research Only** unless separately licensed | **Rejected for v1.** Pre-trained model restrictions and extra runtime complexity (`onnxruntime` / CoreML EP testing) violate clean permissive-weight goals; kept modular behind `FaceEngine` ABC for future swapping. |
| **RetinaFace** | Multi-task 2D/3D face localization + landmarks (Deng et al., CVPR 2020 / arXiv 2019) | Research/code varies by downstream distribution | **Rejected for v1.** Heavier inference footprint than YuNet on CPU-only 8 GB Apple Silicon laptop when baseline speed and simplicity matter first. |
| **`face_recognition` (`dlib` ResNet-34)** | HOG / CNN face detector + ResNet-34 metric learning (~99.38% LFW reported) | **MIT** (`ageitgey/face_recognition`) | **Rejected.** Compiling `dlib` on Apple Silicon adds C++ / CMake build friction and slower HOG/CPU group detection vs native OpenCV YuNet ONNX. |
| **Cloud Vision / LLM APIs** (OpenAI, Gemini Vision, AWS Rekognition) | Hosted HTTPS APIs | Proprietary SaaS / network required | **Explicitly Rejected.** Strictly prohibited at runtime by Task A ("No online APIs or hosted services"). |

### Verified References Checked
- OpenCV Official DNN Face Detection & Recognition Tutorial (`FaceDetectorYN` + `FaceRecognizerSF`, cosine similarity `0.363` / L2 distance `1.128` baselines): <https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html>
- `cv::FaceDetectorYN` Class Reference: <https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html>
- `cv::FaceRecognizerSF` Class Reference: <https://docs.opencv.org/4.x/da/d09/classcv_1_1FaceRecognizerSF.html>
- OpenCV Zoo Repository & Models (`face_detection_yunet`, `face_recognition_sface`): <https://github.com/opencv/opencv_zoo>
- YuNet Model Directory in OpenCV Zoo: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- Upstream `libfacedetection` (YuNet origin, Shiqi Yu): <https://github.com/ShiqiYu/libfacedetection>
- SFace Paper (*SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition*, Zhong et al., IEEE T-PAMI / arXiv): <https://arxiv.org/abs/2205.12010>
- InsightFace Project & License Policy: <https://github.com/deepinsight/insightface> and <https://insightface.ai>
- RetinaFace Paper (Deng et al., *Single-stage Dense Face Localisation in the Wild*): <https://arxiv.org/abs/1905.00641>
- `ageitgey/face_recognition` (`dlib` ResNet wrapper): <https://github.com/ageitgey/face_recognition>

---

## 2. Open-Set Matching, Calibration & Assignment Research

- **Published SFace LFW Baseline vs. Folder Calibration:** OpenCV documents cosine similarity `0.363` as a baseline SFace verification point on LFW (`tutorial_dnn_face.html`). Because verification benchmarks on centered celebrity portraits do not account for lookalikes inside a specific 50-person office folder, PhotoProof floors at `0.363` and computes the `99.5th` percentile across distinct identity pairs (when `N >= 15`, capped at `0.50`) — explicitly disclosing in UI that one-photo-per-person calibration estimates impostor-pair behavior only, not same-person variance.
- **Runner-Up Margin & Lookalike Ambiguity Guard:** High cosine score alone can still fail when two enrolled people look similar. Requiring a margin (`0.05` baseline; `0.10` when enrolled pair similarity `>= 0.42`) prevents forced wrong identifications by sending narrow-margin cases to human review (`Likely`).
- **Linear Sum Assignment (Hungarian Algorithm):** Reviewed global 1-to-1 matching (`scipy.optimize.linear_sum_assignment`: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>). **Decision:** Keep independent per-face matching as default; make Hungarian **opt-in**, augmented with explicit dummy `"unknown"` nodes so duplicate assignments resolve conservatively (`Confirmed`/`Likely` never upgraded, weak faces never forced to an identity).

---

## 3. Datasets & Privacy / Licensing Findings

| Dataset / Source | Characteristics | Suitability & Decision |
| --- | --- | --- |
| **LFW (*Labeled Faces in the Wild*)** (<http://vis-www.cs.umass.edu/lfw/>) | ~13k web-collected individual face images benchmarked for 1:1 face verification pairs | **Not used as calibration truth.** Consists of individual celebrity portraits rather than realistic workplace group scenes, and lacked clear biometric redistribution/calibration clearance for this project. |
| **DigiFace-1M** | Microsoft Research synthetic dataset (~1.2M computer-graphics renders, 110k identities) created to avoid non-consensual web scraping | **Researched as synthetic reference.** Relevant context for why synthetic identities help test mechanics ethically while remaining unfit for real-world camera accuracy claims. |
| **Stock Photos (Unsplash / Pexels)** | General permissive photo sites | **Rejected for biometric IDs.** Terms restrict or do not clear biometric identification of recognizable natural persons. |
| **Consented Local Volunteers (Uncommitted `.gitignore`)** | Real portraits + real group photos with non-enrolled unknowns | **Required for valid real-world calibration & evaluation** at live demo time; never uploaded to GitHub. |
| **Bundled Practice Set (`data/practice/`)** | 6 synthetic portraits (`1024×1024`) + 1 composited scene (`1400×820`) + Demo Mode | **Used strictly for offline smoke tests & UI demonstration.** Never cited for real-world recognition accuracy. |

---

## 4. Application Architecture & Browser UX Choices

- **FastAPI + Uvicorn (127.0.0.1) + ThreadPool vs. CLI / Hosted Cloud:** Chose a single-screen local web studio (`FastAPI` + Vanilla HTML/CSS/ES6 JS) because non-technical office staff use the platform cleanly without terminal usage or external CDN exposure (strict `Content-Security-Policy` enforced).
- **Pre-Scaling Reference Portraits (`1200 px`) vs. Group Photos (`2400 px`):** Observed empirically that YuNet (`2023mar`) splits large close-up faces when giant `3000×4000` portraits run at `1800×2400`; splitting reference pre-scaling (`1200 px` + artifact suppression + original-pixel size gate) from group inspection (`2400 px` + Second Look tiling) solved portrait double-detections without sacrificing group-scene resolution.
