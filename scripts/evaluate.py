"""Reproducible offline evaluation harness for RollCall (Checkpoint 5).

Evaluates detection recall, open-set recognition (Confirmed / Likely / unknown),
false acceptance rate (FAR), false rejection rate (FRR), processing latency,
peak RSS memory on Apple Silicon, and optional Hungarian assignment ablation.

Expected dataset structure:
  <dataset_dir>/
    reference/          # Reference portraits named by person
    group/              # Group photographs
    ground_truth.json   # {"groups": {"photo.jpg": {"expected_total_faces": 10, "faces": [...]}}}
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollcall.config import DEFAULT_THRESHOLDS, RecognitionThresholds
from rollcall.engine import OpenCVYuNetSFaceEngine
from rollcall.filename_parser import parse_identity_filename
from rollcall.image_utils import assess_face_quality, load_image_from_bytes
from rollcall.matcher import (
    STATUS_CONFIRMED,
    STATUS_LIKELY,
    STATUS_UNKNOWN,
    FaceMatchInput,
    match_group_faces,
)


def _peak_rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def _iou(box_a: list[int] | tuple[int, int, int, int], box_b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


def run_evaluation(
    dataset_dir: Path,
    thresholds: RecognitionThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    ref_dir = dataset_dir / "reference"
    group_dir = dataset_dir / "group"
    gt_path = dataset_dir / "ground_truth.json"

    if not ref_dir.is_dir() or not group_dir.is_dir() or not gt_path.is_file():
        raise FileNotFoundError(
            f"Dataset at {dataset_dir} must contain reference/, group/, and ground_truth.json."
        )

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    engine = OpenCVYuNetSFaceEngine()

    # 1. Benchmark Enrollment
    t_enroll_start = time.perf_counter()
    exemplars = []
    enroll_failures = []
    for p in sorted(ref_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        parsed = parse_identity_filename(p.name)
        try:
            loaded = load_image_from_bytes(p.read_bytes(), p.name)
            faces = engine.detect_faces(loaded.bgr)
            if len(faces) != 1:
                enroll_failures.append({"file": p.name, "face_count": len(faces)})
                continue
            emb = engine.extract_embedding(loaded.bgr, faces[0])
            exemplars.append(
                {
                    "photo_id": len(exemplars) + 1,
                    "identity_id": parsed.identity_key,
                    "display_name": parsed.display_name,
                    "original_filename": parsed.original_filename,
                    "embedding": emb,
                }
            )
        except Exception as exc:
            enroll_failures.append({"file": p.name, "error": str(exc)})

    enroll_ms = round((time.perf_counter() - t_enroll_start) * 1000.0, 1)
    peak_rss = _peak_rss_mb()

    # 2. Evaluate Group Photos
    totals = {
        "expected_faces": 0,
        "detected_faces": 0,
        "missed_faces": 0,
        "quality_rejected_unknown": 0,
        "known_gt_matched": 0,
        "known_correct_confirmed": 0,
        "known_correct_likely": 0,
        "known_false_reject_unknown": 0,
        "known_misidentified_confirmed": 0,
        "unknown_gt_matched": 0,
        "unknown_correct_unknown": 0,
        "unknown_false_accept_confirmed": 0,
        "unknown_flagged_likely": 0,
    }
    group_latencies_ms = []
    genuine_sims = []
    impostor_sims = []

    for fname, spec in gt.get("groups", {}).items():
        gpath = group_dir / fname
        if not gpath.is_file():
            continue

        t0 = time.perf_counter()
        loaded = load_image_from_bytes(gpath.read_bytes(), gpath.name)
        detected = engine.detect_faces(loaded.bgr)

        inputs = []
        for idx, face in enumerate(detected):
            q = assess_face_quality(
                loaded.bgr,
                face.bbox,
                face.score,
                thresholds.min_face_size_px,
                thresholds.min_blur_laplacian,
                thresholds.min_detector_score,
            )
            emb = engine.extract_embedding(loaded.bgr, face) if q.passed else None
            inputs.append(FaceMatchInput(idx, face.bbox, q, emb, ""))

        results = match_group_faces(inputs, exemplars, thresholds)
        group_latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 1))
        peak_rss = max(peak_rss, _peak_rss_mb())

        expected_total = int(spec.get("expected_total_faces", len(spec.get("faces", []))))
        totals["expected_faces"] += expected_total
        totals["detected_faces"] += len(detected)
        totals["missed_faces"] += max(0, expected_total - len(detected))
        totals["quality_rejected_unknown"] += sum(1 for r in results if not r.quality.passed)

        # Match detected boxes to annotated ground-truth boxes via IoU >= 0.30
        gt_faces = spec.get("faces", [])
        used_gt = set()
        for res in results:
            best_gt_idx = None
            best_iou = 0.0
            for gi, gf in enumerate(gt_faces):
                if gi in used_gt or "bbox" not in gf:
                    continue
                iou = _iou(res.bbox, gf["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_gt_idx is None or best_iou < 0.30:
                continue
            used_gt.add(best_gt_idx)
            gt_key = gt_faces[best_gt_idx]["identity_key"]

            if gt_key == "unknown":
                totals["unknown_gt_matched"] += 1
                if res.similarity is not None:
                    impostor_sims.append(res.similarity)
                if res.status == STATUS_UNKNOWN:
                    totals["unknown_correct_unknown"] += 1
                elif res.status == STATUS_LIKELY:
                    totals["unknown_flagged_likely"] += 1
                elif res.status == STATUS_CONFIRMED:
                    totals["unknown_false_accept_confirmed"] += 1
            else:
                totals["known_gt_matched"] += 1
                if res.similarity is not None and res.identity_id == gt_key:
                    genuine_sims.append(res.similarity)
                if res.status == STATUS_UNKNOWN:
                    totals["known_false_reject_unknown"] += 1
                elif res.identity_id == gt_key:
                    if res.status == STATUS_CONFIRMED:
                        totals["known_correct_confirmed"] += 1
                    elif res.status == STATUS_LIKELY:
                        totals["known_correct_likely"] += 1
                else:
                    if res.status == STATUS_CONFIRMED:
                        totals["known_misidentified_confirmed"] += 1

    det_recall = (
        round(totals["detected_faces"] / totals["expected_faces"], 4)
        if totals["expected_faces"] > 0
        else None
    )
    far = (
        round(totals["unknown_false_accept_confirmed"] / totals["unknown_gt_matched"], 4)
        if totals["unknown_gt_matched"] > 0
        else None
    )
    frr = (
        round(totals["known_false_reject_unknown"] / totals["known_gt_matched"], 4)
        if totals["known_gt_matched"] > 0
        else None
    )

    return {
        "model_version": engine.model_version,
        "thresholds": asdict(thresholds),
        "enrolled_exemplars": len(exemplars),
        "enroll_failures": enroll_failures,
        "enroll_time_ms": enroll_ms,
        "avg_group_time_ms": round(float(np.mean(group_latencies_ms)), 1) if group_latencies_ms else None,
        "peak_rss_mb": peak_rss,
        "detection_recall": det_recall,
        "false_accept_rate_confirmed": far,
        "false_reject_rate_unknown": frr,
        "genuine_similarity_stats": {
            "count": len(genuine_sims),
            "mean": round(float(np.mean(genuine_sims)), 4) if genuine_sims else None,
            "min": round(float(np.min(genuine_sims)), 4) if genuine_sims else None,
        },
        "impostor_similarity_stats": {
            "count": len(impostor_sims),
            "mean": round(float(np.mean(impostor_sims)), 4) if impostor_sims else None,
            "max": round(float(np.max(impostor_sims)), 4) if impostor_sims else None,
        },
        "counts": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RollCall on a labeled dataset.")
    parser.add_argument("dataset_dir", type=Path, help="Path containing reference/, group/, ground_truth.json")
    parser.add_argument("--hungarian", action="store_true", help="Enable constrained Hungarian assignment")
    args = parser.parse_args()

    thresholds = RecognitionThresholds(
        similarity_high=DEFAULT_THRESHOLDS.similarity_high,
        similarity_low=DEFAULT_THRESHOLDS.similarity_low,
        runner_up_margin=DEFAULT_THRESHOLDS.runner_up_margin,
        min_face_size_px=DEFAULT_THRESHOLDS.min_face_size_px,
        min_blur_laplacian=DEFAULT_THRESHOLDS.min_blur_laplacian,
        min_detector_score=DEFAULT_THRESHOLDS.min_detector_score,
        enable_hungarian=args.hungarian,
    )
    report = run_evaluation(args.dataset_dir, thresholds)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
