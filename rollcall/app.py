"""FastAPI local server for RollCall (strictly bound to 127.0.0.1, 100% offline)."""

import asyncio
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
import threading
import uuid
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psutil

from rollcall.calibration import calibrate_from_exemplars
from rollcall.config import (
    DEFAULT_THRESHOLDS,
    PRACTICE_DATA_DIR,
    RecognitionThresholds,
    STATIC_DIR,
)
from rollcall.db import RollCallDB
from rollcall.engine import ModelMissingError, OpenCVYuNetSFaceEngine
from rollcall.enrollment import (
    build_health_counts,
    check_extra_photo_consistency,
    is_system_file,
    render_thumbnail_b64,
)
from rollcall.exporter import (
    build_attendance_csv,
    build_print_attendance_html,
    render_annotated_png,
)
from rollcall.filename_parser import parse_identity_filename
from rollcall.image_utils import (
    ImageLoadError,
    assess_face_quality,
    crop_face_with_margin,
    encode_jpeg_bytes,
    load_image_from_bytes,
)
from rollcall.matcher import (
    FaceMatchInput,
    FaceMatchResult,
    STATUS_CONFIRMED,
    STATUS_LIKELY,
    STATUS_UNKNOWN,
    match_group_faces,
)
from rollcall.second_look import run_manual_box_detection, run_tiled_second_look
from rollcall.self_check import run_startup_self_check


def _memory_rss_mb() -> float:
    proc = psutil.Process()
    return round(proc.memory_info().rss / (1024 * 1024), 1)


def _b64_jpeg(bgr_crop) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(
        encode_jpeg_bytes(bgr_crop, quality=88)
    ).decode("ascii")


STATUS_CONFIRMED_REVIEWER = "Confirmed by reviewer"


def _serialize_face_result(r: FaceMatchResult, source: str = "first_pass") -> dict:
    return {
        "face_index": r.face_index,
        "bbox": list(r.bbox),
        "status": r.status,
        "identity_id": r.identity_id,
        "display_name": r.display_name,
        "similarity": r.similarity,
        "runner_up_name": r.runner_up_name,
        "runner_up_similarity": r.runner_up_similarity,
        "runner_up_margin": r.runner_up_margin,
        "matched_photo_id": r.matched_photo_id,
        "decision_reason": r.decision_reason,
        "original_status": r.status,
        "original_identity_id": r.identity_id,
        "original_display_name": r.display_name,
        "original_similarity": r.similarity,
        "original_matched_photo_id": r.matched_photo_id,
        "original_decision_reason": r.decision_reason,
        "quality": asdict(r.quality),
        "crop_b64": r.crop_b64,
        "manually_corrected": False,
        "source": source,
        "removed": False,
    }


class AppState:
    """Thread-safe container for engine, DB, background jobs, and bounded sessions."""

    def __init__(self, db: RollCallDB | None = None, engine=None) -> None:
        self.db = db or RollCallDB()
        self.engine = engine
        self.model_error: str | None = None
        if self.engine is None:
            try:
                self.engine = OpenCVYuNetSFaceEngine()
            except ModelMissingError as exc:
                self.model_error = str(exc)

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rollcall-cv")
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.pending_multiface: dict[str, dict] = {}
        self.sessions: OrderedDict[str, dict] = OrderedDict()
        self.last_calibration: dict | None = None
        self.last_enrollment_summary: dict | None = None

    def store_session(self, session_id: str, payload: dict) -> None:
        with self.lock:
            self.sessions[session_id] = payload
            self.sessions.move_to_end(session_id)
            while len(self.sessions) > 8:
                self.sessions.popitem(last=False)


def _require_engine(state: AppState):
    if state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=state.model_error
            or "Face recognition models are not loaded. Run `python scripts/setup_models.py` first.",
        )
    return state.engine


def _enroll_single_candidate(
    state: AppState,
    existing_embs_by_id: dict[str, list[np.ndarray]],
    identity_id: str,
    display_name: str,
    original_filename: str,
    image_sha256: str,
    model_version: str,
    embedding: np.ndarray,
    crop_jpeg: bytes,
    quality_dict: dict,
    thumb_b64: str,
    persist_cache: bool,
    warnings: list[dict],
    infos: list[dict],
) -> dict:
    """Persist one reference photo, check same-person consistency, and build its health record."""
    prev_embs = existing_embs_by_id.get(identity_id, [])
    is_extra = len(prev_embs) > 0
    same_ok, same_sim = check_extra_photo_consistency(prev_embs, embedding)

    state.db.add_reference_photo(
        identity_id=identity_id,
        display_name=display_name,
        original_filename=original_filename,
        image_sha256=image_sha256,
        model_version=model_version,
        embedding=embedding,
        crop_jpeg=crop_jpeg,
        quality_dict=quality_dict,
        persist_cache=persist_cache,
    )
    existing_embs_by_id.setdefault(identity_id, []).append(embedding)

    warning_reasons: list[str] = []
    if not quality_dict.get("passed", True):
        warning_reasons.append(
            "Quality warning: " + "; ".join(quality_dict.get("reasons", []))
        )
    if is_extra and not same_ok:
        warning_reasons.append(
            f"May be a different person (similarity {same_sim:.4f} < 0.363 against existing {display_name})"
        )

    if warning_reasons:
        reason_text = "; ".join(warning_reasons)
        warnings.append(
            {
                "filename": original_filename,
                "display_name": display_name,
                "reason": reason_text,
            }
        )
        return {
            "filename": original_filename,
            "display_name": display_name,
            "identity_id": identity_id,
            "status": "enrolled_with_warnings",
            "reason": reason_text,
            "thumb_b64": thumb_b64,
        }

    if is_extra:
        info_text = f"Extra photo merged into {display_name} (similarity {same_sim:.4f})"
        infos.append(
            {
                "filename": original_filename,
                "display_name": display_name,
                "reason": info_text,
            }
        )
    else:
        w_px = quality_dict.get("width_px", 0)
        h_px = quality_dict.get("height_px", 0)
        info_text = f"Enrolled ({w_px}x{h_px} px)"

    return {
        "filename": original_filename,
        "display_name": display_name,
        "identity_id": identity_id,
        "status": "enrolled",
        "reason": info_text,
        "thumb_b64": thumb_b64,
    }


def _enroll_file_list(
    state: AppState,
    file_items: list[tuple[str, bytes]],
    clear_existing: bool,
    persist_cache: bool,
    job_id: str | None = None,
) -> dict:
    """Synchronous worker to enroll a folder of reference images."""
    engine = _require_engine(state)
    thresholds = DEFAULT_THRESHOLDS

    if clear_existing:
        state.db.clear_enrolled_roster()
        with state.lock:
            state.pending_multiface.clear()
        existing_embs_by_id: dict[str, list[np.ndarray]] = {}
    else:
        existing_embs_by_id = {}
        for ex in state.db.load_all_exemplars(engine.model_version):
            existing_embs_by_id.setdefault(ex["identity_id"], []).append(ex["embedding"])

    cache_hits = 0
    warnings: list[dict] = []
    infos: list[dict] = []
    errors: list[dict] = []
    multiface_items: list[dict] = []
    file_records: list[dict] = []

    total = len(file_items)
    for idx, (filename, raw_bytes) in enumerate(file_items, start=1):
        if job_id:
            with state.lock:
                state.jobs[job_id]["processed"] = idx - 1
                state.jobs[job_id]["current_file"] = Path(filename).name

        if is_system_file(filename):
            file_records.append(
                {
                    "filename": Path(filename).name or filename,
                    "display_name": None,
                    "identity_id": None,
                    "status": "ignored",
                    "reason": "System or hidden file ignored.",
                    "thumb_b64": None,
                }
            )
            continue

        parsed = parse_identity_filename(filename)

        try:
            loaded = load_image_from_bytes(raw_bytes, filename=filename)
        except ImageLoadError as exc:
            errors.append({"filename": parsed.original_filename, "reason": str(exc)})
            file_records.append(
                {
                    "filename": parsed.original_filename,
                    "display_name": parsed.display_name,
                    "identity_id": parsed.identity_key,
                    "status": "failed",
                    "reason": str(exc),
                    "thumb_b64": None,
                }
            )
            continue

        # Check embedding cache first if persistent caching is enabled
        cached = (
            state.db.get_cached_single_face(loaded.sha256, engine.model_version)
            if persist_cache
            else None
        )
        if cached is not None:
            cache_hits += 1
            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(cached["crop_jpeg"]).decode("ascii")
            rec = _enroll_single_candidate(
                state=state,
                existing_embs_by_id=existing_embs_by_id,
                identity_id=parsed.identity_key,
                display_name=parsed.display_name,
                original_filename=parsed.original_filename,
                image_sha256=loaded.sha256,
                model_version=engine.model_version,
                embedding=cached["embedding"],
                crop_jpeg=cached["crop_jpeg"],
                quality_dict=cached["quality"],
                thumb_b64=crop_b64,
                persist_cache=persist_cache,
                warnings=warnings,
                infos=infos,
            )
            file_records.append(rec)
            continue

        faces = engine.detect_faces(loaded.bgr)
        if not faces:
            reason = "No face detected in reference photo."
            errors.append({"filename": parsed.original_filename, "reason": reason})
            file_records.append(
                {
                    "filename": parsed.original_filename,
                    "display_name": parsed.display_name,
                    "identity_id": parsed.identity_key,
                    "status": "failed",
                    "reason": reason,
                    "thumb_b64": render_thumbnail_b64(loaded.bgr),
                }
            )
            continue

        assessed = [
            (
                face,
                assess_face_quality(
                    loaded.bgr,
                    face.bbox,
                    face.score,
                    thresholds.min_face_size_px,
                    thresholds.min_blur_laplacian,
                    thresholds.min_detector_score,
                ),
            )
            for face in faces
        ]

        if len(faces) > 1:
            # Never silently auto-pick: keep in Needs Attention until user confirms a face
            token = uuid.uuid4().hex
            all_boxes = [f.bbox for f, _ in assessed]
            # Suggest largest quality-passing face as initial preview highlight only
            best_idx = max(
                range(len(assessed)),
                key=lambda i: (assessed[i][1].passed, assessed[i][0].bbox[2] * assessed[i][0].bbox[3]),
            )
            candidates_ui = []
            candidates_internal = []
            for f_idx, (face, q) in enumerate(assessed):
                crop = crop_face_with_margin(loaded.bgr, face.bbox)
                crop_jpg = encode_jpeg_bytes(crop, quality=88)
                emb = engine.extract_embedding(loaded.bgr, face)
                boxed_thumb = render_thumbnail_b64(loaded.bgr, boxes=all_boxes, active_idx=f_idx)
                candidates_ui.append(
                    {
                        "candidate_index": f_idx,
                        "bbox": list(face.bbox),
                        "crop_b64": "data:image/jpeg;base64,"
                        + base64.b64encode(crop_jpg).decode("ascii"),
                        "annotated_thumb_b64": boxed_thumb,
                        "quality_passed": q.passed,
                        "quality_reasons": q.reasons,
                    }
                )
                candidates_internal.append(
                    {
                        "embedding": emb,
                        "crop_jpeg": crop_jpg,
                        "quality": asdict(q),
                    }
                )

            with state.lock:
                state.pending_multiface[token] = {
                    "identity_id": parsed.identity_key,
                    "display_name": parsed.display_name,
                    "original_filename": parsed.original_filename,
                    "image_sha256": loaded.sha256,
                    "model_version": engine.model_version,
                    "persist_cache": persist_cache,
                    "candidates": candidates_internal,
                }
            default_thumb = candidates_ui[best_idx]["annotated_thumb_b64"]
            multiface_items.append(
                {
                    "token": token,
                    "filename": parsed.original_filename,
                    "display_name": parsed.display_name,
                    "face_count": len(faces),
                    "suggested_index": best_idx,
                    "candidates": candidates_ui,
                }
            )
            file_records.append(
                {
                    "filename": parsed.original_filename,
                    "display_name": parsed.display_name,
                    "identity_id": parsed.identity_key,
                    "status": "needs_attention",
                    "reason": f"{len(faces)} faces detected — select target face to enroll.",
                    "thumb_b64": default_thumb,
                    "token": token,
                }
            )
            continue

        face, quality = assessed[0]
        emb = engine.extract_embedding(loaded.bgr, face)
        crop = crop_face_with_margin(loaded.bgr, face.bbox)
        crop_jpg = encode_jpeg_bytes(crop, quality=88)
        thumb_b64 = render_thumbnail_b64(crop)
        rec = _enroll_single_candidate(
            state=state,
            existing_embs_by_id=existing_embs_by_id,
            identity_id=parsed.identity_key,
            display_name=parsed.display_name,
            original_filename=parsed.original_filename,
            image_sha256=loaded.sha256,
            model_version=engine.model_version,
            embedding=emb,
            crop_jpeg=crop_jpg,
            quality_dict=asdict(quality),
            thumb_b64=thumb_b64,
            persist_cache=persist_cache,
            warnings=warnings,
            infos=infos,
        )
        file_records.append(rec)

    identities = state.db.list_identities()
    exemplars = state.db.load_all_exemplars(engine.model_version)
    calibration = calibrate_from_exemplars(exemplars).to_dict()
    counts = build_health_counts(file_records)

    summary = {
        "total_files": total,
        "enrolled_photos": counts["enrolled"] + counts["enrolled_with_warnings"],
        "unique_identities": len(identities),
        "cache_hits": cache_hits,
        "counts": counts,
        "file_records": file_records,
        "infos": infos,
        "warnings": warnings,
        "errors": errors,
        "multiface_pending": multiface_items,
        "calibration": calibration,
        "memory_rss_mb": _memory_rss_mb(),
    }

    with state.lock:
        state.last_calibration = calibration
        state.last_enrollment_summary = summary

    if job_id:
        with state.lock:
            state.jobs[job_id]["processed"] = total
            state.jobs[job_id]["status"] = "completed"
            state.jobs[job_id]["summary"] = summary

    return summary


def _identify_group_image(
    state: AppState,
    filename: str,
    raw_bytes: bytes,
    thresholds: RecognitionThresholds,
    is_demo: bool = False,
) -> dict:
    """Synchronous worker to run detection, quality gating, and matching on a group photo."""
    import time

    t0 = time.perf_counter()
    engine = _require_engine(state)
    loaded = load_image_from_bytes(raw_bytes, filename=filename)
    exemplars = state.db.load_all_exemplars(engine.model_version)
    identities = state.db.list_identities()
    calibration = calibrate_from_exemplars(exemplars, base=thresholds)
    effective_thresholds = calibration.to_thresholds(
        base=thresholds, enable_hungarian=thresholds.enable_hungarian
    )
    with state.lock:
        state.last_calibration = calibration.to_dict()

    detected = engine.detect_faces(loaded.bgr)
    match_inputs: list[FaceMatchInput] = []

    for f_idx, face in enumerate(detected):
        quality = assess_face_quality(
            loaded.bgr,
            face.bbox,
            face.score,
            effective_thresholds.min_face_size_px,
            effective_thresholds.min_blur_laplacian,
            effective_thresholds.min_detector_score,
        )
        crop = crop_face_with_margin(loaded.bgr, face.bbox)
        crop_b64 = _b64_jpeg(crop)
        emb = engine.extract_embedding(loaded.bgr, face) if quality.passed else None
        match_inputs.append(
            FaceMatchInput(
                face_index=f_idx,
                bbox=face.bbox,
                quality=quality,
                embedding=emb,
                crop_b64=crop_b64,
            )
        )

    results = match_group_faces(
        match_inputs,
        exemplars,
        effective_thresholds,
        lookalike_pairs=calibration.flagged_pair_set(),
    )
    serialized_faces = [_serialize_face_result(r) for r in results]
    embeddings_by_face = {
        inp.face_index: inp.embedding for inp in match_inputs
    }
    crop_bytes_by_face = {
        inp.face_index: encode_jpeg_bytes(
            crop_face_with_margin(loaded.bgr, inp.bbox), quality=88
        )
        for inp in match_inputs
    }

    session_id = uuid.uuid4().hex
    session_obj = {
        "session_id": session_id,
        "filename": Path(filename).name,
        "group_sha256": loaded.sha256,
        "bgr": loaded.bgr,
        "width": loaded.width,
        "height": loaded.height,
        "faces": serialized_faces,
        "embeddings": embeddings_by_face,
        "crop_bytes": crop_bytes_by_face,
        "model_version": engine.model_version,
        "calibration": calibration.to_dict(),
        "thresholds": effective_thresholds,
        "lookalike_pairs": calibration.flagged_pair_set(),
        "is_demo": is_demo,
        "undo_stack": [],
    }
    state.store_session(session_id, session_obj)

    for face in serialized_faces:
        state.db.append_trail_entry(
            session_id=session_id,
            group_image_sha256=loaded.sha256,
            face_index=face["face_index"],
            face_bbox=face["bbox"],
            actor="automated",
            action="initial_detection",
            previous_status=None,
            previous_identity_id=None,
            previous_display_name=None,
            new_status=face["status"],
            new_identity_id=face["identity_id"],
            new_display_name=face["display_name"],
            reason_note=face["decision_reason"],
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    response_payload = _build_session_response(
        state=state,
        session_obj=session_obj,
        identities=identities,
        image_b64=_b64_jpeg(loaded.bgr),
        processing_time_ms=elapsed_ms,
    )
    return response_payload


def _build_session_response(
    state: AppState,
    session_obj: dict,
    identities: list[dict] | None = None,
    image_b64: str | None = None,
    processing_time_ms: float | None = None,
) -> dict:
    if identities is None:
        identities = state.db.list_identities()

    faces = session_obj["faces"]
    active_faces = [f for f in faces if not f.get("removed")]
    confirmed_auto_ids = {
        f["identity_id"]
        for f in active_faces
        if f["status"] == STATUS_CONFIRMED and f["identity_id"]
    }
    confirmed_reviewer_ids = {
        f["identity_id"]
        for f in active_faces
        if f["status"] == STATUS_CONFIRMED_REVIEWER and f["identity_id"]
    }
    present_ids = confirmed_auto_ids | confirmed_reviewer_ids
    likely_ids = {
        f["identity_id"]
        for f in active_faces
        if f["status"] == STATUS_LIKELY and f["identity_id"]
    }

    enrolled_ids = {p["identity_id"] for p in identities}
    with state.lock:
        not_enrolled_map: dict[str, dict] = {}
        for pend in state.pending_multiface.values():
            iid = pend["identity_id"]
            if iid not in enrolled_ids and iid not in not_enrolled_map:
                not_enrolled_map[iid] = {
                    "identity_id": iid,
                    "display_name": pend["display_name"],
                    "original_filename": pend["original_filename"],
                    "status": "Not enrolled",
                }
    not_enrolled_identities = sorted(
        not_enrolled_map.values(), key=lambda x: x["display_name"].lower()
    )

    # Likely matches are NEVER counted as confirmed attendance automatically
    absent_identities = [
        {
            **person,
            "pending_review": person["identity_id"] in likely_ids,
        }
        for person in identities
        if person["identity_id"] not in present_ids
    ]

    summary = {
        "total_detected": len(active_faces),
        "confirmed": sum(1 for f in active_faces if f["status"] == STATUS_CONFIRMED),
        "confirmed_by_reviewer": sum(1 for f in active_faces if f["status"] == STATUS_CONFIRMED_REVIEWER),
        "present_total": len(present_ids),
        "likely": sum(1 for f in active_faces if f["status"] == STATUS_LIKELY),
        "unknown": sum(1 for f in active_faces if f["status"] == STATUS_UNKNOWN),
        "enrolled_total": len(identities),
        "absent_count": len(absent_identities),
        "not_enrolled_count": len(not_enrolled_identities),
    }

    payload = {
        "session_id": session_obj["session_id"],
        "filename": session_obj["filename"],
        "width": session_obj["width"],
        "height": session_obj["height"],
        "faces": faces,
        "summary": summary,
        "absent_identities": absent_identities,
        "not_enrolled_identities": not_enrolled_identities,
        "identities": identities,
        "calibration": session_obj.get("calibration") or state.last_calibration,
        "decision_trail": state.db.get_session_trail(session_obj["session_id"]),
        "can_undo": len(session_obj.get("undo_stack", [])) > 0,
        "memory_rss_mb": _memory_rss_mb(),
        "model_version": session_obj["model_version"],
    }
    if image_b64 is not None:
        payload["image_b64"] = image_b64
    if processing_time_ms is not None:
        payload["processing_time_ms"] = processing_time_ms
    return payload


class ResolveMultiFaceRequest(BaseModel):
    token: str
    candidate_index: int | None = None  # None skips this image


class EnrollPathRequest(BaseModel):
    folder_path: str
    clear_existing: bool = True
    persist_cache: bool = False


class CorrectionRequest(BaseModel):
    face_index: int
    action: str  # "confirm", "assign", "mark_unknown"
    identity_id: str | None = None
    reason_note: str | None = None


class ManualBoxRequest(BaseModel):
    x: int
    y: int
    w: int
    h: int


_FORBIDDEN_ROOTS = (
    Path("/System"),
    Path("/Library"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/etc"),
    Path("/private/etc"),
    Path("/var/db"),
    Path("/var/root"),
    Path("/private/var/db"),
    Path("/private/var/root"),
)


def _read_folder_images_safely(folder_path: str, max_files: int = 500) -> list[tuple[str, bytes]]:
    raw = (folder_path or "").strip().strip('"').strip("'")
    if not raw:
        raise HTTPException(status_code=400, detail="Folder path cannot be empty.")

    target = Path(raw).expanduser().resolve(strict=False)
    if str(target) in ("/", "/var", "/private", "/private/var") or any(
        target == root or root in target.parents for root in _FORBIDDEN_ROOTS
    ):
        raise HTTPException(status_code=400, detail="Access to system directories is not allowed.")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {target}")

    entries = sorted(target.iterdir())
    file_items: list[tuple[str, bytes]] = []
    for entry in entries:
        if len(file_items) >= max_files:
            break
        if is_system_file(entry.name):
            file_items.append((entry.name, b""))
            continue
        try:
            resolved_entry = entry.resolve(strict=True)
        except OSError:
            continue
        if not resolved_entry.is_relative_to(target) or not resolved_entry.is_file():
            continue
        file_items.append((entry.name, resolved_entry.read_bytes()))

    if not file_items:
        raise HTTPException(status_code=400, detail="No readable files found in that folder.")
    return file_items


def _apply_multiface_resolution(
    state: AppState,
    token: str,
    candidate_index: int | None,
) -> dict:
    """Resolve a pending multi-face reference photo and refresh health counts + calibration."""
    with state.lock:
        pending = state.pending_multiface.pop(token, None)
    if pending is None:
        raise HTTPException(status_code=404, detail="Multi-face selection token expired or invalid.")

    engine = _require_engine(state)
    summary = state.last_enrollment_summary
    new_record: dict | None = None
    outcome_status = "skipped"

    if candidate_index is None:
        new_record = {
            "filename": pending["original_filename"],
            "display_name": pending["display_name"],
            "identity_id": pending["identity_id"],
            "status": "ignored",
            "reason": "Skipped by user during multi-face review.",
            "thumb_b64": None,
        }
    else:
        candidates = pending["candidates"]
        if candidate_index < 0 or candidate_index >= len(candidates):
            with state.lock:
                state.pending_multiface[token] = pending
            raise HTTPException(status_code=400, detail="Invalid candidate face index.")

        chosen = candidates[candidate_index]
        existing_embs_by_id: dict[str, list[np.ndarray]] = {}
        for ex in state.db.load_all_exemplars(engine.model_version):
            existing_embs_by_id.setdefault(ex["identity_id"], []).append(ex["embedding"])

        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(chosen["crop_jpeg"]).decode("ascii")
        warnings_list: list[dict] = summary["warnings"] if summary else []
        infos_list: list[dict] = summary["infos"] if summary else []
        new_record = _enroll_single_candidate(
            state=state,
            existing_embs_by_id=existing_embs_by_id,
            identity_id=pending["identity_id"],
            display_name=pending["display_name"],
            original_filename=pending["original_filename"],
            image_sha256=pending["image_sha256"],
            model_version=pending["model_version"],
            embedding=chosen["embedding"],
            crop_jpeg=chosen["crop_jpeg"],
            quality_dict=chosen["quality"],
            thumb_b64=crop_b64,
            persist_cache=pending["persist_cache"],
            warnings=warnings_list,
            infos=infos_list,
        )
        outcome_status = new_record["status"]

    identities = state.db.list_identities()
    exemplars = state.db.load_all_exemplars(engine.model_version)
    calibration = calibrate_from_exemplars(exemplars).to_dict()

    if summary is not None:
        records = summary["file_records"]
        for i, rec in enumerate(records):
            if rec.get("token") == token:
                records[i] = new_record
                break
        summary["multiface_pending"] = [
            m for m in summary["multiface_pending"] if m["token"] != token
        ]
        counts = build_health_counts(records)
        summary["counts"] = counts
        summary["enrolled_photos"] = counts["enrolled"] + counts["enrolled_with_warnings"]
        summary["unique_identities"] = len(identities)
        summary["calibration"] = calibration
        summary["memory_rss_mb"] = _memory_rss_mb()

    with state.lock:
        state.last_calibration = calibration
        state.last_enrollment_summary = summary

    return {
        "status": outcome_status,
        "display_name": pending["display_name"],
        "identities": identities,
        "summary": summary,
    }


def _start_enroll_job(
    state: AppState,
    loop: asyncio.AbstractEventLoop,
    file_items: list[tuple[str, bytes]],
    clear_existing: bool,
    persist_cache: bool,
) -> dict:
    job_id = uuid.uuid4().hex
    with state.lock:
        state.jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "processed": 0,
            "total": len(file_items),
            "current_file": "",
            "summary": None,
            "error": None,
        }

    def _run():
        try:
            _enroll_file_list(
                state,
                file_items,
                clear_existing=clear_existing,
                persist_cache=persist_cache,
                job_id=job_id,
            )
        except Exception as exc:
            with state.lock:
                state.jobs[job_id]["status"] = "failed"
                state.jobs[job_id]["error"] = str(exc)

    loop.run_in_executor(state.executor, _run)
    return {"job_id": job_id, "total": len(file_items)}


_STRICT_CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


def create_app(db: RollCallDB | None = None, engine=None) -> FastAPI:
    state = AppState(db=db, engine=engine)
    self_check_report = run_startup_self_check(state.db, state.engine, state.model_error)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        state.executor.shutdown(wait=False)

    app = FastAPI(title="RollCall", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.rollcall = state

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _STRICT_CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/api/health")
    def health():
        identities = state.db.list_identities()
        return {
            "status": "ok" if (state.engine is not None and self_check_report["ok"]) else "degraded",
            "offline_mode": True,
            "bind_host": "127.0.0.1",
            "model_ready": state.engine is not None,
            "model_version": state.engine.model_version if state.engine else None,
            "model_error": state.model_error,
            "self_check": self_check_report,
            "enrolled_count": len(identities),
            "memory_rss_mb": _memory_rss_mb(),
            "thresholds": asdict(DEFAULT_THRESHOLDS),
        }

    @app.get("/api/identities")
    def list_identities():
        return {"identities": state.db.list_identities()}

    @app.get("/api/calibration")
    def get_calibration():
        if state.engine is None:
            return {"calibration": None}
        exemplars = state.db.load_all_exemplars(state.engine.model_version)
        cal = calibrate_from_exemplars(exemplars).to_dict()
        with state.lock:
            state.last_calibration = cal
        return {"calibration": cal}

    @app.get("/api/reference-crop/{photo_id}")
    def get_reference_crop(photo_id: int):
        crop = state.db.get_reference_crop(photo_id)
        if crop is None:
            raise HTTPException(status_code=404, detail="Reference crop not found.")
        return Response(content=crop, media_type="image/jpeg")

    @app.post("/api/enroll")
    async def enroll_folder(
        files: list[UploadFile] = File(...),
        clear_existing: bool = Form(True),
        persist_cache: bool = Form(False),
    ):
        _require_engine(state)
        if not files:
            raise HTTPException(status_code=400, detail="No files provided for enrollment.")

        file_items: list[tuple[str, bytes]] = []
        for f in files:
            data = await f.read()
            file_items.append((f.filename or "unnamed.jpg", data))

        return _start_enroll_job(
            state,
            asyncio.get_running_loop(),
            file_items,
            clear_existing=clear_existing,
            persist_cache=persist_cache,
        )

    @app.post("/api/enroll/from-path")
    async def enroll_from_path(req: EnrollPathRequest):
        _require_engine(state)
        file_items = _read_folder_images_safely(req.folder_path)
        return _start_enroll_job(
            state,
            asyncio.get_running_loop(),
            file_items,
            clear_existing=req.clear_existing,
            persist_cache=req.persist_cache,
        )

    @app.get("/api/enroll/{job_id}")
    def enroll_status(job_id: str):
        with state.lock:
            job = state.jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Enrollment job not found.")
            return dict(job)

    @app.post("/api/enroll/resolve-multiface")
    def resolve_multiface(req: ResolveMultiFaceRequest):
        return _apply_multiface_resolution(state, req.token, req.candidate_index)

    @app.post("/api/identify")
    async def identify_group_photo(
        file: UploadFile = File(...),
        enable_hungarian: bool = Form(False),
    ):
        _require_engine(state)
        raw_bytes = await file.read()
        thresholds = RecognitionThresholds(
            similarity_high=DEFAULT_THRESHOLDS.similarity_high,
            similarity_low=DEFAULT_THRESHOLDS.similarity_low,
            runner_up_margin=DEFAULT_THRESHOLDS.runner_up_margin,
            min_face_size_px=DEFAULT_THRESHOLDS.min_face_size_px,
            min_blur_laplacian=DEFAULT_THRESHOLDS.min_blur_laplacian,
            min_detector_score=DEFAULT_THRESHOLDS.min_detector_score,
            enable_hungarian=enable_hungarian,
        )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                state.executor,
                _identify_group_image,
                state,
                file.filename or "group.jpg",
                raw_bytes,
                thresholds,
            )
        except ImageLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    @app.post("/api/sessions/{session_id}/correct")
    def correct_face(session_id: str, req: CorrectionRequest):
        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Identification session not found.")

        faces = session_obj["faces"]
        face = next((f for f in faces if f["face_index"] == req.face_index), None)
        if face is None:
            raise HTTPException(status_code=404, detail="Face index not found in session.")

        prev_snapshot = {
            "face_index": face["face_index"],
            "status": face["status"],
            "identity_id": face["identity_id"],
            "display_name": face["display_name"],
            "matched_photo_id": face["matched_photo_id"],
            "decision_reason": face["decision_reason"],
            "manually_corrected": face["manually_corrected"],
        }
        prev_status = face["status"]
        prev_id = face["identity_id"]
        prev_name = face["display_name"]
        note = (req.reason_note or "").strip() or None

        if req.action == "confirm":
            if prev_status != STATUS_LIKELY or not face["identity_id"]:
                raise HTTPException(
                    status_code=400,
                    detail="Confirm (C) applies only to Likely matches; use Assign (A) for unknown faces.",
                )
            face["status"] = STATUS_CONFIRMED_REVIEWER
            face["decision_reason"] = (
                "Confirmed by reviewer after inspecting reference photo."
                + (f" Note: {note}" if note else "")
            )
            face["manually_corrected"] = True
        elif req.action == "assign":
            if not req.identity_id:
                raise HTTPException(status_code=400, detail="identity_id is required for assign action.")
            identities = {i["identity_id"]: i for i in state.db.list_identities()}
            target = identities.get(req.identity_id)
            if target is None:
                raise HTTPException(status_code=404, detail="Target identity not found.")
            exemplars = state.db.load_all_exemplars(session_obj["model_version"])
            ref_photo = next(
                (ex["photo_id"] for ex in exemplars if ex["identity_id"] == target["identity_id"]),
                None,
            )
            face["identity_id"] = target["identity_id"]
            face["display_name"] = target["display_name"]
            face["status"] = STATUS_CONFIRMED_REVIEWER
            face["matched_photo_id"] = ref_photo
            face["decision_reason"] = (
                f"Assigned to {target['display_name']} by reviewer."
                + (f" Note: {note}" if note else "")
            )
            face["manually_corrected"] = True
        elif req.action == "mark_unknown":
            face["identity_id"] = None
            face["display_name"] = STATUS_UNKNOWN
            face["status"] = STATUS_UNKNOWN
            face["matched_photo_id"] = None
            face["decision_reason"] = (
                "Marked as unknown by reviewer." + (f" Note: {note}" if note else "")
            )
            face["manually_corrected"] = True
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported correction action: {req.action}")

        session_obj.setdefault("undo_stack", []).append(prev_snapshot)

        state.db.record_correction(
            group_image_sha256=session_obj["group_sha256"],
            face_index=req.face_index,
            previous_identity_id=prev_id,
            new_identity_id=face["identity_id"],
            new_status=face["status"],
            action=req.action,
        )
        state.db.append_trail_entry(
            session_id=session_id,
            group_image_sha256=session_obj["group_sha256"],
            face_index=req.face_index,
            face_bbox=face["bbox"],
            actor="reviewer",
            action=req.action,
            previous_status=prev_status,
            previous_identity_id=prev_id,
            previous_display_name=prev_name,
            new_status=face["status"],
            new_identity_id=face["identity_id"],
            new_display_name=face["display_name"],
            reason_note=note,
        )

        return _build_session_response(state=state, session_obj=session_obj)

    @app.post("/api/sessions/{session_id}/undo")
    def undo_last_correction(session_id: str):
        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Identification session not found.")

        undo_stack = session_obj.get("undo_stack", [])
        if not undo_stack:
            raise HTTPException(status_code=400, detail="Nothing to undo in this session.")

        snap = undo_stack.pop()
        face = next(
            (f for f in session_obj["faces"] if f["face_index"] == snap["face_index"]),
            None,
        )
        if face is None:
            raise HTTPException(status_code=404, detail="Face index not found in session.")

        prev_status = face["status"]
        prev_id = face["identity_id"]
        prev_name = face["display_name"]

        face["status"] = snap["status"]
        face["identity_id"] = snap["identity_id"]
        face["display_name"] = snap["display_name"]
        face["matched_photo_id"] = snap["matched_photo_id"]
        face["decision_reason"] = snap["decision_reason"]
        face["manually_corrected"] = snap["manually_corrected"]

        state.db.append_trail_entry(
            session_id=session_id,
            group_image_sha256=session_obj["group_sha256"],
            face_index=face["face_index"],
            face_bbox=face["bbox"],
            actor="reviewer",
            action="undo",
            previous_status=prev_status,
            previous_identity_id=prev_id,
            previous_display_name=prev_name,
            new_status=face["status"],
            new_identity_id=face["identity_id"],
            new_display_name=face["display_name"],
            reason_note="Undid previous reviewer action.",
        )

        return _build_session_response(state=state, session_obj=session_obj)

    def _append_detected_faces_to_session(
        session_obj: dict,
        detected_list: list,
        source: str,
    ) -> list[dict]:
        """Run original-pixel quality gate + SFace + folder matcher and append at end."""
        if not detected_list:
            return []
        engine = _require_engine(state)
        thresholds = session_obj["thresholds"]
        exemplars = state.db.load_all_exemplars(engine.model_version)
        lookalike_pairs = session_obj["lookalike_pairs"]
        bgr = session_obj["bgr"]

        start_idx = len(session_obj["faces"])
        match_inputs: list[FaceMatchInput] = []
        for offset, face in enumerate(detected_list):
            f_idx = start_idx + offset
            # Always evaluate size & blur in ORIGINAL loaded-image pixels
            quality = assess_face_quality(
                bgr,
                face.bbox,
                face.score,
                thresholds.min_face_size_px,
                thresholds.min_blur_laplacian,
                thresholds.min_detector_score,
            )
            crop = crop_face_with_margin(bgr, face.bbox)
            crop_jpg = encode_jpeg_bytes(crop, quality=88)
            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_jpg).decode("ascii")
            emb = engine.extract_embedding(bgr, face) if quality.passed else None
            session_obj["embeddings"][f_idx] = emb
            session_obj["crop_bytes"][f_idx] = crop_jpg
            match_inputs.append(
                FaceMatchInput(
                    face_index=f_idx,
                    bbox=face.bbox,
                    quality=quality,
                    embedding=emb,
                    crop_b64=crop_b64,
                )
            )

        results = match_group_faces(
            match_inputs,
            exemplars,
            thresholds,
            lookalike_pairs=lookalike_pairs,
        )
        added_serialized = [_serialize_face_result(r, source=source) for r in results]
        session_obj["faces"].extend(added_serialized)
        return added_serialized

    @app.post("/api/sessions/{session_id}/second-look")
    async def second_look_rescan(session_id: str):
        import time

        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Identification session not found.")

        engine = _require_engine(state)
        existing_boxes = [
            tuple(f["bbox"]) for f in session_obj["faces"] if not f.get("removed")
        ]

        def _work():
            t0 = time.perf_counter()
            new_faces, meta = run_tiled_second_look(
                session_obj["bgr"], engine, existing_boxes
            )
            added = _append_detected_faces_to_session(
                session_obj, new_faces, source="second_look"
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return added, meta, elapsed_ms

        loop = asyncio.get_running_loop()
        added, meta, elapsed_ms = await loop.run_in_executor(state.executor, _work)

        state.db.append_trail_entry(
            session_id=session_id,
            group_image_sha256=session_obj["group_sha256"],
            face_index=-1,
            face_bbox=[0, 0, session_obj["width"], session_obj["height"]],
            actor="reviewer",
            action="second_look",
            previous_status=None,
            previous_identity_id=None,
            previous_display_name=None,
            new_status="completed",
            new_identity_id=None,
            new_display_name=f"{len(added)} face(s) added",
            reason_note=(
                f"grid={meta['tile_grid']}, overlap={meta['overlap_ratio']}, "
                f"upscale={meta['tile_upscale_factor']}x, first_pass={meta['first_pass_input_size']}, "
                f"faces_added={len(added)}"
            ),
        )
        for item in added:
            state.db.append_trail_entry(
                session_id=session_id,
                group_image_sha256=session_obj["group_sha256"],
                face_index=item["face_index"],
                face_bbox=item["bbox"],
                actor="automated",
                action="second_look_detection",
                previous_status=None,
                previous_identity_id=None,
                previous_display_name=None,
                new_status=item["status"],
                new_identity_id=item["identity_id"],
                new_display_name=item["display_name"],
                reason_note=item["decision_reason"],
            )

        resp = _build_session_response(state=state, session_obj=session_obj)
        resp["second_look"] = {**meta, "elapsed_ms": elapsed_ms}
        return resp

    @app.post("/api/sessions/{session_id}/manual-box")
    async def manual_box_scan(session_id: str, req: ManualBoxRequest):
        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Identification session not found.")

        if req.w < 8 or req.h < 8:
            raise HTTPException(status_code=400, detail="Drawn box is too small.")

        engine = _require_engine(state)
        existing_boxes = [
            tuple(f["bbox"]) for f in session_obj["faces"] if not f.get("removed")
        ]
        user_box = (req.x, req.y, req.w, req.h)

        def _work():
            chosen, msg = run_manual_box_detection(
                session_obj["bgr"], engine, user_box, existing_boxes
            )
            if chosen is None:
                return [], msg
            added = _append_detected_faces_to_session(
                session_obj, [chosen], source="manual_box"
            )
            return added, msg

        loop = asyncio.get_running_loop()
        added, msg = await loop.run_in_executor(state.executor, _work)

        state.db.append_trail_entry(
            session_id=session_id,
            group_image_sha256=session_obj["group_sha256"],
            face_index=added[0]["face_index"] if added else -1,
            face_bbox=list(user_box),
            actor="reviewer",
            action="manual_box",
            previous_status=None,
            previous_identity_id=None,
            previous_display_name=None,
            new_status=added[0]["status"] if added else "no_face",
            new_identity_id=added[0]["identity_id"] if added else None,
            new_display_name=added[0]["display_name"] if added else msg,
            reason_note=f"Manual region {list(user_box)}, faces_added={len(added)}: {msg}",
        )

        resp = _build_session_response(state=state, session_obj=session_obj)
        resp["manual_box"] = {
            "found": len(added) > 0,
            "message": msg,
            "added_face_index": added[0]["face_index"] if added else None,
        }
        return resp

    @app.post("/api/sessions/{session_id}/faces/{face_index}/remove")
    def remove_manual_face(session_id: str, face_index: int):
        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Identification session not found.")

        face = next(
            (f for f in session_obj["faces"] if f["face_index"] == face_index),
            None,
        )
        if face is None:
            raise HTTPException(status_code=404, detail="Face index not found.")
        if face.get("source") != "manual_box":
            raise HTTPException(status_code=400, detail="Only manually drawn boxes can be removed.")
        if face.get("removed"):
            return _build_session_response(state=state, session_obj=session_obj)

        prev_status = face["status"]
        prev_id = face["identity_id"]
        prev_name = face["display_name"]

        face["removed"] = True
        face["status"] = "Removed"
        face["decision_reason"] = "Manual box marked removed by reviewer."

        state.db.append_trail_entry(
            session_id=session_id,
            group_image_sha256=session_obj["group_sha256"],
            face_index=face_index,
            face_bbox=face["bbox"],
            actor="reviewer",
            action="manual_box_removed",
            previous_status=prev_status,
            previous_identity_id=prev_id,
            previous_display_name=prev_name,
            new_status="Removed",
            new_identity_id=None,
            new_display_name="Removed",
            reason_note="Manual box marked removed (preserved in audit trail).",
        )
        return _build_session_response(state=state, session_obj=session_obj)

    @app.get("/api/sessions/{session_id}/export/png")
    def export_png(session_id: str):
        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        active = [f for f in session_obj["faces"] if not f.get("removed")]
        png_bytes = render_annotated_png(session_obj["bgr"], active)
        stem = Path(session_obj["filename"]).stem
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="rollcall_{stem}_annotated.png"'},
        )

    @app.get("/api/sessions/{session_id}/export/csv")
    def export_csv(session_id: str):
        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        resp = _build_session_response(state=state, session_obj=session_obj)
        active = [f for f in resp["faces"] if not f.get("removed")]
        csv_text = build_attendance_csv(active, resp["absent_identities"])
        stem = Path(session_obj["filename"]).stem
        return Response(
            content=csv_text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="rollcall_{stem}_attendance.csv"'},
        )

    @app.get("/api/sessions/{session_id}/export/print")
    def export_print_sheet(session_id: str, include_thumbs: bool = False):
        with state.lock:
            session_obj = state.sessions.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        resp = _build_session_response(state=state, session_obj=session_obj)
        html_doc = build_print_attendance_html(
            resp,
            is_demo=bool(session_obj.get("is_demo", False)),
            include_thumbs=include_thumbs,
        )
        return HTMLResponse(content=html_doc)

    @app.post("/api/demo")
    async def run_demo_mode():
        _require_engine(state)
        ref_dir = PRACTICE_DATA_DIR / "reference"
        group_dir = PRACTICE_DATA_DIR / "group"
        ref_files = sorted(
            p for p in ref_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ) if ref_dir.is_dir() else []
        group_files = sorted(
            p for p in group_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ) if group_dir.is_dir() else []

        if not ref_files or not group_files:
            return JSONResponse(
                status_code=404,
                content={
                    "detail": (
                        "Bundled practice data not found in data/practice/reference/ and "
                        "data/practice/group/. Add practice images first or upload your own folder."
                    )
                },
            )

        file_items = [(p.name, p.read_bytes()) for p in ref_files]
        group_path = group_files[0]
        group_bytes = group_path.read_bytes()

        loop = asyncio.get_running_loop()
        enroll_summary = await loop.run_in_executor(
            state.executor, _enroll_file_list, state, file_items, True, False, None
        )
        # In one-click Demo Mode only, resolve demo multi-face bokeh items with suggested foreground face
        auto_resolved = list(enroll_summary.get("multiface_pending", []))
        for pending in auto_resolved:
            res = _apply_multiface_resolution(state, pending["token"], pending["suggested_index"])
            if res.get("summary"):
                enroll_summary = res["summary"]
        demo_note = (
            f"Demo: dominant face auto-selected for {len(auto_resolved)} photo"
            + ("s" if len(auto_resolved) != 1 else "")
            if auto_resolved
            else None
        )
        if demo_note:
            enroll_summary["demo_note"] = demo_note
        identify_result = await loop.run_in_executor(
            state.executor,
            _identify_group_image,
            state,
            group_path.name,
            group_bytes,
            DEFAULT_THRESHOLDS,
            True,
        )
        identify_result["demo_enrollment"] = enroll_summary
        if demo_note:
            identify_result["demo_note"] = demo_note
        return identify_result

    @app.post("/api/wipe")
    def wipe_all_data():
        counts = state.db.wipe_all_data()
        with state.lock:
            state.sessions.clear()
            state.pending_multiface.clear()
            state.jobs.clear()
            state.last_calibration = None
            state.last_enrollment_summary = None
        return {"status": "wiped", "deleted": counts}

    if STATIC_DIR.is_dir():
        @app.get("/")
        def serve_index():
            return FileResponse(STATIC_DIR / "index.html")

        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rollcall.app:app", host="127.0.0.1", port=8000, reload=False)
