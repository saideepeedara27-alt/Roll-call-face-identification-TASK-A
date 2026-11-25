"""Open-set face matching with quality gating, runner-up margins, and optional Hungarian assignment."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from rollcall.config import DEFAULT_THRESHOLDS, RecognitionThresholds
from rollcall.image_utils import FaceQualityReport


STATUS_CONFIRMED = "High-confidence match"
STATUS_LIKELY = "Needs review"
STATUS_UNKNOWN = "Unknown"


@dataclass
class FaceMatchInput:
    """Single detected face in a group photo awaiting open-set identification."""

    face_index: int
    bbox: tuple[int, int, int, int]
    quality: FaceQualityReport
    embedding: np.ndarray | None
    crop_b64: str


@dataclass
class CandidateScore:
    """Best exemplar score for a single enrolled identity."""

    identity_id: str
    display_name: str
    photo_id: int
    original_filename: str
    similarity: float


@dataclass
class FaceMatchResult:
    """Explainable open-set identification result for one group-photo face."""

    face_index: int
    bbox: tuple[int, int, int, int]
    status: str  # "Confirmed", "Likely", or "unknown"
    identity_id: str | None
    display_name: str  # "unknown" when unrecognized
    similarity: float | None
    runner_up_name: str | None
    runner_up_similarity: float | None
    runner_up_margin: float | None
    matched_photo_id: int | None
    decision_reason: str
    quality: FaceQualityReport
    crop_b64: str


def _rank_identities(
    embedding: np.ndarray, exemplars: list[dict]
) -> list[CandidateScore]:
    """Score one face against all exemplars, taking max similarity per identity."""
    if not exemplars or embedding is None:
        return []

    emb_matrix = np.stack([ex["embedding"] for ex in exemplars], axis=0)
    sims = emb_matrix @ embedding

    best_by_identity: dict[str, CandidateScore] = {}
    for ex, sim in zip(exemplars, sims):
        sim_val = float(sim)
        iid = ex["identity_id"]
        prev = best_by_identity.get(iid)
        if prev is None or sim_val > prev.similarity:
            best_by_identity[iid] = CandidateScore(
                identity_id=iid,
                display_name=ex["display_name"],
                photo_id=ex["photo_id"],
                original_filename=ex["original_filename"],
                similarity=sim_val,
            )

    ranked = sorted(best_by_identity.values(), key=lambda c: c.similarity, reverse=True)
    return ranked


def _decide_single_face(
    face: FaceMatchInput,
    ranked: list[CandidateScore],
    thresholds: RecognitionThresholds,
    lookalike_pairs: set[frozenset[str]] | None = None,
) -> FaceMatchResult:
    """Apply quality gate, similarity thresholds, and runner-up margin independently."""
    if not face.quality.passed:
        return FaceMatchResult(
            face_index=face.face_index,
            bbox=face.bbox,
            status=STATUS_UNKNOWN,
            identity_id=None,
            display_name=STATUS_UNKNOWN,
            similarity=None,
            runner_up_name=None,
            runner_up_similarity=None,
            runner_up_margin=None,
            matched_photo_id=None,
            decision_reason="Quality gate rejected: " + "; ".join(face.quality.reasons),
            quality=face.quality,
            crop_b64=face.crop_b64,
        )

    if not ranked:
        return FaceMatchResult(
            face_index=face.face_index,
            bbox=face.bbox,
            status=STATUS_UNKNOWN,
            identity_id=None,
            display_name=STATUS_UNKNOWN,
            similarity=None,
            runner_up_name=None,
            runner_up_similarity=None,
            runner_up_margin=None,
            matched_photo_id=None,
            decision_reason="No enrolled identities available.",
            quality=face.quality,
            crop_b64=face.crop_b64,
        )

    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = (top1.similarity - top2.similarity) if top2 is not None else None

    if top1.similarity < thresholds.similarity_low:
        return FaceMatchResult(
            face_index=face.face_index,
            bbox=face.bbox,
            status=STATUS_UNKNOWN,
            identity_id=None,
            display_name=STATUS_UNKNOWN,
            similarity=round(top1.similarity, 4),
            runner_up_name=top2.display_name if top2 else None,
            runner_up_similarity=round(top2.similarity, 4) if top2 else None,
            runner_up_margin=round(margin, 4) if margin is not None else None,
            matched_photo_id=None,
            decision_reason=(
                "Marked Unknown because no reference was sufficiently similar "
                f"(best: {top1.similarity:.3f} < {thresholds.similarity_low:.3f})."
            ),
            quality=face.quality,
            crop_b64=face.crop_b64,
        )

    is_lookalike_pair = bool(
        lookalike_pairs
        and top2 is not None
        and frozenset((top1.identity_id, top2.identity_id)) in lookalike_pairs
    )
    required_margin = (
        thresholds.lookalike_margin
        if is_lookalike_pair
        else thresholds.runner_up_margin
    )
    margin_ok = margin is None or margin >= required_margin
    low_detector_score = face.quality.detector_score < thresholds.min_detector_score
    strong_profile = face.quality.pose_asymmetry >= 0.42
    if (
        top1.similarity >= thresholds.similarity_high
        and margin_ok
        and not low_detector_score
        and not strong_profile
    ):
        return FaceMatchResult(
            face_index=face.face_index,
            bbox=face.bbox,
            status=STATUS_CONFIRMED,
            identity_id=top1.identity_id,
            display_name=top1.display_name,
            similarity=round(top1.similarity, 4),
            runner_up_name=top2.display_name if top2 else None,
            runner_up_similarity=round(top2.similarity, 4) if top2 else None,
            runner_up_margin=round(margin, 4) if margin is not None else None,
            matched_photo_id=top1.photo_id,
            decision_reason=(
                f"High-confidence match ({top1.similarity:.3f} >= {thresholds.similarity_high:.3f})"
                + (
                    f" with margin {margin:.3f} >= {required_margin:.3f}."
                    if margin is not None
                    else "."
                )
            ),
            quality=face.quality,
            crop_b64=face.crop_b64,
        )

    # Otherwise: low detector score, profile angle, borderline similarity, or ambiguous runner-up -> Needs review
    if low_detector_score:
        reason = (
            f"Needs review because detector confidence was low ({face.quality.detector_score:.2f} < "
            f"{thresholds.min_detector_score:.2f}; possible occlusion or angle)."
        )
    elif strong_profile and top1.similarity >= thresholds.similarity_high and margin_ok:
        reason = (
            f"Needs review because face is turned toward profile (asymmetry {face.quality.pose_asymmetry:.2f}); "
            f"please confirm '{top1.display_name}'."
        )
    elif top1.similarity >= thresholds.similarity_high and not margin_ok:
        if is_lookalike_pair:
            reason = (
                f"Needs review because '{top1.display_name}' ({top1.similarity:.3f}) and "
                f"lookalike candidate '{top2.display_name}' ({top2.similarity:.3f}) were too close "
                f"(margin {margin:.3f} < {required_margin:.3f})."
            )
        else:
            reason = (
                f"Needs review because the top two candidates were too close: "
                f"'{top1.display_name}' ({top1.similarity:.3f}) vs '{top2.display_name}' ({top2.similarity:.3f}), "
                f"margin {margin:.3f} < {thresholds.runner_up_margin:.3f}."
            )
    else:
        reason = (
            f"Needs review because similarity ({top1.similarity:.3f}) is below high-confidence "
            f"threshold ({thresholds.similarity_high:.3f})."
        )

    return FaceMatchResult(
        face_index=face.face_index,
        bbox=face.bbox,
        status=STATUS_LIKELY,
        identity_id=top1.identity_id,
        display_name=top1.display_name,
        similarity=round(top1.similarity, 4),
        runner_up_name=top2.display_name if top2 else None,
        runner_up_similarity=round(top2.similarity, 4) if top2 else None,
        runner_up_margin=round(margin, 4) if margin is not None else None,
        matched_photo_id=top1.photo_id,
        decision_reason=reason,
        quality=face.quality,
        crop_b64=face.crop_b64,
    )


def match_group_faces(
    faces: list[FaceMatchInput],
    exemplars: list[dict],
    thresholds: RecognitionThresholds = DEFAULT_THRESHOLDS,
    lookalike_pairs: set[frozenset[str]] | None = None,
) -> list[FaceMatchResult]:
    """Identify detected group faces using independent thresholds and optional constrained Hungarian."""
    ranked_per_face: list[list[CandidateScore]] = [
        _rank_identities(f.embedding, exemplars) if (f.quality.passed and f.embedding is not None) else []
        for f in faces
    ]

    results = [
        _decide_single_face(face, ranked, thresholds, lookalike_pairs=lookalike_pairs)
        for face, ranked in zip(faces, ranked_per_face)
    ]

    if not thresholds.enable_hungarian or not exemplars:
        return _resolve_duplicate_identity_conflicts(results)

    # Constrained one-to-one assignment ONLY among faces that already passed quality & similarity_low
    eligible_indices = [
        i for i, r in enumerate(results) if r.status in (STATUS_CONFIRMED, STATUS_LIKELY)
    ]
    if len(eligible_indices) <= 1:
        return results

    identity_ids = sorted({ex["identity_id"] for ex in exemplars})
    id_to_col = {iid: idx for idx, iid in enumerate(identity_ids)}
    n_faces = len(eligible_indices)
    n_ids = len(identity_ids)

    # Add one dummy "unknown" column per eligible face so weak matches are never forced
    total_cols = n_ids + n_faces
    big_cost = 1e6
    cost = np.full((n_faces, total_cols), big_cost, dtype=np.float64)

    for row_idx, face_pos in enumerate(eligible_indices):
        for cand in ranked_per_face[face_pos]:
            if cand.similarity >= thresholds.similarity_low:
                col_idx = id_to_col[cand.identity_id]
                cost[row_idx, col_idx] = -cand.similarity
        # Dummy unknown option has slightly worse reward than similarity_low
        cost[row_idx, n_ids + row_idx] = -(thresholds.similarity_low - 0.001)

    row_ind, col_ind = linear_sum_assignment(cost)
    for row_idx, col_idx in zip(row_ind, col_ind):
        face_pos = eligible_indices[row_idx]
        orig = results[face_pos]
        if col_idx >= n_ids or cost[row_idx, col_idx] >= big_cost / 2:
            # Assigned to dummy unknown node -> downgrade to unknown
            results[face_pos] = FaceMatchResult(
                face_index=orig.face_index,
                bbox=orig.bbox,
                status=STATUS_UNKNOWN,
                identity_id=None,
                display_name=STATUS_UNKNOWN,
                similarity=orig.similarity,
                runner_up_name=orig.runner_up_name,
                runner_up_similarity=orig.runner_up_similarity,
                runner_up_margin=orig.runner_up_margin,
                matched_photo_id=None,
                decision_reason="One-to-one assignment resolved collision to unknown.",
                quality=orig.quality,
                crop_b64=orig.crop_b64,
            )
        else:
            chosen_id = identity_ids[col_idx]
            if chosen_id != orig.identity_id:
                # Re-score with chosen candidate; never upgrade a Likely/unknown match
                chosen_cand = next(
                    c for c in ranked_per_face[face_pos] if c.identity_id == chosen_id
                )
                new_status = (
                    STATUS_LIKELY
                    if orig.status == STATUS_LIKELY
                    or chosen_cand.similarity < thresholds.similarity_high
                    else STATUS_CONFIRMED
                )
                results[face_pos] = FaceMatchResult(
                    face_index=orig.face_index,
                    bbox=orig.bbox,
                    status=new_status,
                    identity_id=chosen_cand.identity_id,
                    display_name=chosen_cand.display_name,
                    similarity=round(chosen_cand.similarity, 4),
                    runner_up_name=orig.display_name,
                    runner_up_similarity=orig.similarity,
                    runner_up_margin=None,
                    matched_photo_id=chosen_cand.photo_id,
                    decision_reason="Reassigned via constrained one-to-one collision resolution.",
                    quality=orig.quality,
                    crop_b64=orig.crop_b64,
                )

    return results


def _resolve_duplicate_identity_conflicts(results: list[FaceMatchResult]) -> list[FaceMatchResult]:
    """Keep only the highest-scoring face per identity and demote duplicates to Needs review."""
    by_identity: dict[str, list[int]] = {}
    for pos, res in enumerate(results):
        if res.identity_id is not None and res.status in (STATUS_CONFIRMED, STATUS_LIKELY):
            by_identity.setdefault(res.identity_id, []).append(pos)

    updated = list(results)
    for positions in by_identity.values():
        if len(positions) < 2:
            continue
        best_pos = max(
            positions,
            key=lambda p: (updated[p].similarity or 0.0, -updated[p].face_index),
        )
        best = updated[best_pos]
        best_sim = best.similarity or 0.0
        for pos in positions:
            if pos == best_pos:
                continue
            orig = updated[pos]
            orig_sim = orig.similarity or 0.0
            conflict_note = (
                f"Needs review because Face {best.face_index + 1} also proposed "
                f"'{best.display_name}' with higher similarity ({best_sim:.3f} vs {orig_sim:.3f}); "
                "confirm or choose another person."
            )
            reason = (
                f"{orig.decision_reason} (Duplicate conflict: Face {best.face_index + 1} also proposed '{best.display_name}'.)"
                if orig.status == STATUS_LIKELY
                else conflict_note
            )
            updated[pos] = FaceMatchResult(
                face_index=orig.face_index,
                bbox=orig.bbox,
                status=STATUS_LIKELY,
                identity_id=orig.identity_id,
                display_name=orig.display_name,
                similarity=orig.similarity,
                runner_up_name=orig.runner_up_name,
                runner_up_similarity=orig.runner_up_similarity,
                runner_up_margin=orig.runner_up_margin,
                matched_photo_id=orig.matched_photo_id,
                decision_reason=reason,
                quality=orig.quality,
                crop_b64=orig.crop_b64,
            )
    return updated
