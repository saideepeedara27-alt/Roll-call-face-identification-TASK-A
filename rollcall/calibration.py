"""Folder-level impostor-pair threshold calibration for RollCall (Differentiator A)."""

from dataclasses import asdict, dataclass
import numpy as np

from rollcall.config import DEFAULT_THRESHOLDS, RecognitionThresholds

SFACE_FLOOR = 0.363
T_LOW_CAP = 0.50
CALIBRATION_PERCENTILE = 99.5
MIN_IDENTITIES_FOR_CALIBRATION = 15

CALIBRATION_LIMITATIONS = [
    "One photo per person means only impostor (different-person) behaviour is calibrated, not same-person variation.",
    "The 99.5th percentile rate is per different-person pair in the folder, not per face in a group photo.",
    "Enrolled folder portraits are typically cleaner (frontal, evenly lit) than group-photo faces.",
]


@dataclass(frozen=True)
class FolderCalibrationResult:
    """Transparent summary of how T_low and T_high were derived for the active folder."""

    enrolled_identities: int
    pair_count: int
    percentile_used: str
    raw_value: float | None
    final_t_low: float
    final_t_high: float
    floor_value: float
    cap_value: float
    fallback_used: bool
    floor_bound: bool
    cap_hit: bool
    warning: str | None
    method_summary: str
    limitations: list[str]

    def to_thresholds(
        self,
        base: RecognitionThresholds = DEFAULT_THRESHOLDS,
        enable_hungarian: bool = False,
    ) -> RecognitionThresholds:
        return RecognitionThresholds(
            similarity_high=self.final_t_high,
            similarity_low=self.final_t_low,
            runner_up_margin=base.runner_up_margin,
            min_face_size_px=base.min_face_size_px,
            min_blur_laplacian=base.min_blur_laplacian,
            min_detector_score=base.min_detector_score,
            enable_hungarian=enable_hungarian,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def compute_distinct_identity_pair_similarities(exemplars: list[dict]) -> tuple[int, np.ndarray]:
    """Compute max cosine similarity for every distinct enrolled identity pair (i < j)."""
    by_id: dict[str, list[np.ndarray]] = {}
    for ex in exemplars:
        by_id.setdefault(ex["identity_id"], []).append(ex["embedding"])

    identity_ids = sorted(by_id.keys())
    n = len(identity_ids)
    if n < 2:
        return n, np.empty(0, dtype=np.float32)

    pair_sims = np.empty(n * (n - 1) // 2, dtype=np.float32)
    idx = 0
    for i in range(n):
        embs_i = np.stack(by_id[identity_ids[i]], axis=0)
        for j in range(i + 1, n):
            embs_j = np.stack(by_id[identity_ids[j]], axis=0)
            sim_mat = embs_i @ embs_j.T
            pair_sims[idx] = float(np.max(sim_mat))
            idx += 1

    return n, pair_sims


def calibrate_from_pair_similarities(
    n_identities: int,
    pair_sims: np.ndarray | list[float],
    base: RecognitionThresholds = DEFAULT_THRESHOLDS,
) -> FolderCalibrationResult:
    """Calibrate T_low from different-person pair similarities with floor, cap, and N < 15 fallback."""
    sims = np.asarray(pair_sims, dtype=np.float64)
    pair_count = int(sims.size)
    raw_val = round(float(np.percentile(sims, CALIBRATION_PERCENTILE)), 4) if pair_count > 0 else None

    if n_identities < MIN_IDENTITIES_FOR_CALIBRATION or pair_count == 0:
        final_t_low = SFACE_FLOOR
        final_t_high = round(max(final_t_low + 0.06, base.similarity_high, 0.44), 4)
        return FolderCalibrationResult(
            enrolled_identities=n_identities,
            pair_count=pair_count,
            percentile_used=f"not applied (N < {MIN_IDENTITIES_FOR_CALIBRATION})",
            raw_value=raw_val,
            final_t_low=final_t_low,
            final_t_high=final_t_high,
            floor_value=SFACE_FLOOR,
            cap_value=T_LOW_CAP,
            fallback_used=True,
            floor_bound=True,
            cap_hit=False,
            warning=f"Small folder (N={n_identities} < {MIN_IDENTITIES_FOR_CALIBRATION}): fell back to published SFace baseline {SFACE_FLOOR:.3f}.",
            method_summary=(
                f"Fell back to SFace baseline {SFACE_FLOOR:.3f} because N={n_identities} "
                f"(< {MIN_IDENTITIES_FOR_CALIBRATION} identities)."
            ),
            limitations=list(CALIBRATION_LIMITATIONS),
        )

    assert raw_val is not None
    floor_bound = raw_val <= SFACE_FLOOR
    cap_hit = raw_val > T_LOW_CAP
    final_t_low = round(min(T_LOW_CAP, max(SFACE_FLOOR, raw_val)), 4)
    final_t_high = round(max(final_t_low + 0.06, 0.44), 4)

    warning = None
    if cap_hit:
        warning = (
            f"Upward cap hit: 99.5th percentile ({raw_val:.4f}) exceeded {T_LOW_CAP:.2f}; "
            f"T_low was capped at {T_LOW_CAP:.2f} to prevent mass false rejections."
        )

    return FolderCalibrationResult(
        enrolled_identities=n_identities,
        pair_count=pair_count,
        percentile_used=f"{CALIBRATION_PERCENTILE}th",
        raw_value=raw_val,
        final_t_low=final_t_low,
        final_t_high=final_t_high,
        floor_value=SFACE_FLOOR,
        cap_value=T_LOW_CAP,
        fallback_used=False,
        floor_bound=floor_bound,
        cap_hit=cap_hit,
        warning=warning,
        method_summary=(
            f"Computed {CALIBRATION_PERCENTILE}th percentile across {pair_count} different-person pairs "
            f"(raw={raw_val:.4f}), clamped to [{SFACE_FLOOR:.3f}, {T_LOW_CAP:.2f}] -> T_low={final_t_low:.4f}."
        ),
        limitations=list(CALIBRATION_LIMITATIONS),
    )


def calibrate_from_exemplars(
    exemplars: list[dict],
    base: RecognitionThresholds = DEFAULT_THRESHOLDS,
) -> FolderCalibrationResult:
    """Run folder calibration directly on loaded reference exemplars."""
    n, pair_sims = compute_distinct_identity_pair_similarities(exemplars)
    return calibrate_from_pair_similarities(n, pair_sims, base=base)
