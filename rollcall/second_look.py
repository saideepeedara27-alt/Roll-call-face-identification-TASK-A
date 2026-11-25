"""Second Look tiled detection and manual-box region re-scan (Differentiator F)."""

import cv2
import numpy as np

from rollcall.engine.base import DetectedFace

TILE_OVERLAP_RATIO = 0.20
MANUAL_BOX_SCORE_FLOOR = 0.35


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


def _center_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    cx = inner[0] + inner[2] / 2.0
    cy = inner[1] + inner[3] / 2.0
    return (
        outer[0] <= cx <= outer[0] + outer[2]
        and outer[1] <= cy <= outer[1] + outer[3]
    )


def is_duplicate_box(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    iou_threshold: float = 0.40,
) -> bool:
    """Return True if boxes overlap significantly (IoU > 0.40) or share center with similar area."""
    if compute_iou(box_a, box_b) > iou_threshold:
        return True
    area_a = max(1, box_a[2] * box_a[3])
    area_b = max(1, box_b[2] * box_b[3])
    ratio = area_a / float(area_b)
    if 0.40 <= ratio <= 2.50:
        if _center_inside(box_a, box_b) or _center_inside(box_b, box_a):
            return True
    return False


def compute_tile_windows(
    width: int, height: int, overlap_ratio: float = TILE_OVERLAP_RATIO
) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Compute overlapping tile windows (2x2 for <1800px wide, 3x2 for >=1800px wide).

    Each tile size satisfies: tile_dim = axis_dim / (n - (n - 1) * overlap_ratio),
    giving exact 20% tile-dimension overlap between adjacent tiles:
      - 2x2: each tile is ~55.6% x 55.6% of the image
      - 3x2: each tile is ~38.5% x 55.6% of the image
    """
    cols = 3 if width >= 1800 else 2
    rows = 2
    tile_w = max(1, min(width, int(round(width / (cols - (cols - 1) * overlap_ratio)))))
    tile_h = max(1, min(height, int(round(height / (rows - (rows - 1) * overlap_ratio)))))

    stride_x = max(1, int(round(tile_w * (1.0 - overlap_ratio))))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap_ratio))))

    xs = [min(width - tile_w, c * stride_x) for c in range(cols)]
    ys = [min(height - tile_h, r * stride_y) for r in range(rows)]
    xs[-1] = max(0, width - tile_w)
    ys[-1] = max(0, height - tile_h)

    windows: list[tuple[int, int, int, int]] = []
    for ty in ys:
        for tx in xs:
            windows.append((tx, ty, tile_w, tile_h))
    return f"{cols}x{rows}", windows


def _map_detection_to_image(
    face: DetectedFace,
    offset_x: int,
    offset_y: int,
    scale: float,
    img_w: int,
    img_h: int,
) -> DetectedFace:
    """Map tile/crop detection bbox and 5-point landmarks back into original image pixels."""
    bx, by, bw, bh = face.bbox
    mx = bx / scale + offset_x
    my = by / scale + offset_y
    mw = bw / scale
    mh = bh / scale

    row = face.raw_row.copy()
    row[0], row[1], row[2], row[3] = mx, my, mw, mh
    for k in range(4, 14, 2):
        row[k] = row[k] / scale + offset_x
        row[k + 1] = row[k + 1] / scale + offset_y

    x = max(0, int(round(mx)))
    y = max(0, int(round(my)))
    x2 = min(img_w, int(round(mx + max(1.0, mw))))
    y2 = min(img_h, int(round(my + max(1.0, mh))))
    return DetectedFace(
        bbox=(x, y, max(1, x2 - x), max(1, y2 - y)),
        score=face.score,
        raw_row=row,
    )


def run_tiled_second_look(
    bgr_image: np.ndarray,
    engine,
    existing_boxes: list[tuple[int, int, int, int]],
    overlap_ratio: float = TILE_OVERLAP_RATIO,
) -> tuple[list[DetectedFace], dict]:
    """Run tiled YuNet re-scan, dedupe tile results against each other and existing faces."""
    img_h, img_w = bgr_image.shape[:2]
    grid_name, windows = compute_tile_windows(img_w, img_h, overlap_ratio=overlap_ratio)

    raw_tile_candidates: list[DetectedFace] = []
    max_scale_used = 1.0

    for tx, ty, tw, th in windows:
        tile = bgr_image[ty : ty + th, tx : tx + tw]
        if min(tw, th) < 640:
            scale = min(2.0, 1280.0 / float(max(tw, th, 1)))
            det_input = cv2.resize(
                tile,
                (max(1, int(round(tw * scale))), max(1, int(round(th * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
            max_scale_used = max(max_scale_used, scale)
        else:
            scale = 1.0
            det_input = tile

        tile_faces = engine.detect_faces(det_input)
        for tf in tile_faces:
            raw_tile_candidates.append(
                _map_detection_to_image(tf, tx, ty, scale, img_w, img_h)
            )

    # 1. Deduplicate tile detections against each other (highest confidence wins)
    raw_tile_candidates.sort(key=lambda f: f.score, reverse=True)
    deduped_tiles: list[DetectedFace] = []
    for cand in raw_tile_candidates:
        if any(is_duplicate_box(cand.bbox, kept.bbox) for kept in deduped_tiles):
            continue
        deduped_tiles.append(cand)

    # 2. Keep only genuinely new detections that don't duplicate existing session faces
    new_faces: list[DetectedFace] = []
    for cand in deduped_tiles:
        if any(is_duplicate_box(cand.bbox, ex_box) for ex_box in existing_boxes):
            continue
        new_faces.append(cand)

    new_faces.sort(key=lambda f: (f.bbox[0], f.bbox[1]))
    meta = {
        "first_pass_input_size": [img_w, img_h],
        "tile_grid": grid_name,
        "tile_count": len(windows),
        "overlap_ratio": overlap_ratio,
        "tile_upscale_factor": round(max_scale_used, 2),
        "raw_tile_detections": len(raw_tile_candidates),
        "deduped_tile_detections": len(deduped_tiles),
        "new_faces_found": len(new_faces),
    }
    return new_faces, meta


def run_manual_box_detection(
    bgr_image: np.ndarray,
    engine,
    user_box: tuple[int, int, int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    score_floor: float = MANUAL_BOX_SCORE_FLOOR,
) -> tuple[DetectedFace | None, str]:
    """Run YuNet at relaxed score_floor inside a padded user-drawn box."""
    img_h, img_w = bgr_image.shape[:2]
    ux, uy, uw, uh = user_box
    pad_x = max(12, int(round(uw * 0.35)))
    pad_y = max(12, int(round(uh * 0.35)))
    cx1 = max(0, ux - pad_x)
    cy1 = max(0, uy - pad_y)
    cx2 = min(img_w, ux + uw + pad_x)
    cy2 = min(img_h, uy + uh + pad_y)

    crop = bgr_image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, "Drawn box is outside image bounds."

    ch, cw = crop.shape[:2]
    if min(cw, ch) < 320:
        scale = min(2.0, 640.0 / float(max(cw, ch, 1)))
        det_input = cv2.resize(
            crop,
            (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        scale = 1.0
        det_input = crop

    found = engine.detect_faces(det_input, score_threshold=score_floor)
    if not found:
        return None, "No aligned face detected in the drawn box."

    mapped = [
        _map_detection_to_image(f, cx1, cy1, scale, img_w, img_h)
        for f in found
    ]
    # Pick candidate best matching the user's drawn box
    mapped.sort(
        key=lambda f: (
            _center_inside(f.bbox, user_box),
            compute_iou(f.bbox, user_box),
            f.score,
        ),
        reverse=True,
    )
    chosen = mapped[0]
    if any(is_duplicate_box(chosen.bbox, ex_box) for ex_box in existing_boxes):
        return None, "This face is already in the results list."

    return chosen, "Aligned face detected in drawn box."
