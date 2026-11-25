"""Startup self-check verifying offline models, checksums, local assets, and DB health."""

import hashlib
import json
from pathlib import Path

from rollcall.config import (
    MODELS_DIR,
    MODEL_VERSION_TAG,
    SFACE_MODEL_NAME,
    STATIC_DIR,
    YUNET_MODEL_NAME,
)
from rollcall.db import RollCallDB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_startup_self_check(
    db: RollCallDB,
    engine,
    model_error: str | None,
    models_dir: Path = MODELS_DIR,
    static_dir: Path = STATIC_DIR,
) -> dict:
    """Verify local models, checksums, zero-CDN static assets, and SQLite connectivity."""
    checks: dict[str, dict] = {}

    # 1. Model & checksum check (only verify disk ONNX files for the real YuNet+SFace engine)
    if engine is None:
        checks["models"] = {
            "ok": False,
            "detail": model_error or "Models not loaded. Run `python scripts/setup_models.py` offline prep.",
        }
    elif engine.model_version == MODEL_VERSION_TAG:
        manifest_path = models_dir / "manifest.json"
        expected_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        problems = []
        for fname in (YUNET_MODEL_NAME, SFACE_MODEL_NAME):
            fpath = models_dir / fname
            if not fpath.is_file() or fpath.stat().st_size == 0:
                problems.append(f"Missing or empty model: {fname}")
                continue
            expected_sha = expected_manifest.get(fname, {}).get("sha256")
            if expected_sha and _sha256(fpath) != expected_sha:
                problems.append(f"Checksum mismatch for {fname}")
        checks["models"] = {
            "ok": len(problems) == 0,
            "detail": "Local YuNet + SFace weights verified."
            if not problems
            else "; ".join(problems),
        }
    else:
        checks["models"] = {
            "ok": True,
            "detail": f"Active engine: {engine.model_version}",
        }

    # 2. Static asset & zero-external-URL check
    asset_problems = []
    for asset_name in ("index.html", "styles.css", "app.js"):
        apath = static_dir / asset_name
        if not apath.is_file():
            asset_problems.append(f"Missing static/{asset_name}")
            continue
        text = apath.read_text(encoding="utf-8")
        if "http://" in text or "https://" in text:
            asset_problems.append(f"External URL found in static/{asset_name}")
    checks["static_assets"] = {
        "ok": len(asset_problems) == 0,
        "detail": "All frontend assets bundled locally with zero external URLs."
        if not asset_problems
        else "; ".join(asset_problems),
    }

    # 3. SQLite connectivity check
    try:
        db.list_identities()
        checks["database"] = {"ok": True, "detail": "Local SQLite database ready."}
    except Exception as exc:
        checks["database"] = {"ok": False, "detail": f"SQLite error: {exc}"}

    all_ok = all(item["ok"] for item in checks.values())
    return {"ok": all_ok, "checks": checks}
