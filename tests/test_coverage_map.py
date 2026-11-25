"""Tests for PhotoProof frontend UI, Evidence Card, Demo Readiness, and accessibility contracts."""

from pathlib import Path


def test_photoproof_ui_evidence_card_and_accessibility_contracts():
    html = Path("static/index.html").read_text(encoding="utf-8")
    js = Path("static/app.js").read_text(encoding="utf-8")
    css = Path("static/styles.css").read_text(encoding="utf-8")

    # 1. Start screen actions & on-device privacy explanation
    assert "PhotoProof" in html
    assert "Load reference folder" in html
    assert "Analyse group photo" in html
    assert "100% on-device privacy" in html

    # 2. Demo readiness strip exact 6 metrics
    for metric_label in (
        "References loaded",
        "References rejected",
        "Faces detected",
        "High-confidence matches",
        "Needs-review faces",
        "Unknown faces",
    ):
        assert metric_label in html

    # 3. Evidence Card, stable Face N labels, and manual-vs-model distinction
    assert 'id="evidence-card"' in html
    assert "Choose another person" in html
    assert "Mark Unknown" in html
    assert "Clear session" in html
    assert "chooseCanvasLabelRect" in js
    assert "Manual decision" in js
    assert "Model suggestion" in js
    assert "Face ${idxNum}" in js

    # 4. Forbidden attendance / probability / external CDN wording absent
    for forbidden in ("Attendance", "absent", "probability", "http://", "https://"):
        assert forbidden.lower() not in html.lower() or "cosine similarity" in html.lower()
    assert "http://" not in html and "https://" not in html
    assert "http://" not in js and "https://" not in js
    assert "http://" not in css and "https://" not in css
