"""Tests for Step 8: Mini Coverage Map UI and accessibility contracts."""

from pathlib import Path


def test_coverage_map_html_js_contracts():
    html = Path("static/index.html").read_text(encoding="utf-8")
    js = Path("static/app.js").read_text(encoding="utf-8")

    # 1. Permanent one-line caption (not hover-only) + aria-label
    assert "Shows viewport and detected faces; does not prove every face was found" in html
    assert 'aria-label="Mini coverage map showing viewport and detected face markers"' in html
    assert 'id="btn-toggle-map"' in html

    # 2. Preserves aspect ratio inside map (Math.min scale, centered offsets)
    assert "Math.min(mw / img.width, mh / img.height)" in js

    # 3. Distinct marker shapes per status + face number
    assert "drawCoverageMarkerShape" in js
    assert 'status === "Confirmed by reviewer"' in js

    # 4. Removed manual boxes excluded & collapsed on narrow windows by default
    assert "if (face.removed) continue;" in js
    assert "window.innerWidth > 960" in js
    assert "window.innerWidth <= 960" in js

    # 5. Zero external URLs introduced
    assert "http://" not in html and "https://" not in html
    assert "http://" not in js and "https://" not in js
