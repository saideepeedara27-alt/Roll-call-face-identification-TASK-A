"""Intelligent filename parsing and identity-key normalization."""

import re
from dataclasses import dataclass
from pathlib import Path

_COPY_SUFFIX_RE = re.compile(r"(?:\s*\(\d+\)|\s+copy(?:\s+\d+)?)$", re.IGNORECASE)
_INDEX_SUFFIX_RE = re.compile(r"[\s_-]+(?:\d+|v\d+|photo\d+|img\d+)$", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"[_\-.]+")
_WHITESPACE_RE = re.compile(r"\s+")
_CAMERA_STEM_RE = re.compile(
    r"^(?:"
    r"img|dsc|dscn|dscf|pxl|mvimg|dji|gopr|pic|photo|image|snapshot|"
    r"screenshot|capture|scan|untitled|unnamed|file|doc|frame|cam|"
    r"whatsapp[\s_\-]*image|photo[\s_\-]*on"
    r")(?:[\s_\-]*(?:\d+|wa\d+|am|pm|at))*$",
    re.IGNORECASE,
)
_NUMERIC_OR_TIMESTAMP_RE = re.compile(r"^[\d\s_\-]+$")


@dataclass(frozen=True)
class ParsedIdentity:
    """Normalized identity extracted from an enrollment image filename."""

    identity_key: str
    display_name: str
    original_filename: str
    is_unclear: bool = False
    unclear_reason: str | None = None


def parse_identity_filename(filename: str) -> ParsedIdentity:
    """Extract a clean display name and normalized identity key from a filename.

    Handles spaces, underscores, hyphens, mixed case, browser copy suffixes
    like `(1)`, and multi-photo numeric suffixes like `Priya_Sharma_2.jpg` so
    multiple reference images map to the same normalized `identity_key`.
    Flags camera/generic stems such as `IMG_1234.jpg` or `12345.png` as unclear.
    """
    base_name = Path(filename).name
    stem = Path(base_name).stem.strip()

    raw_collapsed = _WHITESPACE_RE.sub(" ", _SEPARATOR_RE.sub(" ", stem)).strip()
    is_unclear = False
    unclear_reason: str | None = None
    if not raw_collapsed:
        is_unclear = True
        unclear_reason = "Filename has no readable name."
    elif _NUMERIC_OR_TIMESTAMP_RE.match(raw_collapsed):
        is_unclear = True
        unclear_reason = f"Filename '{base_name}' contains only numbers or timestamps."
    elif _CAMERA_STEM_RE.match(raw_collapsed):
        is_unclear = True
        unclear_reason = (
            f"Filename '{base_name}' looks like a camera or generic filename rather than a person's name."
        )

    # Replace underscores/dots/hyphens with spaces and collapse whitespace
    cleaned = raw_collapsed

    # Strip trailing copy markers e.g. "Priya Sharma (1)" or "Priya Sharma copy"
    cleaned = _COPY_SUFFIX_RE.sub("", cleaned).strip()

    # Strip trailing numeric photo index e.g. "Priya Sharma 2" -> "Priya Sharma"
    without_index = _INDEX_SUFFIX_RE.sub("", cleaned).strip()
    if without_index:
        cleaned = without_index

    alpha_chars = re.findall(r"[A-Za-z\u00C0-\u024F]", cleaned)
    if not is_unclear and len(alpha_chars) < 2:
        is_unclear = True
        unclear_reason = f"Filename '{base_name}' does not contain a readable person name."

    if not cleaned:
        cleaned = "Unnamed Person"

    # Title-case cleanly while preserving already-capitalized initials if mixed
    display_name = " ".join(
        word if (word.isupper() and len(word) <= 3) else word.capitalize()
        for word in cleaned.split(" ")
    )
    identity_key = re.sub(r"\s+", "_", display_name.lower())

    return ParsedIdentity(
        identity_key=identity_key,
        display_name=display_name,
        original_filename=base_name,
        is_unclear=is_unclear,
        unclear_reason=unclear_reason,
    )
