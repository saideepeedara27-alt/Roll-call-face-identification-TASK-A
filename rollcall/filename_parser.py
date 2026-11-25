"""Intelligent filename parsing and identity-key normalization."""

import re
from dataclasses import dataclass
from pathlib import Path

_COPY_SUFFIX_RE = re.compile(r"(?:\s*\(\d+\)|\s+copy(?:\s+\d+)?)$", re.IGNORECASE)
_INDEX_SUFFIX_RE = re.compile(r"[\s_-]+(?:\d+|v\d+|photo\d+|img\d+)$", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"[_\-.]+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedIdentity:
    """Normalized identity extracted from an enrollment image filename."""

    identity_key: str
    display_name: str
    original_filename: str


def parse_identity_filename(filename: str) -> ParsedIdentity:
    """Extract a clean display name and normalized identity key from a filename.

    Handles spaces, underscores, hyphens, mixed case, browser copy suffixes
    like `(1)`, and multi-photo numeric suffixes like `Priya_Sharma_2.jpg` so
    multiple reference images map to the same normalized `identity_key`.
    """
    base_name = Path(filename).name
    stem = Path(base_name).stem.strip()

    # Replace underscores/dots/hyphens with spaces and collapse whitespace
    cleaned = _SEPARATOR_RE.sub(" ", stem)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    # Strip trailing copy markers e.g. "Priya Sharma (1)" or "Priya Sharma copy"
    cleaned = _COPY_SUFFIX_RE.sub("", cleaned).strip()

    # Strip trailing numeric photo index e.g. "Priya Sharma 2" -> "Priya Sharma"
    without_index = _INDEX_SUFFIX_RE.sub("", cleaned).strip()
    if without_index:
        cleaned = without_index

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
    )
