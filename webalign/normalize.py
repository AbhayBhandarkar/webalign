"""Normalization: turn surface claim values into comparable canonical forms.

The symbolic layer compares *normalized* values so that legitimate differences
(units, rounding, date formats) don't look like contradictions. Entity linking
attaches a Wikidata QID when available so that same-surface/different-entity
cases are recognized as ambiguity rather than conflict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

Number = Union[int, float]

# Word-scale multipliers, including South-Asian scales for the multilingual set.
_SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6, "mn": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
}

_NUM_RE = re.compile(
    r"(-?\d[\d,]*\.?\d*)\s*"
    r"(thousand|million|billion|trillion|lakhs?|lac|crores?|cr|mn|bn|[kmb])?",
    re.IGNORECASE,
)


def parse_number(text: str) -> Optional[Number]:
    if text is None:
        return None
    s = str(text).strip().lower().replace(",", "")
    m = _NUM_RE.search(s)
    if not m or m.group(1) in ("", "-", "."):
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    scale = m.group(2)
    if scale:
        val *= _SCALES.get(scale.lower(), 1.0)
    return val


# --- dates ---------------------------------------------------------------
_ISO_RE = re.compile(r"\b(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?\b")
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")


@dataclass(frozen=True)
class DateInterval:
    start: str
    end: str

    def overlaps(self, other: "DateInterval") -> bool:
        return self.start <= other.end and other.start <= self.end


def parse_date(text: str) -> Optional[DateInterval]:
    """Parse a date/year into an interval so year vs full-date compare cleanly."""
    if text is None:
        return None
    s = str(text).strip()
    m = _ISO_RE.search(s)
    if m and m.group(2):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if d:
            return DateInterval(f"{y}-{mo}-{d}", f"{y}-{mo}-{d}")
        return DateInterval(f"{y}-{mo}-01", f"{y}-{mo}-31")
    ym = _YEAR_RE.search(s)
    if ym:
        y = ym.group(1)
        return DateInterval(f"{y}-01-01", f"{y}-12-31")
    return None


def canonical_string(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_value(value: str) -> object:
    """Canonical form: date/year-interval > number > canonical string.

    Dates are tried first so year-like values (e.g. "2016") are treated as exact
    intervals rather than tolerant measurements.
    """
    d = parse_date(value)
    if d is not None:
        return d
    n = parse_number(value)
    if n is not None:
        return round(n, 6)
    return canonical_string(value)
