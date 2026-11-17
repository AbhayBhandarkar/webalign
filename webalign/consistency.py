"""Symbolic consistency engine.

Given the competing claims for one (subject, predicate) slot, decide whether they
can coexist and, if not, what kind of (in)consistency it is. This is the symbolic
half of the neurosymbolic system: the neural models produce normalized claims;
these deterministic rules decide coexistence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from data.schema import Claim, ConflictType, Modality
from webalign.normalize import DateInterval, normalize_value

NUMERIC_REL_TOL = 0.005    # 0.5% relative tolerance (measurement rounding)
NUMERIC_ABS_TOL = 1e-9


def values_equivalent(a, b) -> bool:
    """Compare two normalized values under type-aware tolerance."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) <= max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * denom)
    if isinstance(a, DateInterval) and isinstance(b, DateInterval):
        return a.overlaps(b)
    return str(a) == str(b)


@dataclass
class SlotVerdict:
    subject: str
    predicate: str
    conflict_type: ConflictType
    num_value_clusters: int
    supported_value: Optional[str] = None   # majority/consensus value if any
    note: str = ""


def _ensure_normalized(claims: list[Claim]) -> None:
    for c in claims:
        if c.value_norm is None:
            c.value_norm = normalize_value(c.value)


def _value_clusters(claims: list[Claim], unit: bool = True) -> list[list[Claim]]:
    """Greedy clustering of claims by equivalent value.

    With unit=True, values are compared under numeric/format tolerance; with
    unit=False, only exact surface values match (unit rule ablated).
    """
    clusters: list[list[Claim]] = []
    for c in claims:
        placed = False
        for cl in clusters:
            same = (values_equivalent(c.value_norm, cl[0].value_norm) if unit
                    else str(c.value).strip().lower() == str(cl[0].value).strip().lower())
            if same:
                cl.append(c); placed = True; break
        if not placed:
            clusters.append([c])
    return clusters


ALL_RULES = frozenset({"temporal", "entity", "unit", "modality"})


def classify_slot(claims: list[Claim], enabled=ALL_RULES) -> SlotVerdict:
    subj = claims[0].subject
    pred = claims[0].predicate
    _ensure_normalized(claims)

    # 1) Entity ambiguity: same surface subject resolves to different QIDs.
    qids = {c.subject_qid for c in claims if c.subject_qid}
    if "entity" in enabled and len(qids) > 1:
        return SlotVerdict(subj, pred, ConflictType.ENTITY_AMBIGUITY,
                           len(_value_clusters(claims)),
                           note=f"distinct entities: {sorted(qids)}")

    clusters = _value_clusters(claims, unit=("unit" in enabled))

    # 2) Consensus.
    if len(clusters) == 1:
        return SlotVerdict(subj, pred, ConflictType.AGREEMENT, 1,
                           supported_value=str(clusters[0][0].value))

    # 3) Differing values: decide temporal vs modality vs genuine conflict.
    times = [c.qualifier.time for c in claims]
    distinct_times = {t for t in times if t}
    if "temporal" in enabled and len(distinct_times) > 1 and _time_explains_split(clusters):
        return SlotVerdict(subj, pred, ConflictType.TEMPORAL_CHANGE, len(clusters),
                           note="values differ but along distinct time qualifiers")

    modalities = {c.modality for c in claims}
    if "modality" in enabled and Modality.TEXT in modalities and (modalities - {Modality.TEXT}):
        if _modality_explains_split(clusters):
            return SlotVerdict(subj, pred, ConflictType.MODALITY_CONFLICT, len(clusters),
                               note="text and visual claims disagree")

    # 4) Genuine contradiction.
    return SlotVerdict(subj, pred, ConflictType.TEXT_CONFLICT, len(clusters),
                       note="incompatible values at the same qualifier")


def _time_explains_split(clusters: list[list[Claim]]) -> bool:
    """True if each value-cluster occupies a distinct (non-overlapping) time."""
    intervals = []
    for cl in clusters:
        cl_times = [c.qualifier.time for c in cl if c.qualifier.time]
        if not cl_times:
            return False
        intervals.append(set(cl_times))
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            if intervals[i] & intervals[j]:
                return False
    return True


def _modality_explains_split(clusters: list[list[Claim]]) -> bool:
    """True if the split aligns with a text-vs-visual boundary."""
    for cl in clusters:
        mods = {c.modality for c in cl}
        if Modality.TEXT in mods and (mods - {Modality.TEXT}):
            return False   # a single cluster mixes modalities -> not a clean modality split
    return True
