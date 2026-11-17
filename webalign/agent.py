"""WebAlign agent: extract -> claim graph -> symbolic verdict -> answer or abstain.

Confidence is a rule-assigned score derived from the within-context contradiction
H_within: consensus -> high; genuine contradiction -> low or abstain. Legitimate
(temporal/entity/unit) differences do not lower it, because the symbolic layer
collapses them first. The score ranks answers for abstention; it is not a
calibrated probability.
"""
from __future__ import annotations

from collections import Counter

from data.schema import BenchmarkItem, ConflictType, Prediction, Source
from webalign.extract import extract_item
from webalign.graph import build_graph


def _majority_value(claims) -> str | None:
    vals = [str(c.value) for c in claims]
    return Counter(vals).most_common(1)[0][0] if vals else None


def _target_time_value(item: BenchmarkItem, claims):
    """For temporal questions, pick the value matching the queried year."""
    yr = item.meta.get("target_year") or item.meta.get("year")
    if yr is None:
        return None
    for c in claims:
        if c.qualifier.time and str(yr) in str(c.qualifier.time):
            return str(c.value)
    return None


def run_webalign(item: BenchmarkItem, extract_spec="extract_text", agent="webalign", enabled=None) -> Prediction:
    item.sources = [s if isinstance(s, Source) else Source(**s) for s in item.sources]
    claims = extract_item(extract_spec, item)
    g = build_graph(claims, enabled=enabled)
    frag = g.frag
    h_norm = frag.get("h_true_norm", 0.0)  # genuine (context-aware) fragmentation

    pred = Prediction(qid=item.qid, method="webalign", agent=agent, frag=frag)

    if not claims:
        pred.answer, pred.answerable, pred.confidence = None, False, 0.15
        pred.conflict_type = ConflictType.INSUFFICIENT.value
        pred.rationale = "No source asserts a value for the queried slot."
        return pred

    v = g.primary
    ct = v.conflict_type
    pred.conflict_type = ct.value

    if ct in (ConflictType.AGREEMENT, ConflictType.UNIT_TOLERANCE):
        pred.answer = v.supported_value or _majority_value(claims)
        pred.answerable, pred.confidence = True, round(0.95 - 0.5 * h_norm, 3)
        pred.rationale = "Sources agree after normalization."
    elif ct == ConflictType.TEMPORAL_CHANGE:
        pred.answer = _target_time_value(item, claims) or _majority_value(claims)
        pred.answerable = True
        pred.confidence = 0.9 if _target_time_value(item, claims) else 0.5
        pred.rationale = "Values differ across time qualifiers (not a contradiction)."
    elif ct == ConflictType.ENTITY_AMBIGUITY:
        pred.answer, pred.answerable, pred.confidence = None, False, 0.2
        pred.conflict_detected = False
        pred.rationale = "The name refers to multiple distinct entities."
    else:  # TEXT_CONFLICT (or MODALITY_CONFLICT)
        maj = _majority_value(claims)
        counts = Counter(str(c.value) for c in claims)
        top2 = counts.most_common(2) + [("", 0)] * (2 - len(counts.most_common(2)))
        top, second = top2[0], top2[1]
        clear_majority = top[1] > second[1]
        pred.conflict_detected = True
        if clear_majority:
            # Recoverable: majority is likely correct; confidence = its support share.
            pred.answer = maj
            pred.answerable = True
            pred.confidence = round(top[1] / max(1, len(claims)), 3)
        else:
            # Unresolvable: the Web offers no single supported answer.
            pred.answer = None
            pred.answerable = False
            pred.confidence = round(max(0.05, 0.3 * (1 - h_norm)), 3)
        pred.rationale = "Sources assert incompatible values for the same slot."
    return pred


def run_neural_only(item: BenchmarkItem, extract_spec="extract_text", agent="neural_only") -> Prediction:
    """Ablation: same neural extraction, but NO symbolic consistency layer.

    Clusters by surface value and majority-votes; treats any surface difference as
    conflict and never collapses temporal/entity/unit variation. Isolates the
    contribution of the symbolic layer.
    """
    item.sources = [s if isinstance(s, Source) else Source(**s) for s in item.sources]
    claims = extract_item(extract_spec, item)
    pred = Prediction(qid=item.qid, method="neural_only", agent=agent)
    if not claims:
        pred.answer, pred.answerable, pred.confidence = None, False, 0.15
        pred.rationale = "No value extracted."
        return pred
    counts = Counter(str(c.value).strip().lower() for c in claims)
    top2 = counts.most_common(2) + [("", 0)] * (2 - len(counts.most_common(2)))
    top, second = top2[0], top2[1]
    pred.conflict_detected = len(counts) > 1
    if top[1] > second[1]:
        pred.answer = Counter(str(c.value) for c in claims).most_common(1)[0][0]
        pred.answerable = True
        pred.confidence = round(top[1] / len(claims), 3)
    else:
        pred.answer, pred.answerable = None, False
        pred.confidence = 0.3
    return pred
