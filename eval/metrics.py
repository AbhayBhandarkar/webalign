"""Evaluation metrics for WebAlign.

Definitions:
  * answerability-aware accuracy: correct if (answerable gold & answer matches) or
    (unanswerable gold & the method abstained).
  * conflict detection: gold "genuine conflict" = TEXT_CONFLICT / MODALITY_CONFLICT.
  * False Contradiction Rate: on apparent-non-conflicts (temporal/entity/unit),
    fraction flagged as disagreement.
  * Web Consistency Awareness: on unanswerable items, fraction correctly abstained.
  * ECE: expected calibration error of confidence vs correctness, computed only
    over items where the method commits to an answer (selective ECE).
  * corr(confidence, H_true): Pearson corr of confidence with gold entropy.
"""
from __future__ import annotations

import math
from collections import defaultdict

from data.schema import ConflictType
from webalign.consistency import values_equivalent
from webalign.normalize import normalize_value

GENUINE_CONFLICT = {ConflictType.TEXT_CONFLICT.value, ConflictType.MODALITY_CONFLICT.value}
APPARENT = {ConflictType.TEMPORAL_CHANGE.value, ConflictType.ENTITY_AMBIGUITY.value,
            ConflictType.UNIT_TOLERANCE.value}


def _match(a, b) -> bool:
    if a is None or b is None:
        return False
    return values_equivalent(normalize_value(str(a)), normalize_value(str(b)))


def score(preds: list[dict], items_by_qid: dict[str, dict]) -> dict:
    n = 0
    correct = 0
    tp = fp = fn = 0             # conflict detection
    fc_num = fc_den = 0         # false contradiction
    awa_num = awa_den = 0       # consistency awareness
    conf_correct = []           # (confidence, correct) for ECE
    conf_list, htrue_list = [], []

    for p in preds:
        it = items_by_qid.get(p["qid"])
        if it is None:
            continue
        n += 1
        gold_answerable = it["gold_answerable"]
        gold_ans = it.get("gold_answer")
        gct = it.get("gold_conflict_type")

        # accuracy
        if gold_answerable:
            ok = bool(p.get("answerable", True)) and _match(p.get("answer"), gold_ans)
        else:
            ok = not bool(p.get("answerable", True))
        correct += int(ok)

        # Selective calibration: ECE over items where the method commits an answer.
        committed = bool(p.get("answerable", True)) and p.get("answer") is not None
        if committed:
            ans_ok = gold_answerable and _match(p.get("answer"), gold_ans)
            conf_correct.append((float(p.get("confidence", 0.5)), int(ans_ok)))

        # conflict detection
        gold_conf = gct in GENUINE_CONFLICT
        det = bool(p.get("conflict_detected", False))
        if gold_conf and det: tp += 1
        elif gold_conf and not det: fn += 1
        elif (not gold_conf) and det: fp += 1

        # false contradiction (on apparent non-conflicts)
        if gct in APPARENT:
            fc_den += 1
            fc_num += int(det)

        # consistency awareness (on unanswerable)
        if not gold_answerable:
            awa_den += 1
            awa_num += int(not bool(p.get("answerable", True)))

        # calibration correlation
        if it.get("gold_entropy") is not None:
            conf_list.append(float(p.get("confidence", 0.5)))
            htrue_list.append(float(it["gold_entropy"]))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "conflict_precision": prec,
        "conflict_recall": rec,
        "conflict_f1": f1,
        "false_contradiction_rate": fc_num / fc_den if fc_den else None,
        "consistency_awareness": awa_num / awa_den if awa_den else None,
        "ece": _ece(conf_correct),
        "corr_conf_htrue": _pearson(conf_list, htrue_list) if len(conf_list) > 2 else None,
    }


def _ece(conf_correct: list[tuple[float, int]], bins: int = 10) -> float:
    if not conf_correct:
        return 0.0
    buckets = defaultdict(list)
    for c, ok in conf_correct:
        b = min(bins - 1, int(c * bins))
        buckets[b].append((c, ok))
    n = len(conf_correct)
    ece = 0.0
    for b, items in buckets.items():
        avg_conf = sum(c for c, _ in items) / len(items)
        acc = sum(ok for _, ok in items) / len(items)
        ece += (len(items) / n) * abs(avg_conf - acc)
    return ece


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a - mx) ** 2 for a in x))
    vy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (vx * vy) if vx and vy else 0.0
