"""Leave-one-rule-out ablation: each symbolic rule vs the failure it prevents.

Removing a rule should inflate the false-contradiction rate on exactly the
legitimate-variation category that rule handles (paper Table 4). Extractions are
cached, so after the first variant no new model calls are made.

Usage:
    python -m eval.ablation --spec qwen2.5-32b
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from data.schema import BenchmarkItem, Source, load_items
from webalign.agent import run_webalign
from webalign.consistency import ALL_RULES

RULE_CATEGORY = {"temporal": "temporal_change", "entity": "entity_ambiguity",
                 "unit": "unit_tolerance", "modality": "modality_conflict"}


def _mk(d):
    return BenchmarkItem(qid=d["qid"], question=d["question"], subject=d["subject"],
                         predicate=d["predicate"], sources=[Source(**s) for s in d["sources"]],
                         meta=d.get("meta", {}))


def fcr_by_category(items_raw, enabled, spec="qwen2.5-32b"):
    """Return dict category -> false-contradiction rate under the given rule set."""
    hit = defaultdict(lambda: [0, 0])
    for d in items_raw:
        cat = d["gold_conflict_type"]
        if cat not in ("temporal_change", "entity_ambiguity", "unit_tolerance", "modality_conflict"):
            continue
        p = run_webalign(_mk(d), extract_spec=spec, agent="webalign", enabled=enabled)
        hit[cat][0] += int(p.conflict_detected)
        hit[cat][1] += 1
    return {c: (n / d if d else None) for c, (n, d) in hit.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", default="data/benchmark/webfrag_ctrl.jsonl")
    ap.add_argument("--spec", default="qwen2.5-32b")
    args = ap.parse_args()
    ctrl = load_items(args.ctrl)

    variants = [("full", ALL_RULES)]
    for r in ["temporal", "entity", "unit"]:
        variants.append((f"-{r} rule", ALL_RULES - {r}))

    cats = ["temporal_change", "entity_ambiguity", "unit_tolerance"]
    print(f"{'variant':12s} " + " ".join(f"{c.split('_')[0]:>10}" for c in cats))
    for name, en in variants:
        res = fcr_by_category(ctrl, en, args.spec)
        row = " ".join((f"{res[c]*100:9.1f}%" if res.get(c) is not None else f"{'-':>10}") for c in cats)
        print(f"{name:12s} {row}")


if __name__ == "__main__":
    main()
