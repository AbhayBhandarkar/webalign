"""Recompute the paper's tables from the released raw outputs (no model calls).

Usage:
    python -m eval.reproduce_tables
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from data.schema import load_items
from eval.metrics import _match, score

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "data" / "benchmark"
OUT = REPO / "outputs"

CTRL_METHODS = ["vanilla", "rag", "reflect", "debate", "self_consistency", "cove",
                "nli", "neural_only", "webalign"]
CATEGORIES = ["agreement", "temporal_change", "unit_tolerance", "entity_ambiguity",
              "text_conflict", "insufficient"]
CROSS_MODELS = ["phi-3.5-mini", "qwen2.5-32b", "qwen2.5-72b", "llama-3.3-70b",
                "mistral-large-3", "gemma-4-31b", "deepseek-v3.2", "deepseek-v4-pro",
                "kimi-k2.6", "grok-4", "grok-4.3", "gpt-4o", "gpt-4.1", "gpt-5.1", "o3",
                "gpt-oss-120b"]


def _by_qid(path: Path) -> dict[str, dict]:
    return {d["qid"]: d for d in load_items(str(path))}


def _fmt(x, digits=3):
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "--"


def _preds(name: str, dirs: list[str]) -> list[dict] | None:
    for d in dirs:
        p = OUT / d / f"{name}.jsonl"
        if p.exists():
            return load_items(str(p))
    return None


def table_controlled(ctrl: dict) -> None:
    print("\nTable 2 - WebFrag-Ctrl, Qwen2.5-32B (n=737)")
    print(f"{'method':18s} {'Acc':>6s} {'F1':>6s} {'FCR':>6s} {'CWA':>6s} {'ECE':>6s} {'rho':>6s}")
    for m in CTRL_METHODS:
        s = score(_preds(f"{m}__qwen2.5-32b", ["webfrag_ctrl"]), ctrl)
        print(f"{m:18s} {_fmt(s['accuracy'])} {_fmt(s['conflict_f1'])} "
              f"{_fmt(s['false_contradiction_rate'])} {_fmt(s['consistency_awareness'])} "
              f"{_fmt(s['ece'])} {_fmt(s['corr_conf_htrue'], 2)}")


def table_categories(ctrl: dict) -> None:
    print("\nTable 3 - category-wise accuracy (%), WebFrag-Ctrl, Qwen2.5-32B")
    print(f"{'method':10s} " + " ".join(f"{c.split('_')[0][:8]:>8s}" for c in CATEGORIES))
    for m in ["vanilla", "rag", "reflect", "debate", "webalign"]:
        hit = defaultdict(lambda: [0, 0])
        for p in _preds(f"{m}__qwen2.5-32b", ["webfrag_ctrl"]):
            it = ctrl[p["qid"]]
            answered = bool(p.get("answerable", True))
            ok = (answered and _match(p.get("answer"), it.get("gold_answer"))
                  if it["gold_answerable"] else not answered)
            hit[it["gold_conflict_type"]][0] += int(ok)
            hit[it["gold_conflict_type"]][1] += 1
        print(f"{m:10s} " + " ".join(f"{100 * hit[c][0] / hit[c][1]:8.1f}" for c in CATEGORIES))


def table_cross_model(sub: dict) -> None:
    print("\nTable 5 - cross-model, stratified subset (n=205): FCR / CWA")
    print(f"{'model':16s} {'vanilla':>13s} {'reflection':>13s} {'WebAlign':>13s}")
    for model in CROSS_MODELS:
        cells = []
        for m in ["vanilla", "reflect", "webalign"]:
            s = score(_preds(f"{m}__{model}", ["webfrag_ctrl_subset", "webfrag_ctrl"]), sub)
            cells.append(f"{s['false_contradiction_rate']:.2f} / {s['consistency_awareness']:.2f}")
        print(f"{model:16s} " + " ".join(f"{c:>13s}" for c in cells))


def table_ambigqa(real: dict) -> None:
    print(f"\nTable 6a - WebFrag-Real / AmbigQA (n={len(real)})")
    for m in ["vanilla", "reflect", "neural_only", "webalign"]:
        s = score(_preds(f"{m}__qwen2.5-32b", ["webfrag_real"]), real)
        print(f"{m:12s} Acc {s['accuracy']:.2f}  CWA {s['consistency_awareness']:.2f}")


def main() -> None:
    ctrl = _by_qid(BENCH / "webfrag_ctrl.jsonl")
    table_controlled(ctrl)
    table_categories(ctrl)
    table_cross_model(_by_qid(BENCH / "webfrag_ctrl_subset.jsonl"))
    table_ambigqa(_by_qid(BENCH / "webfrag_real.jsonl"))
    print("\nTables 4 and 6b require model calls: see eval/ablation.py and eval/run_multimodal.py.")


if __name__ == "__main__":
    main()
