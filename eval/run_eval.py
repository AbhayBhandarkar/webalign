"""Run baselines and WebAlign on a benchmark split and score them.

Usage:
    python -m eval.run_eval --items data/benchmark/webfrag_ctrl.jsonl \
        --agents qwen2.5-32b --methods vanilla reflect webalign

Predictions are written to ``runs/<method>__<agent>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from baselines import METHODS
from data.schema import BenchmarkItem, Source, load_items
from eval.metrics import score
from webalign.agent import run_webalign

ALL_METHODS = [*METHODS, "nli", "neural_only", "webalign"]


def run_method_agent(items, method, agent, workers):
    def one(it):
        try:
            if method == "webalign":
                return run_webalign(it, extract_spec=agent, agent=agent).to_dict()
            if method == "neural_only":
                from webalign.agent import run_neural_only
                return run_neural_only(it, extract_spec=agent, agent=agent).to_dict()
            if method == "nli":
                from baselines.methods import run_nli
                return run_nli(it, agent).to_dict()
            return METHODS[method](it, agent).to_dict()
        except Exception as e:
            return {"qid": it.qid, "method": method, "agent": agent,
                    "answer": None, "answerable": True, "conflict_detected": False,
                    "confidence": 0.5, "error": repr(e)[:200]}
    preds = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, it) for it in items]
        for f in as_completed(futs):
            preds.append(f.result())
    return preds


def to_item(d: dict) -> BenchmarkItem:
    srcs = [Source(id=s["id"], name=s.get("name", ""), text=s.get("text", ""),
                   image_path=s.get("image_path"), lang=s.get("lang", "en"))
            for s in d["sources"]]
    return BenchmarkItem(qid=d["qid"], question=d["question"], subject=d["subject"],
                         predicate=d["predicate"], sources=srcs, meta=d.get("meta", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/benchmark/webfrag_ctrl.jsonl")
    ap.add_argument("--agents", nargs="+", default=["qwen2.5-32b"])
    ap.add_argument("--methods", nargs="+", default=["vanilla", "reflect", "webalign"],
                    choices=ALL_METHODS)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N items")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()

    raw = load_items(args.items)
    if args.limit:
        raw = raw[:args.limit]
    items = [to_item(d) for d in raw]
    items_by_qid = {d["qid"]: d for d in raw}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for agent in args.agents:
        for method in args.methods:
            w = 2 if method == "nli" else args.workers  # the NLI pipeline is not thread-safe
            preds = run_method_agent(items, method, agent, w)
            fp = outdir / f"{method}__{agent}.jsonl"
            fp.write_text("".join(json.dumps(p) + "\n" for p in preds))
            m = score(preds, items_by_qid)
            f = lambda k: "--" if m[k] is None else f"{m[k]:.3f}"
            print(f"[{agent}/{method}] acc={f('accuracy')} f1={f('conflict_f1')} "
                  f"FCR={f('false_contradiction_rate')} CWA={f('consistency_awareness')} "
                  f"ECE={f('ece')} rho={f('corr_conf_htrue')}  -> {fp}")


if __name__ == "__main__":
    main()
