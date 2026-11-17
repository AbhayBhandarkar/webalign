"""Cross-modal evaluation on WebFrag-MM (paper Table 6b).

Runs a text-only agent, a vision-language agent that sees the chart (with and
without reflection), and WebAlign, then reports how many of the 40 text-chart
conflicts each one flags and how many of the 40 agreement items it falsely flags.

Usage:
    python -m eval.run_multimodal --items data/benchmark/webfrag_mm.jsonl --outdir runs/mm
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from baselines.methods import run_vanilla, run_vision
from data.schema import BenchmarkItem, Source, load_items
from webalign.agent import run_webalign

RUNS = {
    "text_only": lambda it: run_vanilla(it, "qwen2.5-32b"),
    "vision_vanilla": lambda it: run_vision(it, "qwen2.5-vl-7b"),
    "vision_reflect": lambda it: run_vision(it, "qwen2.5-vl-7b", reflect=True),
    "webalign": lambda it: run_webalign(it, extract_spec="qwen2.5-32b", agent="qwen2.5-32b"),
}


def _item(d: dict) -> BenchmarkItem:
    srcs = [Source(id=s["id"], name=s.get("name", ""), text=s.get("text", ""),
                   image_path=s.get("image_path"), lang=s.get("lang", "en"))
            for s in d["sources"]]
    return BenchmarkItem(qid=d["qid"], question=d["question"], subject=d["subject"],
                         predicate=d["predicate"], sources=srcs, meta=d.get("meta", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/benchmark/webfrag_mm.jsonl")
    ap.add_argument("--outdir", default="runs/mm")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    raw = load_items(args.items)
    gold = {d["qid"]: d["gold_conflict_type"] for d in raw}
    items = [_item(d) for d in raw]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{'method':16s} {'conflicts flagged':>18s} {'false positives':>16s}")
    for name, fn in RUNS.items():
        def one(it):
            try:
                return fn(it).to_dict()
            except Exception as e:
                return {"qid": it.qid, "method": name, "conflict_detected": False,
                        "error": repr(e)[:200]}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            preds = list(ex.map(one, items))
        (outdir / f"{name}.jsonl").write_text("\n".join(json.dumps(p) for p in preds))
        conf = [p for p in preds if gold[p["qid"]] == "modality_conflict"]
        agree = [p for p in preds if gold[p["qid"]] == "agreement"]
        tp = sum(bool(p.get("conflict_detected")) for p in conf)
        fp = sum(bool(p.get("conflict_detected")) for p in agree)
        errs = sum("error" in p for p in preds)
        note = f"  ({errs} errors)" if errs else ""
        print(f"{name:16s} {tp:>12d}/{len(conf):<5d} {fp:>10d}/{len(agree):<5d}{note}")


if __name__ == "__main__":
    main()
