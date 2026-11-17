"""Baseline agents: vanilla, RAG, reflect, debate.

All see the same sources and must self-report an answer, whether the sources
disagree, answerability, and a confidence. Contrasting their *self-reported*
confidence with WebAlign's entropy-derived confidence is the core experiment.
"""
from __future__ import annotations

import json
import re

from data.schema import BenchmarkItem, Prediction, Source
from llm import get_client

_JSON_RE = re.compile(r"\{.*\}", re.S)

_SCHEMA = ('Return ONLY JSON: {"answer": <string, or null if there is no single '
           'well-supported answer>, "answerable": <true|false>, "sources_disagree": '
           '<true|false>, "confidence": <number 0..1>, "rationale": <short string>}')


def _sources_block(item: BenchmarkItem) -> str:
    lines = []
    for i, s in enumerate(item.sources):
        s = s if isinstance(s, Source) else Source(**s)
        lines.append(f"[{i+1}] {s.text}")
    return "\n".join(lines)


def _parse(txt: str) -> dict:
    m = _JSON_RE.search(txt or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _to_pred(item, agent, method, d) -> Prediction:
    ans = d.get("answer")
    if isinstance(ans, str) and ans.strip().lower() in ("null", "none", ""):
        ans = None
    try:
        conf = float(d.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return Prediction(
        qid=item.qid, method=method, agent=agent,
        answer=None if ans is None else str(ans),
        answerable=bool(d.get("answerable", ans is not None)),
        conflict_detected=bool(d.get("sources_disagree", False)),
        confidence=max(0.0, min(1.0, conf)),
        rationale=str(d.get("rationale", ""))[:300],
    )


def run_vanilla(item: BenchmarkItem, agent: str) -> Prediction:
    c = get_client()
    msg = (f"Answer the question using the web sources. Sources may disagree, be "
           f"outdated, or refer to different entities.\n\nQuestion: {item.question}\n\n"
           f"Sources:\n{_sources_block(item)}\n\n{_SCHEMA}")
    out = c.chat(agent, [{"role": "user", "content": msg}], max_tokens=600)
    return _to_pred(item, agent, "vanilla", _parse(out))


def run_reflect(item: BenchmarkItem, agent: str) -> Prediction:
    c = get_client()
    msg = (f"Answer using the web sources. First carefully consider whether the "
           f"sources genuinely contradict each other, refer to DIFFERENT TIME "
           f"PERIODS, use different units, or refer to DIFFERENT ENTITIES with the "
           f"same name. Only report disagreement if it is a genuine contradiction.\n\n"
           f"Question: {item.question}\n\nSources:\n{_sources_block(item)}\n\n{_SCHEMA}")
    out = c.chat(agent, [{"role": "user", "content": msg}], max_tokens=800)
    return _to_pred(item, agent, "reflect", _parse(out))


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    return _embedder


def run_rag(item: BenchmarkItem, agent: str, k: int = 3) -> Prediction:
    """Retrieve top-k sources by semantic similarity (bge embeddings), then answer."""
    srcs = [s if isinstance(s, Source) else Source(**s) for s in item.sources]
    if len(srcs) > k:
        import numpy as np
        emb = _get_embedder()
        qv = emb.encode([item.question], normalize_embeddings=True)
        sv = emb.encode([s.text for s in srcs], normalize_embeddings=True)
        sims = (sv @ qv.T).ravel()
        top = [srcs[i] for i in np.argsort(-sims)[:k]]
    else:
        top = srcs
    block = "\n".join(f"[{i+1}] {s.text}" for i, s in enumerate(top))
    c = get_client()
    msg = (f"Answer using the retrieved web sources.\n\nQuestion: {item.question}\n\n"
           f"Sources:\n{block}\n\n{_SCHEMA}")
    out = c.chat(agent, [{"role": "user", "content": msg}], max_tokens=600)
    return _to_pred(item, agent, "rag", _parse(out))


def run_debate(item: BenchmarkItem, agent: str) -> Prediction:
    c = get_client()
    src = _sources_block(item)
    pos = c.chat(agent, [{"role": "user", "content":
          f"In 2 sentences, argue that the sources support a single clear answer to: {item.question}\n"
          f"Sources:\n{src}"}], max_tokens=160)
    neg = c.chat(agent, [{"role": "user", "content":
          f"In 2 sentences, argue that the sources do NOT support a single answer to: {item.question} "
          f"(they may conflict, be outdated, or refer to different entities).\nSources:\n{src}"}],
          max_tokens=160)
    judge = c.chat(agent, [{"role": "user", "content":
          f"Question: {item.question}\nSources:\n{src}\n\n"
          f"Argument A (single answer exists): {pos}\n\n"
          f"Argument B (no single answer): {neg}\n\n"
          f"As an impartial judge, decide. {_SCHEMA}"}], max_tokens=400)
    return _to_pred(item, agent, "debate", _parse(judge))


def run_self_consistency(item: BenchmarkItem, agent: str, k: int = 5) -> Prediction:
    """Sample k answers at temperature and majority-vote; confidence = agreement."""
    from collections import Counter
    c = get_client()
    base = (f"Answer the question using the web sources. Sources may disagree, be "
            f"outdated, or refer to different entities.\n\nQuestion: {item.question}\n\n"
            f"Sources:\n{_sources_block(item)}\n\n{_SCHEMA}")
    preds, keys = [], []
    for i in range(k):
        out = c.chat(agent, [{"role": "user", "content": base + f"\n\n[independent attempt {i+1}]"}],
                     max_tokens=600, temperature=0.8)
        d = _parse(out)
        preds.append(d)
        a = d.get("answer")
        if isinstance(a, str) and a.strip().lower() in ("null", "none", ""):
            a = None
        keys.append(None if a is None else str(a).strip().lower())
    top, n = Counter(keys).most_common(1)[0]
    agree = n / k
    maj = next((d.get("answer") for d, kk in zip(preds, keys) if kk == top), None) if top is not None else None
    d = {"answer": maj, "answerable": maj is not None,
         "sources_disagree": agree < 0.6, "confidence": agree,
         "rationale": f"self-consistency {n}/{k} agree"}
    return _to_pred(item, agent, "self_consistency", d)


def run_cove(item: BenchmarkItem, agent: str) -> Prediction:
    """Chain-of-Verification: draft, generate checks, verify against sources, finalize."""
    c = get_client()
    src = _sources_block(item)
    draft = c.chat(agent, [{"role": "user", "content":
          f"Draft a concise answer to the question from the sources.\n"
          f"Question: {item.question}\nSources:\n{src}"}], max_tokens=200)
    checks = c.chat(agent, [{"role": "user", "content":
          f"Question: {item.question}\nDraft answer: {draft}\n\nList 2-3 short verification questions that "
          f"check whether the sources truly support this answer, including whether they refer to the same "
          f"entity, the same time period, and the same units. One question per line."}], max_tokens=200)
    verify = c.chat(agent, [{"role": "user", "content":
          f"Sources:\n{src}\n\nAnswer each verification question strictly from the sources:\n{checks}"}],
          max_tokens=400)
    final = c.chat(agent, [{"role": "user", "content":
          f"Question: {item.question}\nSources:\n{src}\n\nDraft: {draft}\nVerification questions: {checks}\n"
          f"Verification answers: {verify}\n\nUsing the verification, give the final answer. If the sources "
          f"refer to different entities or time periods, or do not agree on a single value, report that. "
          f"{_SCHEMA}"}], max_tokens=500)
    return _to_pred(item, agent, "cove", _parse(final))


METHODS = {"vanilla": run_vanilla, "reflect": run_reflect, "rag": run_rag, "debate": run_debate,
           "self_consistency": run_self_consistency, "cove": run_cove}


def _img_url(path):
    import base64
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def run_vision(item: BenchmarkItem, agent: str, reflect: bool = False) -> Prediction:
    """Multimodal baseline: the agent sees text sources AND the chart image."""
    srcs = [s if isinstance(s, Source) else Source(**s) for s in item.sources]
    txt = "\n".join(f"[{i+1}] {s.text}" for i, s in enumerate(srcs) if s.text)
    guide = ("First check whether the text and the chart genuinely disagree, or refer to "
             "different times, units, or entities. " if reflect else "")
    msg = [{"type": "text", "text":
            f"Answer using the web sources, which include text and a chart image. {guide}"
            f"\n\nQuestion: {item.question}\n\nText sources:\n{txt}\n\nChart:"}]
    for s in srcs:
        if s.image_path:
            msg.append({"type": "image_url", "image_url": {"url": _img_url(s.image_path)}})
    msg.append({"type": "text", "text": _SCHEMA})
    out = get_client().chat(agent, [{"role": "user", "content": msg}], max_tokens=700)
    method = "vision_reflect" if reflect else "vision_vanilla"
    return _to_pred(item, agent, method, _parse(out))


# --- Non-LLM NLI conflict-detection baseline -----------------------------
_nli = None


def _get_nli():
    global _nli
    if _nli is None:
        from transformers import pipeline
        import torch
        dev = 0 if torch.cuda.is_available() else -1
        try:
            _nli = pipeline("text-classification", model="microsoft/deberta-large-mnli",
                            device=dev, top_k=None)
        except Exception:
            _nli = pipeline("text-classification", model="microsoft/deberta-large-mnli",
                            device=-1, top_k=None)
    return _nli


def run_nli(item: BenchmarkItem, agent: str = "deberta-mnli") -> Prediction:
    """Conflict detection via pairwise NLI over raw source texts (no LLM).

    Flags a conflict if any source pair is classified CONTRADICTION. Abstains when
    a conflict is found. Scored only on conflict-oriented metrics.
    """
    nli = _get_nli()
    srcs = [s if isinstance(s, Source) else Source(**s) for s in item.sources][:5]
    texts = [s.text for s in srcs if s.text]
    contradiction = False
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            out = nli({"text": texts[i], "text_pair": texts[j]})
            scores = {d["label"].upper(): d["score"] for d in (out if isinstance(out, list) else [out])}
            if scores.get("CONTRADICTION", 0.0) > 0.5:
                contradiction = True
                break
        if contradiction:
            break
    return Prediction(qid=item.qid, method="nli", agent=agent, answer=None,
                      answerable=not contradiction, conflict_detected=contradiction,
                      confidence=0.5, rationale="pairwise NLI contradiction" if contradiction else "no NLI contradiction")
