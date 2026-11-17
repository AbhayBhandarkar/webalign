"""Claim extraction: source text (or image) -> normalized Claim tuples.

Targeted extraction: given the queried (subject, predicate), pull the value each
source asserts, plus a time qualifier and any entity-disambiguation cue. This is
the neural "perception" half of the neurosymbolic system.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Optional

from data.schema import BenchmarkItem, Claim, Modality, Qualifier, Source
from llm import get_client
from webalign.normalize import canonical_string, normalize_value

_JSON_RE = re.compile(r"\{.*\}", re.S)


def _img_data_url(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{b64}"


def _prompt(subject: str, predicate: str, source_text: str) -> str:
    return (
        f'Extract what the source asserts about the "{predicate}" of "{subject}".\n'
        f'Return ONLY a JSON object with keys:\n'
        f'  "value": the asserted value as a short string, or null if not stated;\n'
        f'  "time": the year/date the value refers to (string), or null;\n'
        f'  "entity_note": any parenthetical or description that disambiguates '
        f'which "{subject}" is meant (e.g. a location), or null.\n\n'
        f'Source: "{source_text}"\n\nJSON:'
    )


def _parse(txt: str) -> dict:
    m = _JSON_RE.search(txt or "")
    if not m:
        return {"value": None, "time": None, "entity_note": None}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"value": None, "time": None, "entity_note": None}
    return {"value": d.get("value"), "time": d.get("time"), "entity_note": d.get("entity_note")}


def extract_source(spec, subject: str, predicate: str, source: Source,
                   visual_spec="extract_visual") -> Optional[Claim]:
    client = get_client()
    if source.image_path:
        # Visual source: read the value off the chart/table with a vision model.
        out = client.chat(visual_spec, [
            {"role": "system", "content": "You are a precise extraction system. Output only JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": _prompt(subject, predicate, "(see image)")},
                {"type": "image_url", "image_url": {"url": _img_data_url(source.image_path)}},
            ]},
        ], max_tokens=512)
    else:
        out = client.chat(spec, [
            {"role": "system", "content": "You are a precise extraction system. Output only JSON."},
            {"role": "user", "content": _prompt(subject, predicate, source.text)},
        ], max_tokens=256)
    d = _parse(out)
    if d["value"] in (None, "", "null"):
        return None
    c = Claim(subject=subject, predicate=predicate, value=str(d["value"]),
              source_id=source.id, qualifier=Qualifier(time=(str(d["time"]) if d["time"] else None)),
              modality=source.modality, raw_text=source.text)
    c.value_norm = normalize_value(c.value)
    # Drop a redundant time qualifier equal to the value itself (e.g. a founding
    # year is the answer, not an "as-of" time) so year conflicts aren't mistaken
    # for temporal change.
    if c.qualifier.time is not None:
        if normalize_value(c.qualifier.time) == c.value_norm:
            c.qualifier.time = None
    if d.get("entity_note"):
        # Distinct disambiguation cues => distinct entities (pseudo-QID).
        c.subject_qid = "note:" + canonical_string(str(d["entity_note"]))
    return c


def extract_item(spec, item: BenchmarkItem) -> list[Claim]:
    claims = []
    for s in item.sources:
        s = s if isinstance(s, Source) else Source(**s)
        c = extract_source(spec, item.subject, item.predicate, s)
        if c is not None:
            claims.append(c)
    return claims
