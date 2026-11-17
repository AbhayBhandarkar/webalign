"""Core data structures for WebAlign.

A benchmark item is a question plus a set of sources. Each source asserts one or
more atomic claims. Claims are normalized into tuples so the symbolic layer can
decide whether they can coexist.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"


class ConflictType(str, Enum):
    AGREEMENT = "agreement"                 # sources agree
    TEXT_CONFLICT = "text_conflict"         # genuine contradictory values
    TEMPORAL_CHANGE = "temporal_change"     # same fact, different valid times (NOT a conflict)
    ENTITY_AMBIGUITY = "entity_ambiguity"   # same surface, different entity (NOT a conflict)
    UNIT_TOLERANCE = "unit_tolerance"       # differ only by unit/rounding (NOT a conflict)
    MODALITY_CONFLICT = "modality_conflict" # text vs visual disagree
    INSUFFICIENT = "insufficient"           # no source supports an answer


# Conflict types that represent *apparent* rather than *true* disagreement.
APPARENT_NONCONFLICT = {
    ConflictType.TEMPORAL_CHANGE,
    ConflictType.ENTITY_AMBIGUITY,
    ConflictType.UNIT_TOLERANCE,
}


@dataclass
class Qualifier:
    time: Optional[str] = None      # ISO date or interval "start/end"
    location: Optional[str] = None
    unit: Optional[str] = None

    def slot_key(self) -> tuple:
        return (self.time, self.location, self.unit)


@dataclass
class Claim:
    subject: str
    predicate: str
    value: str
    source_id: str
    qualifier: Qualifier = field(default_factory=Qualifier)
    modality: Modality = Modality.TEXT
    raw_text: str = ""
    # Filled by normalization:
    subject_qid: Optional[str] = None
    value_norm: Optional[Any] = None        # float / normalized string / interval

    def slot(self) -> tuple:
        """Competing claims share the surface subject+predicate.

        Uses the surface form (not the linked QID) so that same-name/different-
        entity claims land in one slot where entity ambiguity can be detected.
        """
        return (self.subject.lower(), self.predicate.lower())


@dataclass
class Source:
    id: str
    name: str = ""
    url: str = ""
    text: str = ""
    image_path: Optional[str] = None
    modality: Modality = Modality.TEXT
    lang: str = "en"


@dataclass
class BenchmarkItem:
    qid: str
    question: str
    subject: str                             # canonical entity the question is about
    predicate: str
    sources: list[Source] = field(default_factory=list)
    # Gold labels (known for controlled set; annotated for real set):
    gold_answer: Optional[str] = None
    gold_conflict_type: Optional[ConflictType] = None
    gold_answerable: bool = True             # False when the Web has no single answer
    gold_entropy: Optional[float] = None     # ground-truth H_true for controlled items
    gold_num_clusters: Optional[int] = None
    split: str = "ctrl"                      # "ctrl" | "real"
    lang: str = "en"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gold_conflict_type"] = self.gold_conflict_type.value if self.gold_conflict_type else None
        for s in d["sources"]:
            s["modality"] = s["modality"].value if isinstance(s["modality"], Modality) else s["modality"]
        return d


@dataclass
class Prediction:
    qid: str
    method: str                              # vanilla | rag | reflect | debate | webalign
    agent: str                               # backend/model identifier
    answer: Optional[str] = None
    answerable: bool = True                  # False => model asserts "no single answer"
    conflict_detected: bool = False
    confidence: float = 0.5                  # score in [0,1]; not a calibrated probability
    conflict_type: Optional[str] = None
    rationale: str = ""
    frag: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def dumps_items(items: list[BenchmarkItem]) -> str:
    return "\n".join(json.dumps(it.to_dict(), ensure_ascii=False) for it in items)


def load_items(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
