"""Claim graph: group claims into slots, classify each, aggregate fragmentation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from data.schema import Claim, ConflictType
from webalign.consistency import SlotVerdict, classify_slot
from webalign.entropy import fragmentation


@dataclass
class WebGraph:
    slots: dict[tuple, list[Claim]] = field(default_factory=dict)
    verdicts: dict[tuple, SlotVerdict] = field(default_factory=dict)
    frag: dict[str, float] = field(default_factory=dict)
    num_sources: int = 0
    num_claims: int = 0

    @property
    def primary(self) -> Optional[SlotVerdict]:
        """Verdict for the most-supported slot (single-slot benchmark => the one)."""
        if not self.verdicts:
            return None
        return max(self.verdicts.values(), key=lambda v: v.num_value_clusters + 0)


def group_by_slot(claims: list[Claim]) -> dict[tuple, list[Claim]]:
    slots: dict[tuple, list[Claim]] = defaultdict(list)
    for c in claims:
        slots[c.slot()].append(c)
    return dict(slots)


def build_graph(claims: list[Claim], enabled=None) -> WebGraph:
    from webalign.consistency import ALL_RULES
    enabled = ALL_RULES if enabled is None else enabled
    slots = group_by_slot(claims)
    verdicts = {k: classify_slot(v, enabled=enabled) for k, v in slots.items()}
    g = WebGraph(slots=slots, verdicts=verdicts,
                 frag=fragmentation(claims) if claims else
                 {"h_raw": 0.0, "h_true": 0.0, "delta_apparent": 0.0, "h_true_norm": 0.0},
                 num_sources=len({c.source_id for c in claims}),
                 num_claims=len(claims))
    return g
