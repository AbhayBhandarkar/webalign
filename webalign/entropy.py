"""Web Epistemic Entropy.

Surface disagreement over sources conflates three distinct things: genuine
contradiction, cross-context heterogeneity (different entities or times), and
formatting variation. We decompose it accordingly:

  * h_raw     : entropy over surface values, ignoring context. Rises on ANY
                surface difference (units, dates, entities, contradictions).
  * h_within  : contradiction. Weighted within-context value entropy
                h_within = sum_g p(g) H(V | G=g), where G is the (entity, time)
                context with source-support distribution p(g) and V is the
                normalized value. Legitimate variation across contexts and
                unit/format differences collapse; only real within-context
                contradiction remains (H_within in the paper).
  * h_between  : ambiguity / heterogeneity. Entropy H(G) over the (entity, time)
                contexts themselves, weighted by distinct sources.

delta_apparent = h_raw - h_within is the apparent (non-contradictory) disagreement:
the part removed by normalization plus the part explained by context, I(V; G).
Gold values are computed from the constructed ground-truth contexts; an agent's
estimate is computed from what it actually extracts, so the two are distinct.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from data.schema import Claim
from webalign.normalize import normalize_value


def entropy_from_clusters(cluster_source_counts: Iterable[int]) -> float:
    counts = [c for c in cluster_source_counts if c > 0]
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts)


def normalized_entropy(cluster_source_counts: Iterable[int]) -> float:
    """Entropy scaled to [0,1] by log(k). 0 = consensus, 1 = maximal split."""
    counts = [c for c in cluster_source_counts if c > 0]
    k = len(counts)
    if k <= 1:
        return 0.0
    return entropy_from_clusters(counts) / math.log(k)


def _ensure_norm(claims: list[Claim]) -> None:
    for c in claims:
        if c.value_norm is None:
            c.value_norm = normalize_value(c.value)


def _value_cluster_counts(claims: list[Claim]) -> list[int]:
    """Distinct-source count per equivalent-value cluster (tolerance-aware)."""
    from webalign.consistency import values_equivalent
    clusters: list[tuple[object, set[str]]] = []
    for c in claims:
        for key, srcs in clusters:
            if values_equivalent(c.value_norm, key):
                srcs.add(c.source_id)
                break
        else:
            clusters.append((c.value_norm, {c.source_id}))
    return [len(s) for _, s in clusters]


def _context_key(c: Claim):
    """Legitimate-variation context: which entity, at which time."""
    return (c.subject_qid or c.subject.lower(), c.qualifier.time or "")


def _contexts(claims: list[Claim]) -> dict[tuple, list[Claim]]:
    _ensure_norm(claims)
    contexts: dict[tuple, list[Claim]] = defaultdict(list)
    for c in claims:
        contexts[_context_key(c)].append(c)
    return contexts


def _best_context(claims: list[Claim]) -> list[Claim]:
    return max(_contexts(claims).values(), key=lambda cl: len({c.source_id for c in cl}))


def h_raw(claims: list[Claim]) -> float:
    if not claims:
        return 0.0
    counts: dict[str, set[str]] = defaultdict(set)
    for c in claims:
        counts[str(c.value).strip().lower()].add(c.source_id)
    return entropy_from_clusters(len(s) for s in counts.values())


def h_within(claims: list[Claim]) -> float:
    """Contradiction: weighted within-context value entropy sum_g p(g) H(V|G=g)."""
    if not claims:
        return 0.0
    contexts = _contexts(claims)
    total = sum(len({c.source_id for c in cl}) for cl in contexts.values())
    if total == 0:
        return 0.0
    return sum((len({c.source_id for c in cl}) / total)
               * entropy_from_clusters(_value_cluster_counts(cl))
               for cl in contexts.values())


def h_between(claims: list[Claim]) -> float:
    """Ambiguity: entropy over (entity,time) contexts, weighted by distinct sources."""
    if not claims:
        return 0.0
    _ensure_norm(claims)
    ctx: dict[tuple, set[str]] = defaultdict(set)
    for c in claims:
        ctx[_context_key(c)].add(c.source_id)
    return entropy_from_clusters(len(s) for s in ctx.values())


# Backward-compatible alias: h_true is the paper's H_within.
def h_true(claims: list[Claim]) -> float:
    return h_within(claims)


def h_true_normalized(claims: list[Claim]) -> float:
    if not claims:
        return 0.0
    contexts = _contexts(claims)
    total = sum(len({c.source_id for c in cl}) for cl in contexts.values())
    if total == 0:
        return 0.0
    return sum((len({c.source_id for c in cl}) / total)
               * normalized_entropy(_value_cluster_counts(cl))
               for cl in contexts.values())


def fragmentation(claims: list[Claim]) -> dict[str, float]:
    hr, hw, hb = h_raw(claims), h_within(claims), h_between(claims)
    return {"h_raw": hr, "h_true": hw, "h_within": hw, "h_between": hb,
            "delta_apparent": hr - hw, "h_true_norm": h_true_normalized(claims)}

