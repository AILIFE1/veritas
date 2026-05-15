from functools import reduce
from .models import Claim, Source, Stance, ConfidenceVector, InferenceType

# How strongly shared upstreams reduce effective independence.
# 1.0 = fully correlated sources contribute nothing beyond the best one.
# 0.8 = at full upstream overlap, independence is reduced by 80%.
UPSTREAM_OVERLAP_PENALTY = 0.8


def _source_overlap(a: Source, b: Source) -> float:
    """Jaccard similarity of two sources' upstream ID sets. 0 if either has no upstreams."""
    if not a.upstream_ids or not b.upstream_ids:
        return 0.0
    sa, sb = set(a.upstream_ids), set(b.upstream_ids)
    return len(sa & sb) / len(sa | sb)


def _graph_independences(sources: list[Source]) -> list[float]:
    """
    Return effective independence for each source after adjusting for shared upstreams.

    Sources that share upstream IDs draw from the same well. The more upstream
    overlap between two sources, the less independently each contributes to the
    noisy-OR pool. This prevents citation laundering: N blog posts all citing
    the same paper should not compound to near-certainty.

    Adjustment: for each source S, find its maximum Jaccard overlap with any
    other source. Reduce independence by (overlap * UPSTREAM_OVERLAP_PENALTY).
    """
    if not any(s.upstream_ids for s in sources):
        return [s.independence for s in sources]

    result = []
    for i, s in enumerate(sources):
        if not s.upstream_ids:
            result.append(s.independence)
            continue
        max_overlap = max(
            (_source_overlap(s, other) for j, other in enumerate(sources) if j != i),
            default=0.0,
        )
        adjusted = s.independence * (1.0 - UPSTREAM_OVERLAP_PENALTY * max_overlap)
        result.append(max(0.0, adjusted))
    return result


def _evidence_mass(sources: list[Source], use_decay: bool = True) -> float:
    """
    Noisy-OR pooling: each independent source reduces remaining uncertainty.
    When use_decay=True, source weights are reduced by temporal decay first.
    Effective independence is adjusted for shared upstream sources.
    """
    if not sources:
        return 0.0
    independences = _graph_independences(sources)

    def eff(s: Source, ind: float) -> float:
        return (s.effective_weight() if use_decay else s.weight) * ind

    return 1.0 - reduce(
        lambda acc, pair: acc * (1.0 - eff(pair[0], pair[1])),
        zip(sources, independences),
        1.0,
    )


def calculate_confidence(sources: list[Source]) -> ConfidenceVector:
    """Confidence from direct sources with temporal decay applied."""
    supporting = [s for s in sources if s.stance == Stance.SUPPORTS]
    contradicting = [s for s in sources if s.stance == Stance.CONTRADICTS]

    # Current (decayed) confidence
    sup_mass = _evidence_mass(supporting, use_decay=True)
    con_mass = _evidence_mass(contradicting, use_decay=True)
    total_mass = sup_mass + con_mass

    if total_mass == 0.0:
        value = 0.5
    else:
        raw = sup_mass / total_mass
        value = 0.5 + (raw - 0.5) * min(1.0, total_mass)

    # Undecayed confidence — diff reveals how much age has cost
    sup_mass_raw = _evidence_mass(supporting, use_decay=False)
    con_mass_raw = _evidence_mass(contradicting, use_decay=False)
    total_raw = sup_mass_raw + con_mass_raw
    if total_raw == 0.0:
        value_raw = 0.5
    else:
        raw2 = sup_mass_raw / total_raw
        value_raw = 0.5 + (raw2 - 0.5) * min(1.0, total_raw)

    staleness_penalty = max(0.0, value_raw - value)
    fragility = _fragility(sources, value)
    diversity = _source_diversity(supporting)
    graph_applied = any(s.upstream_ids for s in sources)

    return ConfidenceVector(
        value=round(value, 4),
        source_count=len(sources),
        support_count=len(supporting),
        contra_count=len(contradicting),
        fragility=round(fragility, 4),
        source_diversity=round(diversity, 4),
        staleness_penalty=round(staleness_penalty, 4),
        upstream_graph_applied=graph_applied,
    )


def propagate(
    claim: Claim,
    all_claims: dict[str, Claim],
    _visited: frozenset | None = None,
) -> ConfidenceVector:
    """
    Confidence with belief propagation.

    Recursively follows provenance links. Each dependency modifies
    the claim's base confidence according to inference type:
      DEDUCTIVE — hard cap: can't exceed dependency's confidence
      INDUCTIVE — medium drag: weak foundation pulls confidence down
      ABDUCTIVE — soft drag: weaker version of inductive

    Cycles return the prior (0.5) to break the loop cleanly.
    """
    if _visited is None:
        _visited = frozenset()

    base_cv = calculate_confidence(claim.sources)

    if not claim.depends_on:
        return base_cv

    if claim.id in _visited:
        return ConfidenceVector(
            value=0.5, source_count=0, support_count=0,
            contra_count=0, fragility=0.0, source_diversity=0.0,
        )

    _visited = _visited | {claim.id}

    value = base_cv.value
    for link in claim.depends_on:
        dep = all_claims.get(link.depends_on_id)
        if not dep:
            continue
        dep_cv = propagate(dep, all_claims, _visited)
        value = _apply_dependency(value, dep_cv.value, link.inference_type)

    return ConfidenceVector(
        value=round(max(0.0, min(1.0, value)), 4),
        source_count=base_cv.source_count,
        support_count=base_cv.support_count,
        contra_count=base_cv.contra_count,
        fragility=base_cv.fragility,
        source_diversity=base_cv.source_diversity,
    )


# How strongly each inference type lets a dependency drag down confidence.
# Asymmetric by design: a weak foundation hurts more than a strong one helps.
_DAMPENING = {
    InferenceType.INDUCTIVE: 0.5,
    InferenceType.ABDUCTIVE: 0.25,
}


def _apply_dependency(base: float, dep: float, inference: InferenceType) -> float:
    if inference == InferenceType.DEDUCTIVE:
        return min(base, dep)
    w = _DAMPENING.get(inference, 0.25)
    if dep < 0.5:
        return max(0.0, base - (0.5 - dep) * w)
    else:
        return min(1.0, base + (dep - 0.5) * w * 0.3)


def _fragility(sources: list[Source], current_value: float) -> float:
    if not sources:
        return 0.0
    max_drop = 0.0
    for s in sources:
        remaining = [x for x in sources if x is not s]
        drop = current_value - calculate_confidence(remaining).value
        if drop > max_drop:
            max_drop = drop
    return max_drop


def _source_diversity(sources: list[Source]) -> float:
    if not sources:
        return 0.0
    if len(sources) == 1:
        return sources[0].independence
    return sum(s.independence for s in sources) / len(sources)


def find_contradictions(claim: Claim, all_claims: list[Claim], db=None) -> list[Claim]:
    """
    Contradiction detection — semantic when available, keyword fallback.

    Semantic mode uses sentence embeddings to find claims about the same
    topic, then flags pairs where exactly one asserts and the other negates.
    Catches contradictions that share no words at all.
    """
    try:
        from .semantic import encoder_available, find_contradictions_semantic
        if encoder_available():
            return find_contradictions_semantic(claim, all_claims, db)
    except Exception:
        pass
    return _find_contradictions_keyword(claim, all_claims)


def _find_contradictions_keyword(claim: Claim, all_claims: list[Claim]) -> list[Claim]:
    STOP = {"the", "a", "an", "is", "are", "was", "were", "it",
            "in", "of", "and", "to", "that", "this", "for", "on"}
    NEGATIONS = {"not", "no", "never", "false", "incorrect", "wrong",
                 "impossible", "cannot", "cant", "doesnt", "isnt", "arent"}

    words = set(claim.statement.lower().split()) - STOP
    claim_negated = bool(words & NEGATIONS)
    results = []

    for other in all_claims:
        if other.id == claim.id:
            continue
        other_words = set(other.statement.lower().split()) - STOP
        overlap = words & other_words - NEGATIONS
        if len(overlap) < 2:
            continue
        other_negated = bool(other_words & NEGATIONS)
        if claim_negated != other_negated:
            results.append(other)

    return results
