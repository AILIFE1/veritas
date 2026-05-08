"""
Epistemic fingerprint — the characteristic reasoning style of a belief system.

Every collection of beliefs has a pattern: what kinds of evidence it relies on,
how fresh that evidence is, how fragile its claims are, whether it acknowledges
contradictions. Two agents with identical beliefs but different epistemic styles
are different kinds of reasoners.

This module measures that style.
"""
from dataclasses import dataclass, field
from typing import Optional

from .db import VeritasDB
from .engine import calculate_confidence, find_contradictions
from .models import Claim, InferenceType, SourceType


BAR = 24


@dataclass
class EpistemicFingerprint:
    label: str
    total_claims: int
    total_sources: int

    # How beliefs are built
    source_type_dist: dict[str, float]   # e.g. {"EMPIRICAL": 0.6, "ANECDOTAL": 0.2}
    avg_sources_per_claim: float
    single_source_rate: float            # fraction of claims with exactly 1 source

    # How confident — and whether that confidence is earned
    avg_confidence: float
    avg_fragility: float
    overconfidence_rate: float           # high-confidence + high-fragility claims

    # Evidence freshness
    avg_source_age_years: float
    avg_staleness_penalty: float

    # Contradiction awareness
    contradiction_rate: float            # fraction of claims with known conflicts

    # Inference style (from provenance links)
    inference_dist: dict[str, float]     # e.g. {"INDUCTIVE": 0.7, "DEDUCTIVE": 0.3}

    @property
    def rigor_score(self) -> float:
        """
        How rigorously is evidence gathered?
        High: multiple independent sources, fresh, low single-source rate.
        """
        density   = min(1.0, (self.avg_sources_per_claim - 1) / 2)  # peaks at 3 sources
        freshness = max(0.0, 1.0 - self.avg_staleness_penalty * 6)
        breadth   = 1.0 - self.single_source_rate
        return round((density + freshness + breadth) / 3, 3)

    @property
    def calibration_score(self) -> float:
        """
        Is confidence appropriate given evidence quality?
        Penalises overconfidence (high confidence + high fragility).
        """
        return round(max(0.0, 1.0 - self.overconfidence_rate * 2), 3)

    @property
    def overall_score(self) -> float:
        return round((self.rigor_score + self.calibration_score) / 2, 3)

    def __str__(self) -> str:
        return _render(self)


def compute(db: VeritasDB, context: Optional[str] = None) -> EpistemicFingerprint:
    """Compute the epistemic fingerprint for all claims in a context (or the whole DB)."""
    claims = db.all_claims(context=context)
    if not claims:
        return _empty(context or "all")

    all_claims_list = db.all_claims()
    label = context or "all"

    # ── sources ──────────────────────────────────────────────────────────────
    all_sources = [s for c in claims for s in c.sources]
    total_sources = len(all_sources)

    type_counts: dict[str, int] = {t.value: 0 for t in SourceType}
    total_age = 0.0
    for s in all_sources:
        type_counts[s.source_type.value] = type_counts.get(s.source_type.value, 0) + 1
        total_age += s.age_years()

    source_type_dist = {
        k: round(v / total_sources, 3) if total_sources else 0.0
        for k, v in type_counts.items()
        if v > 0
    }
    avg_sources_per_claim = round(total_sources / len(claims), 2) if claims else 0.0
    avg_source_age = round(total_age / total_sources, 1) if total_sources else 0.0

    # ── confidence + fragility ────────────────────────────────────────────────
    vectors = [(c, calculate_confidence(c.sources)) for c in claims]

    single_source_count = sum(1 for c in claims if len(c.sources) == 1)
    single_source_rate = round(single_source_count / len(claims), 3)

    avg_conf = round(sum(cv.value for _, cv in vectors) / len(vectors), 3)
    avg_frag = round(sum(cv.fragility for _, cv in vectors) / len(vectors), 3)
    avg_stale = round(sum(cv.staleness_penalty for _, cv in vectors) / len(vectors), 3)

    # Overconfidence: confidence > 0.75 AND fragility > 0.3
    overconf_count = sum(
        1 for _, cv in vectors
        if cv.value > 0.75 and cv.fragility > 0.3
    )
    overconfidence_rate = round(overconf_count / len(claims), 3)

    # ── contradictions ────────────────────────────────────────────────────────
    contra_count = sum(
        1 for c in claims
        if find_contradictions(c, all_claims_list, db=db)
    )
    contradiction_rate = round(contra_count / len(claims), 3)

    # ── inference style ───────────────────────────────────────────────────────
    inf_counts: dict[str, int] = {t.value: 0 for t in InferenceType}
    total_links = 0
    for c in claims:
        for link in c.depends_on:
            inf_counts[link.inference_type.value] = inf_counts.get(link.inference_type.value, 0) + 1
            total_links += 1
    inference_dist = {
        k: round(v / total_links, 3)
        for k, v in inf_counts.items()
        if v > 0
    } if total_links else {}

    return EpistemicFingerprint(
        label=label,
        total_claims=len(claims),
        total_sources=total_sources,
        source_type_dist=source_type_dist,
        avg_sources_per_claim=avg_sources_per_claim,
        single_source_rate=single_source_rate,
        avg_confidence=avg_conf,
        avg_fragility=avg_frag,
        overconfidence_rate=overconfidence_rate,
        avg_source_age_years=avg_source_age,
        avg_staleness_penalty=avg_stale,
        contradiction_rate=contradiction_rate,
        inference_dist=inference_dist,
    )


def compare(a: EpistemicFingerprint, b: EpistemicFingerprint) -> str:
    """Side-by-side diff of two fingerprints."""
    lines = [
        f"  {'Metric':<28} {a.label[:18]:<20} {b.label[:18]}",
        f"  {'-'*28} {'-'*18}   {'-'*18}",
    ]

    def row(label, va, vb, fmt=".2f", higher_is="better"):
        a_str = format(va, fmt)
        b_str = format(vb, fmt)
        if higher_is == "better":
            marker = ">" if va > vb else ("<" if va < vb else "=")
        else:
            marker = "<" if va < vb else (">" if va > vb else "=")
        lines.append(f"  {label:<28} {a_str:<20} {b_str}  {marker}")

    row("Claims",            a.total_claims,          b.total_claims,          fmt="d",   higher_is="neutral")
    row("Avg sources/claim", a.avg_sources_per_claim, b.avg_sources_per_claim, fmt=".2f")
    row("Single-source rate",a.single_source_rate,    b.single_source_rate,    fmt=".2f", higher_is="worse")
    row("Avg confidence",    a.avg_confidence,        b.avg_confidence)
    row("Avg fragility",     a.avg_fragility,         b.avg_fragility,         higher_is="worse")
    row("Overconfidence",    a.overconfidence_rate,   b.overconfidence_rate,   higher_is="worse")
    row("Avg source age (y)",a.avg_source_age_years,  b.avg_source_age_years,  higher_is="worse")
    row("Staleness penalty", a.avg_staleness_penalty, b.avg_staleness_penalty, higher_is="worse")
    row("Contradiction rate",a.contradiction_rate,    b.contradiction_rate,    higher_is="worse")
    row("Rigor score",       a.rigor_score,           b.rigor_score)
    row("Calibration score", a.calibration_score,     b.calibration_score)
    row("Overall score",     a.overall_score,         b.overall_score)

    lines.append("")
    lines.append(f"  Key: > means left is higher  < means left is lower  = means equal")
    return "\n".join(lines)


# ── rendering ─────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = BAR) -> str:
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def _render(fp: EpistemicFingerprint) -> str:
    lines = [
        f"",
        f"  Epistemic Fingerprint: {fp.label}",
        f"  {'='*(BAR+30)}",
        f"  Claims: {fp.total_claims}   Sources: {fp.total_sources}"
        f"   Avg sources/claim: {fp.avg_sources_per_claim:.1f}",
        f"",
        f"  Source composition:",
    ]

    type_order = ["MATHEMATICAL", "THEORETICAL", "EMPIRICAL", "AUTHORITY", "META", "ANECDOTAL"]
    for t in type_order:
        frac = fp.source_type_dist.get(t, 0.0)
        if frac == 0:
            continue
        lines.append(f"    {t:<14} [{_bar(frac)}] {frac*100:.0f}%")

    lines += [
        f"",
        f"  Confidence profile:",
        f"    Average        [{_bar(fp.avg_confidence)}] {fp.avg_confidence:.2f}",
        f"    Fragility      [{_bar(fp.avg_fragility)}] {fp.avg_fragility:.2f}",
        f"    Overconfident  [{_bar(fp.overconfidence_rate)}] {fp.overconfidence_rate*100:.0f}% of claims",
        f"    Single-source  [{_bar(fp.single_source_rate)}] {fp.single_source_rate*100:.0f}% of claims",
        f"",
        f"  Evidence freshness:",
        f"    Avg age        {fp.avg_source_age_years:.1f} years",
        f"    Staleness loss [{_bar(min(1.0, fp.avg_staleness_penalty*5))}] -{fp.avg_staleness_penalty:.3f} avg",
        f"",
        f"  Epistemic health:",
        f"    Contradicted   [{_bar(fp.contradiction_rate)}] {fp.contradiction_rate*100:.0f}% of claims",
        f"    Rigor score    [{_bar(fp.rigor_score)}] {fp.rigor_score:.2f}",
        f"    Calibration    [{_bar(fp.calibration_score)}] {fp.calibration_score:.2f}",
        f"    Overall        [{_bar(fp.overall_score)}] {fp.overall_score:.2f}",
    ]

    if fp.inference_dist:
        lines.append(f"")
        lines.append(f"  Inference style:")
        for inf_type, frac in sorted(fp.inference_dist.items(), key=lambda x: -x[1]):
            lines.append(f"    {inf_type:<14} [{_bar(frac)}] {frac*100:.0f}%")

    lines.append("")
    return "\n".join(lines)


def _empty(label: str) -> EpistemicFingerprint:
    return EpistemicFingerprint(
        label=label, total_claims=0, total_sources=0,
        source_type_dist={}, avg_sources_per_claim=0.0,
        single_source_rate=0.0, avg_confidence=0.5,
        avg_fragility=0.0, overconfidence_rate=0.0,
        avg_source_age_years=0.0, avg_staleness_penalty=0.0,
        contradiction_rate=0.0, inference_dist={},
    )
