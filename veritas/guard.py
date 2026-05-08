"""
Veritas Reasoning Guard

Before an AI agent acts on a belief, check whether that belief holds up
epistemically. Returns a structured verdict so agents can decide whether
to proceed, flag uncertainty, or halt.

Usage:
    from veritas import VeritasDB
    from veritas.guard import ReasoningGuard

    db = VeritasDB("~/.veritas/veritas.db")
    guard = ReasoningGuard(db)

    result = guard.check("persistent memory improves agent reliability")
    if result.should_proceed:
        # act
    else:
        print(result.reason)
"""
from dataclasses import dataclass, field
from typing import Optional

from .db import VeritasDB
from .engine import calculate_confidence, propagate, find_contradictions
from .models import Claim


# Thresholds — tune these per use case
HALT_CONFIDENCE    = 0.30   # below this: don't act
CAUTION_CONFIDENCE = 0.55   # below this: flag uncertainty
FRAGILITY_CAUTION  = 0.25   # drop on best-source removal: warn
STALENESS_CAUTION  = 0.08   # confidence lost to age: warn
CONTRA_HALT_CONF   = 0.45   # a contradiction this confident triggers CAUTION


@dataclass
class GuardResult:
    verdict: str                              # PROCEED | CAUTION | HALT
    confidence: float                         # propagated confidence, 0-1
    reason: str                               # human-readable explanation
    claim: Optional[Claim] = None
    contradictions: list[Claim] = field(default_factory=list)
    is_fragile: bool = False
    is_stale: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def should_proceed(self) -> bool:
        return self.verdict == "PROCEED"

    def __str__(self) -> str:
        lines = [f"[{self.verdict}] confidence={self.confidence:.2f}  {self.reason}"]
        for flag in self.flags:
            lines.append(f"  * {flag}")
        if self.contradictions:
            lines.append(f"  * {len(self.contradictions)} contradiction(s) found")
            for c in self.contradictions[:2]:
                cv = calculate_confidence(c.sources)
                lines.append(f"    - [{cv.value:.2f}] {c.statement[:60]}")
        return "\n".join(lines)


class ReasoningGuard:
    def __init__(self, db: VeritasDB):
        self._db = db

    def check(self, claim_text: str) -> GuardResult:
        """
        Check a belief before acting on it.

        Searches the DB for matching claims. If found, evaluates using full
        propagation, fragility, staleness, and contradiction detection.
        If not found, returns CAUTION — unknown claims shouldn't be acted on
        without registering them first.
        """
        matches = self._db.search(claim_text)
        if not matches:
            return GuardResult(
                verdict="CAUTION",
                confidence=0.5,
                reason=f"Claim not in database — unregistered belief",
                flags=["Add this claim with sources before acting on it"],
            )

        claim = matches[0]
        all_by_id = self._db.all_claims_by_id()
        cv = propagate(claim, all_by_id)
        contras = find_contradictions(claim, list(all_by_id.values()), db=self._db)

        flags = []
        verdict = "PROCEED"

        # Confidence checks
        if cv.value < HALT_CONFIDENCE:
            verdict = "HALT"
        elif cv.value < CAUTION_CONFIDENCE:
            verdict = "CAUTION"
            flags.append(f"Low confidence ({cv.value:.2f} < {CAUTION_CONFIDENCE})")

        # Single source — always fragile, regardless of calculated fragility score
        if cv.source_count == 1:
            if verdict == "PROCEED":
                verdict = "CAUTION"
            flags.append("Single source — needs independent verification")

        # Multi-source fragility
        elif cv.fragility > FRAGILITY_CAUTION:
            if verdict == "PROCEED":
                verdict = "CAUTION"
            flags.append(
                f"Fragile — confidence drops {cv.fragility:.2f} if best source removed"
            )

        # Staleness
        if cv.staleness_penalty > STALENESS_CAUTION:
            if verdict == "PROCEED":
                verdict = "CAUTION"
            flags.append(
                f"Stale — evidence aging has reduced confidence by {cv.staleness_penalty:.2f}"
            )

        # High-confidence contradictions
        strong_contras = []
        for c in contras:
            c_cv = calculate_confidence(c.sources)
            if c_cv.value >= CONTRA_HALT_CONF:
                strong_contras.append(c)
        if strong_contras:
            if verdict == "PROCEED":
                verdict = "CAUTION"
            flags.append(
                f"{len(strong_contras)} well-sourced contradiction(s) in database"
            )

        # Build reason string
        if verdict == "PROCEED":
            reason = f"Belief is well-supported (confidence {cv.value:.2f})"
        elif verdict == "CAUTION":
            reason = f"Belief has weaknesses that should be acknowledged"
        else:
            reason = f"Confidence too low to act ({cv.value:.2f} < {HALT_CONFIDENCE})"

        return GuardResult(
            verdict=verdict,
            confidence=cv.value,
            reason=reason,
            claim=claim,
            contradictions=contras,
            is_fragile=cv.fragility > FRAGILITY_CAUTION,
            is_stale=cv.staleness_penalty > STALENESS_CAUTION,
            flags=flags,
        )

    def check_all(self, claim_texts: list[str]) -> dict[str, GuardResult]:
        """Check multiple claims at once. Returns dict keyed by claim text."""
        return {text: self.check(text) for text in claim_texts}

    def weakest_premise(self, claim_texts: list[str]) -> Optional[str]:
        """
        Given a list of claims that together support a conclusion,
        return the weakest one — the most likely point of failure.
        """
        if not claim_texts:
            return None
        results = self.check_all(claim_texts)
        return min(results.keys(), key=lambda t: results[t].confidence)
