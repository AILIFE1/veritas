import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Stance(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"


class SourceType(str, Enum):
    MATHEMATICAL = "MATHEMATICAL"  # formal proof / theorem — timeless by definition
    THEORETICAL = "THEORETICAL"   # scientific theory or model
    EMPIRICAL = "EMPIRICAL"        # experimental / observational data
    AUTHORITY = "AUTHORITY"        # expert or institutional consensus
    META = "META"                  # literature review / meta-analysis
    ANECDOTAL = "ANECDOTAL"        # personal account


class InferenceType(str, Enum):
    DEDUCTIVE = "DEDUCTIVE"   # must be true if premises hold
    INDUCTIVE = "INDUCTIVE"   # likely true based on evidence
    ABDUCTIVE = "ABDUCTIVE"   # best explanation given evidence


# Annual decay rates per source type.
# Exponential: effective_weight = weight * e^(-rate * years_old)
# Floor: 10% of original — except MATHEMATICAL which has no floor (proofs don't expire).
DECAY_RATE: dict[SourceType, float] = {
    SourceType.MATHEMATICAL: 0.000,  # timeless — Turing 1936 is as valid today
    SourceType.THEORETICAL:  0.005,  # half-life ~140 yrs  (Newton, Darwin still matter)
    SourceType.EMPIRICAL:    0.07,   # half-life  ~10 yrs  (studies, measurements)
    SourceType.AUTHORITY:    0.12,   # half-life   ~6 yrs  (expert consensus shifts)
    SourceType.META:         0.18,   # half-life   ~4 yrs  (reviews age faster)
    SourceType.ANECDOTAL:    0.35,   # half-life   ~2 yrs  (personal accounts fade)
}


@dataclass
class Source:
    citation: str
    weight: float                         # 0-1, stated reliability at publication
    stance: Stance
    source_type: SourceType = SourceType.EMPIRICAL
    independence: float = 1.0             # 0-1, how independent from other sources
    notes: Optional[str] = None
    source_date: Optional[datetime] = None  # when the source was published (None = added_at)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: Optional[str] = None
    added_at: datetime = field(default_factory=datetime.utcnow)

    def effective_date(self) -> datetime:
        return self.source_date or self.added_at

    def age_years(self, as_of: datetime | None = None) -> float:
        ref = as_of or datetime.utcnow()
        return max(0.0, (ref - self.effective_date()).days / 365.25)

    def effective_weight(self, as_of: datetime | None = None) -> float:
        """Weight after applying temporal decay.
        Floor is 10% of original for most types; MATHEMATICAL has no decay at all."""
        years = self.age_years(as_of)
        rate = DECAY_RATE.get(self.source_type, 0.1)
        if rate == 0.0:
            return self.weight
        decayed = self.weight * math.exp(-rate * years)
        return max(self.weight * 0.1, decayed)


@dataclass
class ConfidenceVector:
    value: float           # 0-1, current best estimate (with decay applied)
    source_count: int
    support_count: int
    contra_count: int
    fragility: float       # confidence drop if best source removed
    source_diversity: float
    staleness_penalty: float = 0.0  # how much decay has reduced confidence

    def __str__(self) -> str:
        bar_len = 20
        filled = int(self.value * bar_len)
        bar = "#" * filled + "." * (bar_len - filled)
        flags = ""
        if self.fragility > 0.3:
            flags += " (!)"
        if self.staleness_penalty > 0.05:
            flags += " [stale]"
        return (
            f"[{bar}] {self.value:.2f}  "
            f"src:{self.source_count}  "
            f"contra:{self.contra_count}  "
            f"fragility:{self.fragility:.2f}{flags}"
        )


@dataclass
class ProvenanceLink:
    depends_on_id: str
    inference_type: InferenceType = InferenceType.INDUCTIVE


@dataclass
class Claim:
    statement: str
    context: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    added_at: datetime = field(default_factory=datetime.utcnow)
    sources: list[Source] = field(default_factory=list)
    depends_on: list[ProvenanceLink] = field(default_factory=list)
    confidence: Optional[ConfidenceVector] = None         # computed by engine, not stored
    probe_id: Optional[str] = None                        # registered probe callable ID
