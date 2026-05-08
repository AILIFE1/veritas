"""
Semantic contradiction detection using sentence embeddings.

Replaces keyword matching with meaning-aware similarity. Two claims
are candidates for contradiction when they are semantically close
(about the same thing) but one asserts while the other negates.

Falls back gracefully to keyword detection if sentence-transformers
is not installed.
"""
import re
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Claim
    from .db import VeritasDB

MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.48   # cosine similarity — above this = same topic

_encoder = None


def encoder_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


def encode(texts: list[str]) -> np.ndarray:
    return get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already normalised — dot product equals cosine similarity
    return float(np.dot(a, b))


def pack_vector(v: np.ndarray) -> bytes:
    return np.array(v, dtype=np.float32).tobytes()


def unpack_vector(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


# ── negation detection ────────────────────────────────────────────────────────
# More thorough than keyword matching: catches "there is no evidence",
# "has been disproved", "cannot be proven", etc.

_NEGATION = re.compile(
    r'\b('
    r'not|no\b|never|cannot|can\'t|doesn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|'
    r'has no|have no|there is no|there are no|'
    r'no evidence|no proof|no support|'
    r'disproved|debunked|refuted|rejected|discredited|'
    r'false|incorrect|wrong|untrue|inaccurate|'
    r'impossible|unlikely|implausible|unfounded'
    r')',
    re.IGNORECASE,
)


def _is_negated(text: str) -> bool:
    return bool(_NEGATION.search(text))


# ── main entry point ──────────────────────────────────────────────────────────

def find_contradictions_semantic(
    claim: "Claim",
    all_claims: list["Claim"],
    db: "VeritasDB | None" = None,
) -> list["Claim"]:
    """
    Find contradictions using semantic similarity + negation detection.

    Steps:
      1. Encode all claims (cached in DB when available).
      2. Find claims with cosine similarity > SIMILARITY_THRESHOLD
         — these are about the same topic.
      3. Among similar pairs, flag those where exactly one is negated.
    """
    others = [c for c in all_claims if c.id != claim.id]
    if not others:
        return []

    claim_vec = _get_embedding(claim, db)
    other_vecs = [_get_embedding(c, db) for c in others]

    claim_negated = _is_negated(claim.statement)
    results = []

    for other, other_vec in zip(others, other_vecs):
        sim = cosine_sim(claim_vec, other_vec)
        if sim < SIMILARITY_THRESHOLD:
            continue
        if claim_negated != _is_negated(other.statement):
            results.append((other, sim))

    # Return sorted by similarity — strongest semantic matches first
    results.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in results]


def similarity_score(a: "Claim", b: "Claim", db: "VeritasDB | None" = None) -> float:
    """Cosine similarity between two claims. Useful for related-claim discovery."""
    vec_a = _get_embedding(a, db)
    vec_b = _get_embedding(b, db)
    return cosine_sim(vec_a, vec_b)


def _get_embedding(claim: "Claim", db: "VeritasDB | None") -> np.ndarray:
    if db is not None:
        cached = db.get_embedding(claim.id, MODEL_NAME)
        if cached is not None:
            return cached
    vec = encode([claim.statement])[0]
    if db is not None:
        db.set_embedding(claim.id, MODEL_NAME, vec)
    return vec
