from veritas import VeritasDB, ReasoningGuard
from veritas.models import Claim, Source, Stance, SourceType


def _src(w, stype=SourceType.EMPIRICAL, stance=Stance.SUPPORTS):
    return Source(citation="test", weight=w, stance=stance, source_type=stype)


def test_proceed_for_well_sourced(db):
    db.add_claim(Claim("X is true", sources=[_src(0.9), _src(0.85)]))
    result = ReasoningGuard(db).check("X is true")
    assert result.verdict == "PROCEED"
    assert result.should_proceed


def test_caution_for_weak_source(db):
    # weight=0.1 gives confidence ~0.55 (weak support, not contradicted)
    # Should be CAUTION due to single-source and low confidence
    db.add_claim(Claim("Y is true", sources=[_src(0.1)]))
    result = ReasoningGuard(db).check("Y is true")
    assert result.verdict in ("HALT", "CAUTION")
    assert result.confidence < 0.65


def test_caution_for_unknown_claim(db):
    result = ReasoningGuard(db).check("This claim is not in the database at all")
    assert result.verdict == "CAUTION"
    assert not result.should_proceed


def test_caution_for_single_source(db):
    db.add_claim(Claim("Z is true", sources=[_src(0.7, SourceType.ANECDOTAL)]))
    result = ReasoningGuard(db).check("Z is true")
    assert result.verdict == "CAUTION"
    assert result.is_fragile or any("single" in f.lower() for f in result.flags)


def test_flags_populated_on_caution(db):
    db.add_claim(Claim("A is true", sources=[_src(0.4)]))
    result = ReasoningGuard(db).check("A is true")
    assert len(result.flags) > 0


def test_weakest_premise(db):
    db.add_claim(Claim("Strong claim", sources=[_src(0.95), _src(0.9)]))
    db.add_claim(Claim("Weak claim",   sources=[_src(0.2)]))
    guard = ReasoningGuard(db)
    weakest = guard.weakest_premise(["Strong claim", "Weak claim"])
    assert weakest == "Weak claim"


def test_check_all_returns_dict(db):
    db.add_claim(Claim("P is true", sources=[_src(0.9), _src(0.85)]))
    db.add_claim(Claim("Q is true", sources=[_src(0.3)]))
    guard = ReasoningGuard(db)
    results = guard.check_all(["P is true", "Q is true"])
    assert len(results) == 2
    assert "P is true" in results
    assert "Q is true" in results
