from veritas import VeritasDB, propagate, calculate_confidence
from veritas.models import Claim, Source, Stance, SourceType, InferenceType


def _src(weight, stance=Stance.SUPPORTS):
    return Source(citation="test", weight=weight, stance=stance)


def test_no_dependencies_equals_direct_confidence(db):
    claim = Claim(statement="A", sources=[_src(0.9)])
    db.add_claim(claim)
    direct = calculate_confidence(claim.sources)
    propagated = propagate(db.get_claim(claim.id), db.all_claims_by_id())
    assert direct.value == propagated.value


def test_deductive_dependency_caps_confidence(db):
    # Foundation with weight=0.3 has confidence ~0.65 (0.5 + 0.5*0.3)
    # Deductive cap means dependent cannot exceed foundation confidence
    foundation = Claim(statement="Foundation", sources=[_src(0.3)])
    dependent  = Claim(statement="Dependent",  sources=[_src(0.95)])
    db.add_claim(foundation)
    db.add_claim(dependent)
    db.link_claims(dependent.id, foundation.id, InferenceType.DEDUCTIVE)

    cv     = propagate(db.get_claim(dependent.id), db.all_claims_by_id())
    direct = calculate_confidence(dependent.sources)
    foundation_cv = calculate_confidence(foundation.sources)
    assert cv.value < direct.value                     # capped below direct
    assert cv.value <= foundation_cv.value + 0.05      # capped near foundation


def test_inductive_dependency_pulls_down(db):
    # Foundation with strong contradiction has confidence < 0.5 — pulls claim down
    weak_foundation = Claim(statement="Weak", sources=[
        _src(0.2, Stance.SUPPORTS),
        _src(0.8, Stance.CONTRADICTS),
    ])
    strong_claim = Claim(statement="Strong", sources=[_src(0.95)])
    db.add_claim(weak_foundation)
    db.add_claim(strong_claim)
    db.link_claims(strong_claim.id, weak_foundation.id, InferenceType.INDUCTIVE)

    cv     = propagate(db.get_claim(strong_claim.id), db.all_claims_by_id())
    direct = calculate_confidence(strong_claim.sources)
    assert cv.value < direct.value


def test_cycle_returns_prior(db):
    a = Claim(statement="A", sources=[_src(0.8)])
    b = Claim(statement="B", sources=[_src(0.8)])
    db.add_claim(a)
    db.add_claim(b)
    db.link_claims(a.id, b.id)
    db.link_claims(b.id, a.id)
    # Should not hang or crash
    cv = propagate(db.get_claim(a.id), db.all_claims_by_id())
    assert 0.0 <= cv.value <= 1.0


def test_weakening_foundation_propagates_upward(db):
    bottom = Claim(statement="Bottom", sources=[_src(0.95), _src(0.90)])
    top    = Claim(statement="Top",    sources=[_src(0.85)])
    db.add_claim(bottom)
    db.add_claim(top)
    db.link_claims(top.id, bottom.id)

    cv_before = propagate(db.get_claim(top.id), db.all_claims_by_id())

    db.add_source(bottom.id, Source(
        citation="Contradicting evidence", weight=0.8, stance=Stance.CONTRADICTS
    ))
    cv_after = propagate(db.get_claim(top.id), db.all_claims_by_id())

    assert cv_after.value < cv_before.value
