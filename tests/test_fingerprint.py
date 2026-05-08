from veritas import VeritasDB, compute_fingerprint
from veritas.models import Claim, Source, Stance, SourceType


def _src(w, stype=SourceType.EMPIRICAL):
    return Source(citation="test", weight=w, stance=Stance.SUPPORTS, source_type=stype)


def test_empty_database(db):
    fp = compute_fingerprint(db)
    assert fp.total_claims == 0


def test_claim_count(db):
    for i in range(3):
        db.add_claim(Claim(f"Claim {i}", sources=[_src(0.8)]))
    fp = compute_fingerprint(db)
    assert fp.total_claims == 3


def test_single_source_rate(db):
    db.add_claim(Claim("A", sources=[_src(0.8)]))
    db.add_claim(Claim("B", sources=[_src(0.8), _src(0.7)]))
    fp = compute_fingerprint(db)
    assert fp.single_source_rate == 0.5


def test_source_type_distribution_sums_to_one(db):
    db.add_claim(Claim("A", sources=[_src(0.8, SourceType.EMPIRICAL), _src(0.7, SourceType.AUTHORITY)]))
    fp = compute_fingerprint(db)
    assert abs(sum(fp.source_type_dist.values()) - 1.0) < 0.01


def test_rigor_improves_with_more_sources(db):
    db.add_claim(Claim("sparse", context="a", sources=[_src(0.8)]))
    db.add_claim(Claim("rich", context="b",
                       sources=[_src(0.8), _src(0.75), _src(0.85, SourceType.AUTHORITY)]))
    fp_a = compute_fingerprint(db, context="a")
    fp_b = compute_fingerprint(db, context="b")
    assert fp_b.rigor_score > fp_a.rigor_score


def test_context_filter(db):
    db.add_claim(Claim("A", context="science", sources=[_src(0.9)]))
    db.add_claim(Claim("B", context="science", sources=[_src(0.9)]))
    db.add_claim(Claim("C", context="other",   sources=[_src(0.9)]))
    fp = compute_fingerprint(db, context="science")
    assert fp.total_claims == 2


def test_overconfidence_detected(db):
    # High confidence + high fragility = overconfident
    # Single source with high weight but no corroboration
    db.add_claim(Claim("Very sure about this",
                       sources=[_src(0.99, SourceType.ANECDOTAL)]))
    fp = compute_fingerprint(db)
    assert fp.overconfidence_rate > 0
