from datetime import datetime, timedelta
from veritas.models import Source, Stance, SourceType
from veritas import calculate_confidence


def _src(weight, stype, years_ago):
    date = datetime.utcnow() - timedelta(days=365.25 * years_ago)
    return Source(citation="test", weight=weight, stance=Stance.SUPPORTS,
                  source_type=stype, source_date=date)


def test_fresh_source_minimal_decay():
    src = _src(0.9, SourceType.EMPIRICAL, 0.1)
    assert src.effective_weight() > 0.85


def test_old_anecdotal_decays_significantly():
    src = _src(0.9, SourceType.ANECDOTAL, 5)
    assert src.effective_weight() < 0.3


def test_mathematical_never_decays():
    src = _src(0.99, SourceType.MATHEMATICAL, 90)
    assert src.effective_weight() == 0.99


def test_theoretical_decays_slowly():
    src_new = _src(0.9, SourceType.THEORETICAL, 1)
    src_old = _src(0.9, SourceType.THEORETICAL, 50)
    assert src_old.effective_weight() > 0.7  # still meaningful after 50 years


def test_floor_prevents_zero_weight():
    src = _src(0.8, SourceType.ANECDOTAL, 100)
    assert src.effective_weight() >= src.weight * 0.1


def test_staleness_penalty_increases_with_age():
    fresh = [_src(0.9, SourceType.EMPIRICAL, 1)]
    old   = [_src(0.9, SourceType.EMPIRICAL, 15)]
    cv_fresh = calculate_confidence(fresh)
    cv_old   = calculate_confidence(old)
    assert cv_old.staleness_penalty > cv_fresh.staleness_penalty


def test_staleness_penalty_positive_when_sources_old():
    old_sources = [_src(0.9, SourceType.EMPIRICAL, 20)]
    cv = calculate_confidence(old_sources)
    assert cv.staleness_penalty > 0


def test_decay_rate_ordering():
    # Anecdotal should decay faster than authority, which decays faster than theoretical
    years = 5
    anec  = _src(1.0, SourceType.ANECDOTAL,   years).effective_weight()
    auth  = _src(1.0, SourceType.AUTHORITY,   years).effective_weight()
    theo  = _src(1.0, SourceType.THEORETICAL, years).effective_weight()
    math  = _src(1.0, SourceType.MATHEMATICAL,years).effective_weight()
    assert anec < auth < theo < math
