from veritas import calculate_confidence
from veritas.models import Source, Stance, SourceType


def make_src(weight, stance=Stance.SUPPORTS, stype=SourceType.EMPIRICAL, independence=1.0):
    return Source(citation="test", weight=weight, stance=stance,
                  source_type=stype, independence=independence)


def test_no_sources_returns_prior():
    cv = calculate_confidence([])
    assert cv.value == 0.5
    assert cv.source_count == 0


def test_single_strong_source_high_confidence():
    cv = calculate_confidence([make_src(0.9)])
    assert cv.value > 0.7
    assert cv.source_count == 1


def test_single_weak_source_near_prior():
    cv = calculate_confidence([make_src(0.1)])
    assert 0.5 < cv.value < 0.6


def test_contradicting_source_reduces_confidence():
    sources = [
        make_src(0.9, Stance.SUPPORTS),
        make_src(0.9, Stance.CONTRADICTS),
    ]
    cv = calculate_confidence(sources)
    assert abs(cv.value - 0.5) < 0.1


def test_multiple_sources_higher_than_single():
    single = calculate_confidence([make_src(0.7)])
    multi  = calculate_confidence([make_src(0.7), make_src(0.7)])
    assert multi.value > single.value


def test_correlated_sources_diminishing_returns():
    independent   = calculate_confidence([make_src(0.7, independence=1.0), make_src(0.7, independence=1.0)])
    correlated    = calculate_confidence([make_src(0.7, independence=0.1), make_src(0.7, independence=0.1)])
    assert independent.value > correlated.value


def test_fragility_single_source():
    cv = calculate_confidence([make_src(0.9)])
    # Removing the only source drops to prior 0.5
    assert cv.fragility > 0.2


def test_fragility_multiple_sources_lower():
    single = calculate_confidence([make_src(0.8)])
    multi  = calculate_confidence([make_src(0.8), make_src(0.8)])
    assert multi.fragility < single.fragility


def test_staleness_penalty_zero_for_fresh_sources():
    cv = calculate_confidence([make_src(0.9)])
    assert cv.staleness_penalty == 0.0
