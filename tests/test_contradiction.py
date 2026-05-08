import pytest
from veritas import VeritasDB, find_contradictions
from veritas.models import Claim, Source, Stance


def _claim(stmt, db):
    c = Claim(statement=stmt, sources=[Source(citation="src", weight=0.7, stance=Stance.SUPPORTS)])
    db.add_claim(c)
    return c


def test_keyword_direct_negation(db):
    a = _claim("Memory is necessary for agent identity", db)
    b = _claim("Memory is not necessary for agent identity", db)
    all_claims = db.all_claims()
    contras = find_contradictions(a, all_claims)
    assert any(c.id == b.id for c in contras)


def test_no_false_positive_unrelated(db):
    a = _claim("The sky is blue", db)
    b = _claim("Coffee tastes bitter", db)
    all_claims = db.all_claims()
    contras = find_contradictions(a, all_claims)
    assert not any(c.id == b.id for c in contras)


def test_claim_does_not_contradict_itself(db):
    a = _claim("Memory is necessary for identity", db)
    all_claims = db.all_claims()
    contras = find_contradictions(a, all_claims)
    assert not any(c.id == a.id for c in contras)


def test_empty_database_no_contradictions(db):
    a = _claim("Anything at all", db)
    # Only one claim in DB — nothing to contradict
    contras = find_contradictions(a, db.all_claims())
    assert contras == []


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sentence_transformers"),
    reason="sentence-transformers not installed",
)
def test_semantic_catches_no_shared_words(db):
    a = _claim("Physical activity strengthens the cardiovascular system", db)
    b = _claim("Exercise has no proven benefit for heart health", db)
    all_claims = db.all_claims()
    contras = find_contradictions(a, all_claims, db=db)
    assert any(c.id == b.id for c in contras)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sentence_transformers"),
    reason="sentence-transformers not installed",
)
def test_semantic_no_false_positive_related_claims(db):
    a = _claim("The sky appears blue due to Rayleigh scattering", db)
    b = _claim("Sunsets produce red and orange colors from the same effect", db)
    all_claims = db.all_claims()
    contras = find_contradictions(a, all_claims, db=db)
    # Related but not contradictory — should not flag
    assert not any(c.id == b.id for c in contras)
