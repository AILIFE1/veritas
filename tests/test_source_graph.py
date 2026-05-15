"""Tests for source-graph modeling (upstream ID tracking, citation laundering prevention)."""
import os, tempfile
import pytest

from veritas import VeritasDB, calculate_confidence
from veritas.models import Claim, Source, SourceType, Stance
from veritas.engine import _source_overlap, _graph_independences, UPSTREAM_OVERLAP_PENALTY


def _db():
    path = os.path.join(tempfile.gettempdir(), f"veritas_graph_{os.getpid()}.db")
    if os.path.exists(path):
        os.remove(path)
    return VeritasDB(path), path


def _src(citation, weight=0.85, upstreams=None, is_primary=True):
    return Source(
        citation=citation,
        weight=weight,
        stance=Stance.SUPPORTS,
        source_type=SourceType.EMPIRICAL,
        upstream_ids=upstreams or [],
        is_primary=is_primary,
    )


# ---------------------------------------------------------------------------
# _source_overlap unit tests
# ---------------------------------------------------------------------------

class TestSourceOverlap:
    def test_identical_upstreams(self):
        a = _src("A", upstreams=["paper_x"])
        b = _src("B", upstreams=["paper_x"])
        assert _source_overlap(a, b) == 1.0

    def test_no_upstreams(self):
        a = _src("A")
        b = _src("B")
        assert _source_overlap(a, b) == 0.0

    def test_one_has_no_upstreams(self):
        a = _src("A", upstreams=["paper_x"])
        b = _src("B")
        assert _source_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = _src("A", upstreams=["paper_x", "paper_y"])
        b = _src("B", upstreams=["paper_x", "paper_z"])
        # intersection={x}, union={x,y,z} → 1/3
        assert abs(_source_overlap(a, b) - 1/3) < 1e-9

    def test_no_overlap(self):
        a = _src("A", upstreams=["paper_x"])
        b = _src("B", upstreams=["paper_y"])
        assert _source_overlap(a, b) == 0.0


# ---------------------------------------------------------------------------
# _graph_independences unit tests
# ---------------------------------------------------------------------------

class TestGraphIndependences:
    def test_no_upstreams_unchanged(self):
        sources = [_src("A"), _src("B")]
        inds = _graph_independences(sources)
        assert inds == [1.0, 1.0]

    def test_full_overlap_reduces_independence(self):
        a = _src("A", upstreams=["paper_x"])
        b = _src("B", upstreams=["paper_x"])
        inds = _graph_independences([a, b])
        expected = 1.0 * (1.0 - UPSTREAM_OVERLAP_PENALTY * 1.0)
        assert abs(inds[0] - expected) < 1e-9
        assert abs(inds[1] - expected) < 1e-9

    def test_no_overlap_independence_unchanged(self):
        a = _src("A", upstreams=["paper_x"])
        b = _src("B", upstreams=["paper_y"])
        inds = _graph_independences([a, b])
        assert inds == [1.0, 1.0]

    def test_single_source_with_upstream_unchanged(self):
        a = _src("A", upstreams=["paper_x"])
        inds = _graph_independences([a])
        assert inds == [1.0]

    def test_mixed_pool_only_overlapping_reduced(self):
        # a and b share upstream; c has different upstream
        a = _src("A", upstreams=["paper_x"])
        b = _src("B", upstreams=["paper_x"])
        c = _src("C", upstreams=["paper_y"])
        inds = _graph_independences([a, b, c])
        expected_ab = 1.0 * (1.0 - UPSTREAM_OVERLAP_PENALTY * 1.0)
        assert abs(inds[0] - expected_ab) < 1e-9
        assert abs(inds[1] - expected_ab) < 1e-9
        assert inds[2] == 1.0  # c not penalised

    def test_respects_manual_independence(self):
        a = Source(citation="A", weight=0.9, stance=Stance.SUPPORTS,
                   source_type=SourceType.EMPIRICAL, independence=0.5,
                   upstream_ids=["paper_x"])
        b = Source(citation="B", weight=0.9, stance=Stance.SUPPORTS,
                   source_type=SourceType.EMPIRICAL, independence=0.5,
                   upstream_ids=["paper_x"])
        inds = _graph_independences([a, b])
        expected = 0.5 * (1.0 - UPSTREAM_OVERLAP_PENALTY * 1.0)
        assert abs(inds[0] - expected) < 1e-9


# ---------------------------------------------------------------------------
# calculate_confidence with graph adjustment
# ---------------------------------------------------------------------------

class TestConfidenceWithGraph:
    def test_shared_upstream_reduces_confidence_vs_independent(self):
        """Two sources citing the same paper should give less confidence than two independent sources."""
        w = 0.85
        # Independent sources (no upstreams declared)
        independent = [_src("A", w), _src("B", w)]
        cv_ind = calculate_confidence(independent)

        # Same-weight sources sharing an upstream
        correlated = [_src("A", w, upstreams=["paper_x"]),
                      _src("B", w, upstreams=["paper_x"])]
        cv_cor = calculate_confidence(correlated)

        assert cv_cor.value < cv_ind.value
        assert cv_cor.upstream_graph_applied is True
        assert cv_ind.upstream_graph_applied is False

    def test_no_upstreams_graph_applied_false(self):
        sources = [_src("A"), _src("B")]
        cv = calculate_confidence(sources)
        assert cv.upstream_graph_applied is False

    def test_n_copies_same_upstream_approaches_single_source(self):
        """N copies all citing the same paper shouldn't compound toward certainty."""
        w = 0.90
        single = calculate_confidence([_src("A", w)])
        copies = calculate_confidence([
            _src(f"copy_{i}", w, upstreams=["paper_x"]) for i in range(5)
        ])
        # Copies should not massively exceed single source
        assert copies.value < single.value * 1.3  # some compounding is OK, just not 5x

    def test_different_upstreams_stay_independent(self):
        """Sources with different upstreams should compound normally."""
        w = 0.80
        diverse = [_src("A", w, upstreams=["paper_x"]),
                   _src("B", w, upstreams=["paper_y"]),
                   _src("C", w, upstreams=["paper_z"])]
        cv_diverse = calculate_confidence(diverse)

        no_upstreams = [_src("A", w), _src("B", w), _src("C", w)]
        cv_plain = calculate_confidence(no_upstreams)

        # Diverse upstreams should behave like independent sources
        assert abs(cv_diverse.value - cv_plain.value) < 0.01

    def test_partial_overlap_intermediate_penalty(self):
        """Partial upstream overlap → intermediate confidence between full overlap and independence."""
        w = 0.85
        full_overlap = calculate_confidence([
            _src("A", w, upstreams=["paper_x"]),
            _src("B", w, upstreams=["paper_x"]),
        ])
        partial_overlap = calculate_confidence([
            _src("A", w, upstreams=["paper_x", "paper_y"]),
            _src("B", w, upstreams=["paper_x", "paper_z"]),
        ])
        independent = calculate_confidence([_src("A", w), _src("B", w)])

        assert full_overlap.value <= partial_overlap.value <= independent.value


# ---------------------------------------------------------------------------
# is_primary field persisted and loaded
# ---------------------------------------------------------------------------

class TestIsPrimary:
    def test_primary_default(self):
        db, path = _db()
        claim = Claim(statement="test", sources=[_src("original")])
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded.sources[0].is_primary is True
        os.remove(path)

    def test_derivative_persisted(self):
        db, path = _db()
        s = Source(citation="blog post citing paper_x", weight=0.6, stance=Stance.SUPPORTS,
                   source_type=SourceType.ANECDOTAL, upstream_ids=["paper_x"],
                   is_primary=False)
        claim = Claim(statement="test", sources=[s])
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded.sources[0].is_primary is False
        assert loaded.sources[0].upstream_ids == ["paper_x"]
        os.remove(path)


# ---------------------------------------------------------------------------
# upstream_ids persisted and loaded
# ---------------------------------------------------------------------------

class TestUpstreamPersistence:
    def test_upstream_ids_roundtrip(self):
        db, path = _db()
        s = _src("study A", upstreams=["doi:10.1234/paper", "doi:10.5678/dataset"])
        claim = Claim(statement="test", sources=[s])
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded.sources[0].upstream_ids == ["doi:10.1234/paper", "doi:10.5678/dataset"]
        os.remove(path)

    def test_empty_upstream_ids_roundtrip(self):
        db, path = _db()
        claim = Claim(statement="test", sources=[_src("original")])
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded.sources[0].upstream_ids == []
        os.remove(path)

    def test_migration_existing_db(self):
        """DB without upstream_ids/is_primary columns should migrate cleanly."""
        import json, sqlite3
        path = os.path.join(tempfile.gettempdir(), f"veritas_migrate2_{os.getpid()}.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE claims (id TEXT PRIMARY KEY, statement TEXT NOT NULL, "
            "context TEXT, added_at TEXT NOT NULL, probe_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE sources (id TEXT PRIMARY KEY, claim_id TEXT, citation TEXT, "
            "weight REAL, stance TEXT, source_type TEXT, independence REAL, "
            "notes TEXT, source_date TEXT, added_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE provenance (claim_id TEXT, depends_on TEXT, "
            "inference_type TEXT, PRIMARY KEY (claim_id, depends_on))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(claim_id TEXT, model TEXT, vector BLOB, PRIMARY KEY (claim_id, model))"
        )
        conn.commit()
        conn.close()

        db = VeritasDB(path)  # triggers migration
        s = _src("after migration", upstreams=["paper_x"])
        claim = Claim(statement="migrated", sources=[s])
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded.sources[0].upstream_ids == ["paper_x"]
        assert loaded.sources[0].is_primary is True
        os.remove(path)
