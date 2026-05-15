"""Tests for the runtime probe primitive."""
import os, tempfile
import pytest

from veritas import VeritasDB, ReasoningGuard
from veritas.models import Claim, Source, SourceType, Stance
from veritas.probe import ProbeRegistry, ProbeResult


def _db():
    path = os.path.join(tempfile.gettempdir(), f"veritas_probe_{os.getpid()}.db")
    if os.path.exists(path):
        os.remove(path)
    return VeritasDB(path), path


def _well_sourced_claim(statement: str, probe_id: str | None = None) -> Claim:
    return Claim(
        statement=statement,
        probe_id=probe_id,
        sources=[
            Source(citation="Source A", weight=0.90, stance=Stance.SUPPORTS,
                   source_type=SourceType.EMPIRICAL),
            Source(citation="Source B", weight=0.85, stance=Stance.SUPPORTS,
                   source_type=SourceType.EMPIRICAL),
        ],
    )


def _single_source_claim(statement: str, probe_id: str | None = None) -> Claim:
    return Claim(
        statement=statement,
        probe_id=probe_id,
        sources=[
            Source(citation="One shaky source", weight=0.60, stance=Stance.SUPPORTS,
                   source_type=SourceType.ANECDOTAL),
        ],
    )


# ---------------------------------------------------------------------------
# ProbeRegistry unit tests
# ---------------------------------------------------------------------------

class TestProbeRegistry:
    def test_register_and_fire(self):
        reg = ProbeRegistry()
        reg.register("ok", lambda: ProbeResult(success=True, confidence=0.95, message="ok"))
        result = reg.fire("ok")
        assert result is not None
        assert result.success
        assert result.confidence == 0.95

    def test_fire_unregistered_returns_none(self):
        reg = ProbeRegistry()
        assert reg.fire("missing") is None

    def test_decorator(self):
        reg = ProbeRegistry()

        @reg.probe("my_probe")
        def _probe() -> ProbeResult:
            return ProbeResult(success=True, confidence=0.80, message="decorated")

        result = reg.fire("my_probe")
        assert result is not None
        assert result.message == "decorated"

    def test_probe_exception_returns_failed_result(self):
        reg = ProbeRegistry()
        reg.register("boom", lambda: (_ for _ in ()).throw(RuntimeError("network error")))
        result = reg.fire("boom")
        assert result is not None
        assert result.success is False
        assert result.confidence == 0.0
        assert "network error" in result.message

    def test_default_singleton(self):
        a = ProbeRegistry.default()
        b = ProbeRegistry.default()
        assert a is b


# ---------------------------------------------------------------------------
# probe_id persisted to / loaded from DB
# ---------------------------------------------------------------------------

class TestProbeIdPersistence:
    def test_probe_id_roundtrip(self):
        db, path = _db()
        claim = _well_sourced_claim("the service is up", probe_id="check_health")
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded is not None
        assert loaded.probe_id == "check_health"
        os.remove(path)

    def test_probe_id_none_roundtrip(self):
        db, path = _db()
        claim = _well_sourced_claim("no probe here")
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded.probe_id is None
        os.remove(path)

    def test_migration_existing_db(self):
        """Adding probe_id column to a DB that predates it should not error."""
        import sqlite3
        path = os.path.join(tempfile.gettempdir(), f"veritas_migrate_{os.getpid()}.db")
        # Create a DB without probe_id column
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE claims (id TEXT PRIMARY KEY, statement TEXT NOT NULL, "
            "context TEXT, added_at TEXT NOT NULL)"
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

        db = VeritasDB(path)  # migration runs here
        claim = _well_sourced_claim("migrated claim", probe_id="p1")
        db.add_claim(claim)
        loaded = db.get_claim(claim.id)
        assert loaded.probe_id == "p1"
        os.remove(path)


# ---------------------------------------------------------------------------
# Guard behaviour with probes
# ---------------------------------------------------------------------------

class TestGuardWithProbe:
    def _setup(self, probe_id, probe_fn):
        db, path = _db()
        reg = ProbeRegistry()
        if probe_fn is not None:
            reg.register(probe_id, probe_fn)
        guard = ReasoningGuard(db, registry=reg)
        return db, guard, path

    # --- probe not loaded in this process ---

    def test_unloaded_probe_adds_flag_no_verdict_change(self):
        db, guard, path = self._setup("missing_probe", None)
        claim = _well_sourced_claim("service is reliable", probe_id="missing_probe")
        db.add_claim(claim)
        result = guard.check("service is reliable")
        assert result.verdict == "PROCEED"   # coherence still good
        assert result.probe_result is None
        assert any("not loaded" in f for f in result.flags)
        os.remove(path)

    # --- probe passes ---

    def test_probe_pass_keeps_proceed(self):
        probe = lambda: ProbeResult(success=True, confidence=0.95, message="200 OK")
        db, guard, path = self._setup("health", probe)
        claim = _well_sourced_claim("service is reliable", probe_id="health")
        db.add_claim(claim)
        result = guard.check("service is reliable")
        assert result.verdict == "PROCEED"
        assert result.probe_fired
        assert result.probe_result.success
        assert any("confirmed" in f for f in result.flags)
        os.remove(path)

    def test_probe_pass_upgrades_caution_to_proceed(self):
        # Single-source → internal CAUTION; probe passes → upgrades to PROCEED
        probe = lambda: ProbeResult(success=True, confidence=0.90, message="200 OK")
        db, guard, path = self._setup("health", probe)
        claim = _single_source_claim("the endpoint returns JSON", probe_id="health")
        db.add_claim(claim)
        result = guard.check("the endpoint returns JSON")
        assert result.verdict == "PROCEED"
        assert result.probe_fired
        os.remove(path)

    def test_probe_pass_does_not_upgrade_halt(self):
        # HALT requires contradicting evidence stronger than supporting.
        # A probe pass should not rescue a belief the internal evidence condemns.
        probe = lambda: ProbeResult(success=True, confidence=0.95, message="200 OK")
        db, guard, path = self._setup("health", probe)
        claim = Claim(
            statement="the service has never had an outage",
            probe_id="health",
            sources=[
                Source(citation="one anecdote", weight=0.20, stance=Stance.SUPPORTS,
                       source_type=SourceType.ANECDOTAL),
                Source(citation="Outage log 2025-01", weight=0.90, stance=Stance.CONTRADICTS,
                       source_type=SourceType.EMPIRICAL),
                Source(citation="Outage log 2025-06", weight=0.85, stance=Stance.CONTRADICTS,
                       source_type=SourceType.EMPIRICAL),
            ],
        )
        db.add_claim(claim)
        result = guard.check("the service has never had an outage")
        assert result.verdict == "HALT"   # internal evidence condemned it; probe can't override
        os.remove(path)

    # --- probe fails ---

    def test_probe_fail_downgrades_proceed_to_caution(self):
        probe = lambda: ProbeResult(success=False, confidence=0.10, message="503 Service Unavailable")
        db, guard, path = self._setup("health", probe)
        claim = _well_sourced_claim("service is reliable", probe_id="health")
        db.add_claim(claim)
        result = guard.check("service is reliable")
        assert result.verdict in ("CAUTION", "HALT")
        assert result.probe_fired
        assert any("failed" in f for f in result.flags)
        os.remove(path)

    def test_probe_very_low_confidence_triggers_halt(self):
        probe = lambda: ProbeResult(success=False, confidence=0.05, message="connection refused")
        db, guard, path = self._setup("health", probe)
        claim = _well_sourced_claim("service is reliable", probe_id="health")
        db.add_claim(claim)
        result = guard.check("service is reliable")
        assert result.verdict == "HALT"
        os.remove(path)

    # --- probe uncertain ---

    def test_probe_uncertain_downgrades_proceed(self):
        probe = lambda: ProbeResult(success=True, confidence=0.50, message="degraded response")
        db, guard, path = self._setup("health", probe)
        claim = _well_sourced_claim("service is reliable", probe_id="health")
        db.add_claim(claim)
        result = guard.check("service is reliable")
        assert result.verdict == "CAUTION"
        assert any("uncertain" in f for f in result.flags)
        os.remove(path)

    # --- probe exception ---

    def test_probe_exception_downgrades(self):
        def bad_probe() -> ProbeResult:
            raise ConnectionError("timeout")
        db, guard, path = self._setup("health", bad_probe)
        claim = _well_sourced_claim("service is reliable", probe_id="health")
        db.add_claim(claim)
        result = guard.check("service is reliable")
        # Exception → failed ProbeResult(confidence=0.0) → HALT
        assert result.verdict == "HALT"
        assert result.probe_fired
        assert result.probe_result.confidence == 0.0
        os.remove(path)

    # --- no probe on claim ---

    def test_no_probe_id_probe_result_is_none(self):
        db, guard, path = self._setup("unused", None)
        claim = _well_sourced_claim("no probe here")
        db.add_claim(claim)
        result = guard.check("no probe here")
        assert not result.probe_fired
        assert result.probe_result is None
        os.remove(path)
