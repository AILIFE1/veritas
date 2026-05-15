"""
Runtime probe primitive.

A probe is a registered callable that tests a belief against the real world
at check time. It upgrades the guard from PROCEED-by-coherence to
PROCEED-by-coherence-and-fresh-probe.

Usage:
    from veritas.probe import ProbeRegistry, ProbeResult

    registry = ProbeRegistry.default()

    @registry.probe("check_api_health")
    def _check_api_health() -> ProbeResult:
        resp = requests.get("https://api.example.com/health", timeout=3)
        ok = resp.status_code == 200
        return ProbeResult(
            success=ok,
            confidence=0.95 if ok else 0.05,
            message=f"GET /health -> {resp.status_code}",
        )

    # Attach the probe ID to a claim once, persist it in the DB
    claim.probe_id = "check_api_health"
    db.add_claim(claim)

    # The guard fires it automatically at check time
    result = guard.check("The payment API is reliable")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ProbeResult:
    success: bool
    confidence: float   # 0-1, world-state confidence returned by the probe
    message: str


# A probe is any callable that takes no args and returns a ProbeResult.
Probe = Callable[[], ProbeResult]

# Thresholds for probe verdict effect
PROBE_PASS_THRESHOLD   = 0.70   # >= this: probe confirmed
PROBE_FAIL_THRESHOLD   = 0.30   # <  this: probe failed


class ProbeRegistry:
    """
    Maps probe_id strings to callables. In-process only — callables cannot
    be serialised to the DB. The probe_id is stored; the callable is registered
    at process start.
    """

    _default: Optional["ProbeRegistry"] = None

    def __init__(self) -> None:
        self._probes: dict[str, Probe] = {}

    @classmethod
    def default(cls) -> "ProbeRegistry":
        if cls._default is None:
            cls._default = cls()
        return cls._default

    def register(self, probe_id: str, probe: Probe) -> None:
        self._probes[probe_id] = probe

    def probe(self, probe_id: str) -> Callable[[Probe], Probe]:
        """Decorator: @registry.probe('my_probe_id')"""
        def decorator(fn: Probe) -> Probe:
            self.register(probe_id, fn)
            return fn
        return decorator

    def get(self, probe_id: str) -> Optional[Probe]:
        return self._probes.get(probe_id)

    def fire(self, probe_id: str) -> Optional[ProbeResult]:
        """
        Fire the probe. Returns None if probe_id is not registered.
        Catches all exceptions and returns a failed ProbeResult rather than
        letting the guard crash on a bad probe.
        """
        fn = self.get(probe_id)
        if fn is None:
            return None
        try:
            return fn()
        except Exception as exc:
            return ProbeResult(success=False, confidence=0.0, message=f"Probe error: {exc}")
