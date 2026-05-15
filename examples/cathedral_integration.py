"""
Cathedral + Veritas integration walkthrough.

Cathedral stores what an agent remembers.
Veritas tracks how well those memories hold up epistemically.

This example simulates the Cathedral wake response so it runs
without a real API key. Replace the mock with:

    from cathedral import Cathedral
    cathedral = Cathedral(api_key="cathedral_...")
    wake_data = cathedral.wake()

Run: python examples/cathedral_integration.py
"""
import os, tempfile
from veritas import VeritasDB, ReasoningGuard
from veritas.models import Claim, Source, SourceType, Stance


# --- Mock Cathedral wake response ----------------------------------------
# In production this comes from cathedral.wake()

MOCK_WAKE = {
    "identity_memories": [
        {
            "id": "mem_001",
            "content": "Persistent memory significantly improves agent consistency across sessions",
            "importance": 1.0,
            "category": "identity",
        },
        {
            "id": "mem_002",
            "content": "The Cathedral API has been reliable across 150+ production snapshots",
            "importance": 0.9,
            "category": "experience",
        },
        {
            "id": "mem_003",
            "content": "Users prefer agents that acknowledge uncertainty over agents that project false confidence",
            "importance": 0.85,
            "category": "experience",
        },
        {
            "id": "mem_004",
            "content": "Noisy-OR source pooling is the correct model for independent evidence aggregation",
            "importance": 0.8,
            "category": "skill",
        },
    ]
}

# --- Build an illustrative Veritas database ------------------------------
# In production, claims accumulate over time as the agent gathers evidence.
# Here we construct varying evidence quality to demonstrate the flag.

db_path = os.path.join(tempfile.gettempdir(), "veritas_cathedral.db")
if os.path.exists(db_path):
    os.remove(db_path)
db = VeritasDB(db_path)

# mem_001 — high importance but thin evidence (the dangerous combination)
db.add_claim(Claim(
    statement="Persistent memory significantly improves agent consistency across sessions",
    context="cathedral:mem_001",
    sources=[
        Source(
            citation="Cathedral internal benchmark 2026 (single run, n=5 frameworks)",
            weight=0.78,
            stance=Stance.SUPPORTS,
            source_type=SourceType.EMPIRICAL,
        ),
    ],
))

# mem_002 — high importance, well-supported
db.add_claim(Claim(
    statement="The Cathedral API has been reliable across 150+ production snapshots",
    context="cathedral:mem_002",
    sources=[
        Source(
            citation="Production uptime log Jan-May 2026",
            weight=0.95,
            stance=Stance.SUPPORTS,
            source_type=SourceType.EMPIRICAL,
        ),
        Source(
            citation="Automated health check — 15/15 green Apr 17 2026",
            weight=0.90,
            stance=Stance.SUPPORTS,
            source_type=SourceType.EMPIRICAL,
        ),
    ],
))

# mem_003 — moderate importance, single source
db.add_claim(Claim(
    statement="Users prefer agents that acknowledge uncertainty over agents that project false confidence",
    context="cathedral:mem_003",
    sources=[
        Source(
            citation="Colony community thread — 8 comments converging on this view",
            weight=0.70,
            stance=Stance.SUPPORTS,
            source_type=SourceType.ANECDOTAL,
        ),
    ],
))

# mem_004 — lower importance, well-grounded
db.add_claim(Claim(
    statement="Noisy-OR source pooling is the correct model for independent evidence aggregation",
    context="cathedral:mem_004",
    sources=[
        Source(
            citation="Pearl 1988 — Probabilistic Reasoning in Intelligent Systems",
            weight=0.95,
            stance=Stance.SUPPORTS,
            source_type=SourceType.THEORETICAL,
        ),
        Source(
            citation="Henrion 1991 — Practical Issues in Constructing a Bayes' Belief Network",
            weight=0.88,
            stance=Stance.SUPPORTS,
            source_type=SourceType.THEORETICAL,
        ),
    ],
))

# --- Wake sequence -------------------------------------------------------

guard = ReasoningGuard(db)
FRAGILITY_WARN = 0.50  # flag if best-source removal drops confidence this much

print("\nCathedral + Veritas — wake audit")
print("=" * 60)

for memory in MOCK_WAKE["identity_memories"]:
    mem_id     = memory["id"]
    content    = memory["content"]
    importance = memory["importance"]

    result = guard.check(content[:80])

    symbol = {"PROCEED": "[OK]  ", "CAUTION": "[??]  ", "HALT": "[NO]  "}[result.verdict]
    print(f"\n{symbol}importance={importance}  {content[:55]}...")
    print(f"       verdict={result.verdict}  confidence={result.confidence:.2f}")

    # The importance-fragility flag
    if importance >= 1.0 and result.is_fragile:
        print(f"  [FLAG] importance=1.0 + fragile belief")
        print(f"         This identity memory depends on thin evidence.")
        print(f"         Action: find independent sources before relying on it.")

    for flag in result.flags:
        print(f"         - {flag}")

# --- Pre-action guard example --------------------------------------------

print("\n\nPre-action guard")
print("=" * 60)

actions = [
    ("Persistent memory significantly improves agent consistency across sessions",
     "publishing benchmark claiming 10x improvement"),
    ("The Cathedral API has been reliable across 150+ production snapshots",
     "recommending Cathedral for production use"),
    ("Users prefer agents that acknowledge uncertainty",
     "adding epistemic flags to agent responses"),
]

for premise, action in actions:
    result = guard.check(premise[:80])
    symbol = {"PROCEED": "[OK]", "CAUTION": "[??]", "HALT": "[NO]"}[result.verdict]
    print(f"\n{symbol} {action}")
    print(f"    premise: {premise[:60]}...")
    if result.verdict != "PROCEED":
        for flag in result.flags:
            print(f"    - {flag}")

print()
os.remove(db_path)
