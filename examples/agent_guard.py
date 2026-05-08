"""
Reasoning guard — check beliefs before an AI agent acts on them.

Shows how an agent can use Veritas to audit its own premises before
taking action. Demonstrates PROCEED, CAUTION, and HALT verdicts.

Run: python examples/agent_guard.py
"""
import os, tempfile
from datetime import datetime
from veritas import VeritasDB, ReasoningGuard
from veritas.models import Claim, Source, Stance, SourceType

db_path = os.path.join(tempfile.gettempdir(), "veritas_guard.db")
if os.path.exists(db_path):
    os.remove(db_path)
db = VeritasDB(db_path)


def src(citation, weight, stype=SourceType.EMPIRICAL, year=2025, stance=Stance.SUPPORTS):
    return Source(citation=citation, weight=weight, stance=stance,
                  source_type=stype, source_date=datetime(year, 1, 1))


# Register beliefs the agent will act on
db.add_claim(Claim(
    statement="Python is the dominant language for AI development",
    sources=[
        src("Stack Overflow Developer Survey 2025", 0.92, SourceType.AUTHORITY, 2025),
        src("GitHub Copilot usage stats 2025 — Python #1", 0.88, SourceType.EMPIRICAL, 2025),
        src("Kaggle ML Survey 2024", 0.85, SourceType.META, 2024),
    ],
))

db.add_claim(Claim(
    statement="This specific API endpoint returns JSON",
    sources=[
        src("Checked it once last year", 0.6, SourceType.ANECDOTAL, 2024),
    ],
))

db.add_claim(Claim(
    statement="The production database can handle 10k concurrent connections",
    sources=[
        src("Load test from 2019", 0.8, SourceType.EMPIRICAL, 2019),
    ],
))

db.add_claim(Claim(
    statement="Users want a dark mode in the interface",
    sources=[
        src("One user mentioned it in a support ticket", 0.3, SourceType.ANECDOTAL, 2025),
    ],
))

# Simulate an agent checking its premises before acting
guard = ReasoningGuard(db)

agent_premises = [
    ("Python is the dominant language for AI development",
     "choosing Python for new AI project"),
    ("This specific API endpoint returns JSON",
     "parsing API response without error handling"),
    ("The production database can handle 10k concurrent connections",
     "launching high-traffic feature"),
    ("Users want a dark mode in the interface",
     "prioritising dark mode over bug fixes"),
    ("The competitor went bankrupt last month",
     "adjusting market strategy"),  # unknown claim
]

print("\nAgent reasoning audit\n" + "=" * 50)
for premise, action in agent_premises:
    result = guard.check(premise)
    symbol = {"PROCEED": "[OK]  ", "CAUTION": "[??]  ", "HALT": "[NO]  "}[result.verdict]
    print(f"\n{symbol}Action: {action}")
    print(f"       Premise: {premise[:60]}")
    print(f"       {result.verdict}: {result.reason}")
    for flag in result.flags:
        print(f"         - {flag}")

print()

os.remove(db_path)
