"""
Belief propagation — foundations affect everything built on them.

This example builds a 3-level reasoning chain, then shows how
adding contradicting evidence at the bottom propagates upward
without touching the claims above.

Run: python examples/belief_chain.py
"""
import os, tempfile
from veritas import VeritasDB, propagate
from veritas.models import Claim, Source, Stance, SourceType

db_path = os.path.join(tempfile.gettempdir(), "veritas_chain.db")
if os.path.exists(db_path):
    os.remove(db_path)
db = VeritasDB(db_path)

# Bottom: empirical foundation
bottom = Claim(
    statement="AI agents currently lose all context between sessions",
    sources=[
        Source("OpenAI API docs — stateless by design", 0.95, Stance.SUPPORTS, SourceType.AUTHORITY),
        Source("Tested across LangChain, AutoGPT, CrewAI — confirmed", 0.88, Stance.SUPPORTS),
    ],
)

# Middle: depends on bottom
middle = Claim(
    statement="Developers need persistent memory to build reliable agents",
    sources=[
        Source("Stack Overflow survey 2025 — top agent pain point", 0.72, Stance.SUPPORTS, SourceType.AUTHORITY),
    ],
)

# Top: depends on middle
top = Claim(
    statement="There is a real market for agent memory infrastructure",
    sources=[
        Source("Cathedral early adoption: 200+ installs in 2 weeks", 0.75, Stance.SUPPORTS),
    ],
)

for c in [bottom, middle, top]:
    db.add_claim(c)
db.link_claims(middle.id, bottom.id)
db.link_claims(top.id, middle.id)


def show_chain(label):
    print(f"\n  {label}")
    all_by_id = db.all_claims_by_id()
    for name, claim_id in [("Bottom", bottom.id), ("Middle", middle.id), ("Top", top.id)]:
        c = db.get_claim(claim_id)
        cv = propagate(c, all_by_id)
        print(f"    {name:<8} [{cv.value:.2f}]  {c.statement[:55]}")


show_chain("Chain intact:")

# Now add a contradicting source to the bottom claim
print("\n  Adding: 'GPT-5 and Gemini Ultra now include persistent session memory'")
db.add_source(bottom.id, Source(
    citation="GPT-5 and Gemini Ultra now include persistent session memory",
    weight=0.80,
    stance=Stance.CONTRADICTS,
    source_type=SourceType.AUTHORITY,
))

show_chain("After shaking the foundation:")
print()
print("  Note: Top and Middle claim sources are unchanged.")
print("  Confidence shifted purely through propagation.")

os.remove(db_path)
