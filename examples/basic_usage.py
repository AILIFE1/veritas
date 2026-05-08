"""
Basic Veritas usage — add claims, attach sources, trace confidence.

Run: python examples/basic_usage.py
"""
import os, tempfile
from veritas import VeritasDB, calculate_confidence
from veritas.models import Claim, Source, Stance, SourceType

db_path = os.path.join(tempfile.gettempdir(), "veritas_basic.db")
if os.path.exists(db_path):
    os.remove(db_path)
db = VeritasDB(db_path)

# Add a well-sourced claim
claim = Claim(
    statement="Regular exercise reduces risk of cardiovascular disease",
    context="medicine",
    sources=[
        Source(
            citation="WHO Global Health Report 2023",
            weight=0.95,
            stance=Stance.SUPPORTS,
            source_type=SourceType.AUTHORITY,
        ),
        Source(
            citation="Meta-analysis: 50 RCTs, n=120,000 (Lancet 2022)",
            weight=0.92,
            stance=Stance.SUPPORTS,
            source_type=SourceType.META,
            independence=0.9,
        ),
        Source(
            citation="My gym friend says it worked for him",
            weight=0.3,
            stance=Stance.SUPPORTS,
            source_type=SourceType.ANECDOTAL,
            independence=0.5,
        ),
    ],
)
db.add_claim(claim)

# Add a fragile claim — single weak source
fragile = Claim(
    statement="Cold showers cure depression",
    context="medicine",
    sources=[
        Source(
            citation="One blog post citing one small study",
            weight=0.35,
            stance=Stance.SUPPORTS,
            source_type=SourceType.ANECDOTAL,
        )
    ],
)
db.add_claim(fragile)

# Compute confidence vectors
for c in db.all_claims():
    cv = calculate_confidence(c.sources)
    print(f"\n{c.statement}")
    print(f"  {cv}")
    if cv.fragility > 0.3:
        print(f"  WARNING: fragile — confidence drops {cv.fragility:.2f} if best source removed")
    if cv.source_count == 1:
        print(f"  WARNING: single source — needs independent verification")

os.remove(db_path)
