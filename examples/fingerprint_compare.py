"""
Epistemic fingerprint — comparing reasoning styles across two domains.

Builds two belief sets (scientific vs anecdotal), then fingerprints
each and compares them side by side to reveal the difference in
reasoning style.

Run: python examples/fingerprint_compare.py
"""
import os, tempfile
from datetime import datetime
from veritas import VeritasDB, compute_fingerprint, compare_fingerprints
from veritas.models import Claim, Source, Stance, SourceType

db_path = os.path.join(tempfile.gettempdir(), "veritas_fp.db")
if os.path.exists(db_path):
    os.remove(db_path)
db = VeritasDB(db_path)


def src(citation, weight, stype, year, stance=Stance.SUPPORTS):
    return Source(citation=citation, weight=weight, stance=stance,
                  source_type=stype, source_date=datetime(year, 1, 1))


# Context A: rigorous scientific beliefs
for stmt, sources in [
    ("Vaccines are safe and effective", [
        src("Cochrane review 2023 — 200+ RCTs", 0.97, SourceType.META, 2023),
        src("WHO position paper 2024", 0.95, SourceType.AUTHORITY, 2024),
        src("CDC surveillance data 2024 — 1.2B doses", 0.93, SourceType.EMPIRICAL, 2024),
    ]),
    ("Climate change is driven by human activity", [
        src("IPCC AR6 2023 — 97% scientific consensus", 0.96, SourceType.AUTHORITY, 2023),
        src("NASA temperature records 1880-2024", 0.94, SourceType.EMPIRICAL, 2024),
        src("Ocean acidification measurements 2024", 0.90, SourceType.EMPIRICAL, 2024),
    ]),
    ("DNA carries genetic information", [
        src("Watson and Crick 1953", 0.99, SourceType.EMPIRICAL, 1953),
        src("Human Genome Project completion 2003", 0.99, SourceType.EMPIRICAL, 2003),
        src("CRISPR applications validate mechanism 2020", 0.97, SourceType.EMPIRICAL, 2020),
    ]),
]:
    db.add_claim(Claim(statement=stmt, context="scientific", sources=sources))


# Context B: low-rigour anecdotal beliefs
for stmt, sources in [
    ("This supplement cures arthritis", [
        src("My neighbour swears by it", 0.4, SourceType.ANECDOTAL, 2023),
    ]),
    ("Eating late at night causes weight gain", [
        src("Read it in a magazine", 0.35, SourceType.ANECDOTAL, 2021),
    ]),
    ("Sitting close to the TV damages eyesight", [
        src("My mum always said so", 0.3, SourceType.ANECDOTAL, 1990),
    ]),
]:
    db.add_claim(Claim(statement=stmt, context="folk", sources=sources))


fp_science = compute_fingerprint(db, context="scientific")
fp_folk    = compute_fingerprint(db, context="folk")

print(fp_science)
print(fp_folk)
print(compare_fingerprints(fp_science, fp_folk))

os.remove(db_path)
