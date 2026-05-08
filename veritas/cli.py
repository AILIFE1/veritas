import click
from pathlib import Path

from .models import Source, Stance, SourceType, Claim, InferenceType
from .db import VeritasDB
from .engine import calculate_confidence, propagate, find_contradictions

DEFAULT_DB = Path.home() / ".veritas" / "veritas.db"


def _parse_date(date_str: str | None):
    if not date_str:
        return None
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise click.BadParameter(f"Cannot parse date: {date_str}. Use YYYY-MM-DD.")


def _db(ctx) -> VeritasDB:
    return VeritasDB(ctx.obj.get("db", DEFAULT_DB))


@click.group()
@click.option("--db", default=str(DEFAULT_DB), show_default=True, help="Database path")
@click.pass_context
def cli(ctx, db):
    """Veritas — epistemic confidence engine.

    Every belief carries its provenance. Query not just what you know,
    but how well you actually know it.
    """
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


# ── add ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("statement")
@click.option("--source", "-s", multiple=True, metavar="CITATION",
              help="Source citation (repeat for multiple)")
@click.option("--weight", "-w", default=0.7, show_default=True,
              help="Source reliability 0-1")
@click.option("--stance", default="SUPPORTS",
              type=click.Choice(["SUPPORTS", "CONTRADICTS", "QUALIFIES"]))
@click.option("--type", "source_type", default="EMPIRICAL",
              type=click.Choice(["MATHEMATICAL", "THEORETICAL", "EMPIRICAL", "AUTHORITY", "META", "ANECDOTAL"]))
@click.option("--independence", "-i", default=1.0, show_default=True,
              help="Source independence from others 0-1")
@click.option("--date", "-d", default=None, metavar="YYYY-MM-DD",
              help="Publication date of the source (for decay tracking)")
@click.option("--context", "-c", default=None, help="Domain tag (e.g. physics, medicine)")
@click.pass_context
def add(ctx, statement, source, weight, stance, source_type, independence, date, context):
    """Add a claim, optionally with sources."""
    db = _db(ctx)
    source_date = _parse_date(date)
    sources = [
        Source(
            citation=s,
            weight=weight,
            stance=Stance(stance),
            source_type=SourceType(source_type),
            independence=independence,
            source_date=source_date,
        )
        for s in source
    ]
    claim = Claim(statement=statement, context=context, sources=sources)
    db.add_claim(claim)
    cv = calculate_confidence(claim.sources)
    click.echo(f"  added  {claim.id[:8]}  {cv}")


# ── source ───────────────────────────────────────────────────────────────────

@cli.command("source")
@click.argument("claim_ref", metavar="CLAIM_ID_OR_TEXT")
@click.option("--citation", "-s", required=True)
@click.option("--weight", "-w", default=0.7)
@click.option("--stance", default="SUPPORTS",
              type=click.Choice(["SUPPORTS", "CONTRADICTS", "QUALIFIES"]))
@click.option("--type", "source_type", default="EMPIRICAL",
              type=click.Choice(["MATHEMATICAL", "THEORETICAL", "EMPIRICAL", "AUTHORITY", "META", "ANECDOTAL"]))
@click.option("--independence", "-i", default=1.0)
@click.option("--date", "-d", default=None, metavar="YYYY-MM-DD",
              help="Publication date of the source")
@click.option("--notes", default=None)
@click.pass_context
def add_source(ctx, claim_ref, citation, weight, stance, source_type, independence, date, notes):
    """Attach a new source to an existing claim."""
    db = _db(ctx)
    matches = db.search(claim_ref) or ([db.get_claim(claim_ref)] if len(claim_ref) >= 8 else [])
    matches = [c for c in matches if c]
    if not matches:
        click.echo("No matching claim.", err=True)
        return
    claim = matches[0]
    s = Source(citation=citation, weight=weight, stance=Stance(stance),
               source_type=SourceType(source_type), independence=independence,
               source_date=_parse_date(date), notes=notes)
    db.add_source(claim.id, s)
    updated = db.get_claim(claim.id)
    cv = calculate_confidence(updated.sources)
    click.echo(f"  updated  {claim.id[:8]}  {cv}")


# ── trace ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_ref", metavar="CLAIM_ID_OR_TEXT")
@click.pass_context
def trace(ctx, claim_ref):
    """Show full provenance and confidence breakdown for a claim."""
    db = _db(ctx)
    matches = db.search(claim_ref)
    if not matches:
        click.echo("No matching claims.")
        return
    claim = matches[0]
    all_by_id = db.all_claims_by_id()

    base_cv = calculate_confidence(claim.sources)
    prop_cv = propagate(claim, all_by_id)

    click.echo()
    click.echo(f"  Claim : {claim.statement}")
    if claim.context:
        click.echo(f"Context : {claim.context}")
    click.echo(f"     ID : {claim.id[:8]}  (added {claim.added_at.strftime('%Y-%m-%d')})")
    click.echo()
    click.echo(f"  Direct   {base_cv}")
    if claim.depends_on:
        click.echo(f"  Network  {prop_cv}")
    click.echo()

    if claim.sources:
        click.echo("  Sources:")
        for s in sorted(claim.sources, key=lambda x: x.weight * x.independence, reverse=True):
            icon = "+" if s.stance == Stance.SUPPORTS else ("-" if s.stance == Stance.CONTRADICTS else "~")
            eff = s.effective_weight()
            age = s.age_years()
            decay_str = ""
            if age > 0.5:
                decay_str = f"  age:{age:.1f}y  eff:{eff:.2f}"
            click.echo(
                f"    [{icon}] w={s.weight:.2f}  ind={s.independence:.2f}"
                f"  [{s.source_type.value[:3]}]{decay_str}  {s.citation}"
            )
            if s.notes:
                click.echo(f"         {s.notes}")
    else:
        click.echo("  No sources — confidence is prior (0.50)")

    if claim.depends_on:
        click.echo(f"\n  Depends on ({len(claim.depends_on)}):")
        for link in claim.depends_on:
            dep = all_by_id.get(link.depends_on_id)
            if dep:
                dep_cv = propagate(dep, all_by_id)
                click.echo(
                    f"    [{link.inference_type.value[:3]}]  "
                    f"{dep_cv.value:.2f}  {dep.statement[:55]}"
                )

    contras = find_contradictions(claim, db.all_claims(), db=db)
    if contras:
        click.echo(f"\n  Possible contradictions ({len(contras)}):")
        for c in contras:
            cv2 = calculate_confidence(c.sources)
            click.echo(f"    [{cv2.value:.2f}] {c.statement[:70]}")
    click.echo()


# ── query ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--fragile", is_flag=True, help="Claims fragility >= threshold")
@click.option("--threshold", default=0.3, show_default=True)
@click.option("--low", is_flag=True, help="Claims with confidence < 0.5")
@click.option("--unsourced", is_flag=True, help="Claims with no sources")
@click.option("--context", "-c", default=None, help="Filter by domain tag")
@click.option("--limit", default=20, show_default=True)
@click.pass_context
def query(ctx, fragile, threshold, low, unsourced, context, limit):
    """Query claims by epistemic properties."""
    db = _db(ctx)
    claims = db.all_claims(context=context)

    results = []
    for claim in claims:
        cv = calculate_confidence(claim.sources)
        claim.confidence = cv
        if unsourced and cv.source_count == 0:
            results.append(claim)
        elif fragile and cv.fragility >= threshold:
            results.append(claim)
        elif low and cv.value < 0.5:
            results.append(claim)
        elif not fragile and not low and not unsourced:
            results.append(claim)

    if not results:
        click.echo("No claims match.")
        return

    for c in results[:limit]:
        cv = c.confidence
        tag = f"[{c.context}] " if c.context else ""
        click.echo(f"  {c.id[:8]}  {cv.value:.2f}  frag:{cv.fragility:.2f}  {tag}{c.statement[:60]}")


# ── challenge ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_ref", metavar="CLAIM_ID_OR_TEXT")
@click.pass_context
def challenge(ctx, claim_ref):
    """Find claims in the database that contradict this one."""
    db = _db(ctx)
    matches = db.search(claim_ref)
    if not matches:
        click.echo("No matching claims.")
        return
    claim = matches[0]
    contras = find_contradictions(claim, db.all_claims(), db=db)
    if not contras:
        click.echo(f"  No contradictions found for: {claim.statement[:60]}")
        return
    click.echo(f"\n  Challenging: {claim.statement[:60]}\n")
    for c in contras:
        cv = calculate_confidence(c.sources)
        click.echo(f"  [{cv.value:.2f}]  {c.statement}")
    click.echo()


# ── weakest ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--limit", default=5, show_default=True)
@click.pass_context
def weakest(ctx, limit):
    """Show the most fragile beliefs — the ones resting on the thinnest evidence."""
    db = _db(ctx)
    claims = db.all_claims()
    scored = sorted(
        [(c, calculate_confidence(c.sources)) for c in claims],
        key=lambda x: x[1].fragility,
        reverse=True,
    )
    click.echo("\n  Most fragile beliefs:\n")
    for c, cv in scored[:limit]:
        click.echo(f"  {cv}  {c.statement[:50]}")
    click.echo()


# ── report ───────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def report(ctx):
    """Overall epistemic health of the database."""
    db = _db(ctx)
    claims = db.all_claims()
    if not claims:
        click.echo("  No claims in database.")
        return

    vectors = [(c, calculate_confidence(c.sources)) for c in claims]
    avg_conf = sum(cv.value for _, cv in vectors) / len(vectors)
    low_conf = [c for c, cv in vectors if cv.value < 0.5]
    fragile = [c for c, cv in vectors if cv.fragility > 0.3]
    unsourced = [c for c, cv in vectors if cv.source_count == 0]
    contradicted = [c for c in claims if find_contradictions(c, claims, db=db)]

    bar_len = 20
    filled = int(avg_conf * bar_len)
    avg_bar = "#" * filled + "." * (bar_len - filled)

    click.echo()
    click.echo("  -- Veritas Epistemic Report ------------------")
    click.echo(f"  Total claims     {len(claims)}")
    click.echo(f"  Avg confidence   [{avg_bar}] {avg_conf:.2f}")
    click.echo(f"  Low confidence   {len(low_conf)}")
    click.echo(f"  Fragile beliefs  {len(fragile)}")
    click.echo(f"  Unsourced        {len(unsourced)}")
    click.echo(f"  Contradictions   {len(contradicted)}")

    if fragile:
        click.echo(f"\n  Top fragile:")
        fragile_sorted = sorted(
            fragile,
            key=lambda c: calculate_confidence(c.sources).fragility,
            reverse=True,
        )
        for c in fragile_sorted[:3]:
            cv = calculate_confidence(c.sources)
            click.echo(f"    [{cv.fragility:.2f}]  {c.statement[:55]}")
    click.echo()


# ── depends ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_ref", metavar="CLAIM_ID_OR_TEXT")
@click.option("--on", "dep_ref", required=True, metavar="CLAIM_ID_OR_TEXT",
              help="The claim this one derives from")
@click.option("--inference", default="INDUCTIVE",
              type=click.Choice(["DEDUCTIVE", "INDUCTIVE", "ABDUCTIVE"]),
              help="How strongly the dependency gates confidence")
@click.pass_context
def depends(ctx, claim_ref, dep_ref, inference):
    """Link two claims: CLAIM derives from --on DEPENDENCY.

    DEDUCTIVE  confidence is capped by the dependency
    INDUCTIVE  weak foundation pulls confidence down significantly
    ABDUCTIVE  soft pull — best-explanation relationship
    """
    db = _db(ctx)
    claim_matches = db.search(claim_ref)
    dep_matches = db.search(dep_ref)
    if not claim_matches:
        click.echo(f"No claim matching: {claim_ref}", err=True)
        return
    if not dep_matches:
        click.echo(f"No claim matching: {dep_ref}", err=True)
        return
    claim = claim_matches[0]
    dep = dep_matches[0]
    db.link_claims(claim.id, dep.id, InferenceType(inference))
    all_by_id = db.all_claims_by_id()
    before = calculate_confidence(claim.sources)
    after = propagate(db.get_claim(claim.id), all_by_id)
    click.echo(f"  linked  [{inference[:3]}]  {claim.statement[:45]}  <--  {dep.statement[:45]}")
    if abs(after.value - before.value) > 0.001:
        click.echo(f"  confidence shift: {before.value:.2f} -> {after.value:.2f}")


# ── chain ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_ref", metavar="CLAIM_ID_OR_TEXT")
@click.pass_context
def chain(ctx, claim_ref):
    """Show the full belief dependency tree for a claim."""
    db = _db(ctx)
    matches = db.search(claim_ref)
    if not matches:
        click.echo("No matching claims.")
        return
    all_by_id = db.all_claims_by_id()
    claim = matches[0]

    click.echo()
    _print_chain(claim, all_by_id, depth=0, visited=set())
    click.echo()


def _print_chain(claim: Claim, all_by_id: dict, depth: int, visited: set):
    indent = "  " + "  " * depth
    cv = propagate(claim, all_by_id)
    marker = "(cycle)" if claim.id in visited else ""
    click.echo(f"{indent}[{cv.value:.2f}] {claim.statement[:60]} {marker}")
    if claim.id in visited:
        return
    visited = visited | {claim.id}
    for link in claim.depends_on:
        dep = all_by_id.get(link.depends_on_id)
        if dep:
            click.echo(f"{indent}  |-- [{link.inference_type.value[:3]}] -->")
            _print_chain(dep, all_by_id, depth + 2, visited)


# ── check ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_text")
@click.pass_context
def check(ctx, claim_text):
    """Run the reasoning guard on a belief before acting on it.

    Returns PROCEED, CAUTION, or HALT with a full explanation.
    Designed to be called programmatically by AI agents.
    """
    from .guard import ReasoningGuard
    db = _db(ctx)
    guard = ReasoningGuard(db)
    result = guard.check(claim_text)
    click.echo()
    click.echo(f"  {result}")
    click.echo()


# ── stale ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--threshold", default=0.05, show_default=True,
              help="Minimum staleness penalty to flag")
@click.option("--context", "-c", default=None)
@click.option("--limit", default=15, show_default=True)
@click.pass_context
def stale(ctx, threshold, context, limit):
    """Show claims where evidence has aged enough to meaningfully reduce confidence.

    Useful for deciding what to re-source or revisit.
    """
    db = _db(ctx)
    claims = db.all_claims(context=context)
    if not claims:
        click.echo("  No claims.")
        return

    flagged = []
    for claim in claims:
        cv = calculate_confidence(claim.sources)
        if cv.staleness_penalty >= threshold:
            flagged.append((claim, cv))

    if not flagged:
        click.echo(f"  No claims with staleness penalty >= {threshold:.2f}")
        return

    flagged.sort(key=lambda x: x[1].staleness_penalty, reverse=True)
    click.echo(f"\n  Claims losing confidence to age (penalty >= {threshold:.2f}):\n")

    for claim, cv in flagged[:limit]:
        click.echo(f"  -{cv.staleness_penalty:.2f}  {cv}  {claim.statement[:50]}")
        for s in sorted(claim.sources, key=lambda x: x.age_years(), reverse=True):
            age = s.age_years()
            if age < 0.5:
                continue
            eff = s.effective_weight()
            click.echo(
                f"         {age:.1f}y  {s.weight:.2f}->{eff:.2f}"
                f"  [{s.source_type.value[:3]}]  {s.citation[:50]}"
            )
    click.echo()


# ── fingerprint ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--context", "-c", default=None, help="Context to fingerprint (default: all)")
@click.pass_context
def fingerprint(ctx, context):
    """Show the epistemic fingerprint of a belief system.

    Reveals the characteristic reasoning style: what evidence it relies on,
    how fragile its beliefs are, how fresh its sources are, whether it
    acknowledges contradictions.
    """
    from .fingerprint import compute
    db = _db(ctx)
    fp = compute(db, context=context)
    if fp.total_claims == 0:
        click.echo("  No claims found.")
        return
    click.echo(str(fp))


@cli.command("compare")
@click.argument("context_a")
@click.argument("context_b")
@click.pass_context
def compare_contexts(ctx, context_a, context_b):
    """Compare the epistemic fingerprints of two contexts side by side."""
    from .fingerprint import compute, compare
    db = _db(ctx)
    fp_a = compute(db, context=context_a)
    fp_b = compute(db, context=context_b)
    click.echo()
    click.echo(compare(fp_a, fp_b))


# ── demo ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--db", "demo_db", default=None, help="Database path (default: temp file)")
@click.pass_context
def demo(ctx, demo_db):
    """Run a live demonstration of all Veritas capabilities.

    Builds a belief network from scratch, shows confidence propagation,
    temporal decay, semantic contradictions, the reasoning guard, and
    an epistemic fingerprint. Takes about 10 seconds.
    """
    import os, tempfile, warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from datetime import datetime
    from .fingerprint import compute
    from .guard import ReasoningGuard

    db_path = demo_db or os.path.join(tempfile.gettempdir(), "veritas_demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = VeritasDB(db_path)

    def src(citation, weight, stype=SourceType.EMPIRICAL, year=2024, stance=Stance.SUPPORTS):
        return Source(citation=citation, weight=weight, stance=stance,
                      source_type=stype, source_date=datetime(year, 1, 1))

    click.echo()
    click.echo("  Veritas live demo")
    click.echo("  " + "=" * 50)

    # ── 1. Build a belief chain ───────────────────────────────────────────────
    click.echo()
    click.echo("  1. Building a 3-level belief chain...")

    c_base = Claim(
        statement="AI agents lose all memory between sessions by default",
        context="demo",
        sources=[
            src("OpenAI API documentation 2024", 0.95, SourceType.AUTHORITY, 2024),
            src("Empirical testing across 5 frameworks 2024", 0.88, year=2024),
        ]
    )
    c_mid = Claim(
        statement="Developers need persistent memory to build reliable agents",
        context="demo",
        sources=[
            src("Developer survey Colony 2026", 0.65, SourceType.ANECDOTAL, 2026),
        ]
    )
    c_top = Claim(
        statement="Cathedral solves a real problem in the agent ecosystem",
        context="demo",
        sources=[
            src("Benchmark: 10.8x stability improvement vs stateless", 0.85, year=2026),
        ]
    )

    db.add_claim(c_base)
    db.add_claim(c_mid)
    db.add_claim(c_top)
    db.link_claims(c_mid.id, c_base.id)
    db.link_claims(c_top.id, c_mid.id)

    all_by_id = db.all_claims_by_id()
    from .engine import propagate as _prop
    cv_top = _prop(db.get_claim(c_top.id), all_by_id)
    click.echo(f"     Top claim confidence: {cv_top.value:.2f}  (chain intact)")

    # ── 2. Shake the foundation ───────────────────────────────────────────────
    click.echo()
    click.echo("  2. Adding contradicting evidence to the foundation...")

    db.add_source(c_base.id, src(
        "Some LLMs now offer built-in session memory (GPT-5 memory feature)",
        weight=0.70, stype=SourceType.AUTHORITY, year=2025, stance=Stance.CONTRADICTS
    ))

    all_by_id = db.all_claims_by_id()
    cv_base_new = _prop(db.get_claim(c_base.id), all_by_id)
    cv_top_new  = _prop(db.get_claim(c_top.id),  all_by_id)
    click.echo(f"     Foundation confidence: {cv_base_new.value:.2f}  (was ~0.95)")
    click.echo(f"     Top claim confidence:  {cv_top_new.value:.2f}  (propagated down, unchanged sources)")

    # ── 3. Old source decaying ────────────────────────────────────────────────
    click.echo()
    click.echo("  3. Temporal decay on an old claim...")

    old = Claim(
        statement="Symbolic AI is the dominant paradigm for machine reasoning",
        context="demo",
        sources=[
            src("Minsky 1975 — frames and knowledge representation", 0.9,
                SourceType.AUTHORITY, 1975),
        ]
    )
    db.add_claim(old)
    cv_old = calculate_confidence(db.get_claim(old.id).sources)
    click.echo(f"     Confidence: {cv_old.value:.2f}  (lost {cv_old.staleness_penalty:.2f} to age — 50yr authority source)")

    # ── 4. Semantic contradiction ─────────────────────────────────────────────
    click.echo()
    click.echo("  4. Semantic contradiction detection (no shared words)...")

    ca = Claim(
        statement="Regular exercise strengthens the human cardiovascular system",
        context="demo",
        sources=[src("WHO Global Health Report 2023", 0.95, year=2023)]
    )
    cb = Claim(
        statement="Physical activity has no proven benefit for heart health",
        context="demo",
        sources=[src("Fringe wellness blog 2021", 0.2, SourceType.ANECDOTAL, year=2021)]
    )
    db.add_claim(ca)
    db.add_claim(cb)
    contras = find_contradictions(db.get_claim(ca.id), db.all_claims(), db=db)
    found = any(c.id == cb.id for c in contras)
    click.echo(f"     Contradiction caught: {found}  (zero content words in common)")

    # ── 5. Reasoning guard ────────────────────────────────────────────────────
    click.echo()
    click.echo("  5. Reasoning guard checks...")

    guard = ReasoningGuard(db)
    for label, text in [
        ("well-sourced", "Cathedral solves a real problem"),
        ("stale",        "Symbolic AI is the dominant paradigm"),
    ]:
        result = guard.check(text)
        click.echo(f"     [{result.verdict:<7}] {label}: {result.reason}")
        for flag in result.flags:
            click.echo(f"              {flag}")

    # ── 6. Fingerprint ────────────────────────────────────────────────────────
    click.echo()
    click.echo("  6. Epistemic fingerprint of this belief system:")
    fp = compute(db, context="demo")
    click.echo(str(fp))

    if not demo_db:
        os.remove(db_path)


# ── delete ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_ref", metavar="CLAIM_ID_OR_TEXT")
@click.confirmation_option(prompt="Delete this claim?")
@click.pass_context
def delete(ctx, claim_ref):
    """Delete a claim and all its sources."""
    db = _db(ctx)
    matches = db.search(claim_ref)
    if not matches:
        click.echo("No matching claims.")
        return
    claim = matches[0]
    db.delete_claim(claim.id)
    click.echo(f"  deleted  {claim.id[:8]}  {claim.statement[:60]}")
