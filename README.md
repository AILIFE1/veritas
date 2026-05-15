# Veritas

[![Tests](https://github.com/AILIFE1/veritas/actions/workflows/ci.yml/badge.svg)](https://github.com/AILIFE1/veritas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/veritas-epistemic.svg)](https://pypi.org/project/veritas-epistemic/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Epistemic confidence engine — beliefs with provenance.**

Most knowledge systems store facts. Veritas stores *how well you know them*.

Every claim carries a confidence vector derived from its sources, how those sources have aged, whether its foundations are solid, and whether contradicting evidence exists in the database. Shake a foundation and everything built on it moves automatically.

---

## The problem

AI agents act on beliefs they can't evaluate. "The API is reliable" — based on what? One observation from 2022? Twenty independent tests from this month? A single fragile assumption that everything else depends on?

Without epistemic structure, agents overclaim certainty or collapse into paralysis. Veritas gives beliefs a shape: not just a value, but fragility, staleness, and a provenance chain you can audit.

---

## Install

```bash
pip install veritas-epistemic
# For semantic contradiction detection:
pip install veritas-epistemic[semantic]
```

---

## Quick start

```bash
# Add a claim with sources
veritas add "Persistent memory improves agent consistency" \
  --source "Cathedral benchmark 2026 — 10.8x stability improvement" --weight 0.85 \
  --source "Empirical session tests 2026" --weight 0.75 \
  --context cathedral

# See full provenance breakdown
veritas trace "Persistent memory"

# Run the reasoning guard before acting
veritas check "Persistent memory improves agent consistency"
```

```
  Claim : Persistent memory improves agent consistency
Context : cathedral
     ID : dac6295c  (added 2026-05-08)

  Direct   [##################..] 0.90  src:2  contra:0  fragility:0.12
```

---

## Core concepts

### Confidence vectors

Confidence is not a single number — it has a shape.

| Field | Meaning |
|---|---|
| `value` | Current best estimate (0–1), with temporal decay applied |
| `fragility` | How much confidence drops if the best source is removed |
| `staleness_penalty` | How much evidence aging has already cost |
| `source_diversity` | How independent your sources are from each other |

```python
from veritas import VeritasDB, calculate_confidence
from veritas.models import Claim, Source, Stance

db = VeritasDB("~/.veritas/veritas.db")
claim = db.search("persistent memory")[0]
cv = calculate_confidence(claim.sources)

print(cv.value)            # 0.90
print(cv.fragility)        # 0.12  — reasonably robust
print(cv.staleness_penalty) # 0.00  — sources are fresh
```

---

### Temporal decay

Evidence ages. A 1986 study should carry less weight than a 2024 replication. A theorem from 1936 should carry exactly the same weight as the day it was proved.

| Source type | Half-life | Example |
|---|---|---|
| `MATHEMATICAL` | timeless | Turing 1936, Gödel 1931 |
| `THEORETICAL` | ~140 years | Newton, Darwin |
| `EMPIRICAL` | ~10 years | Studies, benchmarks |
| `AUTHORITY` | ~6 years | Expert consensus |
| `META` | ~4 years | Literature reviews |
| `ANECDOTAL` | ~2 years | Personal accounts |

```bash
# See what's going stale
veritas stale

  Claims losing confidence to age:

  -0.32  [##########..........] 0.54  Minsky 1967: AI will solve all problems
         59.0y  0.70->0.07  [ANEC]  Minsky 1967 interview

# Attach a date to a source so decay starts from publication, not today
veritas source "neural networks can learn" \
  --citation "Hinton 1986 backpropagation" --weight 0.85 --date 1986-01-01
```

---

### Belief propagation

Claims depend on other claims. When a foundation weakens, everything built on it updates automatically.

```bash
veritas add "AI agents currently lose state between sessions" \
  --source "OpenAI API docs — no persistent memory" --weight 0.95

veritas add "Developers need persistent agent memory" \
  --source "Community feedback 2026" --weight 0.65

veritas add "Cathedral will find a market" \
  --source "Personal assessment" --weight 0.60

# Wire the dependency chain
veritas depends "Developers need persistent" --on "AI agents currently lose state"
veritas depends "Cathedral will find a market" --on "Developers need persistent"

veritas chain "Cathedral will find a market"
```

```
  [0.86] Cathedral will find a market
    |-- [IND] -->
      [0.89] Developers need persistent agent memory
        |-- [IND] -->
          [0.95] AI agents currently lose state between sessions
```

Now add a contradicting source to the bottom claim and watch it propagate upward — without touching the top two claims.

---

### Semantic contradiction detection

Finds conflicts across different vocabulary. No shared words needed.

```python
from veritas import VeritasDB, find_contradictions
from veritas.models import Claim, Source, Stance

db = VeritasDB("~/.veritas/veritas.db")
claim = db.search("physical activity strengthens cardiovascular")[0]

contras = find_contradictions(claim, db.all_claims(), db=db)
# Finds: "Exercise has no proven benefit for heart health"
# — zero content words in common, caught via sentence embeddings
```

Falls back to keyword matching if `sentence-transformers` is not installed.

---

### Reasoning guard

Before an agent acts on a belief, check whether it actually holds up.

```python
from veritas import VeritasDB, ReasoningGuard

db = VeritasDB("~/.veritas/veritas.db")
guard = ReasoningGuard(db)

result = guard.check("GPT-3 represents the state of the art in language models")
print(result)
```

```
[CAUTION] confidence=0.87  Belief has weaknesses that should be acknowledged
  * Stale — evidence aging has reduced confidence by 0.12
```

```python
if result.should_proceed:
    # act
else:
    # surface the flags to the user or upstream system
    for flag in result.flags:
        print(flag)
```

Verdicts: `PROCEED` · `CAUTION` · `HALT`

Triggers: low confidence · single source · high fragility · staleness · contradictions

#### Threshold reference

| Condition | Default threshold | Verdict effect |
|---|---|---|
| `confidence < 0.30` | HALT_CONFIDENCE | **HALT** — don't act |
| `confidence < 0.55` | CAUTION_CONFIDENCE | **CAUTION** — flag uncertainty |
| `source_count == 1` | — | **CAUTION** — always, regardless of confidence |
| `fragility > 0.25` | FRAGILITY_CAUTION | **CAUTION** — drops badly if best source removed |
| `staleness_penalty > 0.08` | STALENESS_CAUTION | **CAUTION** — evidence aging has cost confidence |
| contradiction `confidence >= 0.45` | CONTRA_HALT_CONF | **CAUTION** — well-sourced counter-claim exists |

All thresholds are constants at the top of `veritas/guard.py` — tune them per use case.

---

### Epistemic fingerprint

Every belief system has a characteristic reasoning style. The fingerprint measures it.

```bash
veritas fingerprint --context cathedral
```

```
  Epistemic Fingerprint: cathedral
  ========================================================
  Claims: 12   Sources: 31   Avg sources/claim: 2.6

  Source composition:
    EMPIRICAL      [################........] 65%
    AUTHORITY      [########................] 32%
    ANECDOTAL      [##......................] 8%

  Confidence profile:
    Average        [##################......] 0.87
    Fragility      [####....................] 0.18
    Overconfident  [##......................] 8% of claims
    Single-source  [######..................] 25% of claims

  Epistemic health:
    Contradicted   [####....................] 17% of claims
    Rigor score    [################........] 0.68
    Calibration    [####################....] 0.84
    Overall        [##################......] 0.76
```

Compare two contexts side by side:

```bash
veritas compare cathedral philosophy
```

Two agents with the same beliefs but different fingerprints are different kinds of reasoners.

---

## Full command reference

```
veritas add         STATEMENT [--source ...] [--weight] [--date YYYY-MM-DD] [--context]
veritas source      CLAIM_REF --citation ... [--weight] [--stance] [--date]
veritas depends     CLAIM_REF --on CLAIM_REF [--inference DEDUCTIVE|INDUCTIVE|ABDUCTIVE]
veritas trace       CLAIM_REF          full provenance breakdown
veritas chain       CLAIM_REF          dependency tree with live confidence at each level
veritas check       CLAIM_TEXT         reasoning guard verdict
veritas challenge   CLAIM_REF          find contradicting claims
veritas fingerprint [--context]        epistemic style of a belief system
veritas compare     CONTEXT_A CONTEXT_B  side-by-side fingerprint diff
veritas query       [--fragile] [--low] [--unsourced] [--context]
veritas weakest     [--limit]          most fragile beliefs
veritas stale       [--threshold]      claims losing confidence to age
veritas report                         overall epistemic health
veritas demo                           live walkthrough of all capabilities
veritas delete      CLAIM_REF
```

---

## Python API

```python
from veritas import (
    VeritasDB,
    Claim, Source, Stance, SourceType,
    calculate_confidence,  # direct sources only
    propagate,             # with belief propagation through dependency chain
    find_contradictions,   # semantic when available, keyword fallback
    ReasoningGuard,
)
```

---

## Roadmap

- [x] Confidence vectors with noisy-OR source pooling
- [x] Temporal decay by source type (MATHEMATICAL to ANECDOTAL)
- [x] Belief propagation — foundations affect dependent claims
- [x] Semantic contradiction detection via sentence embeddings
- [x] Reasoning guard for agent pre-flight checks
- [x] Published to PyPI: `pip install veritas-epistemic`
- [ ] Active challenge mode — search for contradicting evidence automatically
- [ ] Cathedral integration — epistemic layer on top of persistent agent memory
- [ ] REST API for multi-agent use

---

## Connection to Cathedral

[Cathedral](https://cathedral-ai.com) gives AI agents persistent memory across sessions. Veritas is the reasoning layer that sits on top: Cathedral stores *what an agent remembers*, Veritas tracks *how well those memories hold up*.

Together: an agent knows its history and knows how much to trust it.

---

## License

MIT
