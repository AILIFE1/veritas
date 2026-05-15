import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Claim, Source, Stance, SourceType, ProvenanceLink, InferenceType


SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB NOT NULL,
    PRIMARY KEY (claim_id, model)
);

CREATE TABLE IF NOT EXISTS claims (
    id       TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    context  TEXT,
    added_at TEXT NOT NULL,
    probe_id TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY,
    claim_id     TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    citation     TEXT NOT NULL,
    weight       REAL NOT NULL,
    stance       TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    independence REAL NOT NULL,
    notes        TEXT,
    source_date  TEXT,
    added_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provenance (
    claim_id     TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    depends_on   TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    inference_type TEXT NOT NULL DEFAULT 'INDUCTIVE',
    PRIMARY KEY (claim_id, depends_on)
);
"""


class VeritasDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn):
        source_cols = {r[1] for r in conn.execute("PRAGMA table_info(sources)").fetchall()}
        if "source_date" not in source_cols:
            conn.execute("ALTER TABLE sources ADD COLUMN source_date TEXT")
        claim_cols = {r[1] for r in conn.execute("PRAGMA table_info(claims)").fetchall()}
        if "probe_id" not in claim_cols:
            conn.execute("ALTER TABLE claims ADD COLUMN probe_id TEXT")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_claim(self, claim: Claim) -> Claim:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO claims (id, statement, context, added_at, probe_id) VALUES (?,?,?,?,?)",
                (claim.id, claim.statement, claim.context, claim.added_at.isoformat(), claim.probe_id),
            )
            for source in claim.sources:
                source.claim_id = claim.id
                self._insert_source(conn, source)
        return claim

    def _insert_source(self, conn, source: Source):
        conn.execute(
            "INSERT INTO sources "
            "(id, claim_id, citation, weight, stance, source_type, independence, notes, source_date, added_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                source.id, source.claim_id, source.citation, source.weight,
                source.stance.value, source.source_type.value, source.independence,
                source.notes,
                source.source_date.isoformat() if source.source_date else None,
                source.added_at.isoformat(),
            ),
        )

    def add_source(self, claim_id: str, source: Source) -> Source:
        source.claim_id = claim_id
        with self._conn() as conn:
            self._insert_source(conn, source)
        return source

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
            if not row:
                return None
            return self._hydrate(conn, row)

    def all_claims(self, context: Optional[str] = None) -> list[Claim]:
        with self._conn() as conn:
            if context:
                rows = conn.execute(
                    "SELECT * FROM claims WHERE context=? ORDER BY added_at", (context,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM claims ORDER BY added_at").fetchall()
            return [self._hydrate(conn, r) for r in rows]

    def search(self, query: str) -> list[Claim]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE statement LIKE ? OR id LIKE ? ORDER BY added_at",
                (f"%{query}%", f"{query}%"),
            ).fetchall()
            return [self._hydrate(conn, r) for r in rows]

    def link_claims(
        self,
        from_id: str,
        depends_on_id: str,
        inference_type: InferenceType = InferenceType.INDUCTIVE,
    ):
        """Record that from_id's confidence depends on depends_on_id."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO provenance (claim_id, depends_on, inference_type) VALUES (?,?,?)",
                (from_id, depends_on_id, inference_type.value),
            )

    def unlink_claims(self, from_id: str, depends_on_id: str):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM provenance WHERE claim_id=? AND depends_on=?",
                (from_id, depends_on_id),
            )

    def get_embedding(self, claim_id: str, model: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE claim_id=? AND model=?",
                (claim_id, model),
            ).fetchone()
            if not row:
                return None
            from .semantic import unpack_vector
            return unpack_vector(row["vector"])

    def set_embedding(self, claim_id: str, model: str, vector):
        from .semantic import pack_vector
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (claim_id, model, vector) VALUES (?,?,?)",
                (claim_id, model, pack_vector(vector)),
            )

    def invalidate_embedding(self, claim_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM embeddings WHERE claim_id=?", (claim_id,))

    def all_claims_by_id(self, context: str | None = None) -> dict[str, Claim]:
        return {c.id: c for c in self.all_claims(context=context)}

    def delete_claim(self, claim_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM claims WHERE id=?", (claim_id,))

    def _hydrate(self, conn, row) -> Claim:
        source_rows = conn.execute(
            "SELECT * FROM sources WHERE claim_id=?", (row["id"],)
        ).fetchall()
        sources = [
            Source(
                id=s["id"],
                claim_id=s["claim_id"],
                citation=s["citation"],
                weight=s["weight"],
                stance=Stance(s["stance"]),
                source_type=SourceType(s["source_type"]),
                independence=s["independence"],
                notes=s["notes"],
                source_date=datetime.fromisoformat(s["source_date"]) if s["source_date"] else None,
                added_at=datetime.fromisoformat(s["added_at"]),
            )
            for s in source_rows
        ]
        prov_rows = conn.execute(
            "SELECT depends_on, inference_type FROM provenance WHERE claim_id=?", (row["id"],)
        ).fetchall()
        depends_on = [
            ProvenanceLink(
                depends_on_id=p["depends_on"],
                inference_type=InferenceType(p["inference_type"]),
            )
            for p in prov_rows
        ]
        return Claim(
            id=row["id"],
            statement=row["statement"],
            context=row["context"],
            added_at=datetime.fromisoformat(row["added_at"]),
            sources=sources,
            depends_on=depends_on,
            probe_id=row["probe_id"],
        )
