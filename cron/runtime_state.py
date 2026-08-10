"""Profile-local volatile runtime state for cron job definitions.

Cron job definitions live in ``jobs.json``. Scheduler-owned timestamps, status,
counters, leases, and cross-store recovery metadata live here so ordinary runs
do not rewrite operator intent.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, Mapping, Optional, Sequence


_RUNTIME_DB_NAME = "runtime.db"
_SCHEMA_INIT_ATTEMPTS = 3
_SCHEMA_RETRY_BASE_SECONDS = 0.05


def runtime_db_path(cron_dir: Path) -> Path:
    """Return the runtime database path for one profile-local cron directory."""
    return cron_dir / _RUNTIME_DB_NAME


def _owner_tuple(path: Path) -> Optional[tuple[int, int]]:
    """Return uid/gid when the platform exposes POSIX ownership."""
    try:
        stat_result = path.stat()
    except OSError:
        return None
    uid = getattr(stat_result, "st_uid", None)
    gid = getattr(stat_result, "st_gid", None)
    if uid is None or gid is None:
        return None
    return int(uid), int(gid)


def _secure_runtime_files(path: Path, owner: Optional[tuple[int, int]]) -> None:
    """Best-effort owner/mode repair for the database and SQLite sidecars."""
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not candidate.exists():
            continue
        try:
            os.chmod(candidate, 0o600)
        except OSError:
            pass
        if owner is None or not hasattr(os, "chown"):
            continue
        try:
            current = _owner_tuple(candidate)
            if current != owner:
                os.chown(candidate, owner[0], owner[1])
        except OSError:
            pass


def _connect(cron_dir: Path) -> sqlite3.Connection:
    """Open and initialize one profile's cron runtime database."""
    path = runtime_db_path(cron_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = _owner_tuple(path) or _owner_tuple(path.parent)
    for attempt in range(_SCHEMA_INIT_ATTEMPTS):
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(path, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="cron/runtime.db")
            conn.execute("PRAGMA synchronous=FULL")
            # Serialize schema discovery and migration across gateway replicas.
            # Without a write transaction, two first-openers can both observe the
            # legacy shape and race the same ALTER TABLE statement.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS job_runtime (
                     job_id TEXT PRIMARY KEY,
                     state_json TEXT NOT NULL
                   )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS pending_definitions (
                     singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                     definitions_json TEXT NOT NULL,
                     generation_id TEXT,
                     base_definitions_digest TEXT
                   )"""
            )
            pending_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(pending_definitions)"
                ).fetchall()
            }
            if "generation_id" not in pending_columns:
                conn.execute(
                    "ALTER TABLE pending_definitions ADD COLUMN generation_id TEXT"
                )
            if "base_definitions_digest" not in pending_columns:
                conn.execute(
                    "ALTER TABLE pending_definitions "
                    "ADD COLUMN base_definitions_digest TEXT"
                )
            conn.execute(
                "UPDATE pending_definitions SET generation_id = ? "
                "WHERE generation_id IS NULL OR generation_id = ''",
                (uuid.uuid4().hex,),
            )
            conn.commit()
            _secure_runtime_files(path, owner)
            return conn
        except BaseException as exc:
            if conn is not None:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                conn.close()
            is_transient_lock = (
                isinstance(exc, sqlite3.OperationalError)
                and any(token in str(exc).lower() for token in ("locked", "busy"))
            )
            if not is_transient_lock or attempt + 1 >= _SCHEMA_INIT_ATTEMPTS:
                raise
            time.sleep(_SCHEMA_RETRY_BASE_SECONDS * (2**attempt))

    raise RuntimeError("Cron runtime database initialization exhausted retries")


@contextmanager
def _transaction(cron_dir: Path) -> Iterator[sqlite3.Connection]:
    """Open one write transaction and close its connection deterministically.

    ``BEGIN IMMEDIATE`` is required for expected-state fencing: sqlite3 does
    not start a transaction for a plain ``SELECT``, so a deferred transaction
    would leave a check-to-write window where a sibling could commit a newer
    ownership row before this writer's upsert.
    """
    conn = _connect(cron_dir)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()
        owner = _owner_tuple(runtime_db_path(cron_dir)) or _owner_tuple(cron_dir)
        _secure_runtime_files(runtime_db_path(cron_dir), owner)


def _serialize(state: Mapping[str, Any]) -> str:
    """Serialize runtime state canonically for deterministic updates."""
    return json.dumps(
        dict(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _serialize_definitions(definitions: Sequence[Mapping[str, Any]]) -> str:
    """Serialize a pending definition snapshot canonically."""
    return json.dumps(
        [dict(definition) for definition in definitions],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _deserialize(job_id: str, payload: str) -> Dict[str, Any]:
    """Decode one runtime row and fail closed on corrupt state."""
    try:
        state = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cron runtime state for job {job_id!r} is corrupted"
        ) from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"Cron runtime state for job {job_id!r} must be an object")
    return state


def _replace_rows(
    conn: sqlite3.Connection,
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Replace the complete runtime snapshot without SQL variable limits."""
    normalized = {str(job_id): dict(state) for job_id, state in states.items()}
    conn.execute("DELETE FROM job_runtime")
    if normalized:
        conn.executemany(
            "INSERT INTO job_runtime(job_id, state_json) VALUES (?, ?)",
            [(job_id, _serialize(state)) for job_id, state in normalized.items()],
        )


def _merge_rows(
    conn: sqlite3.Connection,
    states: Mapping[str, Mapping[str, Any]],
    *,
    removed_ids: Collection[str] = (),
    expected_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    """Upsert owned rows and delete only explicitly removed job IDs.

    Ordinary cron writers may proceed after the advisory jobs lock times out.
    Replacing the complete table in that degraded mode can erase a sibling
    create that committed after this writer's snapshot. Targeted upserts retain
    sibling rows while ``removed_ids`` preserves intentional-delete semantics.
    """
    normalized = {str(job_id): dict(state) for job_id, state in states.items()}
    intended_remove = {str(job_id) for job_id in removed_ids if job_id}
    if expected_states is not None:
        expected = {
            str(job_id): dict(state) for job_id, state in expected_states.items()
        }
        for job_id in set(normalized) | intended_remove:
            row = conn.execute(
                "SELECT state_json FROM job_runtime WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job_id not in expected:
                unchanged = row is None
            else:
                unchanged = (
                    row is not None
                    and _deserialize(job_id, row[0]) == expected[job_id]
                )
            if not unchanged:
                raise RuntimeError(
                    f"Cron runtime state for job {job_id!r} changed concurrently; "
                    "refusing to overwrite the newer generation"
                )
    if normalized:
        conn.executemany(
            """INSERT INTO job_runtime(job_id, state_json) VALUES (?, ?)
               ON CONFLICT(job_id) DO UPDATE
               SET state_json=excluded.state_json""",
            [(job_id, _serialize(state)) for job_id, state in normalized.items()],
        )
    if intended_remove:
        conn.executemany(
            "DELETE FROM job_runtime WHERE job_id = ?",
            [(job_id,) for job_id in intended_remove],
        )


def load_runtime_states(cron_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all volatile job state for one profile."""
    path = runtime_db_path(cron_dir)
    if not path.exists():
        return {}
    with _transaction(cron_dir) as conn:
        rows = conn.execute(
            "SELECT job_id, state_json FROM job_runtime ORDER BY job_id"
        ).fetchall()
    return {
        str(row["job_id"]): _deserialize(str(row["job_id"]), row["state_json"])
        for row in rows
    }


def load_pending_definition_record(
    cron_dir: Path,
) -> tuple[Optional[list[Dict[str, Any]]], Optional[str], Optional[str]]:
    """Load a pending snapshot, acknowledgement token, and base digest."""
    path = runtime_db_path(cron_dir)
    if not path.exists():
        return None, None, None
    with _transaction(cron_dir) as conn:
        row = conn.execute(
            "SELECT definitions_json, generation_id, base_definitions_digest "
            "FROM pending_definitions WHERE singleton = 1"
        ).fetchone()
    if row is None:
        return None, None, None
    try:
        definitions = json.loads(row["definitions_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cron pending definition journal is corrupted") from exc
    if not isinstance(definitions, list) or not all(
        isinstance(definition, dict) for definition in definitions
    ):
        raise RuntimeError("Cron pending definition journal must be a list of objects")
    generation_id = str(row["generation_id"] or "").strip()
    if not generation_id:
        raise RuntimeError("Cron pending definition journal has no generation id")
    base_digest = str(row["base_definitions_digest"] or "").strip() or None
    return [dict(definition) for definition in definitions], generation_id, base_digest


def load_pending_definition_generation(
    cron_dir: Path,
) -> tuple[Optional[list[Dict[str, Any]]], Optional[str]]:
    """Load a pending definition snapshot and its acknowledgement token."""
    definitions, generation_id, _base_digest = load_pending_definition_record(cron_dir)
    return definitions, generation_id


def load_pending_definitions(cron_dir: Path) -> Optional[list[Dict[str, Any]]]:
    """Load a journaled definition snapshot awaiting materialization."""
    definitions, _generation_id = load_pending_definition_generation(cron_dir)
    return definitions


def list_live_claims(
    cron_dir: Path,
    *,
    fire_claim_ttl_seconds: float,
    run_claim_ttl_seconds: float,
) -> list[Dict[str, str]]:
    """Return fresh fire/run claims that make destructive restore unsafe.

    Malformed, future-dated, and expired claims are stale by the same bounded
    age rule used by scheduler acquisition.  The caller supplies both TTLs so
    this storage module remains independent of scheduler policy/configuration.
    """
    from hermes_time import now

    current = now()
    live: list[Dict[str, str]] = []
    for job_id, state in load_runtime_states(cron_dir).items():
        for field, ttl in (
            ("fire_claim", fire_claim_ttl_seconds),
            ("run_claim", run_claim_ttl_seconds),
        ):
            claim = state.get(field)
            if not isinstance(claim, dict):
                continue
            try:
                claimed_at = datetime.fromisoformat(str(claim["at"]))
                if claimed_at.tzinfo is None:
                    claimed_at = claimed_at.replace(tzinfo=current.tzinfo)
                age = (current - claimed_at).total_seconds()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= age < ttl:
                live.append({
                    "job_id": job_id,
                    "kind": field,
                    "at": str(claim["at"]),
                    "owner": str(claim.get("id") or claim.get("by") or "unknown"),
                })
    return live


def clear_pending_definitions(
    cron_dir: Path,
    *,
    expected_generation_id: Optional[str] = None,
) -> bool:
    """Acknowledge only the pending definition generation materialized by caller.

    Omitting ``expected_generation_id`` is an explicit administrative clear.
    Ordinary writers pass their generation token so an older materializer can
    never erase a newer writer's forward-recovery journal.
    """
    with _transaction(cron_dir) as conn:
        if expected_generation_id is None:
            cursor = conn.execute(
                "DELETE FROM pending_definitions WHERE singleton = 1"
            )
        else:
            cursor = conn.execute(
                "DELETE FROM pending_definitions "
                "WHERE singleton = 1 AND generation_id = ?",
                (expected_generation_id,),
            )
    return cursor.rowcount == 1


def merge_legacy_runtime_states(
    cron_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Seed migrated legacy state without overwriting a newer runtime row.

    Runtime is committed before the combined definition artifact is stripped.
    If the process dies between those steps, the next migration sees the
    existing authoritative row and safely retries the definition rewrite.
    """
    if not states:
        return
    with _transaction(cron_dir) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO job_runtime(job_id, state_json) VALUES (?, ?)",
            [(str(job_id), _serialize(state)) for job_id, state in states.items()],
        )


def replace_runtime_states(
    cron_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically replace runtime rows for the current definition set."""
    with _transaction(cron_dir) as conn:
        _replace_rows(conn, states)


def merge_runtime_states(
    cron_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
    *,
    removed_ids: Collection[str] = (),
    expected_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    """Atomically update owned runtime rows without erasing sibling writes."""
    with _transaction(cron_dir) as conn:
        _merge_rows(
            conn,
            states,
            removed_ids=removed_ids,
            expected_states=expected_states,
        )


def stage_runtime_and_definitions(
    cron_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
    definitions: Sequence[Mapping[str, Any]],
    *,
    removed_ids: Collection[str] = (),
    replace: bool = True,
    expected_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
    base_definitions_digest: Optional[str] = None,
) -> str:
    """Commit runtime plus a generation-fenced recovery journal atomically."""
    generation_id = uuid.uuid4().hex
    with _transaction(cron_dir) as conn:
        if replace:
            _replace_rows(conn, states)
        else:
            _merge_rows(
                conn,
                states,
                removed_ids=removed_ids,
                expected_states=expected_states,
            )
        conn.execute(
            """INSERT INTO pending_definitions(
                   singleton, definitions_json, generation_id,
                   base_definitions_digest
               ) VALUES (1, ?, ?, ?)
               ON CONFLICT(singleton) DO UPDATE SET
                   definitions_json=excluded.definitions_json,
                   generation_id=excluded.generation_id,
                   base_definitions_digest=excluded.base_definitions_digest""",
            (
                _serialize_definitions(definitions),
                generation_id,
                base_definitions_digest,
            ),
        )
    return generation_id
