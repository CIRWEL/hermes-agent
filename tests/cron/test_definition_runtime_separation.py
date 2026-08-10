"""Contracts for declarative cron definitions and volatile runtime state."""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pytest

import cron.jobs as jobs


_RUNTIME_FIELDS = {
    "created_at",
    "fire_claim",
    "last_delivery_error",
    "last_error",
    "last_run_at",
    "last_status",
    "model_snapshot",
    "next_run_at",
    "paused_at",
    "provider_snapshot",
    "run_claim",
    "state",
}


@pytest.fixture()
def cron_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Route cron definition and runtime storage to a temporary profile."""
    home = tmp_path / "profile"
    cron_dir = home / "cron"
    monkeypatch.setattr(jobs, "HERMES_DIR", home)
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(
        jobs,
        "_compute_provider_model_snapshots",
        lambda **_kwargs: ("provider-at-create", "model-at-create"),
    )
    yield home


def _raw_definitions(home: Path) -> list[dict]:
    """Read only the persisted definitions from a temporary profile."""
    data = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    return data["jobs"]


def test_recurring_run_does_not_rewrite_definition_artifact(cron_store: Path) -> None:
    """Execution metadata and counters must change outside jobs.json."""
    created = jobs.create_job(
        prompt="check health",
        schedule="every 1h",
        name="health",
        repeat=3,
        deliver="local",
    )
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()

    jobs.mark_job_run(created["id"], True)

    assert definitions_path.read_bytes() == before
    loaded = jobs.get_job(created["id"])
    assert loaded is not None
    assert loaded["last_status"] == "ok"
    assert loaded["last_run_at"]
    assert loaded["next_run_at"]
    assert loaded["repeat"] == {"times": 3, "completed": 1}
    assert (cron_store / "cron" / "runtime.db").exists()


def test_legacy_combined_store_migrates_without_losing_runtime(
    cron_store: Path,
) -> None:
    """Migration must preserve cadence, counters, and an active fire lease."""
    cron_dir = cron_store / "cron"
    cron_dir.mkdir(parents=True)
    now = jobs._hermes_now().isoformat()
    legacy = {
        "id": "legacy-job",
        "name": "legacy",
        "prompt": "run",
        "schedule": {"kind": "interval", "minutes": 30, "display": "every 30m"},
        "schedule_display": "every 30m",
        "repeat": {"times": 5, "completed": 2},
        "enabled": True,
        "state": "scheduled",
        "created_at": now,
        "next_run_at": now,
        "last_run_at": now,
        "last_status": "error",
        "last_error": "prior failure",
        "last_delivery_error": "prior delivery failure",
        "fire_claim": {"at": now, "by": "live-owner"},
        "run_claim": None,
        "paused_at": None,
        "provider_snapshot": "provider-at-create",
        "model_snapshot": "model-at-create",
        "deliver": "local",
    }
    (cron_dir / "jobs.json").write_text(
        json.dumps({"jobs": [legacy], "updated_at": now}),
        encoding="utf-8",
    )

    migrated = jobs.load_jobs()

    assert len(migrated) == 1
    assert migrated[0]["next_run_at"] == now
    assert migrated[0]["last_status"] == "error"
    assert migrated[0]["fire_claim"] == {"at": now, "by": "live-owner"}
    assert migrated[0]["repeat"] == {"times": 5, "completed": 2}
    definition = _raw_definitions(cron_store)[0]
    assert not (_RUNTIME_FIELDS & definition.keys())
    assert definition["repeat"] == {"times": 5}
    assert (cron_dir / "runtime.db").exists()


def test_migration_retries_after_definition_write_failure(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the runtime commit must leave a lossless retry path."""
    cron_dir = cron_store / "cron"
    cron_dir.mkdir(parents=True)
    now = jobs._hermes_now().isoformat()
    legacy = {
        "id": "retry-job",
        "prompt": "retry",
        "schedule": {"kind": "interval", "minutes": 15},
        "repeat": {"times": 3, "completed": 1},
        "enabled": True,
        "next_run_at": now,
        "fire_claim": {"at": now, "by": "owner-before-crash"},
    }
    jobs_path = cron_dir / "jobs.json"
    jobs_path.write_text(json.dumps({"jobs": [legacy]}), encoding="utf-8")
    original_write = jobs._write_job_definitions_unlocked

    def fail_definition_write(_definitions: list[dict], **_kwargs) -> None:
        """Simulate termination after runtime.db commits."""
        raise OSError("simulated definition write failure")

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", fail_definition_write)
    with pytest.raises(OSError, match="simulated definition write failure"):
        jobs.load_jobs()

    still_legacy = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"][0]
    assert still_legacy["next_run_at"] == now
    assert still_legacy["fire_claim"]["by"] == "owner-before-crash"
    assert (cron_dir / "runtime.db").exists()

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", original_write)
    recovered = jobs.load_jobs()[0]
    assert recovered["next_run_at"] == now
    assert recovered["fire_claim"]["by"] == "owner-before-crash"
    assert recovered["repeat"] == {"times": 3, "completed": 1}
    assert "next_run_at" not in _raw_definitions(cron_store)[0]


def test_definition_export_excludes_runtime_and_timestamps(cron_store: Path) -> None:
    """Stable export must contain operator intent only."""
    created = jobs.create_job(
        prompt="report",
        schedule="every 2h",
        name="report",
        repeat=4,
        deliver="local",
    )
    jobs.mark_job_run(created["id"], False, "transient failure")

    exported = jobs.export_job_definitions()

    assert len(exported) == 1
    definition = exported[0]
    assert not (_RUNTIME_FIELDS & definition.keys())
    assert definition["repeat"] == {"times": 4}
    assert definition["id"] == created["id"]
    assert definition["prompt"] == "report"


def test_definition_only_reconcile_preserves_existing_runtime(cron_store: Path) -> None:
    """Reapplying stable definitions must not erase cadence or run history."""
    created = jobs.create_job(
        prompt="reconcile",
        schedule="every 3h",
        name="reconcile",
        repeat=4,
        deliver="local",
    )
    jobs.mark_job_run(created["id"], True)
    definitions = jobs.export_job_definitions()
    before = jobs.get_job(created["id"])
    assert before is not None

    jobs.save_jobs(definitions)

    after = jobs.get_job(created["id"])
    assert after is not None
    assert after["next_run_at"] == before["next_run_at"]
    assert after["last_run_at"] == before["last_run_at"]
    assert after["last_status"] == "ok"
    assert after["repeat"] == {"times": 4, "completed": 1}


def test_journal_recovers_interrupted_ordinary_definition_update(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed JSON materialization must roll forward on the next read."""
    created = jobs.create_job(
        prompt="before",
        schedule="every 1h",
        name="journal",
        deliver="local",
    )
    original_write = jobs._write_job_definitions_unlocked
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()
    merged = jobs.load_jobs()
    merged[0]["prompt"] = "after"

    def fail_definition_write(_definitions: list[dict], **_kwargs) -> None:
        """Simulate interruption after the SQLite journal commits."""
        raise OSError("simulated materialization failure")

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", fail_definition_write)
    with pytest.raises(OSError, match="simulated materialization failure"):
        jobs.save_jobs(merged)
    assert definitions_path.read_bytes() == before

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", original_write)
    recovered = jobs.get_job(created["id"])
    assert recovered is not None
    assert recovered["prompt"] == "after"
    assert _raw_definitions(cron_store)[0]["prompt"] == "after"


def test_late_shrink_merge_preserves_the_sibling_runtime_generation(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late definition recovery must retain the sibling's complete runtime."""
    from cron.runtime_state import (
        load_pending_definitions,
        load_runtime_states,
        replace_runtime_states,
    )

    cron_dir = cron_store / "cron"
    created = jobs.create_job(
        prompt="before",
        schedule="every 1h",
        name="journal-race",
        deliver="local",
    )
    snapshot = jobs.load_jobs()
    snapshot[0]["prompt"] = "after"
    late_definition: dict[str, Any] = {
        "id": "bbbbbbbbbbbb",
        "name": "late-create",
        "prompt": "late",
        "schedule": {"kind": "interval", "minutes": 5},
        "enabled": True,
        "deliver": "local",
    }
    now = jobs._hermes_now().isoformat()
    late_runtime: dict[str, Any] = {
        "state": "scheduled",
        "created_at": now,
        "next_run_at": now,
        "last_run_at": now,
        "last_status": "ok",
        "last_error": None,
        "fire_claim": {"at": now, "by": "sibling", "id": "fire-token"},
        "run_claim": {"at": now, "by": "sibling", "id": "run-token"},
        "repeat_completed": 7,
    }
    expected_late_runtime = jobs._reconcile_runtime_state(
        late_definition,
        late_runtime,
    )
    original_stage = jobs.stage_runtime_and_definitions
    original_replace = jobs.atomic_replace
    stage_calls = 0

    def inject_sibling_before_initial_stage(
        stage_cron_dir,
        states,
        definitions,
        **kwargs,
    ):
        """Land a degraded-lock sibling between reconcile and initial stage."""
        nonlocal stage_calls
        stage_calls += 1
        if stage_calls == 1:
            jobs_path = cron_dir / "jobs.json"
            payload = json.loads(jobs_path.read_text(encoding="utf-8"))
            payload["jobs"].append(late_definition)
            jobs_path.write_text(json.dumps(payload), encoding="utf-8")
            sibling_states = load_runtime_states(cron_dir)
            sibling_states[late_definition["id"]] = late_runtime
            replace_runtime_states(cron_dir, sibling_states)
        return original_stage(
            stage_cron_dir,
            states,
            definitions,
            **kwargs,
        )

    def fail_materialization(_source, _destination):
        raise OSError("simulated post-merge materialization failure")

    monkeypatch.setattr(
        jobs,
        "stage_runtime_and_definitions",
        inject_sibling_before_initial_stage,
    )
    monkeypatch.setattr(jobs, "atomic_replace", fail_materialization)
    with pytest.raises(OSError, match="post-merge materialization failure"):
        jobs.save_jobs(snapshot)

    pending = load_pending_definitions(cron_dir)
    assert pending is not None
    assert {item["id"] for item in pending} == {
        created["id"],
        late_definition["id"],
    }
    assert load_runtime_states(cron_dir)[late_definition["id"]] == (
        expected_late_runtime
    )

    monkeypatch.setattr(jobs, "stage_runtime_and_definitions", original_stage)
    monkeypatch.setattr(jobs, "atomic_replace", original_replace)
    assert {item["id"] for item in jobs.load_jobs()} == {
        created["id"],
        late_definition["id"],
    }


def test_runtime_only_save_preserves_sibling_definition_and_runtime(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling create after final verification must keep its runtime row."""
    from cron.runtime_state import load_runtime_states, replace_runtime_states

    cron_dir = cron_store / "cron"
    created = jobs.create_job(
        prompt="stable",
        schedule="every 1h",
        name="runtime-only-race",
        deliver="local",
    )
    snapshot = jobs.load_jobs()
    late_definition = {
        "id": "cccccccccccc",
        "name": "late-runtime-only",
        "prompt": "late",
        "schedule": {"kind": "interval", "minutes": 10},
        "enabled": True,
        "deliver": "local",
    }
    now = jobs._hermes_now().isoformat()
    late_runtime = {
        "state": "scheduled",
        "created_at": now,
        "next_run_at": now,
        "last_run_at": now,
        "last_status": "error",
        "last_error": "preserve this exact sibling state",
        "fire_claim": {"at": now, "by": "sibling", "id": "late-fire"},
        "repeat_completed": 3,
    }
    original_merge = jobs._merge_unexpected_disk_jobs
    merge_calls = 0

    def inject_sibling_after_runtime_only_verification(
        definitions,
        *,
        removed_ids=None,
    ):
        """Land after the verification read but before runtime persistence."""
        nonlocal merge_calls
        merge_calls += 1
        merged = original_merge(definitions, removed_ids=removed_ids)
        if merge_calls == 2:
            jobs_path = cron_dir / "jobs.json"
            payload = json.loads(jobs_path.read_text(encoding="utf-8"))
            payload["jobs"].append(late_definition)
            jobs_path.write_text(json.dumps(payload), encoding="utf-8")
            sibling_states = load_runtime_states(cron_dir)
            sibling_states[late_definition["id"]] = late_runtime
            replace_runtime_states(cron_dir, sibling_states)
        return merged

    monkeypatch.setattr(
        jobs,
        "_merge_unexpected_disk_jobs",
        inject_sibling_after_runtime_only_verification,
    )
    jobs.save_jobs(snapshot)

    assert {item["id"] for item in _raw_definitions(cron_store)} == {
        created["id"],
        late_definition["id"],
    }
    assert load_runtime_states(cron_dir)[late_definition["id"]] == late_runtime


def test_runtime_only_save_fails_closed_on_same_job_state_conflict(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded stale snapshot cannot erase a newer ownership claim."""
    from cron.runtime_state import load_runtime_states, replace_runtime_states

    cron_dir = cron_store / "cron"
    created = jobs.create_job(
        prompt="stable",
        schedule="every 1h",
        name="same-job-runtime-race",
        deliver="local",
    )
    snapshot = jobs.load_jobs()
    claim = {
        "id": "newer-fire-owner",
        "at": jobs._hermes_now().isoformat(),
        "by": "sibling-writer",
    }
    original_merge = jobs._merge_unexpected_disk_jobs
    merge_calls = 0

    def inject_conflict_after_runtime_only_verification(
        definitions,
        *,
        removed_ids=None,
    ):
        """Advance this job's state after verify but before persistence."""
        nonlocal merge_calls
        merge_calls += 1
        merged = original_merge(definitions, removed_ids=removed_ids)
        if merge_calls == 2:
            sibling_states = load_runtime_states(cron_dir)
            sibling_states[created["id"]]["fire_claim"] = claim
            replace_runtime_states(cron_dir, sibling_states)
        return merged

    monkeypatch.setattr(
        jobs,
        "_merge_unexpected_disk_jobs",
        inject_conflict_after_runtime_only_verification,
    )

    with pytest.raises(RuntimeError, match="changed concurrently"):
        jobs.save_jobs(snapshot)

    assert load_runtime_states(cron_dir)[created["id"]]["fire_claim"] == claim


def test_stale_loaded_snapshot_cannot_overwrite_claim_acquired_before_save(
    cron_store: Path,
) -> None:
    """CAS must compare with the caller's load, not a save-time reread."""
    from cron.runtime_state import load_runtime_states, replace_runtime_states

    cron_dir = cron_store / "cron"
    created = jobs.create_job(
        prompt="stable",
        schedule="every 1h",
        name="pre-save-claim-race",
        deliver="local",
    )
    stale_snapshot = jobs.load_jobs()
    claim = {
        "id": "claim-after-load",
        "at": jobs._hermes_now().isoformat(),
        "by": "new-owner",
    }
    newer_states = load_runtime_states(cron_dir)
    newer_states[created["id"]]["fire_claim"] = claim
    replace_runtime_states(cron_dir, newer_states)

    with pytest.raises(RuntimeError, match="changed concurrently"):
        jobs.save_jobs(stale_snapshot)

    assert load_runtime_states(cron_dir)[created["id"]]["fire_claim"] == claim


@pytest.mark.parametrize("claim_field", ["fire_claim", "run_claim"])
def test_stale_public_get_job_snapshot_cannot_clear_newer_ownership_claim(
    cron_store: Path,
    claim_field: str,
) -> None:
    """Public single-job snapshots retain their ownership CAS expectation."""
    from cron.runtime_state import load_runtime_states, replace_runtime_states

    cron_dir = cron_store / "cron"
    created = jobs.create_job(prompt="stable", schedule="30m")
    stale_snapshot = jobs.get_job(created["id"])
    assert stale_snapshot is not None

    claim_id = "newer-public-snapshot-owner"
    newer_states = load_runtime_states(cron_dir)
    newer_states[created["id"]][claim_field] = {
        "id": claim_id,
        "at": jobs._hermes_now().isoformat(),
        "by": "new-owner",
    }
    replace_runtime_states(cron_dir, newer_states)

    with pytest.raises(RuntimeError, match="changed concurrently"):
        jobs.save_jobs([stale_snapshot])

    claim = load_runtime_states(cron_dir)[created["id"]][claim_field]
    assert claim is not None
    assert claim["id"] == claim_id


def test_removed_runtime_row_participates_in_expected_state_cas(
    cron_store: Path,
) -> None:
    """A stale delete must not erase a row absent from its CAS expectations."""
    from cron.runtime_state import load_runtime_states, merge_runtime_states

    cron_dir = cron_store / "cron"
    created = jobs.create_job(prompt="delete", schedule="every 1h")
    current = load_runtime_states(cron_dir)

    with pytest.raises(RuntimeError, match="changed concurrently"):
        merge_runtime_states(
            cron_dir,
            {},
            removed_ids={created["id"]},
            expected_states={},
        )

    assert created["id"] in load_runtime_states(cron_dir)


def test_runtime_cas_validation_and_write_share_one_sqlite_transaction(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling cannot commit between the expected-state read and owned write."""
    import cron.runtime_state as runtime_state

    cron_dir = cron_store / "cron"
    created = jobs.create_job(prompt="atomic cas", schedule="every 1h")
    original = runtime_state.load_runtime_states(cron_dir)[created["id"]]
    stale_write = {**original, "last_status": "stale-writer"}
    newer_write = {**original, "last_status": "newer-writer"}
    cas_read = threading.Event()
    release_cas_writer = threading.Event()
    sibling_committed = threading.Event()
    writer_errors: list[BaseException] = []
    real_deserialize = runtime_state._deserialize
    blocked_once = False

    def block_after_cas_read(job_id: str, payload: str) -> dict:
        nonlocal blocked_once
        state = real_deserialize(job_id, payload)
        if threading.current_thread().name == "stale-cas-writer" and not blocked_once:
            blocked_once = True
            cas_read.set()
            assert release_cas_writer.wait(timeout=5)
        return state

    monkeypatch.setattr(runtime_state, "_deserialize", block_after_cas_read)

    def run_stale_writer() -> None:
        try:
            runtime_state.merge_runtime_states(
                cron_dir,
                {created["id"]: stale_write},
                expected_states={created["id"]: original},
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            writer_errors.append(exc)

    def run_sibling_writer() -> None:
        with sqlite3.connect(cron_dir / "runtime.db", timeout=5) as conn:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "UPDATE job_runtime SET state_json = ? WHERE job_id = ?",
                (runtime_state._serialize(newer_write), created["id"]),
            )
        sibling_committed.set()

    stale_thread = threading.Thread(target=run_stale_writer, name="stale-cas-writer")
    stale_thread.start()
    assert cas_read.wait(timeout=5)

    sibling_thread = threading.Thread(target=run_sibling_writer, name="newer-cas-writer")
    sibling_thread.start()
    sibling_committed_before_release = sibling_committed.wait(timeout=0.2)
    release_cas_writer.set()
    stale_thread.join(timeout=5)
    sibling_thread.join(timeout=5)

    assert not stale_thread.is_alive()
    assert not sibling_thread.is_alive()
    assert writer_errors == []
    assert sibling_committed_before_release is False
    assert runtime_state.load_runtime_states(cron_dir)[created["id"]] == newer_write


def test_pending_definition_acknowledgement_is_generation_fenced(
    cron_store: Path,
) -> None:
    """An older materializer must never acknowledge a newer journal entry."""
    from cron.runtime_state import (
        clear_pending_definitions,
        load_pending_definition_generation,
        load_runtime_states,
        stage_runtime_and_definitions,
    )

    cron_dir = cron_store / "cron"
    created = jobs.create_job(prompt="initial", schedule="every 1h")
    states = load_runtime_states(cron_dir)
    definition_a = jobs.export_job_definitions()
    definition_a[0]["prompt"] = "generation-a"
    definition_b = jobs.export_job_definitions()
    definition_b[0]["prompt"] = "generation-b"

    generation_a = stage_runtime_and_definitions(cron_dir, states, definition_a)
    generation_b = stage_runtime_and_definitions(cron_dir, states, definition_b)

    assert generation_a != generation_b
    assert clear_pending_definitions(
        cron_dir,
        expected_generation_id=generation_a,
    ) is False
    pending, pending_generation = load_pending_definition_generation(cron_dir)
    assert pending_generation == generation_b
    assert pending is not None
    assert pending[0]["id"] == created["id"]
    assert pending[0]["prompt"] == "generation-b"


def test_explicit_replace_recovers_corrupt_runtime_and_pending_rows(
    cron_store: Path,
) -> None:
    """Authorized replacement must not deserialize artifacts it will replace."""
    from cron.runtime_state import load_pending_definitions, load_runtime_states

    cron_dir = cron_store / "cron"
    created = jobs.create_job(prompt="before", schedule="every 1h")
    replacement = jobs.export_job_definitions()
    replacement[0]["prompt"] = "after"
    db_path = cron_dir / "runtime.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE job_runtime SET state_json = ? WHERE job_id = ?",
            ("{not-json", created["id"]),
        )
        conn.execute(
            "UPDATE pending_definitions SET definitions_json = ? "
            "WHERE singleton = 1",
            ("{not-json",),
        )

    jobs.save_jobs(replacement, replace=True)

    runtime = load_runtime_states(cron_dir)
    assert set(runtime) == {created["id"]}
    assert runtime[created["id"]]["state"] == "scheduled"
    assert load_pending_definitions(cron_dir) is None
    recovered = jobs.get_job(created["id"], include_terminal=True)
    assert recovered is not None
    assert recovered["prompt"] == "after"


def test_interrupted_explicit_replace_records_base_and_rolls_forward(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authorized replacement remains recoverable without unproven replay."""
    from cron.runtime_state import load_pending_definition_record

    cron_dir = cron_store / "cron"
    created = jobs.create_job(prompt="before", schedule="every 1h")
    original_definitions = jobs.export_job_definitions()
    replacement = [dict(definition) for definition in original_definitions]
    replacement[0]["prompt"] = "after"
    real_write = jobs._write_job_definitions_unlocked

    def interrupt_materialization(*_args, **_kwargs) -> None:
        raise OSError("simulated explicit replacement interruption")

    monkeypatch.setattr(
        jobs,
        "_write_job_definitions_unlocked",
        interrupt_materialization,
    )
    with pytest.raises(OSError, match="explicit replacement interruption"):
        jobs.save_jobs(replacement, replace=True)

    pending, generation_id, base_digest = load_pending_definition_record(cron_dir)
    assert pending == replacement
    assert generation_id
    assert base_digest == jobs._canonical_digest(original_definitions)

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", real_write)
    recovered = jobs.get_job(created["id"], include_terminal=True)
    assert recovered is not None
    assert recovered["prompt"] == "after"


def test_runtime_only_exact_replace_removes_unlisted_runtime_rows(
    cron_store: Path,
) -> None:
    """Explicit replacement stays exact even when definitions need no rewrite."""
    from cron.runtime_state import load_runtime_states

    cron_dir = cron_store / "cron"
    retained = jobs.create_job(prompt="keep", schedule="every 1h")
    removed = jobs.create_job(prompt="remove", schedule="every 2h")
    snapshot = [job for job in jobs.load_jobs() if job["id"] == retained["id"]]

    jobs.save_jobs(snapshot, replace=True)

    assert {item["id"] for item in _raw_definitions(cron_store)} == {retained["id"]}
    assert set(load_runtime_states(cron_dir)) == {retained["id"]}
    assert jobs.get_job(removed["id"], include_terminal=True) is None


def test_journal_recovers_interrupted_definition_delete(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted explicit deletion must roll forward, never resurrect runtime."""
    created = jobs.create_job(
        prompt="delete",
        schedule="every 1h",
        name="delete",
        deliver="local",
    )
    original_write = jobs._write_job_definitions_unlocked

    def fail_definition_write(_definitions: list[dict], **_kwargs) -> None:
        """Simulate interruption after deletion is journaled."""
        raise OSError("simulated delete materialization failure")

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", fail_definition_write)
    with pytest.raises(OSError, match="simulated delete materialization failure"):
        jobs.remove_job(created["id"])

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", original_write)
    assert jobs.get_job(created["id"]) is None
    assert _raw_definitions(cron_store) == []


@pytest.mark.parametrize("record_base_digest", [False, True])
def test_unproven_pending_generation_cannot_replace_newer_definitions(
    cron_store: Path,
    record_base_digest: bool,
) -> None:
    """A stale recovery journal must fail closed instead of deleting a new job."""
    from cron.runtime_state import (
        load_pending_definitions,
        load_runtime_states,
        stage_runtime_and_definitions,
    )

    cron_dir = cron_store / "cron"
    original = jobs.create_job(prompt="original", schedule="every 1h")
    stale_definitions = jobs.export_job_definitions()
    base_digest = jobs._canonical_digest(stale_definitions)
    sibling = jobs.create_job(prompt="newer sibling", schedule="every 2h")
    runtime = load_runtime_states(cron_dir)

    stale_definitions[0]["prompt"] = "stale writer update"
    stage_runtime_and_definitions(
        cron_dir,
        runtime,
        stale_definitions,
        replace=False,
        expected_states=runtime,
        base_definitions_digest=base_digest if record_base_digest else None,
    )

    with pytest.raises(RuntimeError, match="pending definition generation"):
        jobs.load_jobs()

    assert {item["id"] for item in _raw_definitions(cron_store)} == {
        original["id"],
        sibling["id"],
    }
    assert load_pending_definitions(cron_dir) == stale_definitions


def test_authoritative_definition_load_prunes_orphaned_runtime_rows(
    cron_store: Path,
) -> None:
    """A hand-deployed removal cannot leave claims or tombstones for reuse."""
    from cron.runtime_state import load_runtime_states

    cron_dir = cron_store / "cron"
    created = jobs.create_job(prompt="remove by deploy", schedule="1m")
    jobs.mark_job_run(created["id"], True)
    assert created["id"] in load_runtime_states(cron_dir)

    (cron_dir / "jobs.json").write_text(
        json.dumps(
            {"jobs": [], "updated_at": jobs._hermes_now().isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )

    assert jobs.load_jobs() == []
    assert load_runtime_states(cron_dir) == {}


def test_runtime_database_is_owner_only(cron_store: Path) -> None:
    """Volatile state must retain the jobs store's owner-only mode."""
    jobs.create_job(
        prompt="secure",
        schedule="every 1h",
        name="secure",
        deliver="local",
    )
    mode = (cron_store / "cron" / "runtime.db").stat().st_mode & 0o777
    assert mode == 0o600


def test_schedule_reconcile_resets_stale_cadence_and_claims(cron_store: Path) -> None:
    """Source-controlled cadence changes must not inherit old leases/counters."""
    created = jobs.create_job(
        prompt="cadence",
        schedule="every 1h",
        name="cadence",
        repeat=5,
        deliver="local",
    )
    jobs.mark_job_run(created["id"], True)
    definitions = jobs.export_job_definitions()
    definitions[0]["schedule"] = {
        "kind": "interval",
        "minutes": 10,
        "display": "every 10m",
    }
    definitions[0]["schedule_display"] = "every 10m"
    definitions[0]["repeat"] = {"times": 2}

    jobs.save_jobs(definitions)

    reconciled = jobs.get_job(created["id"])
    assert reconciled is not None
    assert reconciled["repeat"] == {"times": 2, "completed": 0}
    assert reconciled.get("next_run_at") is None
    assert reconciled.get("fire_claim") is None
    assert reconciled.get("run_claim") is None
    assert reconciled["state"] == "scheduled"
    assert reconciled["last_status"] == "ok"


def test_terminal_one_shot_tombstones_without_definition_write(
    cron_store: Path,
) -> None:
    """Terminal lifecycle must hide the job but retain reproducible intent."""
    created = jobs.create_job(
        prompt="once",
        schedule="1m",
        name="once",
        deliver="local",
    )
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()

    jobs.mark_job_run(created["id"], True)

    assert definitions_path.read_bytes() == before
    assert jobs.get_job(created["id"]) is None
    assert jobs.list_jobs(include_disabled=True) == []
    internal = jobs.load_jobs()
    assert internal[0]["runtime_tombstone"]["reason"] == "repeat_limit"
    assert jobs.export_job_definitions()[0]["id"] == created["id"]


def test_missing_definitions_fail_closed_without_erasing_runtime(
    cron_store: Path,
) -> None:
    """Transient JSON absence cannot destroy cadence or terminal tombstones."""
    from cron.runtime_state import load_runtime_states

    created = jobs.create_job(
        prompt="once",
        schedule="1m",
        name="preserve-on-missing",
        deliver="local",
    )
    jobs_path = cron_store / "cron" / "jobs.json"
    definitions_bytes = jobs_path.read_bytes()
    jobs.mark_job_run(created["id"], True)
    runtime_before = load_runtime_states(cron_store / "cron")
    assert runtime_before[created["id"]]["runtime_tombstone"]

    jobs_path.unlink()
    with pytest.raises(RuntimeError, match="jobs.json is missing"):
        jobs.load_jobs()
    assert load_runtime_states(cron_store / "cron") == runtime_before

    jobs_path.write_bytes(definitions_bytes)
    recovered = jobs.load_jobs()
    assert recovered[0]["runtime_tombstone"]["reason"] == "repeat_limit"


def test_missing_definitions_are_valid_for_true_first_run(cron_store: Path) -> None:
    """No JSON and no runtime rows still represents an empty first-run store."""
    assert jobs.load_jobs() == []


def test_large_runtime_snapshot_avoids_sql_variable_limit(cron_store: Path) -> None:
    """Complete snapshots must work above common 999-variable SQLite limits."""
    records = [
        {
            "id": f"job-{index}",
            "prompt": "bulk",
            "schedule": {"kind": "interval", "minutes": 60},
            "repeat": {"times": None, "completed": index},
            "enabled": True,
            "state": "scheduled",
            "next_run_at": jobs._hermes_now().isoformat(),
        }
        for index in range(1_100)
    ]

    jobs.save_jobs(records)

    assert len(jobs.load_jobs()) == 1_100
    assert len(_raw_definitions(cron_store)) == 1_100


def test_pending_definition_schema_migration_is_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first opens cannot race the pending-journal migration."""
    from cron import runtime_state

    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    database = cron_dir / "runtime.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE job_runtime (job_id TEXT PRIMARY KEY, state_json TEXT NOT NULL)"
        )
        connection.execute(
            """CREATE TABLE pending_definitions (
                 singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                 definitions_json TEXT NOT NULL
               )"""
        )
        connection.execute(
            "INSERT INTO pending_definitions(singleton, definitions_json) "
            "VALUES (1, '[]')"
        )

    contenders = 8
    barrier = threading.Barrier(contenders)
    real_connect = runtime_state.sqlite3.connect

    class RacingConnection(sqlite3.Connection):
        """Force the unfenced implementation to share one stale schema view."""

        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if (
                "PRAGMA table_info(pending_definitions)" in sql
                and not self.in_transaction
            ):
                barrier.wait(timeout=5)
            return cursor

    def connect_with_race(*args, **kwargs):
        kwargs["factory"] = RacingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(runtime_state.sqlite3, "connect", connect_with_race)

    def initialize() -> None:
        connection = runtime_state._connect(cron_dir)
        connection.close()

    with ThreadPoolExecutor(max_workers=contenders) as executor:
        futures = [executor.submit(initialize) for _ in range(contenders)]
    errors = [future.exception() for future in futures if future.exception()]

    assert errors == []
    with real_connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(pending_definitions)")
        }
    assert {"generation_id", "base_definitions_digest"} <= columns


def test_runtime_schema_initialization_retries_transient_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brief replica lock must not make first-open migration permanently fail."""
    from cron import runtime_state

    cron_dir = tmp_path / "cron"
    real_connect = runtime_state.sqlite3.connect
    attempts = 0
    injected_lock = False

    class LockedOnceConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal injected_lock
            if sql.strip().upper() == "BEGIN IMMEDIATE" and not injected_lock:
                injected_lock = True
                raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, parameters)

    def connect_with_one_lock(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        kwargs["factory"] = LockedOnceConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(runtime_state.sqlite3, "connect", connect_with_one_lock)

    connection = runtime_state._connect(cron_dir)
    connection.close()

    assert injected_lock
    assert attempts == 2


def test_runtime_state_is_profile_local(tmp_path: Path) -> None:
    """Identical job IDs in two profiles must never share runtime state."""
    job_id = "shared-id"
    definition = {
        "id": job_id,
        "name": "shared",
        "prompt": "run",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "deliver": "local",
    }
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"

    with jobs.use_cron_store(home_a):
        jobs.save_jobs([definition])
        jobs.mark_job_run(job_id, True)
    with jobs.use_cron_store(home_b):
        jobs.save_jobs([definition])
        jobs.mark_job_run(job_id, False, "profile-b-only")

    with jobs.use_cron_store(home_a):
        loaded_a = jobs.get_job(job_id)
        assert loaded_a is not None
        assert loaded_a["last_status"] == "ok"
    with jobs.use_cron_store(home_b):
        loaded_b = jobs.get_job(job_id)
        assert loaded_b is not None
        assert loaded_b["last_status"] == "error"
        assert loaded_b["last_error"] == "profile-b-only"


def test_concurrent_fire_claims_remain_atomic_without_definition_write(
    cron_store: Path,
) -> None:
    """Exactly one contender wins while jobs.json remains byte-stable."""
    created = jobs.create_job(
        prompt="claim",
        schedule="every 1h",
        name="claim",
        deliver="local",
    )
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()
    barrier = threading.Barrier(3)
    results: list[bool] = []

    def contend() -> None:
        """Attempt one fire claim after both contenders are ready."""
        barrier.wait()
        results.append(jobs.claim_job_for_fire(created["id"]))

    threads = [threading.Thread(target=contend) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(results) == [False, True]
    assert definitions_path.read_bytes() == before
