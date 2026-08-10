"""Regression coverage for cron execution recovery after provider startup.

A direct cron owner can still be alive when the scheduler provider starts and
exit later. Runtime maintenance must eventually classify that persisted attempt
as ``unknown`` without restarting the provider or manufacturing another run.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from hermes_time import now as hermes_now


_IMMUTABLE_AUDIT_FIELDS = (
    "id",
    "job_id",
    "source",
    "host_id",
    "process_id",
    "pid",
    "process_started_at",
    "claimed_at",
    "started_at",
)
_OWNER_START_TIME = int(hermes_now().timestamp())


class _FakeClock:
    def __init__(self) -> None:
        self.current = hermes_now()

    def now(self) -> datetime:
        return self.current

    def advance(self, *, seconds: int = 60) -> None:
        self.current += timedelta(seconds=seconds)


class _OwnerProbe:
    """Deterministic replacement for PID and process-birth liveness probes."""

    def __init__(self, state: str = "live") -> None:
        self.state = state
        self.calls: list[str] = []

    def pid_exists(self, _pid: int) -> bool:
        self.calls.append(self.state)
        if self.state == "indeterminate":
            raise PermissionError("process liveness unavailable")
        return self.state == "live"

    @staticmethod
    def process_start_time(_pid: int) -> int:
        return _OWNER_START_TIME


class _ScriptedStopEvent:
    """Event-like scheduler clock that advances cycles without sleeping."""

    def __init__(self, *, stop_after: int = 2, on_wait=None) -> None:
        self.stop_after = stop_after
        self.on_wait = on_wait
        self.wait_calls = 0
        self._stopped = False

    def is_set(self) -> bool:
        return self._stopped

    def wait(self, _timeout=None) -> bool:
        if self._stopped:
            return True
        self.wait_calls += 1
        if self.on_wait is not None:
            self.on_wait(self.wait_calls)
        if self.wait_calls >= self.stop_after:
            self._stopped = True
        return self._stopped


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    clock = _FakeClock()
    monkeypatch.setattr(
        executions,
        "EXECUTIONS_FILE",
        tmp_path / "cron" / "executions.db",
    )
    monkeypatch.setattr(executions, "_hermes_now", clock.now)
    monkeypatch.setattr(executions, "_emit_execution_state", lambda *_a, **_k: None)
    return executions, clock


def _persist_foreign_running_attempt(
    executions,
    monkeypatch,
    clock: _FakeClock,
    probe: _OwnerProbe,
    *,
    job_id: str,
):
    """Create a direct attempt, then switch identity to the scheduler owner."""
    import gateway.status as process_status

    monkeypatch.setattr(executions, "_process_start_time", probe.process_start_time)
    monkeypatch.setattr(process_status, "_pid_exists", probe.pid_exists)
    monkeypatch.setattr(executions, "_PROCESS_ID", "direct-cron-owner")

    claimed = executions.create_execution(job_id, source="direct")
    clock.advance()
    running = executions.mark_execution_running(claimed["id"])
    assert running is not None

    monkeypatch.setattr(executions, "_PROCESS_ID", "scheduler-provider")
    probe.calls.clear()
    return running


def _run_inprocess_provider(monkeypatch, stop_event) -> None:
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    from cron.scheduler_provider import InProcessCronScheduler

    monkeypatch.setattr(scheduler, "tick", lambda *_a, **_k: 0)
    monkeypatch.setattr(jobs, "record_ticker_heartbeat", lambda **_k: None)
    monkeypatch.setattr(jobs, "record_ticker_error", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "clear_ticker_error", lambda: None)

    InProcessCronScheduler().start(stop_event, interval=60)


def _assert_single_unknown_attempt(executions, before):
    records = executions.list_executions(job_id=before["job_id"], limit=10)
    assert len(records) == 1, "reconciliation must not create a retry or replacement"
    after = records[0]
    assert after["id"] == before["id"]
    assert after["status"] == "unknown"
    assert after["status"] not in {"completed", "failed"}
    assert after["finished_at"]
    assert after["error"] and "unknown" in after["error"].lower()
    assert "restart" not in after["error"].lower()
    assert {field: after[field] for field in _IMMUTABLE_AUDIT_FIELDS} == {
        field: before[field] for field in _IMMUTABLE_AUDIT_FIELDS
    }
    return after


def _assert_single_unchanged_attempt(executions, before) -> None:
    records = executions.list_executions(job_id=before["job_id"], limit=10)
    assert records == [before]
    assert records[0]["status"] == "running"
    assert records[0]["status"] not in {"completed", "failed", "unknown"}


def test_owner_dying_after_provider_start_is_reconciled_without_restart(
    ledger,
    monkeypatch,
):
    executions, clock = ledger
    probe = _OwnerProbe("live")
    before = _persist_foreign_running_attempt(
        executions,
        monkeypatch,
        clock,
        probe,
        job_id="late-owner-death",
    )

    def advance_owner_lifecycle(wait_number: int) -> None:
        if wait_number == 1:
            probe.state = "dead"
            clock.advance()

    stop = _ScriptedStopEvent(stop_after=2, on_wait=advance_owner_lifecycle)

    _run_inprocess_provider(monkeypatch, stop)

    assert probe.calls[0] == "live", "owner must be alive during startup recovery"
    assert "dead" in probe.calls, "the same provider must probe again while running"
    assert stop.wait_calls == 2
    _assert_single_unknown_attempt(executions, before)


def test_runtime_reconciliation_leaves_live_owner_untouched(ledger, monkeypatch):
    executions, clock = ledger
    probe = _OwnerProbe("live")
    before = _persist_foreign_running_attempt(
        executions,
        monkeypatch,
        clock,
        probe,
        job_id="live-owner",
    )

    _run_inprocess_provider(monkeypatch, _ScriptedStopEvent(stop_after=2))

    assert len(probe.calls) >= 2, "runtime cycles must re-check an owner that stays live"
    assert set(probe.calls) == {"live"}
    _assert_single_unchanged_attempt(executions, before)


def test_runtime_reconciliation_leaves_other_host_owner_untouched(
    ledger,
    monkeypatch,
):
    """A shared ledger never interprets a foreign replica's PID namespace."""
    executions, clock = ledger
    probe = _OwnerProbe("dead")
    monkeypatch.setattr(
        executions,
        "_PROCESS_HOST_ID",
        "replica-a",
        raising=False,
    )
    before = _persist_foreign_running_attempt(
        executions,
        monkeypatch,
        clock,
        probe,
        job_id="foreign-host-owner",
    )
    monkeypatch.setattr(
        executions,
        "_PROCESS_HOST_ID",
        "replica-b",
        raising=False,
    )

    assert executions.recover_interrupted_executions() == 0
    assert probe.calls == [], "foreign-host PIDs must never be probed locally"
    _assert_single_unchanged_attempt(executions, before)


def test_runtime_reconciliation_fails_safe_when_liveness_is_indeterminate(
    ledger,
    monkeypatch,
):
    executions, clock = ledger
    probe = _OwnerProbe("indeterminate")
    before = _persist_foreign_running_attempt(
        executions,
        monkeypatch,
        clock,
        probe,
        job_id="indeterminate-owner",
    )

    _run_inprocess_provider(monkeypatch, _ScriptedStopEvent(stop_after=2))

    assert len(probe.calls) >= 2, "runtime cycles must retry an inconclusive probe"
    assert set(probe.calls) == {"indeterminate"}
    _assert_single_unchanged_attempt(executions, before)


def test_unreadable_process_birth_is_treated_as_indeterminate(ledger, monkeypatch):
    executions, clock = ledger
    probe = _OwnerProbe("live")
    before = _persist_foreign_running_attempt(
        executions,
        monkeypatch,
        clock,
        probe,
        job_id="unreadable-process-birth",
    )
    monkeypatch.setattr(executions, "_process_start_time", lambda _pid: None)

    assert executions.recover_interrupted_executions() == 0
    _assert_single_unchanged_attempt(executions, before)


def test_execution_ledger_routes_to_active_profile_home(tmp_path):
    import cron.executions as executions
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "secondary-profile"
    job_id = "profile-scoped-ledger-probe"
    home_token = set_hermes_home_override(str(profile_home))
    try:
        created = executions.create_execution(job_id, source="direct")
        scoped_records = executions.list_executions(job_id=job_id)
    finally:
        reset_hermes_home_override(home_token)

    assert (profile_home / "cron" / "executions.db").is_file()
    assert scoped_records == [created]
    assert executions.list_executions(job_id=job_id) == []


def test_explicit_execution_file_override_wins_active_profile(
    ledger,
    tmp_path,
):
    executions, _clock = ledger
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "secondary-profile"
    job_id = "explicit-ledger-override"
    home_token = set_hermes_home_override(str(profile_home))
    try:
        created = executions.create_execution(job_id, source="direct")
    finally:
        reset_hermes_home_override(home_token)

    assert executions.EXECUTIONS_FILE.is_file()
    assert executions.list_executions(job_id=job_id) == [created]
    assert not (profile_home / "cron" / "executions.db").exists()


def test_repeated_reconciliation_is_idempotent_and_audit_only(ledger, monkeypatch):
    executions, clock = ledger
    probe = _OwnerProbe("dead")
    before = _persist_foreign_running_attempt(
        executions,
        monkeypatch,
        clock,
        probe,
        job_id="repeated-recovery",
    )
    clock.advance()

    assert executions.recover_interrupted_executions() == 1
    after_first = _assert_single_unknown_attempt(executions, before)

    clock.advance()
    assert executions.recover_interrupted_executions() == 0
    assert executions.list_executions(job_id=before["job_id"]) == [after_first]

    assert executions.finish_execution(before["id"], success=True) is None
    assert executions.finish_execution(
        before["id"],
        success=False,
        error="late owner result",
    ) is None
    assert executions.list_executions(job_id=before["job_id"]) == [after_first]


def test_concurrent_reconciliation_classifies_attempt_exactly_once(
    ledger,
    monkeypatch,
):
    executions, clock = ledger
    probe = _OwnerProbe("dead")
    before = _persist_foreign_running_attempt(
        executions,
        monkeypatch,
        clock,
        probe,
        job_id="concurrent-recovery",
    )
    clock.advance()
    ready = threading.Barrier(3)

    def reconcile() -> int:
        ready.wait()
        return executions.recover_interrupted_executions()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reconcile) for _ in range(2)]
        ready.wait()
        results = [future.result(timeout=5) for future in futures]

    assert sorted(results) == [0, 1]
    _assert_single_unknown_attempt(executions, before)


def test_transient_reconciliation_error_does_not_skip_dispatch(monkeypatch):
    from cron.scheduler_provider import InProcessCronScheduler

    stop = _ScriptedStopEvent(stop_after=2)
    recovery_calls: list[int] = []
    tick_calls: list[int] = []
    provider = InProcessCronScheduler()

    def recover() -> int:
        recovery_calls.append(len(recovery_calls) + 1)
        if len(recovery_calls) == 2:
            raise OSError("temporary ledger failure")
        return 0

    monkeypatch.setattr(provider, "recover_interrupted", recover)

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    monkeypatch.setattr(
        scheduler,
        "tick",
        lambda *_a, **_k: tick_calls.append(len(tick_calls) + 1) or 0,
    )
    monkeypatch.setattr(jobs, "record_ticker_heartbeat", lambda **_k: None)
    monkeypatch.setattr(jobs, "record_ticker_error", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "clear_ticker_error", lambda: None)

    provider.start(stop, interval=60)

    assert recovery_calls == [1, 2]
    assert tick_calls == [1, 2]


def test_provider_shutdown_stops_runtime_reconciliation(monkeypatch):
    from cron.scheduler_provider import InProcessCronScheduler

    stop = _ScriptedStopEvent(stop_after=2)
    recovery_states: list[bool] = []
    provider = InProcessCronScheduler()

    def recover() -> int:
        recovery_states.append(stop.is_set())
        return 0

    monkeypatch.setattr(provider, "recover_interrupted", recover)

    import cron.jobs as jobs
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "tick", lambda *_a, **_k: 0)
    monkeypatch.setattr(jobs, "record_ticker_heartbeat", lambda **_k: None)
    monkeypatch.setattr(jobs, "record_ticker_error", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "clear_ticker_error", lambda: None)

    provider.start(stop, interval=60)

    assert stop.is_set()
    assert stop.wait_calls == 2
    assert len(recovery_states) >= 2, "maintenance must run during provider lifetime"
    assert not any(recovery_states), "reconciliation ran after shutdown was requested"
    calls_at_shutdown = len(recovery_states)
    provider.stop()
    assert len(recovery_states) == calls_at_shutdown


def test_external_provider_reconciles_again_during_warm_maintenance(monkeypatch):
    """Scale-to-zero providers piggyback recovery on existing housekeeping."""
    from plugins.cron_providers.chronos import ChronosCronScheduler

    provider = ChronosCronScheduler()
    stop = threading.Event()
    calls: list[str] = []
    monkeypatch.setattr(
        provider,
        "recover_interrupted",
        lambda: calls.append("recover") or 0,
    )
    monkeypatch.setattr(provider, "reconcile", lambda: calls.append("reconcile"))

    provider.start(stop)
    provider.maintenance()

    assert calls == ["recover", "reconcile", "recover"]


def test_external_provider_maintenance_retries_after_transient_error(monkeypatch):
    """One failed warm scan must not disable later reconciliation cycles."""
    from plugins.cron_providers.chronos import ChronosCronScheduler

    provider = ChronosCronScheduler()
    stop = threading.Event()
    calls: list[int] = []

    def recover() -> int:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise OSError("ledger busy")
        return 0

    monkeypatch.setattr(provider, "recover_interrupted", recover)
    monkeypatch.setattr(provider, "reconcile", lambda: None)
    provider.start(stop)
    provider.maintenance()

    assert calls == [1, 2]


def test_external_provider_maintenance_routes_and_stops_between_profiles(
    tmp_path,
    monkeypatch,
):
    """Multiplex recovery uses each store and never starts a post-stop scan."""
    from hermes_constants import get_hermes_home
    from plugins.cron_providers.chronos import ChronosCronScheduler

    provider = ChronosCronScheduler()
    stop = threading.Event()
    homes = [tmp_path / "alpha", tmp_path / "beta"]
    provider._stop_event = stop
    provider._profile_homes = [("alpha", homes[0]), ("beta", homes[1])]
    observed = []

    def recover() -> int:
        observed.append(get_hermes_home().resolve())
        stop.set()
        return 0

    monkeypatch.setattr(provider, "recover_interrupted", recover)

    provider.maintenance()

    assert observed == [homes[0].resolve()]


def test_inherited_process_identity_is_regenerated_after_fork(
    ledger,
    monkeypatch,
):
    """A fork child must not mint rows carrying its parent's process UUID."""
    executions, _clock = ledger
    import gateway.status as process_status

    child_pid = 41_001
    parent_pid = 41_002
    monkeypatch.setattr(executions, "_PROCESS_ID", "inherited-parent-identity")
    monkeypatch.setattr(executions.os, "getpid", lambda: child_pid)
    monkeypatch.setattr(executions, "_process_start_time", lambda _pid: 123)
    created = executions.create_execution("fork-child", source="direct")
    executions.mark_execution_running(created["id"])

    monkeypatch.setattr(executions.os, "getpid", lambda: parent_pid)
    monkeypatch.setattr(process_status, "_pid_exists", lambda pid: pid != child_pid)

    assert executions.recover_interrupted_executions() == 1
    assert executions.latest_execution("fork-child")["status"] == "unknown"


def test_execution_connection_follows_active_profile(tmp_path, monkeypatch):
    """The opened ledger must match the profile resolved after module import."""
    import cron.executions as executions

    import_home = tmp_path / "import-home"
    active_home = tmp_path / "active-profile"
    import_file = import_home / "cron" / "executions.db"
    active_file = active_home / "cron" / "executions.db"

    # Preserve the compatibility seam as if the module had originally loaded
    # under ``import_home``. A later profile context must still choose its own
    # ledger rather than opening that stale import-time path.
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", import_file)
    monkeypatch.setattr(executions, "_IMPORT_EXECUTIONS_FILE", import_file)
    monkeypatch.setattr(executions, "get_hermes_home", lambda: active_home)
    monkeypatch.setattr(executions, "_emit_execution_state", lambda *_a, **_k: None)

    created = executions.create_execution("profile-routed", source="direct")

    assert active_file.is_file()
    assert not import_file.exists()
    latest = executions.latest_execution("profile-routed")
    assert latest is not None
    assert latest["id"] == created["id"]
