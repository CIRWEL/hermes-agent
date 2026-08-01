"""End-to-end management contracts for runtime-completed (tombstoned) jobs.

A repeat-exhausted declaration is retained for reproducibility but must stay
reachable through the supported management surfaces: it can be listed
(``include_completed``), removed, edited, and revived (resume/trigger reset
the repeat budget together with the tombstone).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

import cron.jobs as jobs
from cron.runtime_state import load_runtime_states


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


def _completed_one_shot() -> dict:
    """Create a one-shot job and drive it to its terminal tombstone."""
    created = jobs.create_job(
        prompt="once",
        schedule="1m",
        name="finished-once",
        deliver="local",
    )
    assert jobs.mark_job_run(created["id"], True)
    return created


def _completed_recurring(times: int = 2) -> dict:
    """Create a finite recurring job and exhaust its repeat budget."""
    created = jobs.create_job(
        prompt="recur",
        schedule="every 1h",
        name="finished-recurring",
        repeat=times,
        deliver="local",
    )
    for _ in range(times):
        assert jobs.mark_job_run(created["id"], True)
    return created


def _completed_past_one_shot() -> dict:
    """Persist a one-shot whose run_at is far in the past, then tombstone it."""
    past = "2001-01-01T00:00:00"
    definition = {
        "id": "past-oneshot",
        "name": "past-oneshot",
        "prompt": "ran long ago",
        "schedule": {"kind": "once", "run_at": past, "display": past},
        "schedule_display": past,
        "repeat": {"times": 1, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "next_run_at": past,
        "deliver": "local",
    }
    jobs.save_jobs([definition])
    assert jobs.mark_job_run("past-oneshot", True)
    return definition


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_completed_job_hidden_by_default_but_listable(cron_store: Path) -> None:
    """include_completed exposes terminal declarations with state=completed."""
    created = _completed_one_shot()

    assert jobs.list_jobs() == []
    assert jobs.list_jobs(include_disabled=True) == []

    listed = jobs.list_jobs(include_completed=True)
    assert [job["id"] for job in listed] == [created["id"]]
    assert listed[0]["state"] == "completed"
    assert listed[0]["runtime_tombstone"]["reason"] == "repeat_limit"


def test_completed_job_reachable_via_get_and_resolve(cron_store: Path) -> None:
    """Explicit opt-in lookups reach the terminal record; defaults never do."""
    created = _completed_one_shot()

    assert jobs.get_job(created["id"]) is None
    assert jobs.resolve_job_ref(created["id"]) is None
    assert jobs.resolve_job_ref("finished-once") is None

    fetched = jobs.get_job(created["id"], include_completed=True)
    assert fetched is not None
    assert fetched["state"] == "completed"
    by_name = jobs.resolve_job_ref("finished-once", include_completed=True)
    assert by_name is not None
    assert by_name["id"] == created["id"]


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def test_completed_job_can_be_removed(cron_store: Path) -> None:
    """Removal deletes the retained declaration and its runtime row."""
    created = _completed_one_shot()

    assert jobs.remove_job(created["id"]) is True

    assert jobs.load_jobs() == []
    assert jobs.export_job_definitions() == []
    assert created["id"] not in load_runtime_states(cron_store / "cron")
    raw = json.loads(
        (cron_store / "cron" / "jobs.json").read_text(encoding="utf-8")
    )
    assert raw["jobs"] == []


def test_completed_job_can_be_removed_by_name(cron_store: Path) -> None:
    """Name-based references resolve terminal declarations for removal."""
    _completed_one_shot()

    assert jobs.remove_job("finished-once") is True
    assert jobs.load_jobs() == []


# ---------------------------------------------------------------------------
# Revive
# ---------------------------------------------------------------------------


def test_resume_revives_completed_recurring_job(cron_store: Path) -> None:
    """Resume clears the tombstone and restores the full repeat budget."""
    created = _completed_recurring(times=2)
    assert jobs.get_job(created["id"]) is None

    revived = jobs.resume_job(created["id"])

    assert revived is not None
    assert not revived.get("runtime_tombstone")
    assert revived["state"] == "scheduled"
    assert revived["repeat"] == {"times": 2, "completed": 0}
    assert revived["next_run_at"]
    # Visible through default surfaces again.
    assert jobs.get_job(created["id"]) is not None
    assert [job["id"] for job in jobs.list_jobs()] == [created["id"]]


def test_trigger_revives_completed_one_shot_for_immediate_fire(
    cron_store: Path,
) -> None:
    """Trigger revives a terminal one-shot so its dispatch is claimable again."""
    _completed_past_one_shot()

    revived = jobs.trigger_job("past-oneshot")

    assert revived is not None
    assert not revived.get("runtime_tombstone")
    assert revived["repeat"] == {"times": 1, "completed": 0}
    assert revived["next_run_at"]
    # The dispatch-limit guard must accept the revived fire instead of
    # re-tombstoning it as a stale claim.
    assert jobs.claim_dispatch("past-oneshot") is True
    # And completing that run re-tombstones the declaration cleanly.
    assert jobs.mark_job_run("past-oneshot", True)
    assert jobs.get_job("past-oneshot") is None
    assert jobs.get_job("past-oneshot", include_completed=True) is not None


def test_resume_completed_past_one_shot_raises_actionable_error(
    cron_store: Path,
) -> None:
    """A past one-shot cannot silently re-arm; the error points at revival paths."""
    _completed_past_one_shot()

    with pytest.raises(ValueError, match="Cannot revive completed job"):
        jobs.resume_job("past-oneshot")
    # Still retained and reachable afterwards.
    assert jobs.get_job("past-oneshot", include_completed=True) is not None


def test_definition_edit_revives_with_fresh_repeat_budget(cron_store: Path) -> None:
    """An operator edit revives the job AND resets the exhausted repeat count.

    Without the budget reset the revived job would satisfy the
    ``completed >= times`` dispatch-limit guard on the very next due scan and
    be re-tombstoned before it ever fired.
    """
    created = _completed_one_shot()

    updated = jobs.update_job(created["id"], {"prompt": "edited prompt"})

    assert updated is not None
    assert not updated.get("runtime_tombstone")
    assert updated["repeat"] == {"times": 1, "completed": 0}
    assert jobs.get_job(created["id"]) is not None
    assert jobs.claim_dispatch(created["id"]) is True


def test_update_without_definition_change_keeps_job_completed(
    cron_store: Path,
) -> None:
    """A no-op update is not a revival: the terminal state must survive."""
    created = _completed_one_shot()

    updated = jobs.update_job(created["id"], {"prompt": created["prompt"]})

    assert updated is not None
    assert updated["state"] == "completed"
    assert updated["runtime_tombstone"]["reason"] == "repeat_limit"
    assert jobs.get_job(created["id"]) is None


# ---------------------------------------------------------------------------
# cronjob tool surface
# ---------------------------------------------------------------------------


def test_cronjob_tool_lists_removes_and_revives_completed_jobs(
    cron_store: Path,
) -> None:
    """The agent-facing tool exposes the full terminal-management pathway."""
    from tools.cronjob_tools import cronjob

    created = _completed_recurring(times=1)

    hidden = json.loads(cronjob(action="list"))
    assert hidden["count"] == 0
    shown = json.loads(cronjob(action="list", include_completed=True))
    assert shown["count"] == 1
    assert shown["jobs"][0]["state"] == "completed"
    assert shown["jobs"][0]["completed_reason"] == "repeat_limit"

    paused = json.loads(cronjob(action="pause", job_id=created["id"]))
    assert paused["success"] is False
    assert "resume" in paused["error"]

    resumed = json.loads(cronjob(action="resume", job_id=created["id"]))
    assert resumed["success"] is True
    assert resumed["job"]["state"] == "scheduled"
    assert resumed["job"]["repeat"] == "once"

    # Complete it again, then remove the terminal declaration via the tool.
    assert jobs.mark_job_run(created["id"], True)
    removed = json.loads(cronjob(action="remove", job_id=created["id"]))
    assert removed["success"] is True
    assert jobs.load_jobs() == []
