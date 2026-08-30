"""Tests for rl_training_tool.py — file handle lifecycle and cleanup.

Verifies that _stop_training_run properly closes log file handles,
terminates processes, and handles edge cases on failure paths.
Inspired by PR #715 (0xbyt4).
"""

import json
from unittest.mock import MagicMock

import pytest

import tools.rl_training_tool as rl_tool
from tools.rl_training_tool import RunState, _stop_training_run


def _make_run_state(**overrides) -> RunState:
    """Create a minimal RunState for testing."""
    defaults = {
        "run_id": "test-run-001",
        "environment": "test_env",
        "config": {},
    }
    defaults.update(overrides)
    return RunState(**defaults)


class TestStopTrainingRunFileHandles:
    """Verify that _stop_training_run closes log file handles stored as attributes."""

    def test_closes_all_log_file_handles(self):
        state = _make_run_state()
        files = {}
        for attr in ("api_log_file", "trainer_log_file", "env_log_file"):
            fh = MagicMock()
            setattr(state, attr, fh)
            files[attr] = fh

        _stop_training_run(state)

        for attr, fh in files.items():
            fh.close.assert_called_once()
            assert getattr(state, attr) is None

    def test_clears_file_attrs_to_none(self):
        state = _make_run_state()
        state.api_log_file = MagicMock()

        _stop_training_run(state)

        assert state.api_log_file is None

    def test_close_exception_does_not_propagate(self):
        """If a file handle .close() raises, it must not crash."""
        state = _make_run_state()
        bad_fh = MagicMock()
        bad_fh.close.side_effect = OSError("already closed")
        good_fh = MagicMock()
        state.api_log_file = bad_fh
        state.trainer_log_file = good_fh

        _stop_training_run(state)  # should not raise

        bad_fh.close.assert_called_once()
        good_fh.close.assert_called_once()

    def test_handles_missing_file_attrs(self):
        """RunState without log file attrs should not crash."""
        state = _make_run_state()
        # No log file attrs set at all — getattr(..., None) should handle it
        _stop_training_run(state)  # should not raise


class TestStopTrainingRunProcesses:
    """Verify that _stop_training_run terminates processes correctly."""

    def test_terminates_running_processes(self):
        state = _make_run_state()
        for attr in ("api_process", "trainer_process", "env_process"):
            proc = MagicMock()
            proc.poll.return_value = None  # still running
            setattr(state, attr, proc)

        _stop_training_run(state)

        for attr in ("api_process", "trainer_process", "env_process"):
            getattr(state, attr).terminate.assert_called_once()

    def test_does_not_terminate_exited_processes(self):
        state = _make_run_state()
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited
        state.api_process = proc

        _stop_training_run(state)

        proc.terminate.assert_not_called()

    def test_handles_none_processes(self):
        state = _make_run_state()
        # All process attrs are None by default
        _stop_training_run(state)  # should not raise

    def test_handles_mixed_running_and_exited_processes(self):
        state = _make_run_state()
        # api still running
        api = MagicMock()
        api.poll.return_value = None
        state.api_process = api
        # trainer already exited
        trainer = MagicMock()
        trainer.poll.return_value = 0
        state.trainer_process = trainer
        # env is None
        state.env_process = None

        _stop_training_run(state)

        api.terminate.assert_called_once()
        trainer.terminate.assert_not_called()


class TestStopTrainingRunStatus:
    """Verify status transitions in _stop_training_run."""

    def test_sets_status_to_stopped_when_running(self):
        state = _make_run_state(status="running")
        _stop_training_run(state)
        assert state.status == "stopped"

    def test_does_not_change_status_when_failed(self):
        state = _make_run_state(status="failed")
        _stop_training_run(state)
        assert state.status == "failed"

    def test_does_not_change_status_when_pending(self):
        state = _make_run_state(status="pending")
        _stop_training_run(state)
        assert state.status == "pending"

    def test_no_crash_with_no_processes_and_no_files(self):
        state = _make_run_state()
        _stop_training_run(state)  # should not raise
        assert state.status == "pending"


class TestEnvironmentDiscovery:
    """Verify RL environment discovery covers both upstream and Hermes env trees."""

    def test_scans_upstream_and_recursive_hermes_environments(self, tmp_path, monkeypatch):
        """Hermes envs under environments/** should be listed with upstream Atropos envs."""
        tinker_env_dir = tmp_path / "tinker-atropos" / "tinker_atropos" / "environments"
        tinker_env_dir.mkdir(parents=True)
        (tinker_env_dir / "gsm8k_tinker.py").write_text(
            '''
class GSM8kEnv(BaseEnv):
    """Upstream GSM8K environment."""
    name = "gsm8k"
''',
            encoding="utf-8",
        )

        hermes_env_dir = tmp_path / "environments" / "terminal_test_env"
        hermes_env_dir.mkdir(parents=True)
        (hermes_env_dir / "terminal_test_env.py").write_text(
            '''
class TerminalTestEnv(HermesAgentBaseEnv):
    """Hermes terminal smoke environment."""
    name = "terminal-test"
''',
            encoding="utf-8",
        )

        benchmark_dir = tmp_path / "environments" / "benchmarks" / "tblite"
        benchmark_dir.mkdir(parents=True)
        (benchmark_dir / "tblite_env.py").write_text(
            '''
class TBLiteEvalEnv(TerminalBench2EvalEnv):
    """Indirect Hermes benchmark environment."""
    name = "openthoughts-tblite"
''',
            encoding="utf-8",
        )

        monkeypatch.setattr(rl_tool, "HERMES_ROOT", tmp_path)
        monkeypatch.setattr(rl_tool, "ENVIRONMENTS_DIR", tinker_env_dir)

        names = {env.name for env in rl_tool._scan_environments()}

        assert "gsm8k" in names
        assert "terminal-test" in names
        assert "openthoughts-tblite" in names


class TestRLCredentialGating:
    """Verify read-only RL tools stay visible without training credentials."""

    def test_read_only_tools_are_visible_without_tinker_or_wandb_keys(self, monkeypatch):
        monkeypatch.delenv("TINKER_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from model_tools import get_tool_definitions

        tools = get_tool_definitions(enabled_toolsets=["rl"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert {
            "rl_list_environments",
            "rl_select_environment",
            "rl_get_current_config",
            "rl_edit_config",
            "rl_list_runs",
        }.issubset(names)
        assert "rl_start_training" not in names
        assert "rl_test_inference" not in names

    @pytest.mark.asyncio
    async def test_start_training_reports_all_missing_training_keys(self, monkeypatch):
        monkeypatch.delenv("TINKER_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setattr(rl_tool, "_current_env", "terminal-test")

        result = json.loads(await rl_tool.rl_start_training())

        assert result["error"] == "Missing required training credentials"
        assert result["missing"] == ["TINKER_API_KEY", "WANDB_API_KEY"]
