"""Memory-trim coverage for the long-lived messaging gateway housekeeper."""

import gateway.run as gateway_run


class _OneTickStopEvent:
    """Run one housekeeping tick without a sleep or background thread."""

    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_calls_periodic_memory_trim(monkeypatch):
    import hermes_cli.mem_trim as mem_trim

    calls = []
    monkeypatch.setattr(
        mem_trim,
        "trim_memory",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert calls == [{"reason": "messaging gateway housekeeping"}]


def test_gateway_housekeeping_isolates_cron_maintenance_errors(monkeypatch):
    """A failed reconciliation scan must not suppress sibling housekeeping."""
    import hermes_cli.mem_trim as mem_trim

    trim_calls = []
    monkeypatch.setattr(
        mem_trim,
        "trim_memory",
        lambda **kwargs: trim_calls.append(kwargs) or True,
    )

    class _FailingProvider:
        def maintenance(self):
            raise OSError("ledger busy")

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(),
        interval=0,
        cron_provider=_FailingProvider(),
    )

    assert trim_calls == [{"reason": "messaging gateway housekeeping"}]
