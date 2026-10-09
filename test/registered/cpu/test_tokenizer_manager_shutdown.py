from __future__ import annotations

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.managers import tokenizer_manager as tokenizer_manager_module
from sglang.srt.managers.io_struct import ShutdownReq
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager


class _Watchdog:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def stop(self) -> None:
        self.events.append("watchdog.stop")


def _manager(events: list[object]) -> SimpleNamespace:
    return SimpleNamespace(
        gracefully_exit=True,
        rid_to_state={},
        server_status=ServerStatus.Up,
        _subprocess_watchdog=_Watchdog(events),
        _dispatch_to_scheduler=lambda request: events.append(
            ("dispatch", type(request))
        ),
        dump_requests_before_crash=lambda: events.append("dump_requests"),
        force_exit_handler=lambda: events.append("force_exit_handler"),
    )


def _record_child_cleanup(monkeypatch, events: list[object]) -> None:
    def kill_process_tree(parent_pid, *, include_parent, wait_timeout=None):
        events.append(
            ("kill_process_tree", parent_pid, include_parent, wait_timeout)
        )

    monkeypatch.setattr(
        tokenizer_manager_module, "kill_process_tree", kill_process_tree
    )


def test_sigterm_zero_work_budget_fits_launcher_target_grace() -> None:
    assert (
        tokenizer_manager_module.SIGTERM_WATCHDOG_POLL_INTERVAL_S,
        tokenizer_manager_module.SIGTERM_SCHEDULER_GRACE_TIMEOUT_S,
        tokenizer_manager_module.SIGTERM_CHILD_REAP_TIMEOUT_S,
        tokenizer_manager_module.SIGTERM_ZERO_WORK_SHUTDOWN_BUDGET_S,
    ) == (5, 15, 5, 25)
    assert tokenizer_manager_module.SIGTERM_ZERO_WORK_SHUTDOWN_BUDGET_S < 30


def test_sigterm_watchdog_reaps_children_then_exits_parent_zero(monkeypatch) -> None:
    events: list[object] = []
    manager = _manager(events)

    monkeypatch.setattr(
        tokenizer_manager_module,
        "collect_scheduler_processes",
        lambda: [],
    )
    monkeypatch.setattr(
        tokenizer_manager_module, "get_bool_env_var", lambda _name: False
    )
    _record_child_cleanup(monkeypatch, events)

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(TokenizerManager.sigterm_watchdog(manager))

    assert exc_info.value.code == 0
    assert events == [
        "watchdog.stop",
        ("dispatch", ShutdownReq),
        (
            "kill_process_tree",
            tokenizer_manager_module.os.getpid(),
            False,
            5,
        ),
    ]


def test_sigterm_watchdog_unhealthy_cleanup_exits_nonzero(monkeypatch) -> None:
    events: list[object] = []
    manager = _manager(events)
    manager.server_status = ServerStatus.UnHealthy
    monkeypatch.setattr(
        tokenizer_manager_module,
        "collect_scheduler_processes",
        lambda: [],
    )
    _record_child_cleanup(monkeypatch, events)

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(TokenizerManager.sigterm_watchdog(manager))

    assert exc_info.value.code == 1
    assert events == [
        "dump_requests",
        "force_exit_handler",
        "watchdog.stop",
        ("dispatch", ShutdownReq),
        (
            "kill_process_tree",
            tokenizer_manager_module.os.getpid(),
            False,
            tokenizer_manager_module.SIGTERM_CHILD_REAP_TIMEOUT_S,
        ),
    ]


def test_sigterm_watchdog_force_shutdown_cleanup_exits_nonzero(monkeypatch) -> None:
    events: list[object] = []
    manager = _manager(events)
    monkeypatch.setattr(
        tokenizer_manager_module,
        "collect_scheduler_processes",
        lambda: [],
    )
    monkeypatch.setattr(
        tokenizer_manager_module,
        "get_bool_env_var",
        lambda name: name == "SGL_FORCE_SHUTDOWN",
    )
    _record_child_cleanup(monkeypatch, events)

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(TokenizerManager.sigterm_watchdog(manager))

    assert exc_info.value.code == 1
    assert events == [
        "force_exit_handler",
        "watchdog.stop",
        ("dispatch", ShutdownReq),
        (
            "kill_process_tree",
            tokenizer_manager_module.os.getpid(),
            False,
            tokenizer_manager_module.SIGTERM_CHILD_REAP_TIMEOUT_S,
        ),
    ]


def test_sigterm_watchdog_scheduler_timeout_cleanup_exits_nonzero(
    monkeypatch,
) -> None:
    events: list[object] = []
    manager = _manager(events)
    scheduler = SimpleNamespace(pid=4567)
    monotonic_values = iter((0.0, 16.0))
    monkeypatch.setattr(
        tokenizer_manager_module,
        "collect_scheduler_processes",
        lambda: [scheduler],
    )
    monkeypatch.setattr(
        tokenizer_manager_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        tokenizer_manager_module, "get_bool_env_var", lambda _name: False
    )
    _record_child_cleanup(monkeypatch, events)

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(TokenizerManager.sigterm_watchdog(manager))

    assert exc_info.value.code == 1
    assert events[-2:] == [
        ("dispatch", ShutdownReq),
        (
            "kill_process_tree",
            tokenizer_manager_module.os.getpid(),
            False,
            tokenizer_manager_module.SIGTERM_CHILD_REAP_TIMEOUT_S,
        ),
    ]


def test_sigterm_watchdog_scheduler_clears_before_deadline_exits_zero(
    monkeypatch,
) -> None:
    events: list[object] = []
    manager = _manager(events)
    scheduler = SimpleNamespace(pid=4567)
    scheduler_snapshots = iter(([scheduler], [], []))
    monkeypatch.setattr(
        tokenizer_manager_module,
        "collect_scheduler_processes",
        lambda: next(scheduler_snapshots),
    )
    monkeypatch.setattr(
        tokenizer_manager_module.time,
        "sleep",
        lambda seconds: events.append(("sleep", seconds)),
    )
    monkeypatch.setattr(
        tokenizer_manager_module, "get_bool_env_var", lambda _name: False
    )
    _record_child_cleanup(monkeypatch, events)

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(TokenizerManager.sigterm_watchdog(manager))

    assert exc_info.value.code == 0
    assert events == [
        "watchdog.stop",
        ("dispatch", ShutdownReq),
        ("sleep", 0.1),
        (
            "kill_process_tree",
            tokenizer_manager_module.os.getpid(),
            False,
            tokenizer_manager_module.SIGTERM_CHILD_REAP_TIMEOUT_S,
        ),
    ]


def test_sigterm_watchdog_child_reap_failure_forces_nonclean_exit(
    monkeypatch,
) -> None:
    events: list[object] = []
    manager = _manager(events)

    monkeypatch.setattr(
        tokenizer_manager_module,
        "collect_scheduler_processes",
        lambda: [],
    )
    monkeypatch.setattr(
        tokenizer_manager_module, "get_bool_env_var", lambda _name: False
    )

    def kill_process_tree(parent_pid, *, include_parent, wait_timeout=None):
        events.append(
            ("kill_process_tree", parent_pid, include_parent, wait_timeout)
        )
        if not include_parent:
            raise RuntimeError("injected child reap timeout")

    class ForcedSigkill(BaseException):
        pass

    def fatal_kill(pid, signum):
        events.append(("os.kill", pid, signum))
        raise ForcedSigkill

    monkeypatch.setattr(
        tokenizer_manager_module, "kill_process_tree", kill_process_tree
    )
    monkeypatch.setattr(tokenizer_manager_module.os, "kill", fatal_kill)

    with pytest.raises(ForcedSigkill):
        asyncio.run(TokenizerManager.sigterm_watchdog(manager))

    pid = tokenizer_manager_module.os.getpid()
    assert events == [
        "watchdog.stop",
        ("dispatch", ShutdownReq),
        (
            "kill_process_tree",
            pid,
            False,
            tokenizer_manager_module.SIGTERM_CHILD_REAP_TIMEOUT_S,
        ),
        ("kill_process_tree", pid, True, None),
        ("os.kill", pid, tokenizer_manager_module.signal.SIGKILL),
    ]


@pytest.mark.parametrize("exit_code", (0, 1))
def test_task_wrapper_preserves_parent_exit_code(exit_code: int) -> None:
    script = f"""
import asyncio
import uvloop
from sglang.srt.managers.tokenizer_manager import print_exception_wrapper

async def exit_task():
    raise SystemExit({exit_code})

async def main():
    asyncio.create_task(print_exception_wrapper(exit_task))
    await asyncio.sleep(60)

uvloop.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == exit_code
    assert f"SystemExit: {exit_code}" in completed.stderr
