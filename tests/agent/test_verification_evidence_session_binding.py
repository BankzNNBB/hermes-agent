"""Regression: terminal-recorded evidence must land under the agent's real session.

``terminal_tool`` records verification evidence keyed by ``session_id``, and
``build_verify_on_stop_nudge`` reads it back keyed by ``agent.session_id``.
When the tool call arrives without an explicit ``session_id`` argument the
recorder used to fall back to the literal ``"default"``, so the write and the
read landed on two different ledger rows: a passing ``hermes verify`` could
never satisfy the stop-guard for the session that actually edited the files.

The fix resolves the session from ``gateway.session_context.get_session_env``
(the same source the rest of the tool consults) before falling back. These
tests fail if that resolution is removed.
"""

import pytest

from agent import verification_evidence as ve


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(ve, "_db_path", lambda: tmp_path / "verification_evidence.db")
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    return tmp_path


def _record(cwd, session_id):
    return ve.record_terminal_result(
        command="python -m pytest -q",
        cwd=cwd,
        session_id=session_id,
        exit_code=0,
        output="1 passed",
    )


def test_resolver_prefers_explicit_session(monkeypatch):
    """An explicitly passed session id always wins over the environment."""
    from tools.terminal_tool import _resolve_evidence_session_id

    monkeypatch.setenv("HERMES_SESSION_ID", "from-env")
    assert _resolve_evidence_session_id("explicit", None, None) == "explicit"


def test_resolver_falls_back_to_session_env(monkeypatch):
    """No explicit id: use the agent's real session, not the ``default`` bucket."""
    from tools.terminal_tool import _resolve_evidence_session_id

    monkeypatch.setenv("HERMES_SESSION_ID", "20260904_041951_13e2bebd")
    resolved = _resolve_evidence_session_id(None, None, None)
    assert resolved == "20260904_041951_13e2bebd"
    assert resolved != "default"


def test_resolver_uses_task_id_before_default(monkeypatch):
    """Task id keeps its place in the chain when no session is bound."""
    from tools.terminal_tool import _resolve_evidence_session_id

    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    assert _resolve_evidence_session_id(None, "task-7", None) == "task-7"


def test_resolver_defaults_when_nothing_is_bound(monkeypatch):
    """With no session and no task the literal default is still the last resort."""
    from tools.terminal_tool import _resolve_evidence_session_id

    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    assert _resolve_evidence_session_id(None, None, None) == "default"


def test_verify_evidence_satisfies_stop_guard_for_same_session(ledger, monkeypatch):
    """End-to-end: edit + record under the resolved session clears the nudge.

    This is the exact shape of the reported bug — the edit is marked under the
    agent's session id while the terminal recorded evidence with no explicit
    session argument.
    """
    from agent.verification_stop import build_verify_on_stop_nudge
    from tools.terminal_tool import _resolve_evidence_session_id

    session = "20260904_041951_13e2bebd"
    monkeypatch.setenv("HERMES_SESSION_ID", session)
    changed = str(ledger / "mod.py")
    (ledger / "mod.py").write_text("x = 1\n")

    ve.mark_workspace_edited(session_id=session, cwd=ledger, paths=[changed])
    assert build_verify_on_stop_nudge(session_id=session, changed_paths=[changed]) is not None

    # The tool call carries no session_id — exactly how the agent invokes it.
    _record(ledger, _resolve_evidence_session_id(None, None, None))

    assert build_verify_on_stop_nudge(session_id=session, changed_paths=[changed]) is None


def test_default_bucket_does_not_satisfy_other_sessions(ledger):
    """Guard against over-correcting: evidence must stay session-scoped."""
    from agent.verification_stop import build_verify_on_stop_nudge

    changed = str(ledger / "mod.py")
    (ledger / "mod.py").write_text("x = 1\n")

    ve.mark_workspace_edited(session_id="session-a", cwd=ledger, paths=[changed])
    _record(ledger, "session-b")

    # Proof recorded under a different session must NOT clear session-a.
    assert build_verify_on_stop_nudge(session_id="session-a", changed_paths=[changed]) is not None
