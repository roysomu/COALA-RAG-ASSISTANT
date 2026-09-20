from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock
from pathlib import Path
import re
import pytest
from src.checkpoint_retention import select_candidates, CheckpointRetention
from src.config import SetupError

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def thread(name='old', age=40, summary=True, status='archived', user='u'):
    return {'thread_id': name, 'user_id': user, 'last_activity': NOW - timedelta(days=age),
            'summary_saved_at': NOW if summary else None, 'status': status, 'checkpoint_count': 7}


def test_current_and_unsummarized_excluded():
    candidates, skipped = select_candidates([thread('current'), thread('unsaved', summary=False), thread('old')], 30, 20, 'current', now=NOW)
    assert [r['thread_id'] for r in candidates] == ['old']
    assert {r['reason'] for r in skipped} == {'current thread', 'no saved episodic summary'}


def test_force_does_not_override_current():
    candidates, _ = select_candidates([thread('current', summary=False), thread('other', summary=False)], 30, 20, 'current', force=True, now=NOW)
    assert [r['thread_id'] for r in candidates] == ['other']


def test_max_threads_policy_respects_minimum_age():
    rows = [thread('recent', 1), thread('recent-overflow', 2), thread('old-overflow', 40)]
    candidates, _ = select_candidates(rows, 30, 1, now=NOW)
    assert [r['thread_id'] for r in candidates] == ['old-overflow']
    assert candidates[0]['reason'] == 'inactive excess archived thread'


def test_max_thread_ranking_per_user():
    rows = [thread('a1', 40, user='a'), thread('a2', 41, user='a'), thread('b1', 40, user='b')]
    candidates, _ = select_candidates(rows, 30, 1, now=NOW)
    reasons = {r['thread_id']: r['reason'] for r in candidates}
    assert reasons['b1'] == 'inactive'
    assert 'excess' in reasons['a2']


def test_dry_run_never_deletes(settings):
    db, saver = Mock(), Mock()
    db.query.return_value = [thread(age=1000)]
    result = CheckpointRetention(db, saver, settings).prune()
    assert result['dry_run'] and result['candidates']
    saver.delete_thread.assert_not_called()
    db.execute.assert_not_called()


def test_apply_uses_official_api_and_preserves_memories(settings):
    db, saver = Mock(), Mock()
    db.query.return_value = [thread(age=1000)]
    db.maintenance_lock.return_value = nullcontext(True)
    calls = Mock()
    calls.attach_mock(saver.delete_thread, 'delete_thread')
    calls.attach_mock(db.execute, 'execute')
    result = CheckpointRetention(db, saver, settings).prune(apply=True)
    assert result['deleted'] == ['old']
    assert calls.mock_calls[0][0] == 'delete_thread'
    assert calls.mock_calls[1].args[0] == 'DELETE FROM app_threads WHERE thread_id=%s'
    assert not any('memories' in str(call) for call in db.execute.call_args_list)


def test_delete_failure_keeps_tracking(settings):
    db, saver = Mock(), Mock()
    db.query.return_value = [thread(age=1000)]
    db.maintenance_lock.return_value = nullcontext(True)
    saver.delete_thread.side_effect = RuntimeError('mock')
    result = CheckpointRetention(db, saver, settings).prune(apply=True)
    assert result['failed'] == ['old']
    db.execute.assert_not_called()


def test_invocation_excludes_cleanup(settings):
    db = Mock()
    db.maintenance_lock.return_value = nullcontext(False)
    with pytest.raises(SetupError, match='active'):
        CheckpointRetention(db, Mock(), settings).prune(apply=True)


def test_auto_once_per_day(settings):
    settings.auto_prune_checkpoints = True
    db, saver = Mock(), Mock()
    db.maintenance_lock.return_value = nullcontext(True)
    db.query.return_value = [{'last_run_at': datetime.now(timezone.utc)}]
    assert CheckpointRetention(db, saver, settings).auto_prune('current') is None
    saver.delete_thread.assert_not_called()
    db.execute.assert_not_called()


def test_auto_opt_in(settings):
    db = Mock()
    assert CheckpointRetention(db, Mock(), settings).auto_prune('t') is None
    db.maintenance_lock.assert_not_called()


def test_no_direct_checkpoint_table_deletion():
    for root in ('src', 'scripts'):
        for path in Path(root).glob('*.py'):
            assert not re.search(r'DELETE\s+FROM\s+(checkpoints|checkpoint_blobs|checkpoint_writes)\b', path.read_text(), re.I), path


def test_apply_only_previewed_ids(settings):
    db, saver = Mock(), Mock()
    db.query.return_value = [thread('old', age=1000), thread('new-candidate', age=1000)]
    db.maintenance_lock.return_value = nullcontext(True)
    CheckpointRetention(db, saver, settings).prune(apply=True, only_ids={'old'})
    saver.delete_thread.assert_called_once_with('old')


def test_auto_records_run_without_llm(settings):
    settings.auto_prune_checkpoints = True
    db, saver = Mock(), Mock()
    db.maintenance_lock.return_value = nullcontext(True)
    db.query.side_effect = [[], [thread(age=1000)]]
    result = CheckpointRetention(db, saver, settings).auto_prune('current')
    assert result['deleted'] == ['old']
    saver.delete_thread.assert_called_once_with('old')
    assert 'maintenance_runs' in db.execute.call_args.args[0]


def test_manual_delete_checks_owner_and_preserves_memory(settings):
    db, saver = Mock(), Mock()
    db.maintenance_lock.return_value = nullcontext(True)
    db.query.return_value = [{'thread_id': 't'}]
    CheckpointRetention(db, saver, settings).delete_conversation('t', 'alice')
    saver.delete_thread.assert_called_once_with('t')
    assert db.execute.call_args.args == ('DELETE FROM app_threads WHERE thread_id=%s AND user_id=%s', ('t', 'alice'))


def test_manual_delete_rejects_other_user(settings):
    db, saver = Mock(), Mock()
    db.maintenance_lock.return_value = nullcontext(True)
    db.query.return_value = []
    with pytest.raises(SetupError):
        CheckpointRetention(db, saver, settings).delete_conversation('t', 'wrong-user')
    saver.delete_thread.assert_not_called()
