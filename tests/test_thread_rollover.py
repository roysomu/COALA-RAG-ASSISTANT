from contextlib import nullcontext
from unittest.mock import Mock
from datetime import datetime, timezone
from src.thread_service import ThreadService
from src.service import AssistantService


def test_activity_sql_increment(settings):
    db = Mock()
    service = ThreadService(db, settings, Mock())
    service.record_success('thread', 'user', True)
    sql, params = db.execute.call_args.args
    assert 'message_count=app_threads.message_count+1' in sql
    assert 'last_activity=now()' in sql
    assert params == ('thread', 'user', True, True)


def test_rollover_at_fifty_stores_summary_first(settings):
    db, memories = Mock(), Mock()
    db.query.return_value = [{'message_count': 50, 'summary_saved_at': None}]
    memories.save.return_value = 'summary-id'
    conn = Mock()
    db.transaction.return_value = nullcontext(conn)
    calls = Mock()
    calls.attach_mock(memories.save, 'save')
    calls.attach_mock(db.execute, 'execute')
    calls.attach_mock(conn.execute, 'transaction_write')
    next_id = ThreadService(db, settings, memories).rollover('old-thread', 'u', 'Summary')
    assert next_id != 'old-thread'
    assert calls.mock_calls[0][0] == 'save'
    assert memories.save.call_args.args[:4] == ('u', 'old-thread', 'episodic', 'Summary')
    assert "status='archived'" in conn.execute.call_args_list[0].args[0]
    assert conn.execute.call_args_list[1].args[1][2].obj == {'rollover_from': 'old-thread'}


def test_no_rollover_before_limit(settings):
    db, memories = Mock(), Mock()
    db.query.return_value = [{'message_count': 49, 'summary_saved_at': None}]
    assert ThreadService(db, settings, memories).rollover('t', 'u', 'summary') is None
    memories.save.assert_not_called()
    db.execute.assert_not_called()


def test_existing_summary_reused(settings):
    db, memories, conn = Mock(), Mock(), Mock()
    db.query.return_value = [{'message_count': 50, 'summary_saved_at': datetime.now(timezone.utc)}]
    memories.latest_episode.return_value = {'id': 'episode', 'content': 'summary'}
    db.transaction.return_value = nullcontext(conn)
    assert ThreadService(db, settings, memories).rollover('t', 'u', 'summary')
    memories.save.assert_not_called()


def test_invocation_exit_durability_and_success(settings):
    db, graph, threads, retention = Mock(), Mock(), Mock(), Mock()
    db.invocation_lock.return_value = nullcontext()
    threads.get.return_value = {'message_count': 2, 'status': 'active', 'metadata': {}}
    graph.invoke.return_value = {'summary': 'summary', 'summary_saved': True, 'final_answer': 'answer'}
    threads.rollover.return_value = None
    result = AssistantService(settings, db, graph, threads, retention).ask('Question', 't', 'u')
    assert result['final_answer'] == 'answer'
    assert graph.invoke.call_args.kwargs['durability'] == 'exit'
    assert graph.invoke.call_args.kwargs['config']['configurable']['thread_id'] == 't'
    threads.record_success.assert_called_once_with('t', 'u', True)


def test_new_thread_carries_episode_without_old_messages(settings):
    db, graph, threads = Mock(), Mock(), Mock()
    db.invocation_lock.return_value = nullcontext()
    threads.get.return_value = {'message_count': 0, 'status': 'active', 'metadata': {'rollover_from': 'old'}}
    threads.memories.latest_episode.return_value = {'content': 'Prior rolling summary'}
    graph.invoke.return_value = {'summary': 'new', 'summary_saved': True}
    AssistantService(settings, db, graph, threads, Mock()).ask('New question', 'new', 'user')
    graph_input = graph.invoke.call_args.args[0]
    assert graph_input['summary'] == 'Prior rolling summary'
    assert len(graph_input['messages']) == 1
    threads.memories.latest_episode.assert_called_once_with('user', 'old')


def test_failed_invocation_does_not_increment(settings):
    import pytest
    db, graph, threads = Mock(), Mock(), Mock()
    db.invocation_lock.return_value = nullcontext()
    threads.get.return_value = {'message_count': 0, 'status': 'active', 'metadata': {}}
    graph.invoke.side_effect = RuntimeError('mock failure')
    with pytest.raises(RuntimeError):
        AssistantService(settings, db, graph, threads, Mock()).ask('Question', 't', 'u')
    threads.record_success.assert_not_called()
