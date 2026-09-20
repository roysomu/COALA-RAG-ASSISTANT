from unittest.mock import Mock
import pytest
from src.memory import MemoryStore, MEMORY_TYPES


@pytest.mark.parametrize('memory_type', MEMORY_TYPES)
def test_types(memory_type):
    db, embed = Mock(), Mock()
    db.query.side_effect = [[], [{'id': 'new'}]]
    embed.embed.return_value = [[.1] * 768]
    assert MemoryStore(db, embed).save('user', 'thread', memory_type, 'Prefers concise examples.') == 'new'
    assert memory_type in db.query.call_args.args[1]


def test_dedup_before_embedding():
    db, embed = Mock(), Mock()
    db.query.return_value = [{'id': 'existing'}]
    assert MemoryStore(db, embed).save('u', 't', 'semantic', 'Stable preference') == 'existing'
    embed.query.assert_not_called()
    embed.embed.assert_not_called()


@pytest.mark.parametrize('content', ['password=supersecret', 'postgresql://u:p@host/db', 'sk-proj-123456789', 'API_KEY: confidential'])
def test_secret_rejection(content):
    db, embed = Mock(), Mock()
    assert MemoryStore(db, embed).save('u', 't', 'semantic', content) is None
    db.query.assert_not_called()
    embed.query.assert_not_called()


def test_invalid_type():
    with pytest.raises(ValueError):
        MemoryStore(Mock(), Mock()).save('u', 't', 'working', 'bad')


def test_retrieval_scoped_and_compact():
    db, embed = Mock(), Mock()
    db.query.return_value = []
    embed.query.return_value = [.1] * 768
    MemoryStore(db, embed).retrieve('alice', 'question', 'thread-1')
    assert db.query.call_count == 3
    for call in db.query.call_args_list:
        sql, params = call.args
        assert 'left(content, 500)' in sql
        assert 'user_id=%s' in sql and 'alice' in params and 'thread-1' in params


def test_memory_delete_scoped():
    db = Mock()
    MemoryStore(db, Mock()).delete('alice', 'id-1')
    assert db.execute.call_args.args[1] == ('alice', 'id-1', 'alice')
    assert 'summary_saved_at=NULL' in db.execute.call_args.args[0]


@pytest.mark.parametrize('content', ['My password is hunter2', 'OPENAI_API_KEY=confidentialvalue', 'password="multiple word secret"'])
def test_secret_assignment_variants(content):
    db, embed = Mock(), Mock()
    assert MemoryStore(db, embed).save('u', 't', 'semantic', content) is None
    embed.query.assert_not_called()
