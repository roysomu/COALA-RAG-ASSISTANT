from contextlib import nullcontext
from unittest.mock import Mock
from src.db import initialize_database, create_pool


def test_idempotent_setup_and_official_migrations(settings):
    db, saver, conn = Mock(), Mock(), Mock()
    db.transaction.side_effect = lambda: nullcontext(conn)
    initialize_database(db, saver, settings)
    initialize_database(db, saver, settings)
    assert saver.setup.call_count == 2
    statements = [call.args[0] for call in conn.execute.call_args_list]
    assert any('CREATE EXTENSION IF NOT EXISTS vector' in sql for sql in statements)
    assert all('IF NOT EXISTS' in sql for sql in statements if sql.strip().startswith('CREATE'))
    assert any('VECTOR(768)' in sql for sql in statements)


def test_hnsw_unavailable_does_not_prevent_setup(settings):
    db, saver, conn = Mock(), Mock(), Mock()
    db.transaction.return_value = nullcontext(conn)
    db.execute.side_effect = RuntimeError('index unavailable')
    assert initialize_database(db, saver, settings) == ['document_chunks', 'memories']
    saver.setup.assert_called_once()


def test_small_pool_config(settings, monkeypatch):
    from pydantic import SecretStr
    settings.database_url = SecretStr('postgresql://u:p@ep-test.neon.tech/db?sslmode=require')
    constructor = Mock()
    monkeypatch.setattr('src.db.ConnectionPool', constructor)
    create_pool(settings)
    kwargs = constructor.call_args.kwargs
    assert kwargs['min_size'] == 0 and kwargs['max_size'] == 5
    assert kwargs['kwargs']['prepare_threshold'] is None
    assert kwargs['kwargs']['autocommit']
