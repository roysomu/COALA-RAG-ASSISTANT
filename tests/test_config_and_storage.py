import pytest
from pydantic import ValidationError
from unittest.mock import Mock
from src.config import Settings
from src.storage_monitor import exceeds_threshold, storage_report


@pytest.mark.parametrize('kwargs', [
    {'embedding_dimensions': 1536}, {'checkpoint_durability': 'sync'},
    {'checkpoint_retention_days': 0}, {'max_retained_threads': -1},
    {'recent_messages_to_keep': 0}, {'max_messages_per_thread': 0},
])
def test_config_invariants(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_db_secret_hidden():
    secret = 'postgresql://user:super-secret@localhost/db'
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, database_url=secret)
    assert secret not in str(caught.value)
    assert 'super-secret' not in str(caught.value)


def test_pooled_neon_url():
    settings = Settings(_env_file=None, database_url='postgresql://u:p@ep-demo-pooler.us-east-2.aws.neon.tech/db?sslmode=require')
    assert 'u:p' not in repr(settings)


def test_supabase_url():
    settings = Settings(_env_file=None, database_url='postgresql://postgres:encoded@db.project.supabase.co:5432/postgres?sslmode=require')
    assert settings.database_provider == 'Supabase'


def test_supabase_pooler_url():
    settings = Settings(_env_file=None, database_url='postgresql://postgres.project:encoded@aws-0-us-west-1.pooler.supabase.com:6543/postgres?sslmode=require')
    assert settings.database_provider == 'Supabase'


def test_storage_threshold():
    assert not exceeds_threshold(350 * 1024 * 1024, 350)
    assert exceeds_threshold(350 * 1024 * 1024 + 1, 350)


def test_storage_report():
    db = Mock()
    db.query.side_effect = [[{'bytes': 400 * 1024 * 1024, 'pretty': '400 MB'}],
        [{'table_name': 'checkpoints', 'total_size': '10 MB'}], [],
        [{'threads': 1, 'checkpoints': 2, 'documents': 3, 'chunks': 4}], [{'memory_type': 'episodic', 'count': 2}]]
    report = storage_report(db)
    assert report['warning'] and report['counts']['chunks'] == 4


def test_disabled_tracing_needs_no_key():
    settings = Settings(_env_file=None)
    assert not settings.langsmith_tracing
    from src.service import AssistantService
    service = AssistantService(settings, Mock(), Mock(), Mock(), Mock())
    assert service.trace_client is None
    with service.tracing('t'):
        from langsmith.run_helpers import get_tracing_context
        assert get_tracing_context()['enabled'] is False
