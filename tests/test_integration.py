"""Explicit opt-in tests. Use a dedicated hosted PostgreSQL test database for DB checks."""
import os
from uuid import uuid4
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv('RUN_DB_TESTS') != '1', reason='Set RUN_DB_TESTS=1 for live hosted PostgreSQL checks.')
def test_live_database_round_trip():
    from src.config import load_settings
    from src.db import Database, create_pool, create_checkpointer, initialize_database
    from src.ingestion import DocumentStore
    from src.chunking import split_text
    from src.memory import MemoryStore
    settings = load_settings()
    settings.require_database()
    from unittest.mock import Mock
    fake_embeddings = Mock()
    fake_embeddings.query.return_value = [0.01] * 768
    fake_embeddings.embed.return_value = [[0.01] * 768]
    with create_pool(settings) as pool:
        db, saver = Database(pool), create_checkpointer(pool)
        initialize_database(db, saver, settings)
        initialize_database(db, saver, settings)
        suffix = uuid4().hex
        store = DocumentStore(db)
        memories = MemoryStore(db, fake_embeddings)
        doc_id = None
        try:
            assert store.replace(f'test-{suffix}.md', suffix, split_text('Temporary DB integration test.'), [[0.01] * 768], settings.embedding_model) == 'added'
            doc_id = store.find_hash(suffix)['id']
            assert store.replace(f'test-{suffix}.md', suffix, split_text('Temporary DB integration test.'), [[0.01] * 768], settings.embedding_model) == 'skipped'
            mid = memories.save(suffix, suffix, 'episodic', 'Temporary integration memory.')
            assert mid
            # Exercise real PostgreSQL checkpoints with mocked model calls, so DB tests are unpaid.
            from src.agents import Agents
            from src.graph import build_graph
            from langchain_core.messages import HumanMessage
            from src import prompts
            router, retriever = Mock(), Mock()
            retriever.search.return_value = []
            from pydantic import SecretStr
            test_settings = settings.model_copy(update={'gemini_api_key': SecretStr('offline-test-key')})
            def complete(model, messages, metadata):
                if messages[0]['content'] == prompts.CRITIC:
                    return '{"decision":"accept","reason":"Appropriately abstains"}', model
                if messages[0]['content'] == prompts.CURATOR:
                    return '{"summary":"A temporary database checkpoint test.","memories":[]}', model
                return 'The evidence is insufficient.', model
            router.complete.side_effect = complete
            graph = build_graph(Agents(test_settings, router, retriever, memories), saver)
            result = graph.invoke({'question': 'Unanswerable test question', 'user_id': suffix, 'thread_id': suffix,
                'messages': [HumanMessage(content='Unanswerable test question')]},
                config={'configurable': {'thread_id': suffix}}, durability='exit')
            assert result['evidence'] == [] and result['summary_saved']
            assert len(list(saver.list({'configurable': {'thread_id': suffix}}))) == 1
            saver.delete_thread(suffix)
            assert saver.get_tuple({'configurable': {'thread_id': suffix}}) is None
            assert memories.list(suffix)
        finally:
            saver.delete_thread(suffix)
            if doc_id:
                store.delete(doc_id)
            memories.delete_all(suffix)
            db.execute('DELETE FROM ingestion_records WHERE filename=%s', (f'test-{suffix}.md',))


@pytest.mark.skipif(os.getenv('RUN_LLM_TESTS') != '1', reason='Set RUN_LLM_TESTS=1 for paid Gemini embedding check.')
def test_live_gemini_embeddings():
    from src.config import load_settings
    from src.embeddings import Embeddings
    settings = load_settings()
    settings.require_embeddings()
    assert len(Embeddings(settings).query('An original test sentence.')) == 768


@pytest.mark.parametrize('provider', ['gemini', 'openai'])
@pytest.mark.skipif(os.getenv('RUN_LLM_TESTS') != '1', reason='Set RUN_LLM_TESTS=1 for paid chat checks.')
def test_live_chat(provider):
    from src.config import load_settings
    from src.llm_router import LLMRouter
    from langsmith import tracing_context
    settings = load_settings()
    if provider not in settings.providers:
        pytest.skip(f'Configure the {provider} API key for this opt-in check.')
    model = settings.simple_chat_model if provider == 'gemini' else settings.complex_chat_model
    with tracing_context(enabled=False):
        result, actual = LLMRouter(settings).complete(model, [{'role': 'user', 'content': 'Reply with the word ready.'}])
    assert result and actual


@pytest.mark.skipif(os.getenv('RUN_LANGSMITH_TESTS') != '1', reason='Set RUN_LANGSMITH_TESTS=1 for a live trace check.')
def test_live_langsmith():
    from datetime import datetime, timezone
    import time
    from src.config import load_settings
    from langsmith import Client
    settings = load_settings()
    if not settings.langsmith_api_key.get_secret_value():
        pytest.fail('Configure LANGSMITH_API_KEY before the opt-in trace check.')
    client = Client(api_key=settings.langsmith_api_key.get_secret_value(), api_url=settings.langsmith_endpoint or None, auto_batch_tracing=False)
    run_id = uuid4()
    try:
        client.create_run(name='credential-opt-in-smoke', run_type='chain', inputs={'test': 'synthetic'},
                          outputs={'result': 'ok'}, end_time=datetime.now(timezone.utc), id=run_id,
                          project_name=settings.langsmith_project)
        client.flush(timeout=10)
        for attempt in range(10):
            try:
                assert str(client.read_run(run_id).id) == str(run_id)
                break
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(.5)
    except Exception:
        pytest.fail('LangSmith trace verification failed. Check the key, endpoint, workspace access, quota, and network.', pytrace=False)
