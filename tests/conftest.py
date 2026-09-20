import os
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

import socket
import pytest
from unittest.mock import Mock
from src.config import Settings


@pytest.fixture(autouse=True)
def offline_defaults(monkeypatch, request):
    if request.node.get_closest_marker('integration'):
        return
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    def blocked(*args, **kwargs):
        raise AssertionError('Network access is forbidden in default tests.')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


@pytest.fixture
def settings():
    return Settings(_env_file=None, gemini_api_key='test-gemini-key', openai_api_key='test-openai-key')


@pytest.fixture
def evidence():
    return [{'id': 'chunk-1', 'document_id': 'doc-1', 'filename': 'security_policy.md',
             'chunk_index': 3, 'page': None, 'score': .9, 'content': 'Security audit events are retained for 365 days.'}]


@pytest.fixture
def graph_dependencies(settings, evidence):
    from src.agents import Agents
    router, retriever, memories = Mock(), Mock(), Mock()
    retriever.search.return_value = evidence
    memories.retrieve.return_value = []
    memories.save.return_value = 'memory-1'
    router.complete.side_effect = [
        ('Audit events are retained for 365 days [security_policy.md, chunk 3].', settings.simple_chat_model),
        ('{"decision":"accept","reason":"supported"}', settings.simple_chat_model),
        ('{"summary":"Discussed 365-day audit retention.","memories":[]}', settings.simple_chat_model),
    ]
    return Agents(settings, router, retriever, memories), router, retriever, memories
