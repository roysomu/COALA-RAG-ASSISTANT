from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.config import Settings, SetupError
from src.llm_router import choose_route, LLMRouter


def response(text='answer'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.mark.parametrize('question,category', [
    ('What is audit retention?', 'simple'), ('Compare Cove and Beacon.', 'complex'),
    ('Who approves access? When does it expire?', 'complex'), ('x' * 501, 'complex'),
    ('Return a JSON table of tier limits.', 'complex'), ('Diagnose latency across documents.', 'complex'),
])
def test_auto_routing(settings, question, category):
    route = choose_route(question, settings)
    assert route.category == category
    assert route.model == (settings.simple_chat_model if category == 'simple' else settings.complex_chat_model)


@pytest.mark.parametrize('manual,provider', [('Gemini', 'gemini/'), ('OpenAI', 'openai/')])
def test_manual(settings, manual, provider):
    assert choose_route('Compare tiers', settings, manual).model.startswith(provider)


def test_one_provider():
    settings = Settings(_env_file=None, openai_api_key='test')
    assert choose_route('Hello', settings).model.startswith('openai/')
    assert choose_route('Hello', settings, 'Gemini').model.startswith('openai/')


def test_missing_provider():
    with pytest.raises(SetupError, match='GEMINI_API_KEY or OPENAI_API_KEY'):
        choose_route('hello', Settings(_env_file=None))


def test_greeting_skips_rag(settings):
    assert not choose_route('Hello!', settings).rag_needed


@pytest.mark.parametrize('status,attempts', [(429, 2), (503, 2), (401, 1), (400, 1), (403, 1)])
def test_bounded_fallback(settings, status, attempts):
    error = RuntimeError('SECRET raw provider error')
    error.status_code = status
    completion = Mock(side_effect=[error, response()])
    router = LLMRouter(settings, completion)
    if attempts == 2:
        text, model = router.complete(settings.complex_chat_model, [{'role': 'user', 'content': 'hi'}])
        assert model == settings.fallback_chat_model and text == 'answer'
    else:
        with pytest.raises(SetupError) as caught:
            router.complete(settings.complex_chat_model, [{'role': 'user', 'content': 'hi'}])
        assert 'SECRET' not in str(caught.value)
    assert completion.call_count == attempts
    assert all(c.kwargs['num_retries'] == 0 for c in completion.call_args_list)


def test_fallback_failure_stops(settings):
    error = TimeoutError('sensitive')
    completion = Mock(side_effect=error)
    with pytest.raises(SetupError):
        LLMRouter(settings, completion).complete(settings.complex_chat_model, [])
    assert completion.call_count == 2


def test_output_and_input_redaction(settings):
    completion = Mock(return_value=response('test-gemini-key'))
    text, _ = LLMRouter(settings, completion).complete(settings.simple_chat_model, [{'role': 'user', 'content': 'test-openai-key'}])
    assert text == '[REDACTED]'
    assert completion.call_args.kwargs['messages'][0]['content'] == '[REDACTED]'


def test_real_litellm_library_mock_path(settings):
    import litellm
    def completion(**kwargs):
        return litellm.completion(**kwargs, mock_response='Mock provider response.')
    result, model = LLMRouter(settings, completion).complete(settings.simple_chat_model, [{'role': 'user', 'content': 'hello'}])
    assert result == 'Mock provider response.'
    assert model == settings.simple_chat_model
