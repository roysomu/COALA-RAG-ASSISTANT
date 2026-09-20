from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.embeddings import Embeddings, vector_literal, transient
from src.config import SetupError


def test_batch_dimensions(settings):
    client = Mock()
    client.models.embed_content.side_effect = lambda **kw: SimpleNamespace(embeddings=[SimpleNamespace(values=[.01] * 768) for _ in kw['contents']])
    result = Embeddings(settings, client).embed(['x'] * 33)
    assert len(result) == 33 and all(len(v) == 768 for v in result)
    assert client.models.embed_content.call_count == 2
    for call in client.models.embed_content.call_args_list:
        assert call.kwargs['model'] == 'gemini-embedding-2'
        assert call.kwargs['config'].output_dimensionality == 768


@pytest.mark.parametrize('vector', [[.1] * 767, [0.] * 768, [float('nan')] * 768, [float('inf')] * 768])
def test_bad_vectors(vector):
    with pytest.raises(ValueError):
        vector_literal(vector)


def test_no_silent_dimension_change(settings):
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.] * 3072)])
    with pytest.raises(SetupError):
        Embeddings(settings, client).embed(['test'])
    assert client.models.embed_content.call_count == 1


def test_transient_classification():
    assert transient(TimeoutError())
    for code in (400, 401, 403):
        error = RuntimeError()
        error.code = code
        assert not transient(error)


def test_sdk_constructor_offline(settings):
    client = Embeddings(settings)
    assert client.client is not None
    client.client.close()


def test_auth_failure_not_retried(settings):
    client = Mock()
    error = RuntimeError('do not leak this')
    error.code = 401
    client.models.embed_content.side_effect = error
    with pytest.raises(SetupError) as caught:
        Embeddings(settings, client).embed(['text'])
    assert 'do not leak' not in str(caught.value)
    assert client.models.embed_content.call_count == 1


def test_transient_embedding_retry(settings, monkeypatch):
    client = Mock()
    client.models.embed_content.side_effect = [TimeoutError(), SimpleNamespace(embeddings=[SimpleNamespace(values=[.1] * 768)])]
    monkeypatch.setattr(Embeddings._batch.retry, 'sleep', lambda _: None)
    assert len(Embeddings(settings, client).embed(['text'])[0]) == 768
    assert client.models.embed_content.call_count == 2


def test_embedding2_separate_content_objects(settings):
    from google.genai import _transformers
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[.1] * 768)] * 2)
    Embeddings(settings, client).embed(['first chunk', 'second chunk'])
    submitted = client.models.embed_content.call_args.kwargs['contents']
    # Exercise the installed SDK conversion that otherwise aggregates bare strings.
    normalized = _transformers.t_contents(submitted)
    assert len(normalized) == 2
    assert normalized[0].parts[0].text == 'title: none | text: first chunk'
    assert normalized[1].parts[0].text == 'title: none | text: second chunk'


def test_query_uses_search_instruction(settings):
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[.1] * 768)])
    Embeddings(settings, client).query('What is retention?')
    contents = client.models.embed_content.call_args.kwargs['contents']
    assert contents[0].parts[0].text == 'task: search result | query: What is retention?'


def test_sdk_transport_failures_are_transient():
    import httpx
    assert transient(httpx.ReadTimeout('timeout'))
    assert transient(httpx.ConnectError('connection lost'))
