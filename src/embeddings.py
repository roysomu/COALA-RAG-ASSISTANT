import math
from httpx import TimeoutException, NetworkError
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from src.config import SetupError
from src.security import redact


def transient(exc):
    code = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    return code in (408, 429, 500, 502, 503, 504) or isinstance(exc, (TimeoutError, ConnectionError, TimeoutException, NetworkError))


def vector_literal(vector):
    if len(vector) != 768 or not all(math.isfinite(float(x)) for x in vector):
        raise ValueError('Expected a finite 768-dimensional embedding.')
    if not any(float(x) for x in vector):
        raise ValueError('Zero embeddings cannot be used for cosine retrieval.')
    return '[' + ','.join(str(float(x)) for x in vector) + ']'


class Embeddings:
    def __init__(self, settings, client=None):
        self.settings = settings
        if client is None:
            settings.require_embeddings()
            client = genai.Client(api_key=settings.gemini_api_key.get_secret_value(), vertexai=False,
                                  http_options=types.HttpOptions(timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.client = client

    @retry(retry=retry_if_exception(transient), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, max=4), reraise=True)
    def _batch(self, texts):
        return self.client.models.embed_content(
            model=self.settings.embedding_model,
            # Embedding 2 aggregates bare strings; Content objects preserve one vector per chunk.
            contents=[types.Content(parts=[types.Part.from_text(text=text)]) for text in texts],
            config=types.EmbedContentConfig(output_dimensionality=768))

    def embed(self, texts, *, task='document'):
        result = []
        try:
            for start in range(0, len(texts), 32):
                batch = [redact(t) for t in texts[start:start + 32]]
                if self.settings.embedding_model.startswith('gemini-embedding-2'):
                    batch = [('task: search result | query: ' if task == 'query' else 'title: none | text: ') + text for text in batch]
                response = self._batch(batch)
                vectors = [list(item.values) for item in response.embeddings or []]
                if len(vectors) != len(batch):
                    raise ValueError('Embedding count mismatch.')
                for vector in vectors:
                    vector_literal(vector)
                result.extend(vectors)
            return result
        except Exception:
            raise SetupError('Gemini embedding failed. Check GEMINI_API_KEY, model access, quota, and network; no document replacement was committed.') from None

    def query(self, text):
        return self.embed([text], task='query')[0]
