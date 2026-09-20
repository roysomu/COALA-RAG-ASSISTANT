import os
import re
from dataclasses import dataclass
from langsmith import traceable
from src.config import SetupError
from src.security import redact, configure_logging

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('DO_NOT_TRACK', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


class ProviderFailure(RuntimeError):
    def __init__(self, retryable=False):
        super().__init__('Chat provider request failed. Check the selected model, API key, quota, and connectivity.')
        self.retryable = retryable


@dataclass(frozen=True)
class Route:
    model: str
    category: str
    reason: str
    rag_needed: bool


def classify(question):
    indicators = []
    if len(question) > 500:
        indicators.append('long prompt')
    if question.count('?') > 1:
        indicators.append('multiple questions')
    if re.search(r'\b(compar\w*|analy\w*|design|diagnos\w*|evaluat\w*|synthesi\w*|trade.?offs?|across|combine|multiple documents|json|table|structured|step.by.step)\b', question, re.I):
        indicators.append('analysis, synthesis, or structured output')
    return ('complex' if indicators else 'simple', ', '.join(indicators) or 'short factual request')


def choose_route(question, settings, manual='Auto'):
    category, reason = classify(question)
    if manual not in ('Auto', 'Gemini', 'OpenAI'):
        raise SetupError('Choose Auto, Gemini, or OpenAI.')
    if not settings.providers:
        raise SetupError('Set GEMINI_API_KEY or OPENAI_API_KEY in .env for chat. RAG also requires GEMINI_API_KEY for embeddings.')
    desired = settings.complex_chat_model if category == 'complex' else settings.simple_chat_model
    if manual != 'Auto':
        desired = settings.simple_chat_model if manual == 'Gemini' else settings.complex_chat_model
        reason = f'Manual {manual} override; {reason}'
    if desired.split('/')[0] not in settings.providers:
        configured = [settings.simple_chat_model, settings.complex_chat_model, settings.fallback_chat_model]
        desired = next((m for m in configured if m.split('/')[0] in settings.providers), '')
        if not desired:
            raise SetupError('Configure a chat model for the available API provider.')
        reason += '; using the only configured provider'
    rag_needed = not bool(re.fullmatch(r'\s*(hi|hello|hey|thanks|thank you)[!. ]*', question, re.I))
    return Route(desired, category, reason, rag_needed)


@traceable(run_type='llm', name='litellm_completion',
           process_inputs=lambda inputs: {k: v for k, v in inputs.items() if k != 'invoke'})
def traced_completion(model, messages, metadata, invoke):
    """Only safe request fields reach LangSmith; keys are supplied inside invoke."""
    return invoke(model, messages)


class LLMRouter:
    def __init__(self, settings, completion=None):
        self.settings = settings
        self.completion = completion
        configure_logging()

    def _invoke(self, model, messages):
        failed = False
        retryable = False
        try:
            if self.completion is None:
                import litellm
                litellm.telemetry = False
                litellm.suppress_debug_info = True
                litellm.set_verbose = False
                configure_logging()
                completion = litellm.completion
            else:
                completion = self.completion
            provider = model.split('/')[0]
            key = getattr(self.settings, f'{provider}_api_key').get_secret_value()
            response = completion(model=model, messages=messages, api_key=key,
                                  temperature=0.2, max_tokens=1800, timeout=60, num_retries=0)
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Empty provider response.')
            return redact(content, self._secrets())
        except Exception as exc:
            failed = True
            code = getattr(exc, 'status_code', None)
            retryable = code in (408, 429, 500, 502, 503, 504) or type(exc).__name__ in ('Timeout', 'APIConnectionError', 'RateLimitError', 'ServiceUnavailableError') or isinstance(exc, (TimeoutError, ConnectionError))
        if failed:
            # Raised outside the except block: raw SDK errors cannot enter trace context.
            raise ProviderFailure(retryable)

    def _secrets(self):
        return [getattr(self.settings, field).get_secret_value() for field in
                ('gemini_api_key', 'openai_api_key', 'database_url', 'langsmith_api_key')]

    def complete(self, model, messages, metadata=None):
        messages = [{'role': m['role'], 'content': redact(m['content'], self._secrets())} for m in messages]
        metadata = metadata or {}
        attempts = [model]
        fallback = self.settings.fallback_chat_model
        if fallback != model and fallback.split('/')[0] in self.settings.providers:
            attempts.append(fallback)
        for index, candidate in enumerate(attempts):
            safe_metadata = {**metadata, 'model': candidate, 'provider': candidate.split('/')[0]}
            try:
                text = traced_completion(candidate, messages, safe_metadata, self._invoke,
                    langsmith_extra={'metadata': safe_metadata})
                return text, candidate
            except ProviderFailure as exc:
                if not exc.retryable or index == len(attempts) - 1:
                    raise SetupError(str(exc)) from None
        raise SetupError('No configured chat provider is available.')
