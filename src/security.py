"""Conservative redaction at persistence, model, logging, and UI boundaries."""
import logging
import re
import warnings

_PATTERNS = [
    r'(?i)postgres(?:ql)?://[^\s\]\)\"\'<>]+',
    r'\bsk-[A-Za-z0-9_-]{8,}',
    r'\bAIza[A-Za-z0-9_-]{20,}',
    r'\blsv2_[A-Za-z0-9_-]+',
    r'''(?i)\b(?:[a-z0-9]+[_-])?(?:api[_ -]?key|password|passwd|secret|access[_ -]?token|authorization)\s*(?:[:=]|\bis\b)\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)''',
    r'(?i)\bBearer\s+[A-Za-z0-9._-]+',
    r'-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----',
]


def redact(text, secrets=()):
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, '[REDACTED]')
    for pattern in _PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text)
    return text


def contains_secret(text):
    return redact(text) != text


def configure_logging():
    # Some LiteLLM/Pydantic diagnostic warnings embed complete provider response objects.
    warnings.filterwarnings('ignore', message='Pydantic serializer warnings:', category=UserWarning, module=r'pydantic\.main')
    # Third-party debug/exception logs can contain credentials or full provider requests.
    for name in ('LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'httpx', 'httpcore', 'google.genai', 'psycopg.pool', 'langsmith', 'pypdf'):
        logger = logging.getLogger(name)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        logger.disabled = True
