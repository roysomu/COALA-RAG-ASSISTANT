from typing import Literal
from urllib.parse import urlsplit, parse_qs
from pydantic import SecretStr, Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class SetupError(ValueError):
    """An intentionally safe, actionable error suitable for the UI."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', hide_input_in_errors=True)
    database_url: SecretStr = SecretStr('')
    gemini_api_key: SecretStr = SecretStr('')
    openai_api_key: SecretStr = SecretStr('')
    embedding_model: str = Field('gemini-embedding-2', pattern=r'^gemini-embedding-[a-z0-9-]{1,64}$')
    embedding_dimensions: int = 768
    simple_chat_model: str = 'gemini/gemini-2.5-flash-lite'
    complex_chat_model: str = 'openai/gpt-4.1-mini'
    fallback_chat_model: str = 'gemini/gemini-2.5-flash'
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr('')
    langsmith_project: str = 'local-coala-rag-assistant'
    langsmith_endpoint: str = ''
    default_user_id: str = 'local-user'
    default_top_k: int = Field(5, gt=0, le=20)
    max_upload_mb: int = Field(10, gt=0, le=50)
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    checkpoint_durability: Literal['exit'] = 'exit'
    recent_messages_to_keep: int = Field(12, ge=2, le=100)
    max_messages_per_thread: int = Field(50, gt=0)
    checkpoint_retention_days: int = Field(30, gt=0)
    max_retained_threads: int = Field(20, gt=0)
    auto_prune_checkpoints: bool = False
    database_warning_mb: int = Field(350, gt=0)

    @field_validator('simple_chat_model', 'complex_chat_model', 'fallback_chat_model')
    @classmethod
    def model_provider(cls, value, info):
        if not value.startswith(('gemini/', 'openai/')) or len(value) > 120 or not all(c.isalnum() or c in '/-._' for c in value):
            raise ValueError('Chat model must be an OpenAI or Gemini model identifier.')
        expected = {'simple_chat_model': 'gemini/', 'complex_chat_model': 'openai/'}.get(info.field_name)
        if expected and not value.startswith(expected):
            raise ValueError('Simple chat must use Gemini and complex chat must use OpenAI so manual provider overrides remain unambiguous.')
        return value

    @field_validator('database_url')
    @classmethod
    def secure_hosted_postgres(cls, value):
        raw = value.get_secret_value()
        if not raw:
            return value
        try:
            parts = urlsplit(raw)
            valid_host = bool(parts.hostname) and parts.hostname.endswith(
                ('.neon.tech', '.supabase.co', '.supabase.com'))
            valid = (parts.scheme in ('postgres', 'postgresql') and valid_host
                     and parse_qs(parts.query).get('sslmode', [''])[0] in ('require', 'verify-full'))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError('DATABASE_URL must be a Neon or Supabase PostgreSQL URL with sslmode=require or verify-full.')
        return value

    @property
    def database_provider(self):
        host = urlsplit(self.database_url.get_secret_value()).hostname or ''
        return 'Supabase' if host.endswith(('.supabase.co', '.supabase.com')) else 'Neon'

    @field_validator('embedding_dimensions')
    @classmethod
    def fixed_embedding_dimensions(cls, value):
        if value != 768:
            raise ValueError('Embedding dimensions must equal 768.')
        return value

    @property
    def providers(self):
        return [p for p, key in [('gemini', self.gemini_api_key), ('openai', self.openai_api_key)] if key.get_secret_value()]

    def require_database(self):
        if not self.database_url.get_secret_value():
            raise SetupError('Set DATABASE_URL to your SSL-enabled Supabase or Neon PostgreSQL connection string in .env; then run make init-db.')

    def require_embeddings(self):
        if not self.gemini_api_key.get_secret_value():
            raise SetupError('Set GEMINI_API_KEY in .env. Gemini is required for document, query, and memory embeddings.')


def load_settings():
    try:
        return Settings()
    except ValidationError as exc:
        fields = ', '.join(sorted({str(e['loc'][0]) for e in exc.errors(include_input=False)}))
        raise SetupError(f'Invalid configuration fields: {fields}. Check .env.example; secret values are hidden.') from None
