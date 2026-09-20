import argparse
import importlib
import platform
from _common import run, load_settings

MODULES = ['streamlit', 'langgraph', 'langgraph.checkpoint.postgres', 'langsmith', 'litellm',
           'google.genai', 'psycopg', 'psycopg_pool', 'pgvector', 'pypdf', 'src.runtime', 'app']


def main():
    parser = argparse.ArgumentParser(description='No network calls. Use --smoke for credential-free import checks.')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    print('Python:', platform.python_version())
    # Initialize privacy defaults before importing provider libraries.
    import src.llm_router
    for module in MODULES:
        importlib.import_module(module)
    print('All runtime imports passed.')
    if args.smoke:
        return 0
    settings = load_settings()
    missing = []
    if not settings.database_url.get_secret_value():
        missing.append('DATABASE_URL: set your Supabase or Neon SSL connection string')
    if not settings.gemini_api_key.get_secret_value():
        missing.append('GEMINI_API_KEY: required for all embeddings and Gemini chat')
    if not settings.providers:
        missing.append('GEMINI_API_KEY or OPENAI_API_KEY: required for chat')
    if settings.langsmith_tracing and not settings.langsmith_api_key.get_secret_value():
        missing.append('LANGSMITH_API_KEY: required when tracing is enabled')
    print('Chat providers:', ', '.join(settings.providers) or 'none')
    if settings.database_url.get_secret_value():
        print('Database provider:', settings.database_provider)
    print('Tracing:', 'enabled' if settings.langsmith_tracing else 'disabled')
    for item in missing:
        print('SETUP REQUIRED:', item)
    print('Edit .env using .env.example. No connections or paid API calls were made.')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(run(main))
