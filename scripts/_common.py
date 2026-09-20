import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SetupError, load_settings
from src.db import Database, create_pool, create_checkpointer
from src.security import configure_logging


def run(main):
    configure_logging()
    try:
        return main()
    except SetupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print('Operation failed. Check .env, hosted PostgreSQL SSL/network access, database initialization, and provider configuration. Sensitive error details are suppressed.', file=sys.stderr)
        return 1
