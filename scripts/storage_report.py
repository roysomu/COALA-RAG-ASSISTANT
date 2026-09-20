import json
from _common import run, load_settings, create_pool, Database
from src.storage_monitor import storage_report


def main():
    settings = load_settings()
    with create_pool(settings) as pool:
        print(json.dumps(storage_report(Database(pool), settings.database_warning_mb), indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(run(main))
