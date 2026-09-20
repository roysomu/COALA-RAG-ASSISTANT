from _common import run, load_settings, create_pool, Database, create_checkpointer
from src.db import initialize_database


def main():
    settings = load_settings()
    with create_pool(settings) as pool:
        unavailable = initialize_database(Database(pool), create_checkpointer(pool), settings)
    print('Application schema and official LangGraph checkpoints initialized.')
    if unavailable:
        print('HNSW unavailable for ' + ', '.join(unavailable) + '; exact vector scans will be used.')
    return 0


if __name__ == '__main__':
    raise SystemExit(run(main))
