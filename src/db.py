from contextlib import contextmanager
from pathlib import Path
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from src.config import SetupError
from src.security import configure_logging

# A transaction-scoped advisory lock works with hosted PostgreSQL transaction pooling.
MAINTENANCE_LOCK = 271834101


def create_pool(settings):
    settings.require_database()
    configure_logging()
    return ConnectionPool(settings.database_url.get_secret_value(), min_size=0, max_size=5,
                          timeout=15, open=True,
                          kwargs={'autocommit': True, 'row_factory': dict_row,
                                  'prepare_threshold': None, 'connect_timeout': 10},
                          reconnect_timeout=15)


def create_checkpointer(pool):
    return PostgresSaver(pool)


class Database:
    def __init__(self, pool):
        self.pool = pool

    def query(self, sql, params=()):
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def execute(self, sql, params=()):
        with self.pool.connection() as conn:
            return conn.execute(sql, params).rowcount

    @contextmanager
    def transaction(self):
        with self.pool.connection() as conn, conn.transaction():
            yield conn

    @contextmanager
    def invocation_lock(self, thread_id):
        with self.transaction() as conn:
            conn.execute('SELECT pg_advisory_xact_lock_shared(%s)', (MAINTENANCE_LOCK,))
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (thread_id,))
            yield

    @contextmanager
    def maintenance_lock(self):
        with self.transaction() as conn:
            acquired = conn.execute('SELECT pg_try_advisory_xact_lock(%s) AS acquired', (MAINTENANCE_LOCK,)).fetchone()['acquired']
            yield acquired

    def verify_embedding_config(self, settings):
        rows = self.query('SELECT model, dimensions FROM embedding_config WHERE singleton = TRUE')
        if not rows or rows[0] != {'model': settings.embedding_model, 'dimensions': settings.embedding_dimensions}:
            raise SetupError('Embedding configuration differs from the database. Restore the original settings or perform a full re-embedding migration.')


def initialize_database(db, saver, settings):
    schema = (Path(__file__).resolve().parents[1] / 'sql/schema.sql').read_text()
    with db.transaction() as conn:
        for statement in schema.split(';'):
            if statement.strip():
                conn.execute(statement)
        conn.execute('INSERT INTO embedding_config(singleton, model, dimensions) VALUES(TRUE, %s, %s) ON CONFLICT DO NOTHING',
                     (settings.embedding_model, settings.embedding_dimensions))
    db.verify_embedding_config(settings)
    # Official migrations require autocommit (including concurrent index creation).
    saver.setup()
    unavailable = []
    for statement, label in [
        ('CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops)', 'document_chunks'),
        ('CREATE INDEX IF NOT EXISTS memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops)', 'memories'),
    ]:
        try:
            db.execute(statement)
        except Exception:
            unavailable.append(label)
    return unavailable  # Exact cosine scans remain available without these indexes.
