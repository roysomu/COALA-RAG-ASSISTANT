from uuid import uuid4
from psycopg.types.json import Jsonb
from src.chunking import content_hash
from src.embeddings import vector_literal
from src.security import contains_secret, redact

MEMORY_TYPES = ('episodic', 'semantic', 'procedural')


class MemoryStore:
    def __init__(self, db, embeddings):
        self.db, self.embeddings = db, embeddings

    def save(self, user_id, thread_id, memory_type, content, metadata=None):
        if memory_type not in MEMORY_TYPES:
            raise ValueError('Unknown CoALA memory type.')
        content = ' '.join(content.split())[:2000]
        if not content or contains_secret(content) or '[REDACTED]' in content:
            return None
        digest = content_hash(content.casefold())
        old = self.db.query('SELECT id::text FROM memories WHERE user_id=%s AND memory_type=%s AND content_hash=%s', (user_id, memory_type, digest))
        if old:
            # Record duplicate provenance for rollover even when the summary is unchanged.
            self.db.execute('UPDATE memories SET updated_at=now(), metadata=metadata || %s WHERE id=%s',
                            (Jsonb({'last_source_thread': thread_id}), old[0]['id']))
            return old[0]['id']
        vector = vector_literal(self.embeddings.embed([content])[0])
        memory_id = str(uuid4())
        safe_meta = {str(k): redact(str(v))[:500] for k, v in (metadata or {}).items()}
        rows = self.db.query('''INSERT INTO memories(id,user_id,thread_id,memory_type,content,content_hash,embedding,metadata)
            VALUES(%s,%s,%s,%s,%s,%s,%s::vector,%s)
            ON CONFLICT(user_id,memory_type,content_hash) DO UPDATE SET updated_at=now()
            RETURNING id::text''', (memory_id, user_id, thread_id, memory_type, content, digest, vector, Jsonb(safe_meta)))
        return rows[0]['id']

    def retrieve(self, user_id, question, thread_id=None):
        vector = vector_literal(self.embeddings.query(question))
        results = []
        # Per-type queries keep episodic, semantic, and procedural categories represented.
        for memory_type in MEMORY_TYPES:
            rows = self.db.query('''SELECT id::text, memory_type, left(content, 500) AS content,
                thread_id, 1 - (embedding <=> %s::vector) AS score FROM memories
                WHERE user_id=%s AND memory_type=%s AND (%s::text IS NULL OR thread_id=%s)
                ORDER BY embedding <=> %s::vector LIMIT 2''',
                (vector, user_id, memory_type, thread_id, thread_id, vector))
            results.extend(rows)
        return results

    def list(self, user_id, memory_type=None):
        return self.db.query('SELECT id::text,user_id,thread_id,memory_type,content,metadata,created_at FROM memories WHERE user_id=%s AND (%s::text IS NULL OR memory_type=%s) ORDER BY created_at DESC LIMIT 200', (user_id, memory_type, memory_type))

    def latest_episode(self, user_id, thread_id):
        rows = self.db.query('''SELECT id::text, content FROM memories WHERE user_id=%s
            AND memory_type='episodic' AND (thread_id=%s OR metadata->>'last_source_thread'=%s)
            ORDER BY updated_at DESC LIMIT 1''', (user_id, thread_id, thread_id))
        return rows[0] if rows else None

    def delete(self, user_id, memory_id):
        self.db.execute('''WITH removed AS (
            DELETE FROM memories WHERE user_id=%s AND id=%s
            RETURNING thread_id, memory_type, metadata)
            UPDATE app_threads SET summary_saved_at=NULL WHERE user_id=%s AND thread_id IN (
                SELECT thread_id FROM removed WHERE memory_type='episodic'
                UNION SELECT metadata->>'last_source_thread' FROM removed WHERE memory_type='episodic')''',
            (user_id, memory_id, user_id))

    def delete_all(self, user_id, memory_type=None):
        self.db.execute('''WITH removed AS (
            DELETE FROM memories WHERE user_id=%s AND (%s::text IS NULL OR memory_type=%s)
            RETURNING thread_id, memory_type, metadata)
            UPDATE app_threads SET summary_saved_at=NULL WHERE user_id=%s AND thread_id IN (
                SELECT thread_id FROM removed WHERE memory_type='episodic'
                UNION SELECT metadata->>'last_source_thread' FROM removed WHERE memory_type='episodic')''',
            (user_id, memory_type, memory_type, user_id))
