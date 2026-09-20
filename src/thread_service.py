from uuid import uuid4
from psycopg.types.json import Jsonb
from src.config import SetupError


class ThreadService:
    def __init__(self, db, settings, memories):
        self.db, self.settings, self.memories = db, settings, memories

    def create(self, user_id, source_thread=None):
        thread_id = str(uuid4())
        self.db.execute('INSERT INTO app_threads(thread_id,user_id,metadata) VALUES(%s,%s,%s)',
                        (thread_id, user_id, Jsonb({'rollover_from': source_thread} if source_thread else {})))
        return thread_id

    def get(self, thread_id, user_id):
        rows = self.db.query('SELECT * FROM app_threads WHERE thread_id=%s AND user_id=%s', (thread_id, user_id))
        return rows[0] if rows else None

    def list(self, user_id, status=None):
        return self.db.query('SELECT * FROM app_threads WHERE user_id=%s AND (%s::text IS NULL OR status=%s) ORDER BY last_activity DESC', (user_id, status, status))

    def record_success(self, thread_id, user_id, summary_saved):
        self.db.execute('''INSERT INTO app_threads(thread_id,user_id,message_count,last_activity,summary_saved_at)
            VALUES(%s,%s,1,now(),CASE WHEN %s THEN now() ELSE NULL END)
            ON CONFLICT(thread_id) DO UPDATE SET last_activity=now(), message_count=app_threads.message_count+1,
              summary_saved_at=CASE WHEN %s THEN now() ELSE NULL END
            WHERE app_threads.user_id=EXCLUDED.user_id''', (thread_id, user_id, summary_saved, summary_saved))

    def archive(self, thread_id, user_id):
        self.db.execute("UPDATE app_threads SET status='archived' WHERE thread_id=%s AND user_id=%s", (thread_id, user_id))

    def rollover(self, thread_id, user_id, summary):
        row = self.get(thread_id, user_id)
        if not row or row['message_count'] < self.settings.max_messages_per_thread:
            return None
        if not row['summary_saved_at'] or not self.memories.latest_episode(user_id, thread_id):
            memory_id = self.memories.save(user_id, thread_id, 'episodic', summary, {'source': 'thread rollover'})
            if not memory_id:
                raise SetupError('Rollover needs a saved episodic summary. Remove secret-like content from the conversation and try again.')
            self.db.execute('UPDATE app_threads SET summary_saved_at=now() WHERE thread_id=%s AND user_id=%s', (thread_id, user_id))
        next_id = str(uuid4())
        with self.db.transaction() as conn:
            conn.execute("UPDATE app_threads SET status='archived' WHERE thread_id=%s AND user_id=%s", (thread_id, user_id))
            conn.execute('INSERT INTO app_threads(thread_id,user_id,metadata) VALUES(%s,%s,%s)',
                         (next_id, user_id, Jsonb({'rollover_from': thread_id})))
        return next_id
