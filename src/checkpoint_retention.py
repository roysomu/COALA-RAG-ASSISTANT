from collections import defaultdict
from datetime import datetime, timezone, timedelta
from psycopg.types.json import Jsonb
from src.config import SetupError


def select_candidates(threads, days, max_threads, current_thread=None, force=False, now=None):
    """Prune inactive threads; flag excess archived threads without bypassing the age gate."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    groups = defaultdict(list)
    for row in threads:
        groups[row['user_id']].append(row)
    candidates, skipped = [], []
    for rows in groups.values():
        ranked = sorted(rows, key=lambda r: r['last_activity'], reverse=True)
        for rank, row in enumerate(ranked):
            expired = row['last_activity'] < cutoff
            excess = rank >= max_threads and row['status'] == 'archived'
            if not expired:
                continue
            if row['thread_id'] == current_thread:
                skipped.append({'thread_id': row['thread_id'], 'reason': 'current thread'})
            elif not row['summary_saved_at'] and not force:
                skipped.append({'thread_id': row['thread_id'], 'reason': 'no saved episodic summary'})
            else:
                candidates.append({**row, 'reason': 'inactive excess archived thread' if excess else 'inactive'})
    return candidates, skipped


class CheckpointRetention:
    def __init__(self, db, saver, settings):
        self.db, self.saver, self.settings = db, saver, settings

    def preview(self, current_thread=None, user_id=None, days=None, force=False):
        days = self.settings.checkpoint_retention_days if days is None else days
        if days <= 0:
            raise SetupError('Retention days must be positive.')
        rows = self.db.query('''SELECT t.thread_id, t.user_id, t.status, t.last_activity,
            CASE WHEN EXISTS (SELECT 1 FROM memories m WHERE m.user_id=t.user_id AND m.memory_type='episodic'
                AND (m.thread_id=t.thread_id OR m.metadata->>'last_source_thread'=t.thread_id))
                THEN t.summary_saved_at ELSE NULL END AS summary_saved_at,
            (SELECT count(*) FROM checkpoints c WHERE c.thread_id=t.thread_id) AS checkpoint_count
            FROM app_threads t WHERE (%s::text IS NULL OR t.user_id=%s)''', (user_id, user_id))
        return select_candidates(rows, days, self.settings.max_retained_threads, current_thread, force)

    def prune(self, current_thread=None, user_id=None, days=None, force=False, apply=False, only_ids=None):
        if not apply:
            candidates, skipped = self.preview(current_thread, user_id, days, force)
            return {'dry_run': True, 'candidates': candidates, 'skipped': skipped, 'deleted': [], 'failed': []}
        with self.db.maintenance_lock() as acquired:
            if not acquired:
                raise SetupError('A graph invocation or cleanup is active. Retry cleanup when it finishes.')
            return self._apply_locked(current_thread, user_id, days, force, only_ids)

    def _apply_locked(self, current_thread=None, user_id=None, days=None, force=False, only_ids=None):
        candidates, skipped = self.preview(current_thread, user_id, days, force)
        deleted, failed = [], []
        for row in candidates:
            thread_id = row['thread_id']
            if only_ids is not None and thread_id not in only_ids:
                continue
            try:
                self.saver.delete_thread(thread_id)
                self.db.execute('DELETE FROM app_threads WHERE thread_id=%s', (thread_id,))
                deleted.append(thread_id)
            except Exception:
                failed.append(thread_id)
        return {'dry_run': False, 'candidates': candidates, 'skipped': skipped, 'deleted': deleted, 'failed': failed}

    def delete_conversation(self, thread_id, user_id):
        # Explicit manual deletion is separate from retention and can remove the current thread.
        with self.db.maintenance_lock() as acquired:
            if not acquired:
                raise SetupError('Wait for active graph invocations before deleting a conversation.')
            if not self.db.query('SELECT thread_id FROM app_threads WHERE thread_id=%s AND user_id=%s', (thread_id, user_id)):
                raise SetupError('Conversation does not belong to the current user.')
            self.saver.delete_thread(thread_id)
            self.db.execute('DELETE FROM app_threads WHERE thread_id=%s AND user_id=%s', (thread_id, user_id))

    def auto_prune(self, current_thread):
        if not self.settings.auto_prune_checkpoints:
            return None
        with self.db.maintenance_lock() as acquired:
            if not acquired:
                return None
            rows = self.db.query("SELECT last_run_at FROM maintenance_runs WHERE job_name='checkpoint_prune'")
            if rows and rows[0]['last_run_at'] and rows[0]['last_run_at'] > datetime.now(timezone.utc) - timedelta(hours=24):
                return None
            result = self._apply_locked(current_thread=current_thread)
            self.db.execute('''INSERT INTO maintenance_runs(job_name,last_run_at,result) VALUES('checkpoint_prune',now(),%s)
                ON CONFLICT(job_name) DO UPDATE SET last_run_at=now(),result=EXCLUDED.result''',
                (Jsonb({'deleted': result['deleted'], 'failed': result['failed']}),))
            return result
