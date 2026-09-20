CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS embedding_config (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions = 768)
);
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL UNIQUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, chunk_index),
    UNIQUE(document_id, content_hash)
);
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, memory_type, content_hash)
);
CREATE TABLE IF NOT EXISTS app_threads (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT now(),
    message_count INTEGER NOT NULL DEFAULT 0,
    summary_saved_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS ingestion_records (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('added', 'skipped', 'replaced', 'failed')),
    detail TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS maintenance_runs (
    job_name TEXT PRIMARY KEY,
    last_run_at TIMESTAMPTZ,
    result JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_hash ON document_chunks(content_hash);
CREATE INDEX IF NOT EXISTS memories_user_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS memories_thread ON memories(thread_id);
CREATE INDEX IF NOT EXISTS memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS threads_user_activity ON app_threads(user_id, last_activity DESC);
CREATE INDEX IF NOT EXISTS threads_activity ON app_threads(last_activity);
CREATE INDEX IF NOT EXISTS ingestion_hash ON ingestion_records(content_hash);
