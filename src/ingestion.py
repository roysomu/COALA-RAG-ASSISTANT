from io import BytesIO
from pathlib import PurePosixPath
from uuid import uuid4
from dataclasses import replace
import re
from psycopg.types.json import Jsonb
from pypdf import PdfReader
from src.chunking import content_hash, split_text
from src.embeddings import vector_literal
from src.config import SetupError
from src.security import redact, contains_secret


def sanitize_filename(name):
    name = PurePosixPath(name.replace('\\', '/')).name
    clean = re.sub(r'[^A-Za-z0-9._ -]', '_', name).strip(' .')
    if contains_secret(clean):
        raise SetupError('Filename appears to contain a secret. Rename the file before uploading.')
    if not clean or len(clean) > 160:
        raise SetupError('Use a filename between 1 and 160 safe characters.')
    return clean


def extract_chunks(filename, data, max_upload_mb=10):
    if not data or len(data) > max_upload_mb * 1024 * 1024:
        raise SetupError('File is empty or exceeds MAX_UPLOAD_MB.')
    suffix = PurePosixPath(filename).suffix.lower()
    pages = []
    if suffix in ('.txt', '.md'):
        try:
            pages = [(None, data.decode('utf-8-sig'))]
        except UnicodeDecodeError:
            raise SetupError('Text files must use UTF-8 encoding.') from None
    elif suffix == '.pdf':
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted or len(reader.pages) > 500:
                raise ValueError()
            pages = [(i + 1, page.extract_text() or '') for i, page in enumerate(reader.pages)]
        except Exception:
            raise SetupError('PDF could not be read. Use an unencrypted text PDF with at most 500 pages.') from None
    else:
        raise SetupError('Unsupported file type. Upload .txt, .md, or .pdf.')
    if sum(len(text) for _, text in pages) > 2_000_000:
        raise SetupError('Extracted text exceeds the 2 million character limit. Split the document.')
    chunks, seen = [], set()
    for page, text in pages:
        for chunk in split_text(redact(text), page=page, start_index=len(chunks) + 1):
            if chunk.content_hash not in seen:
                chunks.append(replace(chunk, index=len(chunks) + 1))
                seen.add(chunk.content_hash)
    if not chunks:
        raise SetupError('No extractable text found. Scanned PDFs require external text extraction.')
    return chunks


class DocumentStore:
    def __init__(self, db):
        self.db = db

    def find_hash(self, digest):
        rows = self.db.query('SELECT id FROM documents WHERE content_hash = %s', (digest,))
        return rows[0] if rows else None

    def record(self, filename, digest, status, detail=''):
        self.db.execute('INSERT INTO ingestion_records(id, filename, content_hash, status, detail) VALUES(%s,%s,%s,%s,%s)',
                        (uuid4(), filename, digest, status, detail))

    def replace(self, filename, digest, chunks, vectors, model):
        # All paid embedding work completes before the replacement transaction.
        with self.db.transaction() as conn:
            conn.execute('SELECT pg_advisory_xact_lock(%s)', (271834102,))
            if conn.execute('SELECT id FROM documents WHERE content_hash=%s', (digest,)).fetchone():
                status = 'skipped'
            else:
                old = conn.execute('SELECT id FROM documents WHERE filename=%s FOR UPDATE', (filename,)).fetchone()
                doc_id = old['id'] if old else uuid4()
                status = 'replaced' if old else 'added'
                metadata = Jsonb({'embedding_model': model, 'dimensions': 768})
                if old:
                    conn.execute('DELETE FROM document_chunks WHERE document_id=%s', (doc_id,))
                    conn.execute('UPDATE documents SET content_hash=%s, metadata=%s WHERE id=%s', (digest, metadata, doc_id))
                else:
                    conn.execute('INSERT INTO documents(id, filename, content_hash, metadata) VALUES(%s,%s,%s,%s)', (doc_id, filename, digest, metadata))
                for chunk, vector in zip(chunks, vectors, strict=True):
                    conn.execute('INSERT INTO document_chunks(id, document_id, chunk_index, content, content_hash, embedding, metadata) VALUES(%s,%s,%s,%s,%s,%s::vector,%s)',
                                 (uuid4(), doc_id, chunk.index, chunk.text, chunk.content_hash, vector_literal(vector),
                                  Jsonb({'filename': filename, 'page': chunk.page, 'start': chunk.start, 'end': chunk.end})))
            conn.execute('INSERT INTO ingestion_records(id, filename, content_hash, status) VALUES(%s,%s,%s,%s)', (uuid4(), filename, digest, status))
        return status

    def list(self):
        return self.db.query('SELECT d.id, d.filename, d.created_at, count(c.id) AS chunks FROM documents d LEFT JOIN document_chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.filename')

    def delete(self, document_id):
        self.db.execute('DELETE FROM documents WHERE id=%s', (document_id,))


class IngestionService:
    def __init__(self, store, embeddings, settings):
        self.store, self.embeddings, self.settings = store, embeddings, settings

    def ingest(self, name, data):
        filename, digest = 'invalid-filename', content_hash(data)
        try:
            filename = sanitize_filename(name)
            chunks = extract_chunks(filename, data, self.settings.max_upload_mb)
            if self.store.find_hash(digest):
                self.store.record(filename, digest, 'skipped')
                return {'filename': filename, 'status': 'skipped', 'chunks': 0}
            vectors = self.embeddings.embed([c.text for c in chunks])
            status = self.store.replace(filename, digest, chunks, vectors, self.settings.embedding_model)
            return {'filename': filename, 'status': status, 'chunks': len(chunks)}
        except Exception as exc:
            detail = str(exc) if isinstance(exc, SetupError) else 'Ingestion failed; the previous document remains intact. Check database connectivity and setup.'
            try:
                self.store.record(filename, digest, 'failed', detail)
            except Exception:
                pass
            return {'filename': filename, 'status': 'failed', 'chunks': 0, 'detail': detail}
