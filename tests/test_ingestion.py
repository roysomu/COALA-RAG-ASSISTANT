from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import Mock
import pytest
from src.ingestion import IngestionService, DocumentStore, extract_chunks, sanitize_filename
from src.chunking import content_hash, split_text
from src.config import SetupError


class Cursor:
    def __init__(self, row=None):
        self.row = row
    def fetchone(self):
        return self.row


class TransactionDB:
    """Minimal transactional fake exercising DocumentStore's actual write sequence."""
    def __init__(self):
        self.document = {'id': 'old-id', 'filename': 'file.md', 'content_hash': 'old-hash'}
        self.chunks = ['old content']
        self.records = []
        self.fail_insert = False

    @contextmanager
    def transaction(self):
        snapshot = deepcopy((self.document, self.chunks, self.records))
        try:
            yield self
        except Exception:
            self.document, self.chunks, self.records = snapshot
            raise

    def execute(self, sql, params=()):
        if sql.startswith('SELECT id FROM documents WHERE content_hash'):
            return Cursor(self.document if self.document and self.document['content_hash'] == params[0] else None)
        if sql.startswith('SELECT id FROM documents WHERE filename'):
            return Cursor(self.document)
        if sql.startswith('DELETE FROM document_chunks'):
            self.chunks = []
        elif sql.startswith('UPDATE documents'):
            self.document['content_hash'] = params[0]
        elif sql.startswith('INSERT INTO document_chunks'):
            if self.fail_insert:
                raise RuntimeError('simulated write failure')
            self.chunks.append(params[3])
        elif sql.startswith('INSERT INTO ingestion_records'):
            self.records.append(params)
        return Cursor()


def test_transaction_rollback_restores_old_chunks():
    db = TransactionDB()
    db.fail_insert = True
    with pytest.raises(RuntimeError):
        DocumentStore(db).replace('file.md', 'new-hash', split_text('new content'), [[.1] * 768], 'model')
    assert db.chunks == ['old content']
    assert db.document['content_hash'] == 'old-hash'
    assert db.records == []


def test_replace_atomic_and_race_dedup():
    db = TransactionDB()
    store = DocumentStore(db)
    chunks = split_text('new content')
    assert store.replace('file.md', 'new-hash', chunks, [[.1] * 768], 'model') == 'replaced'
    assert db.chunks == ['new content']
    assert store.replace('file.md', 'new-hash', chunks, [[.1] * 768], 'model') == 'skipped'
    assert db.chunks == ['new content']


def test_unchanged_ingestion_skips_api(settings):
    store, embed = Mock(), Mock()
    store.find_hash.return_value = {'id': 'existing'}
    result = IngestionService(store, embed, settings).ingest('file.md', b'content')
    assert result['status'] == 'skipped'
    embed.embed.assert_not_called()
    store.replace.assert_not_called()
    assert store.find_hash.call_args.args == (content_hash(b'content'),)


def test_embedding_failure_never_replaces(settings):
    store, embed = Mock(), Mock()
    store.find_hash.return_value = None
    embed.embed.side_effect = SetupError('Gemini embedding failed.')
    result = IngestionService(store, embed, settings).ingest('file.md', b'content')
    assert result['status'] == 'failed'
    store.replace.assert_not_called()
    assert store.record.call_args.args[2] == 'failed'


@pytest.mark.parametrize('filename,data', [('x.py', b'print(1)'), ('x.md', b''), ('x.md', b'\xff'), ('x.pdf', b'not a pdf')])
def test_unsupported_or_invalid(filename, data):
    with pytest.raises(SetupError):
        extract_chunks(filename, data)


def test_size_limit():
    with pytest.raises(SetupError):
        extract_chunks('x.md', b'x' * (1024 * 1024 + 1), max_upload_mb=1)


def test_sanitize_filename():
    assert sanitize_filename('../../security_policy.md') == 'security_policy.md'
    assert sanitize_filename('C:\\dir\\file.md') == 'file.md'
    assert '[' not in sanitize_filename('[x].md')


def test_pdf_page_metadata():
    from io import BytesIO
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 10 100 Td (Test PDF text.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    chunks = extract_chunks('test.pdf', buffer.getvalue())
    assert chunks[0].page == 1 and 'Test PDF text.' in chunks[0].text


def test_duplicate_pdf_chunks_keep_unique_indices(monkeypatch):
    import src.ingestion as ingestion
    from types import SimpleNamespace
    # Repeated paragraph pages are deduplicated without reusing a stored chunk index.
    pages = [Mock(), Mock()]
    pages[0].extract_text.return_value = 'a' * 900 + 'b' * 900 + 'a' * 900
    pages[1].extract_text.return_value = 'c' * 2000
    monkeypatch.setattr(ingestion, 'PdfReader', lambda _: SimpleNamespace(is_encrypted=False, pages=pages))
    chunks = extract_chunks('test.pdf', b'fake')
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
