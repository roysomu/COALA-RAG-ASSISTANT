from difflib import SequenceMatcher
from src.embeddings import vector_literal


def diverse_results(rows, top_k):
    selected = []
    for row in rows:
        if any(row['content'] == prev['content'] or
               (row['document_id'] == prev['document_id'] and
                SequenceMatcher(None, row['content'], prev['content'], autojunk=False).ratio() > .78)
               for prev in selected):
            continue
        selected.append(row)
        if len(selected) == top_k:
            break
    return selected


class Retriever:
    def __init__(self, db, embeddings):
        self.db, self.embeddings = db, embeddings

    def search(self, question, top_k=5):
        vector = vector_literal(self.embeddings.query(question))
        rows = self.db.query('''SELECT c.id::text, c.document_id::text, d.filename,
          c.chunk_index, c.content, c.metadata->'page' AS page,
          1 - (c.embedding <=> %s::vector) AS score
          FROM document_chunks c JOIN documents d ON d.id=c.document_id
          ORDER BY c.embedding <=> %s::vector LIMIT %s''', (vector, vector, min(top_k, 20) * 4))
        return diverse_results(rows, min(top_k, 20))

    def fetch_refs(self, refs):
        if not refs:
            return []
        return self.db.query('SELECT c.id::text, d.filename, c.chunk_index, c.content, c.metadata FROM document_chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=ANY(%s::uuid[])',
                             ([r['id'] for r in refs],))
