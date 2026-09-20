from pathlib import Path
from collections import Counter
from _common import run, load_settings, create_pool, Database
from src.embeddings import Embeddings
from src.ingestion import DocumentStore, IngestionService


def main():
    settings = load_settings()
    with create_pool(settings) as pool:
        db = Database(pool)
        db.verify_embedding_config(settings)
        service = IngestionService(DocumentStore(db), Embeddings(settings), settings)
        results = [service.ingest(path.name, path.read_bytes()) for path in sorted((Path(__file__).resolve().parents[1] / 'sample_docs').glob('*.md')) if path.name != 'questions.md']
    for row in results:
        print(row)
    counts = Counter(row['status'] for row in results)
    print({status: counts[status] for status in ('added', 'skipped', 'replaced', 'failed')})
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(run(main))
