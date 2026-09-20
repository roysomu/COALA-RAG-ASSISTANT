from src.retrieval import diverse_results


def test_dedup_and_limit():
    rows = [
        {'id': '1', 'document_id': 'a', 'content': 'some exact identical passage'},
        {'id': '2', 'document_id': 'b', 'content': 'some exact identical passage'},
        {'id': '3', 'document_id': 'c', 'content': 'a completely distinct result'},
        {'id': '4', 'document_id': 'd', 'content': 'yet another result'},
    ]
    assert [r['id'] for r in diverse_results(rows, 2)] == ['1', '3']
