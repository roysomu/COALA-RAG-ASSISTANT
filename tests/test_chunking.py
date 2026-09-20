import pytest
from src.chunking import split_text, content_hash


def test_exact_overlap_and_coverage():
    text = ''.join(str(i % 10) for i in range(2700))
    chunks = split_text(text)
    assert all(len(c.text) <= 900 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for left, right in zip(chunks, chunks[1:]):
        assert left.text[-150:] == right.text[:150]
        assert right.start == left.end - 150
    assert chunks[0].index == 1


def test_paragraph_boundary_and_pages():
    chunks = split_text('a' * 600 + '\n\n' + 'b' * 800, page=4)
    assert chunks[0].text.endswith('\n\n')
    assert all(c.page == 4 for c in chunks)


@pytest.mark.parametrize('text', ['', '  \n\n'])
def test_empty(text):
    assert split_text(text) == []


@pytest.mark.parametrize('size,overlap', [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text('test', size, overlap)


def test_hashing_stable():
    assert content_hash('abc') == content_hash(b'abc')
    assert content_hash('abc') != content_hash('abcd')
