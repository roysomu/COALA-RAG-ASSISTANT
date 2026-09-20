from src.citations import validate_citations, citation_label, has_citation


def test_valid(evidence):
    label = citation_label(evidence[0])
    assert label == '[security_policy.md, chunk 3]'
    assert has_citation(label)
    assert validate_citations('365 days ' + label, evidence) == []


def test_unknown_file_or_chunk(evidence):
    answer = '[fake.md, chunk 3] [security_policy.md, chunk 9]'
    assert len(validate_citations(answer, evidence)) == 2


def test_citation_not_allowed_without_evidence():
    assert validate_citations('[x.md, chunk 1]', []) == ['[x.md, chunk 1]']


def test_malformed_citation_rejected(evidence):
    assert validate_citations('[security_policy.md, chunk banana]', evidence)
    assert validate_citations('[security_policy.md chunk 3]', evidence)
