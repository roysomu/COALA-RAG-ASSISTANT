import re

CITATION = re.compile(r'\[([^\[\]\n]+?),\s*chunk\s+(\d+)\]')
CITATION_LIKE = re.compile(r'\[[^\[\]\n]*(?:\bchunk\b|\.(?:md|txt|pdf)\b)[^\[\]\n]*\]', re.I)


def citation_label(evidence):
    return f"[{evidence['filename']}, chunk {evidence['chunk_index']}]"


def validate_citations(answer, evidence):
    allowed = {(e['filename'], int(e['chunk_index'])) for e in evidence}
    invalid = []
    for match in CITATION_LIKE.finditer(answer):
        parsed = CITATION.fullmatch(match.group(0))
        if not parsed or (parsed.group(1), int(parsed.group(2))) not in allowed:
            invalid.append(match.group(0))
    return invalid


def has_citation(answer):
    return bool(CITATION.search(answer))
