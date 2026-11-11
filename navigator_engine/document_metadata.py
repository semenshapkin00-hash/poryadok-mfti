"""Conservative explicit document metadata; no filename or approval-date inference.

Offsets and quotes refer to the supplied text (or explicitly named metadata.title).
Relationship resolution never interprets a paragraph amendment as a full replacement.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import re
import unicodedata

MONTHS = dict(zip(('января февраля марта апреля мая июня июля августа сентября октября ноября декабря').split(), range(1, 13)))
DATE = r'(?:\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{4}|[«"“]?\d{1,2}[»"”]?\s+(?:' + '|'.join(MONTHS) + r')\s+\d{4}(?:\s*г(?:ода|\.)?)?)'
NUMBER = r'[0-9A-Za-zА-Яа-яЁё]+(?:(?:[ \t]*[-/–—][ \t]*|\.)[0-9A-Za-zА-Яа-яЁё]+)*'
MARKER = r'(?:№|N(?:o\.?)?|номер(?:ом)?)\s*'
PREFIX = r'(?m)(?:^|(?<=;))[ \t]*(?:#{1,6}\s*|\d+(?:\.\d+)*[.)]\s*)?'
SUBJECT = r'(?:(?:настоящ(?:ий|ая|ее)\s+)?(?:документ|приказ|регламент|положение|инструкция|редакция)\s+)?'
DATE_RULES = {
    'effective_from': re.compile(PREFIX + r'(?:действует\s+с\s*:?|дата\s+(?:вступления\s+в\s+силу|начала\s+действия)\s*:|' + SUBJECT + r'вступает\s+в\s+силу\s+с)\s*(?P<date>' + DATE + r')(?!\d)', re.I),
    'effective_to': re.compile(PREFIX + r'(?:действует\s+(?:до|по)\s*:?|дата\s+окончания\s+действия\s*:)\s*(?P<date>' + DATE + r')(?!\d)', re.I),
}
VERSION = re.compile(PREFIX + r'(?:редакция|версия)\s*:\s*(?P<value>[^\n;]+)', re.I)
TITLE_NUMBER = re.compile(r'\b(?:приказ|регламент|положение|инструкция|документ|распоряжение)\s+' + MARKER + r'(?P<number>' + NUMBER + r')', re.I)
FULL_REPLACEMENT = re.compile(
    PREFIX + r'(?:' + SUBJECT + r'(?:полностью\s+)?заменяет\s+(?:документ\s+|приказ\s+|регламент\s+|положение\s+|инструкцию\s+)?|'
    r'признать\s+утративш(?:им|ей)\s+силу\s+(?:документ|приказ|регламент|положение|инструкцию)\s+)'
    + MARKER + r'(?P<number>' + NUMBER + r')(?P<tail>[^\n;]*)', re.I)
PARTIAL = re.compile(r'\b(?:пункт\w*|подпункт\w*|раздел\w*|абзац\w*|приложени\w*|части|изменени\w*|дополнени\w*)\b', re.I)
CONDITIONAL = re.compile(r'\b(?:если|когда|после|при\s+условии|в\s+случае|при\s+наличии|при\s+отсутствии)\b', re.I)


def _statement_tail(value):
    for boundary in re.finditer(r'[.!?;\n]', value):
        index = boundary.start()
        if boundary.group() == '.':
            # Dots within dates/decimals are not sentence boundaries.
            if 0 < index < len(value) - 1 and value[index - 1].isdigit() and value[index + 1].isdigit():
                continue
            # A year abbreviation followed by a comma or lower-case
            # continuation can precede a condition in the same statement.
            if re.search(r'(?<!\w)г$', value[:index], re.I) and re.match(r'\s*(?:[,:([—–-]|[а-яёa-z])', value[index + 1:]):
                continue
        return value[:index]
    return value


def _date(value):
    value = re.sub(r'[«»"“”]', '', value).strip().lower()
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return date.fromisoformat(value).isoformat()
    if re.fullmatch(r'\d{1,2}\.\d{1,2}\.\d{4}', value):
        day, month, year = map(int, value.split('.'))
    else:
        match = re.fullmatch(r'(\d{1,2})\s+(\w+)\s+(\d{4})(?:\s*г(?:ода|\.)?)?', value)
        if not match:
            raise ValueError('Invalid explicit date')
        day, month, year = int(match[1]), MONTHS[match[2]], int(match[3])
    return date(year, month, day).isoformat()


def _number(value):
    value = str(value).strip()
    value = re.sub(r'^(?:№\s*|N(?:o\.?)?\s+|номер(?:ом)?\s+)', '', value, flags=re.I)
    value = unicodedata.normalize('NFKC', value)
    if not re.fullmatch(NUMBER, value):
        return None
    return re.sub(r'\s+', '', value).replace('–', '-').replace('—', '-').casefold()


def _span(text, match, value, rule, source='text'):
    return dict(source=source, start=match.start(), end=match.end(), quote=text[match.start():match.end()], value=value, rule=rule)


def _replacement_references(text):
    references = []
    for match in FULL_REPLACEMENT.finditer(text):
        # A partial change in the same statement does not revoke a document.
        if PARTIAL.search(match.group()) or CONDITIONAL.search(_statement_tail(match['tail'])):
            continue
        number = _number(match['number'])
        if number:
            references.append(_span(text, match, number, 'explicit_full_document_replacement'))
    return references


def infer_metadata_from_explicit_text(text, existing_metadata=None):
    """Return ``(metadata, provenance)`` without mutating either input.

    Explicit existing fields win. Conflicting/invalid source dates remain in
    provenance. A missing field is filled only when all valid mentions agree
    and no malformed mention was found. End dates are literal inclusive dates,
    matching the engine contract; no expiry is derived from an approval date.
    """
    if not isinstance(text, str) or not isinstance(existing_metadata, (dict, type(None))):
        raise TypeError('Expected text and an optional metadata dictionary')
    metadata = deepcopy(existing_metadata or {})
    provenance = dict(fields={}, ambiguities=[], replacement_references=_replacement_references(text))
    for field, pattern in DATE_RULES.items():
        matches, invalid = [], []
        for match in pattern.finditer(text):
            tail = _statement_tail(text[match.end():])
            if CONDITIONAL.search(tail):
                item = _span(text, match, match['date'], 'conditional_explicit_date')
                item.update(end=match.end() + len(tail), quote=text[match.start():match.end() + len(tail)])
                invalid.append(item)
                continue
            try:
                matches.append(_span(text, match, _date(match['date']), 'explicit_' + field))
            except (ValueError, KeyError):
                invalid.append(_span(text, match, match['date'], 'invalid_explicit_date'))
        provenance['fields'][field] = matches + invalid
        values = sorted({item['value'] for item in matches})
        existing = metadata.get(field)
        try:
            comparable = _date(str(existing)) if existing else None
        except (ValueError, KeyError):
            comparable = str(existing)
        if len(values) > 1 or invalid or (existing and values and comparable not in values):
            provenance['ambiguities'].append(dict(field=field, values=values, existing=existing, invalid_spans=invalid, reason='conflicting_or_invalid_explicit_dates'))
        if not existing and len(values) == 1 and not invalid:
            metadata[field] = values[0]
    for field, matches in [('version', [ _span(text, m, m['value'].strip(), 'explicit_version_label') for m in VERSION.finditer(text)])]:
        provenance['fields'][field] = matches
        values = sorted({item['value'] for item in matches})
        if len(values) > 1:
            provenance['ambiguities'].append(dict(field=field, values=values, reason='conflicting_explicit_values'))
        elif values and not metadata.get(field):
            metadata[field] = values[0]
    # Number references in body paragraphs are never treated as this document's number.
    title = str(metadata.get('title') or '')
    title_source = 'metadata.title'
    if not title:
        heading = next((m for m in re.finditer(r'[^\n]+', text) if m.group().strip()), None)
        title = heading.group() if heading else ''
        title_source = 'text_heading'
    numbers = []
    for match in TITLE_NUMBER.finditer(title):
        item = _span(title, match, match['number'].strip(), 'number_in_own_title', title_source)
        if title_source == 'text_heading':
            item['start'] += heading.start(); item['end'] += heading.start()
        numbers.append(item)
    provenance['fields']['doc_number'] = numbers
    values = {item['value'] for item in numbers}
    if len(values) == 1 and not metadata.get('doc_number'):
        metadata['doc_number'] = next(iter(values))
    elif len(values) > 1:
        provenance['ambiguities'].append(dict(field='doc_number', values=sorted(values), reason='ambiguous_own_title_number'))
    try:
        inverted = bool(metadata.get('effective_from') and metadata.get('effective_to') and _date(str(metadata['effective_from'])) > _date(str(metadata['effective_to'])))
    except (ValueError, KeyError):
        inverted = False  # Existing invalid user fields remain for the parser's validation.
    if inverted:
        provenance['ambiguities'].append(dict(field='effective_period', reason='start_after_end'))
        for field in DATE_RULES:
            if not (existing_metadata or {}).get(field):
                metadata.pop(field, None)
    return metadata, provenance


def resolve_document_relationships(documents):
    """Copy documents and link only a unique, exact document-number reference.

    Does not expire old documents retroactively: core applies replacements only
    when the replacing document itself is active. Explicit nonempty ID links
    are retained; duplicate numbers, self-links and partial amendments stay unresolved.
    """
    result = deepcopy(documents)
    numbers = {}
    for document in result:
        metadata, inferred = infer_metadata_from_explicit_text('', document.get('metadata'))
        document['metadata'] = metadata
        document.setdefault('metadata_provenance', {}).setdefault('fields', {}).update({key: value for key, value in inferred['fields'].items() if value})
        number = _number(metadata.get('doc_number', ''))
        if number:
            numbers.setdefault(number, []).append(document)
    for document in result:
        report = dict(resolved=[], unresolved=[], explicit_links_preserved=bool(document['metadata'].get('supersedes')))
        references = list(document.get('metadata_provenance', {}).get('replacement_references', []))
        for clause in document.get('clauses', []):
            if clause.get('unsafe') or clause.get('requires_review'):
                continue
            for reference in _replacement_references(clause.get('text', '')):
                references.append(dict(reference, source='clause', clause_id=clause.get('id')))
        seen = set()
        for reference in references:
            number = reference['value']
            if number in seen:
                continue
            seen.add(number)
            targets = numbers.get(number, [])
            if report['explicit_links_preserved']:
                reason = 'explicit_metadata_links_retained'
            elif len(targets) != 1:
                reason = 'missing_or_ambiguous_exact_number'
            elif targets[0]['id'] == document['id']:
                reason = 'self_reference'
            else:
                document['metadata'].setdefault('supersedes', []).append(targets[0]['id'])
                report['resolved'].append(dict(reference, target_document_id=targets[0]['id']))
                continue
            report['unresolved'].append(dict(reference, reason=reason))
        document.setdefault('metadata_provenance', {})['relationship_resolution'] = report
    return result
