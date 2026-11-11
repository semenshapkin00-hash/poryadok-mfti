"""Narrow, literal spreadsheet lookups; no calculation or inferred table joins."""
from __future__ import annotations

import re

from .text import normalize


_QUOTES = re.compile(r'[«"“\']([^»"”\']+)[»"”\']')
_CELL = re.compile(r'(?<![\w])\$?([A-Za-z]{1,3})\$?([1-9]\d{0,6})(?![\w])')
_LOCATION = re.compile(r'(?i)\b(?:ячейк\w*|лист(?:а|е|у|ом|ы|ов)?|xlsx|excel)\b|\.xlsx\b|!\$?[a-z]{1,3}\$?\d+')
_FACT = re.compile(r'(?i)\b(?:код\w*|значени\w*|сумм\w*|наименовани\w*|числ\w*|содержим\w*|указан\w*|записан\w*|содержится|стоит)\b|\bчто\s+(?:в|на)\b')
_PROCEDURE = re.compile(
    r'(?i)\b(?:порядок|процедур\w*|инструкци\w*|шаг\w*)\b|'
    r'\bкак\s+(?:мне\s+|нужно\s+)?(?:заполн\w*|оформ\w*|подат\w*|соглас\w*|созда\w*|измен\w*|исправ\w*|рассчита\w*|вычисл\w*|удал\w*|добав\w*|отправ\w*)|'
    r'\b(?:нужн\w*|надо|следует|должн\w*|обязан\w*|можно)\s+ли\b|'
    r'\bчто\s+(?:мне\s+)?(?:нужно\s+|надо\s+)?(?:делать|сделать)\b|'
    r'\bкакие\s+(?:(?:нужны|необходимы)\s+(?:действия|документы)|(?:действия|документы)\s+(?:нужны|необходимы))\b')
_HEADERS = {'код', 'значение', 'наименование', 'сумма', 'название', 'цфо', 'подразделение'}
_DETAILS = ('provenance', 'requires_review', 'numeric_uncertainty', 'sheet', 'row',
            'cell_refs', 'spreadsheet_cells', 'formulas_evaluated')


def _cells(clause):
    return clause.get('spreadsheet_cells', (clause.get('evidence') or {}).get('spreadsheet_cells', []))


def _row_key(item):
    doc, clause = item
    return doc['id'], clause.get('sheet'), clause.get('row')


def _has_phrase(haystack, needle):
    return bool(re.search(r'(?<!\w)' + re.escape(normalize(needle)) + r'(?!\w)', normalize(haystack)))


def _citation(doc, clause, index):
    metadata = doc.get('metadata') or {}
    value = dict(id='cite-' + str(index), clause_id=clause['id'], document_id=doc['id'],
                 document_name=doc['name'], page=clause.get('page'), section=clause.get('section', ''),
                 paragraph=clause.get('paragraph'), quote=clause['text'], bbox=clause.get('bbox'),
                 version=metadata.get('version'), effective_from=metadata.get('effective_from'),
                 effective_to=metadata.get('effective_to'))
    evidence = clause.get('evidence') or {}
    for key in _DETAILS:
        if key in clause or key in evidence:
            value[key] = clause.get(key, evidence.get(key))
    return value


def answer_spreadsheet(question, active, result, invalid_versions=(), version_issues=()):
    """Return a literal fact/clarification, or None to preserve procedural search.

    ``active`` must already have passed the core's date, role and version filter.
    No workbook is preselected by a caller or a curated document identifier.
    """
    location = bool(_LOCATION.search(question))
    code_request = bool(re.search(r'(?i)\b(?:како\w*\s+код|код\w*\s+(?:у|для)|что\s+означает\s+код)', question))
    if _PROCEDURE.search(question) or not ((location and (_FACT.search(question) or _CELL.search(question))) or code_request):
        return None

    sheets = [doc for doc in active if str(doc.get('name', '')).lower().endswith('.xlsx') or
              doc.get('extraction', {}).get('method') == 'xlsx_xml']
    if not location and not sheets:
        return None
    quoted = _QUOTES.findall(question)
    filenames = [value for value in quoted if value.lower().endswith('.xlsx')]
    filenames += re.findall(r'(?i)(?<![\w])([^\s«»"\']+\.xlsx)\b', question)
    if filenames:
        # A quoted filename may contain spaces; ignore its unquoted last word.
        filenames = [name for name in filenames if not any(name != other and other.endswith(name) for other in filenames)]
        sheets = [doc for doc in sheets if any(normalize(doc['name']) == normalize(name) for name in filenames)]
    candidates = [(doc, clause) for doc in sheets for clause in doc.get('clauses', [])
                  if clause.get('sheet') is not None and not clause.get('unsafe')]
    if len(candidates) > 5000:
        raise ValueError('Не более 5000 пунктов в активной базе')
    known_sheets = {normalize(clause['sheet']) for _, clause in candidates}
    sheet_names = re.findall(r'(?i)\bлист(?:а|е|у|ом)?\s+[«"“\']([^»"”\']+)[»"”\']', question)
    sheet_names += re.findall(r"'([^']+)'!\$?[a-zA-Z]{1,3}\$?\d+", question)
    sheet_names += re.findall(r'(?<![\w\'])([A-Za-zА-Яа-яЁё_][\w.-]*)!\$?[a-zA-Z]{1,3}\$?\d+', question)
    if not sheet_names and re.search(r'(?i)\bлист\w*\b', question):
        sheet_names = [name for name in known_sheets if _has_phrase(question, name)]
    if sheet_names:
        candidates = [(doc, clause) for doc, clause in candidates if normalize(clause['sheet']) in {normalize(name) for name in sheet_names}]
    elif re.search(r'(?i)\bлист(?:а|е|у|ом)?\s+\S+', question):
        candidates = []  # An explicit unknown sheet is never silently ignored.
    contextual = candidates[:4]

    refs = list(dict.fromkeys(column.upper() + row for column, row in _CELL.findall(question)))
    row_numbers = [int(number) for number in re.findall(r'(?i)\bстрок[аеуи]\s*№?\s*(\d+)', question)]
    codes = [value for value in re.findall(r'(?i)\bкод\w*\s*[№:=]?\s*[«"“\']?([\w./-]+)', question) if any(character.isdigit() for character in value)]
    if not re.search(r'(?i)ячейк\w*|!', question):
        refs = [ref for ref in refs if normalize(ref) not in {normalize(code) for code in codes}]
    anchors = [value for value in quoted if value not in filenames and normalize(value) not in {normalize(name) for name in sheet_names}
               and value.replace('$', '').upper() not in refs]
    anchors += codes
    if not refs and not row_numbers and not anchors:
        # Exact complete cell labels are permitted; fuzzy similarity alone is
        # insufficient to choose a financial code or a numeric value.
        anchors = list(dict.fromkeys(str(cell['value']) for _, clause in candidates for cell in _cells(clause)
                       if cell.get('value') and len(str(cell['value'])) >= 4 and normalize(cell['value']) not in _HEADERS
                       and _has_phrase(question, cell['value'])))
    if not location and not anchors:
        return None

    reason = None
    if re.search(r'(?i)\$?[a-z]{1,3}\$?\d+\s*:\s*\$?[a-z]{1,3}\$?\d+', question) or len(refs) > 8:
        reason = 'Уточните отдельные ячейки: диапазон не раскрывается и не вычисляется автоматически.'
    elif refs:
        selected = []
        for ref in refs:
            matches = [(doc, clause) for doc, clause in candidates if ref in clause.get('cell_refs', [])]
            if len({_row_key(item) for item in matches}) != 1:
                reason = 'Адрес ячейки отсутствует или встречается в нескольких таблицах. Уточните файл и лист.'
                break
            selected.extend(matches)
        if reason is None:
            candidates = selected
    elif row_numbers:
        selected = [(doc, clause) for doc, clause in candidates if clause.get('row') in row_numbers]
        if any(len({_row_key(item) for item in selected if item[1].get('row') == row}) != 1 for row in row_numbers):
            reason = 'Строка отсутствует или неоднозначна. Уточните файл и лист.'
        else:
            candidates = selected
    elif anchors:
        candidates = [(doc, clause) for doc, clause in candidates if all(
            any(normalize(cell.get('value')) == normalize(anchor) for cell in _cells(clause)) for anchor in anchors)]
        if len({_row_key(item) for item in candidates}) != 1:
            reason = 'Точное значение или код не найдено однозначно. Укажите файл, лист и ячейку либо точное название строки.'
    else:
        reason = 'Укажите адрес ячейки или точное значение кода/названия строки, которое нужно найти в таблице.'

    if not reason and (refs or row_numbers) and any(not any(
            normalize(cell.get('value')) == normalize(anchor) for _, clause in candidates for cell in _cells(clause)) for anchor in anchors):
        reason = 'Указанные ячейки не подтверждают дополнительное название или код из вопроса. Уточните адрес и название строки.'

    if not candidates:
        reason = reason or 'В доступных действующих таблицах не найден указанный фрагмент. Проверьте файл, лист, дату и роль.'
    if reason:
        candidates = contextual
    candidates = list({clause['id']: (doc, clause) for doc, clause in candidates}.values())
    selected_ids = {doc['id'] for doc, _ in candidates}
    uncertain = any(clause.get('requires_review') or (clause.get('evidence') or {}).get('requires_review') or
                    doc.get('extraction', {}).get('requires_review') or
                    any(cell.get('requires_review') or cell.get('address_inferred') or cell.get('value') is None for cell in _cells(clause))
                    for doc, clause in candidates)
    version_ambiguity = bool(selected_ids & set(invalid_versions))
    unknown_dates = bool(re.search(r'(?i)действующ\w*|актуальн\w*|сегодня|сейчас', question)) and any(
        not (doc.get('metadata') or {}).get('effective_from') and not (doc.get('metadata') or {}).get('effective_to') for doc, _ in candidates)
    if uncertain:
        reason = 'Фрагмент содержит формулу, ошибку, неполное извлечение или неподтверждённые координаты. Проверьте оригинал; формулы не пересчитывались.'
    if version_ambiguity or unknown_dates:
        reason = 'Не удалось подтвердить применимую редакцию таблицы. Уточните связи версий и даты действия.'

    result.update(steps=[], facts=[], citations=[], sources=[], spreadsheet_lookup=True,
                  status='needs_clarification' if reason else 'answered',
                  coverage=dict(complete=not bool(reason), source_requires_review=uncertain,
                                version_ambiguity=version_ambiguity, unknown_effective_dates=unknown_dates))
    seen = set()
    for index, (doc, clause) in enumerate(candidates, 1):
        citation = _citation(doc, clause, index)
        result['citations'].append(citation)
        if not reason:
            result['facts'].append(dict(text=clause['text'], evidence_ids=[citation['id']]))
        if doc['id'] not in seen:
            seen.add(doc['id']); metadata = doc.get('metadata') or {}
            result['sources'].append(dict(document_id=doc['id'], document_name=doc['name'], title=metadata.get('title', doc['name']),
                                          version=metadata.get('version'), effective_from=metadata.get('effective_from'),
                                          effective_to=metadata.get('effective_to'), score=1.0))
            if doc.get('status') == 'review':
                result['warnings'].append(doc['name'] + ': при извлечении таблицы отмечены ограничения; проверьте предупреждения документа.')
    result['warnings'].extend(issue['message'] for issue in version_issues if selected_ids & set(issue['document_ids']))
    if reason:
        result['answer'] = reason
        result['warnings'].append('Табличный фрагмент не выдан как подтверждённый факт; вычисления и подстановка пропущенных значений не выполнялись.')
        result['followups'].append('Уточните файл, лист и ячейку или подтвердите значение по оригиналу.')
    else:
        result['answer'] = '\n\n'.join(str(index) + '. ' + fact['text'] for index, fact in enumerate(result['facts'], 1))
    result['score'] = result['confidence'] = 1.0 if not reason else 0.0
    result['warnings'] = list(dict.fromkeys(result['warnings']))
    return result
