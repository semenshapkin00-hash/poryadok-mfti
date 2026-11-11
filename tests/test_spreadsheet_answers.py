"""Synthetic spreadsheets only; no internal corpus, models, OCR or network."""
from copy import deepcopy

import pytest

from navigator_engine import DocumentInput, answer, ingest
from test_xlsx import workbook


def document(identifier='book', sheets=None, metadata=None, name='Демо.xlsx'):
    rows = '<row r="1"><c r="A1" t="inlineStr"><is><t>Отдел Демо</t></is></c><c r="B1"><v>00712</v></c></row>'
    content = workbook(sheets or [('Структура', rows, 'visible')])
    return ingest(DocumentInput(identifier, name, content, metadata or {'effective_from': '2026-01-01', 'roles': ['employee']}))


def ask(question, documents, **kwargs):
    result = answer(question, documents, as_of=kwargs.get('as_of', '2026-11-11'), role=kwargs.get('role', 'employee'))
    clauses = {clause['id']: clause for doc in documents for clause in doc.get('clauses', [])}
    citations = {citation['id']: citation for citation in result['citations']}
    for citation in citations.values():
        clause = clauses[citation['clause_id']]
        assert citation['quote'] == clause['text']
        if clause.get('sheet'):
            assert all(citation[key] == clause[key] for key in ['sheet', 'row', 'cell_refs', 'spreadsheet_cells'])
    for fact in result.get('facts', []):
        assert fact['evidence_ids'] and all(ref in citations for ref in fact['evidence_ids'])
        assert fact['text'] in ' '.join(citations[ref]['quote'] for ref in fact['evidence_ids'])
    if result['status'] == 'answered':
        assert result['answer'] == '\n\n'.join(str(i) + '. ' + item['text'] for i, item in enumerate(result['steps'] + result.get('facts', []), 1))
    return result


@pytest.mark.parametrize('question', [
    'Что в ячейке B1 на листе «Структура»?',
    'Что в ячейке «B1» на листе «Структура»?',
    'Какое значение в Структура!B1?',
    "Что записано в 'Структура'!$B$1?",
    'Какой код у подразделения «Отдел Демо»?',
    'Что означает код 00712?',
    'Какой код у Отдел Демо в Excel?',
    'Какое значение в строке 1 на листе «Структура»?',
])
def test_literal_cell_code_and_exact_row_label_produce_facts_not_steps(question):
    result = ask(question, [document()])
    assert result['status'] == 'answered' and result['spreadsheet_lookup']
    assert result['facts'] and not result['steps'] and '00712' in result['answer']
    assert result['citations'][0]['sheet'] == 'Структура' and result['citations'][0]['row'] == 1


@pytest.mark.parametrize('query', ['Что означает код AB12?', 'Что означает код «AB12»?', 'Что означает код AB12 на листе «Коды»?'])
def test_alphanumeric_code_is_a_literal_value_not_an_implicit_cell_address(query):
    rows = '<row r="1"><c r="A1" t="inlineStr"><is><t>AB12</t></is></c><c r="B1" t="inlineStr"><is><t>Демо-статья</t></is></c></row>'
    result = ask(query, [document(sheets=[('Коды', rows, 'visible')])])
    assert result['status'] == 'answered' and result['facts'] and 'Демо-статья' in result['answer']


def test_whole_active_corpus_lookup_requires_unique_sheet_and_workbook():
    one = document('one', name='Книга первая.xlsx')
    two = document('two', name='Книга вторая.xlsx')
    ambiguous = ask('Что в ячейке B1 на листе «Структура»?', [one, two])
    assert ambiguous['status'] == 'needs_clarification' and not ambiguous['facts'] and not ambiguous['steps']
    assert len(ambiguous['citations']) == 2
    precise = ask('Что в ячейке B1 на листе «Структура» файла «Книга вторая.xlsx»?', [one, two])
    assert precise['status'] == 'answered' and precise['citations'][0]['document_id'] == 'two'


@pytest.mark.parametrize('question', [
    'Что в ячейке C99 на листе «Структура»?',
    'Что в ячейке B1 на листе «Несуществующий»?',
    'Что в ячейке B1 на листе Несуществующий?',
    'Что в ячейке B1 файла «Нет.xlsx»?',
    'Что означает код 99999 в Excel?',
    'Что в ячейке B1 для «Чужое подразделение» на листе «Структура»?',
    'Какое значение в диапазоне A1:B9 на листе «Структура»?',
])
def test_missing_or_unsupported_scope_is_not_silently_dropped(question):
    result = ask(question, [document()])
    assert result['status'] == 'needs_clarification' and not result['facts'] and not result['steps']
    assert not result['coverage']['complete'] and result['followups']


def test_same_cell_on_two_sheets_is_ambiguous_even_with_identical_values():
    rows = '<row r="1"><c r="A1"><v>42</v></c></row>'
    doc = document(sheets=[('Первый', rows, 'visible'), ('Второй', rows, 'visible')])
    result = ask('Что в ячейке A1 в Excel?', [doc])
    assert result['status'] == 'needs_clarification' and len(result['citations']) == 2
    assert ask('Что в ячейке A1 на листе «Второй»?', [doc])['status'] == 'answered'


@pytest.mark.parametrize('formula', ['<f>SUM(A1:A1)</f><v>999</v>', '<f t="shared" si="1"/>', '<v>#REF!</v>'])
def test_formula_cache_missing_formula_and_errors_are_review_only(formula):
    kind = ' t="e"' if '#REF!' in formula else ''
    rows = '<row r="1"><c r="A1"><v>10</v></c><c r="B1"' + kind + '>' + formula + '</c></row>'
    result = ask('Что в ячейке B1 на листе «Расчёт»?', [document(sheets=[('Расчёт', rows, 'visible')])])
    assert result['status'] == 'needs_clarification' and not result['facts']
    assert result['citations'] and result['citations'][0]['requires_review']
    assert result['coverage']['source_requires_review'] and 'не пересчитывались' in result['answer']
    if '999' in formula:
        assert '999' in result['citations'][0]['quote'] and not result['steps']


def test_unreadable_workbook_and_unsafe_rows_are_never_facts():
    unreadable = ingest(DocumentInput('bad', 'Повреждённый.xlsx', b'not a zip', {}))
    result = ask('Что в ячейке A1 файла «Повреждённый.xlsx»?', [unreadable])
    assert result['status'] == 'needs_clarification' and not result['facts']
    assert result['excluded'][0]['reason'] == 'unreadable'
    unsafe = document(); unsafe['clauses'][0]['unsafe'] = True
    assert not ask('Что в ячейке B1 на листе «Структура»?', [unsafe])['facts']


@pytest.mark.parametrize(('metadata', 'kwargs', 'excluded'), [
    ({'effective_to': '2026-01-01'}, {}, 'expired'),
    ({'effective_from': '2027-01-01'}, {}, 'not_yet_effective'),
    ({'roles': ['manager']}, {'role': 'employee'}, 'role_mismatch'),
])
def test_core_date_and_role_filters_happen_before_spreadsheet_lookup(metadata, kwargs, excluded):
    result = ask('Что в ячейке B1 на листе «Структура»?', [document(metadata=metadata)], **kwargs)
    assert result['status'] == 'needs_clarification' and not result['citations']
    assert result['excluded'][0]['reason'] == excluded


def test_superseded_and_cyclic_versions_keep_existing_safety_rules():
    old = document('old', metadata={'effective_from': '2025-01-01', 'version': '1'})
    new = document('new', metadata={'effective_from': '2026-01-01', 'version': '2', 'supersedes': ['old']})
    result = ask('Что в ячейке B1 на листе «Структура»?', [old, new])
    assert result['status'] == 'answered' and result['citations'][0]['document_id'] == 'new'
    old['metadata']['supersedes'] = ['new']
    result = ask('Что в ячейке B1 файла «Демо.xlsx» на листе «Структура»?', [old, new])
    assert result['status'] == 'needs_clarification' and result['coverage']['version_ambiguity']


def test_literal_rows_remain_usable_when_another_row_contains_a_formula():
    rows = '<row r="1"><c r="A1"><v>10</v></c></row><row r="2"><c r="A2"><f>A1+1</f><v>11</v></c></row>'
    doc = document(sheets=[('Расчёт', rows, 'visible')])
    assert doc['status'] == 'review'
    result = ask('Что в ячейке A1 на листе «Расчёт»?', [doc])
    assert result['status'] == 'answered' and not result['citations'][0]['requires_review']


def test_multiple_factual_facets_keep_verified_facts_and_canonical_answer():
    rows = '<row r="1"><c r="A1"><v>10</v></c></row><row r="2"><c r="A2"><v>20</v></c></row>'
    doc = document(sheets=[('Расчёт', rows, 'visible')])
    result = ask('Что в ячейке A1 на листе «Расчёт» и что в ячейке A2 на листе «Расчёт»?', [doc])
    assert result['status'] == 'answered' and len(result['facts']) == 2 and not result['steps']


def test_procedures_and_ambiguous_regulatory_queries_preserve_original_fallback():
    regulation = ingest(DocumentInput('rule', 'Регламент.md', '# Талон\nТалон создают в системе.'.encode(), {'effective_from': '2026-01-01'}))
    for question in ['Как создать талон?', 'Как заполнить ячейку A1 в Excel?', 'Какой порядок оформления?', 'Какой код нужен для оформления?']:
        before = answer(question, [regulation], as_of='2026-11-11')
        after = answer(question, [regulation, document()], as_of='2026-11-11')
        assert not before.get('spreadsheet_lookup') and not after.get('spreadsheet_lookup')
        assert before['status'] == after['status'] and before['steps'] == after['steps']


@pytest.mark.parametrize('question', [
    'Нужно ли согласовать значение ячейки B1 на листе «Структура»?',
    'Что делать со значением ячейки B1 на листе «Структура»?',
    'Следует ли изменить значение ячейки B1 на листе «Структура»?',
    'Можно ли использовать код из ячейки B1 на листе «Структура»?',
    'Какие документы нужны для изменения ячейки B1 на листе «Структура»?',
    'Какие нужны действия для изменения ячейки B1 на листе «Структура»?',
])
def test_procedural_intent_with_literal_address_is_not_answered_as_a_cell_fact(question):
    result = ask(question, [document()])
    assert not result.get('spreadsheet_lookup') and not result.get('facts')


def test_lookup_does_not_mutate_source_and_missing_current_dates_are_not_asserted():
    doc = document(metadata={'synthetic': True}); original = deepcopy(doc)
    result = ask('Какое актуальное значение в ячейке B1 на листе «Структура»?', [doc])
    assert result['status'] == 'needs_clarification' and result['coverage']['unknown_effective_dates']
    assert doc == original
