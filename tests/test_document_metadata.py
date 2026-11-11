"""Pure metadata fixtures: no original corpus, models, OCR, filesystem or network."""
from copy import deepcopy

import pytest

from navigator_engine.document_metadata import infer_metadata_from_explicit_text, resolve_document_relationships


@pytest.mark.parametrize('line', [
    'Действует с: 2027-04-09', 'Дата вступления в силу: 09.04.2027',
    'Настоящий регламент вступает в силу с 9 апреля 2027 года.',
    '1. Документ вступает в силу с «9» апреля 2027 г.',
])
def test_only_explicit_effective_dates_with_exact_provenance(line):
    text = '# Учебный регламент\n' + line + '\nУтверждено 01.02.2027.'
    metadata, proof = infer_metadata_from_explicit_text(text)
    assert metadata['effective_from'] == '2027-04-09'
    assert not proof['ambiguities']
    for match in proof['fields']['effective_from']:
        assert text[match['start']:match['end']] == match['quote']


@pytest.mark.parametrize('text', [
    'УТВЕРЖДЕНО 09.04.2027\nПлан мероприятий на 2027–2029 годы.',
    'Приказ от 09.04.2027 № А-17\nСрок подачи: 19.04.2027.',
    'Согласовано 09.04.2027.\nДата регистрации: 10.04.2027.',
    'Для заявления, которое вступает в силу с 09.04.2027, нужен паспорт.',
])
def test_approval_filename_like_plan_and_unrelated_dates_are_not_effective(text):
    metadata, proof = infer_metadata_from_explicit_text(text, {'title': 'План 2027.xlsx'})
    assert 'effective_from' not in metadata and 'effective_to' not in metadata


def test_explicit_existing_metadata_wins_and_inputs_are_immutable():
    existing = {'effective_from': '2027-01-01', 'version': 'manual', 'roles': ['employee']}
    original = deepcopy(existing)
    metadata, proof = infer_metadata_from_explicit_text('Действует с: 2027-04-09\nВерсия: 2', existing)
    assert metadata == existing == original and proof['ambiguities']
    metadata['roles'].append('manager')
    assert existing == original


def test_conflicting_or_invalid_source_dates_never_choose_one():
    for text in ['Действует с: 2027-04-09\nДействует с: 2027-04-10',
                 'Действует с: 31.02.2027\nДействует с: 2027-04-09']:
        metadata, proof = infer_metadata_from_explicit_text(text)
        assert 'effective_from' not in metadata and proof['ambiguities']


def test_expiry_version_and_inverted_interval_are_explicit():
    metadata, proof = infer_metadata_from_explicit_text('Действует с: 2027-04-09; Действует по: 31.12.2027 включительно\nРедакция: 2')
    assert metadata == {'effective_from': '2027-04-09', 'effective_to': '2027-12-31', 'version': '2'}
    assert not proof['ambiguities']
    metadata, proof = infer_metadata_from_explicit_text('Действует с: 2028-01-01\nДействует до: 2027-12-31')
    assert not metadata and proof['ambiguities'][-1]['field'] == 'effective_period'


@pytest.mark.parametrize(('begin', 'end'), [
    ('Дата начала действия: 2029-02-08', 'Дата окончания действия: 30.06.2029'),
    ('Дата начала действия: 8 февраля 2029 года', 'Дата окончания действия: 2029-06-30'),
])
def test_explicit_begin_and_end_label_aliases_keep_exact_source_spans(begin, end):
    text = '# Учебная политика\n' + begin + '\n' + end
    metadata, proof = infer_metadata_from_explicit_text(text)
    assert metadata == {'effective_from': '2029-02-08', 'effective_to': '2029-06-30'}
    assert not proof['ambiguities']
    for field in ['effective_from', 'effective_to']:
        span = proof['fields'][field][0]
        assert span['quote'] == text[span['start']:span['end']]


def test_begin_date_label_conflict_and_condition_still_require_clarification():
    for extra in ['Действует с: 2029-02-09', 'Дата начала действия: 2029-02-08, если получено согласование.']:
        metadata, proof = infer_metadata_from_explicit_text('Дата начала действия: 2029-02-08\n' + extra)
        assert 'effective_from' not in metadata and proof['ambiguities']


def test_number_is_from_own_title_never_a_body_reference():
    metadata, proof = infer_metadata_from_explicit_text('# Приказ № УЧ-017/а\nСм. приказ № УЧ-018.')
    assert metadata['doc_number'] == 'УЧ-017/а'
    assert proof['fields']['doc_number'][0]['source'] == 'text_heading'
    metadata, proof = infer_metadata_from_explicit_text('# Порядок работы\nСм. приказ № УЧ-018.')
    assert 'doc_number' not in metadata
    metadata, proof = infer_metadata_from_explicit_text('', {'title': 'Регламент № 004-Р'})
    assert metadata['doc_number'] == '004-Р' and proof['fields']['doc_number'][0]['source'] == 'metadata.title'


def test_number_sentence_boundary_markers_and_existing_date_forms():
    metadata, proof = infer_metadata_from_explicit_text('Приказ № N-17. Дата утверждения: 01.01.2027.')
    assert metadata['doc_number'] == 'N-17'
    metadata, proof = infer_metadata_from_explicit_text('Действует с: 2027-04-09', {'effective_from': '09.04.2027'})
    assert metadata['effective_from'] == '09.04.2027' and not proof['ambiguities']
    result = resolve_document_relationships([document('old', '№ 004-Р'), document('new', 'N-17', 'Заменяет документ № 004-Р. Новое правило действует отдельно.')])
    assert result[1]['metadata']['supersedes'] == ['old']


def document(identifier, number, text='', **metadata):
    return {'id': identifier, 'name': identifier + '.md', 'metadata': {'doc_number': number, **metadata},
            'clauses': [{'id': identifier + ':1', 'text': text}]}


@pytest.mark.parametrize('reference', [
    'Заменяет документ № 004-Р.',
    'Настоящий регламент полностью заменяет приказ № 004-Р.',
    '2. Признать утратившим силу приказ № 004-Р.',
])
def test_exact_full_replacement_links_and_preserves_span(reference):
    documents = [document('old', '004-Р'), document('new', '005-Р', reference, effective_from='2028-01-01')]
    original = deepcopy(documents)
    resolved = resolve_document_relationships(documents)
    assert documents == original
    assert resolved[1]['metadata']['supersedes'] == ['old']
    assert 'effective_to' not in resolved[0]['metadata']
    proof = resolved[1]['metadata_provenance']['relationship_resolution']['resolved'][0]
    assert proof['quote'] == reference[proof['start']:proof['end']]
    assert proof['clause_id'] == 'new:1'


@pytest.mark.parametrize('reference', [
    'Признать утратившим силу пункт 2 приказа № 004-Р.',
    'Заменяет документ № 004-Р в части пункта 2.',
    'Не заменяет документ № 004-Р.',
    'Если понадобится, признать утратившим силу приказ № 004-Р.',
    'Внести изменения в приказ № 004-Р.',
    'Заменяет документ № 004.',
])
def test_partial_conditional_negative_or_prefix_number_never_replaces_full_doc(reference):
    result = resolve_document_relationships([document('old', '004-Р'), document('new', '005-Р', reference)])
    assert not result[1]['metadata'].get('supersedes')


def test_duplicate_number_self_reference_explicit_links_and_review_text_do_not_override():
    for documents in [
        [document('one', '004-Р'), document('two', '004-Р'), document('new', '005-Р', 'Заменяет документ № 004-Р.')],
        [document('new', '004-Р', 'Заменяет документ № 004-Р.')],
    ]:
        result = resolve_document_relationships(documents)
        assert not result[-1]['metadata'].get('supersedes')
        assert result[-1]['metadata_provenance']['relationship_resolution']['unresolved']
    documents = [document('old', '004-Р'), document('new', '005-Р', 'Заменяет документ № 004-Р.', supersedes=['manual'])]
    assert resolve_document_relationships(documents)[-1]['metadata']['supersedes'] == ['manual']
    documents[-1]['metadata'].pop('supersedes')
    documents[-1]['clauses'][0]['requires_review'] = True
    assert not resolve_document_relationships(documents)[-1]['metadata'].get('supersedes')


@pytest.mark.parametrize('condition', [
    ', если получено согласование', ', при условии согласования',
    ', в случае согласования', ', при наличии согласования', ' после подписания акта',
])
def test_trailing_conditions_do_not_create_unconditional_dates_or_replacements(condition):
    statement = 'Документ вступает в силу с 09.04.2027' + condition + '.'
    metadata, proof = infer_metadata_from_explicit_text(statement)
    assert 'effective_from' not in metadata and proof['ambiguities']
    span = proof['fields']['effective_from'][0]
    assert span['rule'] == 'conditional_explicit_date' and span['quote'] == statement[span['start']:span['end']]
    documents = [document('old', '004-Р'), document('new', '005-Р', 'Заменяет документ № 004-Р' + condition + '.')]
    assert not resolve_document_relationships(documents)[-1]['metadata'].get('supersedes')


def test_dotted_reference_date_does_not_hide_a_later_replacement_condition():
    prefix = 'Заменяет документ № 004-Р от 01.02.2025'
    documents = [document('old', '004-Р'), document('new', '005-Р', prefix + ', если получено согласование.')]
    assert not resolve_document_relationships(documents)[-1]['metadata'].get('supersedes')
    # A genuinely separate next sentence must not make the first rule conditional.
    documents[-1]['clauses'][0]['text'] = prefix + '. Если потерян экземпляр, запросите копию.'
    assert resolve_document_relationships(documents)[-1]['metadata']['supersedes'] == ['old']


def test_year_abbreviation_does_not_hide_a_later_effective_date_condition():
    prefix = 'Действует с: 09.04.2027 г.'
    metadata, proof = infer_metadata_from_explicit_text(prefix + ', если получено согласование.')
    assert 'effective_from' not in metadata and proof['ambiguities']
    span = proof['fields']['effective_from'][0]
    assert ', если получено согласование' in span['quote']
    metadata, proof = infer_metadata_from_explicit_text(prefix + ' Если потерян экземпляр, запросите копию.')
    assert metadata['effective_from'] == '2027-04-09' and not proof['ambiguities']
