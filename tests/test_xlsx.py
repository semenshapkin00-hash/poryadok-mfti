"""Synthetic ZIP/XML only: no official/internal documents, models or network."""
import io
import socket
import zipfile
from xml.sax.saxutils import escape, quoteattr

import pytest

from app_services import xlsx
from navigator_engine import DocumentInput, ingest

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'


def workbook(sheets=None, shared=None, styles=None, extra=None, workbook_extra=''):
    sheets = sheets or [('Смета', '<row r="1"><c r="A1" t="inlineStr"><is><t>Демо</t></is></c></row>', 'visible')]
    entries = {
        '[Content_Types].xml': '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        '_rels/.rels': f'<Relationships xmlns="{PKG}"><Relationship Id="office" Type="{REL}officeDocument" Target="xl/workbook.xml"/></Relationships>',
        'xl/workbook.xml': f'<workbook xmlns="{MAIN}" xmlns:r="{REL[:-1]}"><sheets>' + ''.join(
            '<sheet name='+quoteattr(name)+' sheetId="'+str(i)+'" r:id="s'+str(i)+'" state='+quoteattr(state)+'/>'
            for i, (name, _, state) in enumerate(sheets, 1)) + '</sheets></workbook>',
    }
    relations = []
    for i, (_, rows, _) in enumerate(sheets, 1):
        entries[f'xl/worksheets/sheet{i}.xml'] = f'<worksheet xmlns="{MAIN}"><sheetData>{rows}</sheetData></worksheet>'
        relations.append(f'<Relationship Id="s{i}" Type="{REL}worksheet" Target="worksheets/sheet{i}.xml"/>')
    if shared is not None:
        entries['xl/sharedStrings.xml'] = f'<sst xmlns="{MAIN}">{shared}</sst>'
        relations.append(f'<Relationship Id="strings" Type="{REL}sharedStrings" Target="sharedStrings.xml"/>')
    if styles is not None:
        entries['xl/styles.xml'] = f'<styleSheet xmlns="{MAIN}">{styles}</styleSheet>'
        relations.append(f'<Relationship Id="styles" Type="{REL}styles" Target="styles.xml"/>')
    entries['xl/_rels/workbook.xml.rels'] = f'<Relationships xmlns="{PKG}">' + ''.join(relations) + workbook_extra + '</Relationships>'
    entries.update(extra or {})
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return result.getvalue()


def parse(content):
    return ingest(DocumentInput('synthetic-xlsx', 'Вымышленный бланк.xlsx', content, {'synthetic': True}))


def test_sheets_sparse_row_cell_ids_shared_and_inline_text_are_preserved():
    rows = '<row r="3"><c r="A3" t="s"><v>0</v></c><c r="C3"><v>1250.50</v></c></row>'
    data = workbook([('Бюджет', rows, 'visible'), ('Пояснение', '<row r="7"><c r="B7" t="inlineStr"><is><t>Синтетический текст</t></is></c></row>', 'hidden')],
                    shared='<si><r><t>Статья </t></r><r><t>расходов</t></r><rPh><t>НЕ ДУБЛИРОВАТЬ</t></rPh></si>')
    doc = parse(data)
    assert doc['status'] == 'review'  # Hidden sheet is disclosed, not silently dropped.
    one, two = doc['clauses']
    assert one['sheet'] == 'Бюджет' and one['row'] == one['paragraph'] == 3
    assert one['cell_refs'] == ['A3', 'C3'] and 'Статья расходов' in one['text'] and '1250.50' in one['text']
    assert 'НЕ ДУБЛИРОВАТЬ' not in one['text'] and 'B3' not in one['cell_refs']
    assert two['row'] == 7 and two['sheet'] == 'Пояснение' and 'скрытая' in two['section']
    assert one['evidence']['sheet'] == 'Бюджет' and one['evidence']['cell_refs'] == ['A3', 'C3']


def test_formula_cached_missing_error_and_shared_formula_never_become_calculated():
    rows = '<row r="1"><c r="A1"><v>10</v></c><c r="B1"><f>SUM(A1:A1)</f><v>999</v></c></row>' \
           '<row r="2"><c r="B2"><f t="shared" si="2"/></c><c r="C2" t="e"><v>#REF!</v></c></row>'
    doc = parse(workbook([('Расчёт', rows, 'visible')]))
    cells = doc['clauses'][0]['spreadsheet_cells']; formula = cells[1]
    assert formula['formula'] == 'SUM(A1:A1)' and formula['cached_value'] == '999'
    assert 'не пересчитано: 999' in doc['clauses'][0]['text']  # Deliberately stale cache is not changed to10.
    assert doc['clauses'][0]['requires_review'] and doc['clauses'][0]['evidence']['requires_review']
    shared = doc['clauses'][1]['spreadsheet_cells'][0]
    assert shared['formula'] is None and shared['formula_attributes'] == {'t': 'shared', 'si': '2'}
    assert shared['cached_value'] is None and 'отсутствует' in shared['text']
    assert '#REF!' in doc['clauses'][1]['text'] and doc['clauses'][1]['requires_review']
    assert doc['extraction']['formulas_evaluated'] is False and doc['extraction']['formula_cells'] == 2


def test_external_links_and_formula_text_are_not_opened(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('External network attempted')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    formula = escape("'[1]Таблица'!A1")
    rows = f'<row r="1"><c r="A1"><f>{formula}</f><v>42</v></c></row>'
    link = f'<Relationship Id="external" Type="{REL}externalLink" Target="https://example.invalid/private.xlsx" TargetMode="External"/>'
    doc = parse(workbook([('Ссылки', rows, 'visible')], workbook_extra=link))
    assert doc['status'] == 'review' and doc['extraction']['external_links_not_followed']
    assert doc['extraction']['external_links_detected']
    assert doc['clauses'][0]['spreadsheet_cells'][0]['cached_value'] == '42'


def test_number_formats_are_saved_without_inventing_dates_or_percentages():
    styles = '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.00%"/></numFmts><cellXfs count="2"><xf numFmtId="14"/><xf numFmtId="164"/></cellXfs>'
    rows = '<row r="1"><c r="A1" s="0"><v>45000</v></c><c r="B1" s="1"><v>0.25</v></c><c r="C1" t="b"><v>1</v></c></row>'
    doc = parse(workbook([('Даты', rows, 'visible')], styles=styles)); cells = doc['clauses'][0]['spreadsheet_cells']
    assert cells[0]['value'] == '45000' and cells[0]['number_format'] == 'mm-dd-yy'
    assert cells[1]['value'] == '0.25' and cells[1]['number_format'] == '0.00%'
    assert cells[2]['value'] == 'TRUE' and 'без преобразования числа' in doc['clauses'][0]['text']


def test_merged_cells_are_not_forward_filled_and_inferred_addresses_are_marked():
    data = workbook([('Форма', '<row r="2"><c t="inlineStr"><is><t>Общий заголовок</t></is></c><c><v>0</v></c></row>', 'visible')])
    result = xlsx.extract(data); cells = result['rows'][0]['cells']
    assert [c['ref'] for c in cells] == ['A2', 'B2'] and all(c['address_inferred'] for c in cells)
    # Merged range is structural metadata only; do not invent a value for B1/C1.
    raw = workbook(extra={'xl/worksheets/sheet1.xml': f'<worksheet xmlns="{MAIN}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Один текст</t></is></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:C1"/></mergeCells></worksheet>'})
    result = xlsx.extract(raw)
    assert result['sheets'][0]['merged_ranges'] == ['A1:C1'] and len(result['rows'][0]['cells']) == 1


@pytest.mark.parametrize('extra', [
    {'xl/vbaProject.bin': b'not executed'}, {'xl/embeddings/oleObject1.bin': b'not executed'},
    {'../escape.xml': '<x/>'}, {'xl/worksheets/sheet1.xml': '<!DOCTYPE x [<!ENTITY a "bad">]><worksheet>&a;</worksheet>'},
    {'xl/worksheets/sheet1.xml': '<worksheet><broken>'},
])
def test_active_content_paths_dtd_and_broken_xml_fail_closed(extra):
    doc = parse(workbook(extra=extra))
    assert doc['status'] == 'unreadable' and doc['clauses'] == [] and doc['warnings']


@pytest.mark.parametrize('rows', [
    '<row r="1"><c r="A1" t="s"><v>9</v></c></row>',
    '<row r="1"><c r="XFE1"><v>1</v></c></row>',
    '<row r="1"><c r="A2"><v>1</v></c></row>',
    '<row r="1"><c r="A1"><v>1</v></c><c r="A1"><v>2</v></c></row>',
    '<row r="2"><c r="A2"><v>1</v></c></row><row r="1"><c r="A1"><v>2</v></c></row>',
])
def test_invalid_shared_indices_coordinates_and_duplicate_cells_fail_closed(rows):
    doc = parse(workbook([('Ошибки', rows, 'visible')], shared='<si><t>Демо</t></si>'))
    assert doc['status'] == 'unreadable' and not doc['clauses']


def test_limits_are_enforced_without_dense_excel_dimension_allocation(monkeypatch):
    sparse = '<row r="1048576"><c r="XFD1048576" t="inlineStr"><is><t>Край</t></is></c></row>'
    result = xlsx.extract(workbook([('Разреженный', sparse, 'visible')]))
    assert len(result['rows']) == 1 and result['rows'][0]['cells'][0]['ref'] == 'XFD1048576'
    monkeypatch.setattr(xlsx, 'MAX_CELLS', 1)
    doc = parse(workbook([('Лимит', '<row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row>', 'visible')]))
    assert doc['status'] == 'unreadable'


def test_long_rows_keep_only_overlapping_cell_evidence():
    a, b = 'первый ' * 180, 'второй ' * 180
    data = workbook([('Длинная', '<row r="1"><c r="A1" t="inlineStr"><is><t>'+a+'</t></is></c><c r="B1" t="inlineStr"><is><t>'+b+'</t></is></c></row>', 'visible')])
    doc = parse(data)
    assert len(doc['clauses']) >= 2 and doc['status'] == 'parsed'
    assert doc['clauses'][0]['cell_refs'] == ['A1'] and doc['clauses'][-1]['cell_refs'] == ['B1']
    assert all(c['cell_refs'] and c['evidence']['quote'] == c['text'] for c in doc['clauses'])


def test_empty_cells_do_not_invent_zero_or_action():
    rows = '<row r="1"><c r="A1"/><c r="B1"><v/></c><c r="C1" t="inlineStr"><is><t/></is></c></row>'
    doc = parse(workbook([('Пустой', rows, 'visible')]))
    assert doc['status'] == 'review' and doc['clauses'] == []
    assert doc['extraction']['external_links_not_followed'] is True


def test_chunking_does_not_drop_formula_review_context_from_the_same_row():
    text = 'Описание синтетической статьи расходов ' * 60
    row = '<row r="1"><c r="A1" t="inlineStr"><is><t>'+text+'</t></is></c><c r="B1"><f>1+1</f><v>2</v></c></row>'
    doc = parse(workbook([('Строка', row, 'visible')]))
    assert len(doc['clauses']) > 1
    assert all(c['requires_review'] and c['evidence']['requires_review'] for c in doc['clauses'])


def test_external_worksheet_and_duplicate_archive_parts_are_rejected():
    raw = workbook()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    entries['xl/_rels/workbook.xml.rels'] = f'<Relationships xmlns="{PKG}"><Relationship Id="s1" Type="{REL}worksheet" Target="https://example.invalid/sheet.xml" TargetMode="External"/></Relationships>'.encode()
    target = io.BytesIO()
    with zipfile.ZipFile(target, 'w') as z:
        for n, value in entries.items():
            z.writestr(n, value)
    assert parse(target.getvalue())['status'] == 'unreadable'
    target = io.BytesIO()
    with zipfile.ZipFile(target, 'w') as z:
        z.writestr('xl/workbook.xml', 'first')
        with pytest.warns(UserWarning):
            z.writestr('xl/workbook.xml', 'second')
    assert parse(target.getvalue())['status'] == 'unreadable'
