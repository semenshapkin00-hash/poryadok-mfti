"""Bounded, offline OOXML spreadsheet extraction; never evaluates a formula.

Only ZIP members and XML text are read. No Excel engine, macros, external files,
network, calculation, date-serial conversion or implicit merged-cell filling.
"""
import io
import posixpath
import re
import zipfile
from decimal import Decimal, InvalidOperation
from urllib.parse import unquote
from xml.etree import ElementTree as ET

MAX_BYTES = 5 * 1024**2
MAX_UNPACKED = 50 * 1024**2
MAX_XML_BYTES = 8 * 1024**2
MAX_ENTRIES = 1000
MAX_SHEETS = 32
MAX_ROWS = 20000
MAX_CELLS = 100000
MAX_TEXT = 2 * 1024**2
MAX_CELL_CHARS = 32767
CELL = re.compile(r'([A-Z]{1,3})([1-9][0-9]{0,6})\Z')
BUILTIN_FORMATS = {9: '0%', 10: '0.00%', 14: 'mm-dd-yy', 15: 'd-mmm-yy',
    16: 'd-mmm', 17: 'mmm-yy', 18: 'h:mm AM/PM', 19: 'h:mm:ss AM/PM',
    20: 'h:mm', 21: 'h:mm:ss', 22: 'm/d/yy h:mm'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _tag(node):
    return node.tag.rsplit('}', 1)[-1]


def _child(node, name):
    return next((x for x in node if _tag(x) == name), None)


def _children(node, name):
    return [] if node is None else [x for x in node if _tag(x) == name]


def _texts(node):
    if node is None or _tag(node) in {'rPh', 'phoneticPr'}:
        return ''
    return (node.text or '') if _tag(node) == 't' else ''.join(_texts(x) for x in node)


def _part(base, target):
    target = unquote(target)
    _need(target and '\\' not in target and ':' not in target and not target.startswith('//'), 'Некорректная внутренняя ссылка XLSX')
    name = posixpath.normpath(target.lstrip('/') if target.startswith('/') else posixpath.join(posixpath.dirname(base), target))
    _need(name not in {'.', '..'} and not name.startswith('../'), 'Ссылка XLSX выходит за пределы архива')
    return name


def _xml(archive, name):
    _need(name in archive.namelist(), 'В XLSX отсутствует обязательная XML-часть')
    info = archive.getinfo(name)
    _need(info.file_size <= MAX_XML_BYTES, 'Слишком большая XML-часть XLSX')
    data = archive.read(name)
    # Reject DTD/entities even for UTF-16 XML; ElementTree never fetches links.
    probe = data.replace(b'\x00', b'').upper()
    _need(b'<!DOCTYPE' not in probe and b'<!ENTITY' not in probe, 'DTD и XML-сущности в XLSX запрещены')
    return ET.fromstring(data)


def _relations(archive, part):
    name = posixpath.join(posixpath.dirname(part), '_rels', posixpath.basename(part) + '.rels') if part else '_rels/.rels'
    if name not in archive.namelist():
        return {}
    result = {}
    for node in _xml(archive, name):
        if _tag(node) != 'Relationship':
            continue
        rid = node.get('Id')
        _need(rid and rid not in result, 'Повторная связь XLSX')
        result[rid] = dict(type=node.get('Type', ''), target=node.get('Target', ''), external=node.get('TargetMode', '').casefold() == 'external')
    return result


def _coordinate(ref):
    match = CELL.fullmatch(ref)
    _need(match is not None, 'Некорректный адрес ячейки XLSX')
    col = 0
    for char in match[1]:
        col = col * 26 + ord(char) - 64
    row = int(match[2])
    _need(col <= 16384 and row <= 1048576, 'Адрес ячейки за пределами XLSX')
    return col, row


def _letters(col):
    out = ''
    while col:
        col, remainder = divmod(col - 1, 26)
        out = chr(65 + remainder) + out
    return out


def _value(cell, shared):
    value = _child(cell, 'v')
    raw = None if value is None else (value.text or '')
    kind = cell.get('t', 'n')
    if kind == 'inlineStr':
        return _texts(_child(cell, 'is')), kind
    if raw is None:
        return None, kind
    if kind == 's':
        _need(raw.isdecimal() and int(raw) < len(shared), 'Некорректная ссылка на строку XLSX')
        return shared[int(raw)], 'string'
    if kind == 'b':
        _need(raw in {'0', '1'}, 'Некорректное логическое значение XLSX')
        return ('TRUE' if raw == '1' else 'FALSE'), 'boolean'
    if kind == 'n' and raw:
        try:
            _need(len(raw) <= 1024 and Decimal(raw).is_finite(), 'Некорректное число XLSX')
        except InvalidOperation as exc:
            raise ValueError('Некорректное число XLSX') from exc
    _need(kind in {'n', 'str', 'e', 'd'}, 'Неподдерживаемый тип ячейки XLSX')
    return raw, {'n': 'number', 'str': 'string', 'e': 'error', 'd': 'ISO_date'}[kind]


def _formats(archive, relations, workbook):
    found = [v for v in relations.values() if v['type'].endswith('/styles') and not v['external']]
    if not found:
        return [None]
    root = _xml(archive, _part(workbook, found[0]['target']))
    custom = {int(x.get('numFmtId')): x.get('formatCode', '') for x in _children(_child(root, 'numFmts'), 'numFmt')}
    return [custom.get(int(x.get('numFmtId', '0')), BUILTIN_FORMATS.get(int(x.get('numFmtId', '0'))))
            for x in _children(_child(root, 'cellXfs'), 'xf')] or [None]


def extract(content):
    """Return rows/cells with source locators and explicit saved-formula metadata."""
    _need(isinstance(content, bytes) and 0 < len(content) <= MAX_BYTES, 'XLSX должен быть непустым файлом не более 5 МБ')
    rows, sheets, warnings = [], [], []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        infos = archive.infolist(); names = [x.filename for x in infos]
        _need(len(infos) <= MAX_ENTRIES and len(names) == len(set(names)), 'Слишком много или повторные части XLSX')
        _need(sum(x.file_size for x in infos) <= MAX_UNPACKED and not any(x.flag_bits & 1 for x in infos), 'Слишком большой или зашифрованный XLSX')
        _need(not any(x.compress_size and x.file_size / x.compress_size > 2000 for x in infos), 'Недопустимая степень сжатия XLSX')
        _need(not any(n.startswith('/') or '\\' in n or '..' in n.split('/') for n in names), 'Небезопасный путь в XLSX')
        _need(not any('vbaproject' in n.casefold() or '/activex/' in n.casefold() or '/embeddings/' in n.casefold() for n in names), 'Макросы и встроенные активные объекты XLSX не поддерживаются')
        packages = [v for v in _relations(archive, '').values() if v['type'].endswith('/officeDocument')]
        _need(len(packages) == 1 and not packages[0]['external'], 'Не найден внутренний workbook XLSX')
        workbook = _part('', packages[0]['target']); root = _xml(archive, workbook)
        relations = _relations(archive, workbook); formats = _formats(archive, relations, workbook)
        shared, chars = [], 0
        for rel in relations.values():
            if rel['type'].endswith('/sharedStrings') and not rel['external']:
                for node in _children(_xml(archive, _part(workbook, rel['target'])), 'si'):
                    text = _texts(node); chars += len(text)
                    _need(len(text) <= MAX_CELL_CHARS and chars <= MAX_TEXT and len(shared) < MAX_CELLS, 'Слишком много текста XLSX')
                    shared.append(text)
        sheet_nodes = _children(_child(root, 'sheets'), 'sheet')
        _need(0 < len(sheet_nodes) <= MAX_SHEETS, 'В XLSX допускается от 1 до 32 листов')
        seen_names, cell_count, row_count, text_count, formula_count = set(), 0, 0, 0, 0
        external = any(v['external'] for v in relations.values()) or any('/externalLinks/' in n for n in names)
        visual = any('/drawings/' in n or '/charts/' in n for n in names)
        for node in sheet_nodes:
            name, state = node.get('name', ''), node.get('state', 'visible')
            _need(name and len(name) <= 31 and name.casefold() not in seen_names and state in {'visible', 'hidden', 'veryHidden'}, 'Некорректное имя или состояние листа XLSX')
            seen_names.add(name.casefold())
            rid = next((v for k, v in node.attrib.items() if k.rsplit('}', 1)[-1] == 'id'), None)
            rel = relations.get(rid, {})
            _need(rel.get('type', '').endswith('/worksheet') and not rel.get('external'), 'Неподдерживаемая или внешняя ссылка на лист XLSX')
            part = _part(workbook, rel['target']); sheet = _xml(archive, part)
            external |= any(v['external'] for v in _relations(archive, part).values())
            merges = [x.get('ref', '') for x in _children(_child(sheet, 'mergeCells'), 'mergeCell')]
            _need(len(merges) <= MAX_CELLS, 'Слишком много объединений XLSX')
            for merge in merges:
                for endpoint in merge.split(':'):
                    _coordinate(endpoint)
            previous_row, extracted = 0, 0
            for row_node in _children(_child(sheet, 'sheetData'), 'row'):
                row_count += 1; _need(row_count <= MAX_ROWS, 'В XLSX слишком много строк')
                cells = _children(row_node, 'c'); first_ref = next((c.get('r') for c in cells if c.get('r')), None)
                row = int(row_node.get('r', str(_coordinate(first_ref)[1] if first_ref else previous_row + 1)))
                _need(previous_row < row <= 1048576, 'Повторная или неупорядоченная строка XLSX'); previous_row = row
                values, previous_col = [], 0
                for cell in cells:
                    cell_count += 1; _need(cell_count <= MAX_CELLS, 'В XLSX слишком много ячеек')
                    ref = cell.get('r') or (_letters(previous_col + 1) + str(row))
                    col, cell_row = _coordinate(ref)
                    _need(col > previous_col and cell_row == row, 'Повторная или несогласованная ячейка XLSX'); previous_col = col
                    value, kind = _value(cell, shared); formula = _child(cell, 'f')
                    style = int(cell.get('s', '0')); _need(0 <= style < len(formats), 'Некорректный формат ячейки XLSX')
                    if value in {None, ''} and formula is None:
                        continue
                    record = dict(ref=ref, value=value, value_type=kind, number_format=formats[style], address_inferred=cell.get('r') is None)
                    if formula is not None:
                        formula_count += 1; record.update(formula=formula.text or None, formula_attributes=dict(formula.attrib), cached_value=value, requires_review=True)
                        rendered = 'формула: ' + ('=' + formula.text if formula.text else '(текст отсутствует; ' + formula.get('t', 'связанная') + ')')
                        rendered += '; сохранённое значение, не пересчитано: ' + (value if value not in {None, ''} else 'отсутствует')
                    else:
                        rendered = value or ''
                    if formats[style] and kind == 'number':
                        rendered += '; формат ячейки: ' + formats[style] + ' (без преобразования числа)'
                    record['requires_review'] = bool(formula is not None or kind == 'error')
                    _need(len(rendered) <= MAX_CELL_CHARS, 'Слишком длинный текст ячейки XLSX')
                    record['text'] = rendered; values.append(record)
                if values:
                    text = ' | '.join('[' + v['ref'] + '] ' + v['text'] for v in values)
                    text_count += len(text); _need(text_count <= MAX_TEXT, 'Слишком большой извлечённый текст XLSX')
                    rows.append(dict(sheet=name, row=row, cells=values, text=text, hidden=state != 'visible' or row_node.get('hidden') in {'1', 'true'}, requires_review=any(v['requires_review'] for v in values)))
                    extracted += 1
            sheets.append(dict(name=name, state=state, source_part=part, extracted_rows=extracted, merged_ranges=merges))
        if formula_count:
            warnings.append('Формулы XLSX не вычислялись. Сохранённые значения могут быть устаревшими; ячейки с формулами требуют сверки в оригинале.')
        if external:
            warnings.append('Внешние ссылки XLSX не открывались и не обновлялись.')
        if any(row['hidden'] for row in rows):
            warnings.append('Извлечён также текст скрытых строк или листов; это отмечено в координатах.')
        if visual:
            warnings.append('Изображения и диаграммы внутри XLSX не распознавались; требуется сверка оригинала.')
    return dict(rows=rows, sheets=sheets, warnings=warnings, formula_cells=formula_count,
                external_links_not_followed=True, external_links_detected=external,
                unsupported_visual_content=visual, formulas_evaluated=False)
