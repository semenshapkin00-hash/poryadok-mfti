"""Fast corpus helpers used by the local application and evaluation harness."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent


def demo_manifest():
    value=json.loads((ROOT/'manifest.json').read_text())
    return [dict(d,path=ROOT/d['path']) for d in value['documents']]


def demo_file_paths():return [d['path'] for d in demo_manifest()]


def demo_documents():
    from navigator_engine import DocumentInput
    return [DocumentInput(d['id'],d['name'],d['path'].read_bytes(),d['metadata']) for d in demo_manifest()]


def demo_examples():
    examples=[
        dict(id='xlsx_code',question='Что указано в ячейке B2 листа «Подразделения» файла «Учебная структура.xlsx»?',as_of='2026-11-11',role='employee'),
        dict(id='xlsx_formula',question='Какая формула находится в ячейке B4 листа «Смета» файла «Учебная структура.xlsx»?',as_of='2026-11-11',role='employee'),
        dict(id='trip',question='Как оформить командировку и за сколько дней подать заявку?',as_of='2026-11-08',role='employee'),
        dict(id='access',question='Как получить удалённый доступ к рабочей системе?',as_of='2026-11-08',role='employee'),
        dict(id='receipts',question='Что делать, если потерял кассовый чек после командировки?',as_of='2026-11-08',role='employee'),
        dict(id='conflict',question='Каков срок согласования заявки на закупку?',as_of='2026-11-08',role='employee'),
        dict(id='future',question='Какая компенсация служебного переезда предусмотрена сейчас?',as_of='2026-11-08',role='employee'),
        dict(id='unknown',question='Как оформить декретный отпуск?',as_of='2026-11-08',role='employee'),
    ]
    return [dict(e,label=e['question']) for e in examples]
