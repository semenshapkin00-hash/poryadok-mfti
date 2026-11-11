"""Bounded local TXT/Markdown, selectable PDF, DOCX and XLSX ingestion."""
from datetime import date,datetime
from pathlib import Path
import hashlib,io,json,math,re,zipfile
from .models import DocumentInput
from .text import normalize

MAX_BYTES=20*1024**2;MAX_CLAUSES=5000;MAX_PAGES=150
INJECTION=re.compile(r'(?i)ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|игнориру[йя].{0,40}(?:инструкц|правил|системн)|(?:system|developer)\s*prompt|системн.{0,15}промпт|(?:отправь|раскрой|выведи).{0,50}(?:api.?key|секретн.{0,10}ключ|системн.{0,10}инструкц)')

def iso_date(value):
    if value is None or str(value).strip()=='':return None
    if isinstance(value,date):return value.isoformat()
    text=str(value).strip()
    for fmt in ['%Y-%m-%d','%d.%m.%Y','%d/%m/%Y']:
        try:return datetime.strptime(text,fmt).date().isoformat()
        except ValueError:pass
    raise ValueError('Некорректная дата: '+text[:60])

def decode(content):
    for encoding in ['utf-8-sig','utf-16','cp1251']:
        try:
            text=content.decode(encoding)
            if '\x00' not in text:return text
        except UnicodeError:pass
    raise ValueError('Кодировка текста не распознана')

def list_value(value):
    if value is None or value=='':return []
    if isinstance(value,(tuple,list,set)):return [str(x).strip() for x in value if str(x).strip()]
    text=str(value).strip()
    try:
        parsed=json.loads(text)
        if isinstance(parsed,list):return [str(x).strip() for x in parsed]
    except (ValueError,TypeError):pass
    return [x.strip().strip('"\'') for x in text.strip('[]').split(',') if x.strip()]

ALIASES={'действует с':'effective_from','действует до':'effective_to','редакция':'version','версия':'version','название':'title','роли':'roles','заменяет':'supersedes','тема':'topic','дата вступления в силу':'effective_from'}

def frontmatter(text):
    metadata={}
    if text.startswith('---'):
        match=re.match(r'^---\s*\n(.*?)\n---\s*\n',text,re.S)
        if match:
            for line in match.group(1).splitlines():
                if ':' in line:
                    key,value=line.split(':',1);metadata[ALIASES.get(normalize(key),key.strip())]=value.strip().strip('"\'')
            return metadata,text[match.end():]
    return metadata,text


def _chunks(text,max_chars=1200):
    """Contiguous chunks retain exact source strings; never summarize values."""
    text=text.strip()
    while len(text)>max_chars:
        split=max(text.rfind('. ',0,max_chars),text.rfind('; ',0,max_chars),text.rfind('\n',0,max_chars))
        if split<max_chars//3:split=text.rfind(' ',0,max_chars)
        if split<=0:split=max_chars
        else:split+=1
        yield text[:split].strip();text=text[split:].strip()
    if text:yield text


def _is_notice(text):
    """Identify non-normative synthetic/example disclosures, not rule content.

    Extra descriptive words and line wrapping do not turn an engineering/demo
    notice into an instruction. A paragraph that also states an operational
    requirement remains searchable rather than silently losing that rule.
    """
    value=normalize(text)
    starts=bool(re.match(r'^(?:синтетическ\w*|демонстрацион\w*|учебн\w*)\s+(?:[а-яa-z-]+\s+){0,4}(?:пример|документ|набор|материал)\b',value))
    explicit=bool(re.search(r'вымышлен\w*',value) and re.search(r'не (?:является|реальн\w*|правило|нормативн\w*|юридическ\w*|производствен\w*)',value))
    operational=bool(re.search(r'\b(?:должен|должна|обязан|обязана|запрещено|разрешается|допускается|требуется|пода[ею]тся|регистрируется|исполняется|обрабатывается|не позднее|в течение\s+\d|за\s+\d)',value))
    return (starts or (explicit and len(value)<1000)) and not operational


def _append(document,text,section,paragraph,page=None,bbox=None,provenance="text",ocr_confidence=None,word_confidences=None,page_width=None,page_height=None,coordinate_system=None):
    for part in _chunks(text):
        if len(document['clauses'])>=MAX_CLAUSES:raise ValueError('Не более 5000 пунктов в документе')
        if not part.strip():continue
        cid=f"{document['id']}:clause:{len(document['clauses'])+1}"
        item=dict(id=cid,document_id=document['id'],document_name=document['name'],text=part,section=section,paragraph=paragraph,page=page,bbox=bbox,
                  unsafe=bool(INJECTION.search(part)),kind='notice' if _is_notice(part) else 'rule')
        item['evidence']=dict(document_id=document['id'],document_name=document['name'],page=page,section=section,paragraph=paragraph,quote=part,bbox=bbox)
        item['provenance']=provenance
        item['evidence']['provenance']=provenance
        if page_width is not None:item['evidence']['page_width']=page_width;item['page_width']=page_width
        if page_height is not None:item['evidence']['page_height']=page_height;item['page_height']=page_height
        if coordinate_system:item['evidence']['coordinate_system']=coordinate_system;item['coordinate_system']=coordinate_system
        if provenance in {'ocr','pdf_text+ocr'}:
            details=word_confidences or []
            numeric=any(re.search(r'\d',word.get('text','')) for word in details)
            item.update(ocr_confidence=ocr_confidence,requires_review=True,numeric_uncertainty=numeric,word_confidences=details)
            item['evidence'].update(ocr_confidence=ocr_confidence,requires_review=True,numeric_uncertainty=numeric,word_confidences=details)
        document['clauses'].append(item)
        if item['unsafe']:document['warnings'].append('Пункт содержит признаки инструкции для модели и исключён из ответов: '+cid)


def _text(document,text):
    fm,text=frontmatter(text);document['metadata']={**fm,**document['metadata']}
    section='';paragraph=0;buffer=[]
    def flush():
        nonlocal paragraph
        if buffer:
            paragraph+=1;_append(document,'\n'.join(buffer),section,paragraph);buffer.clear()
    for line in text.splitlines():
        stripped=line.strip()
        if re.match(r'^#{1,6}\s+',stripped):
            flush();section=re.sub(r'^#{1,6}\s+','',stripped).strip()
            if not document['metadata'].get('title'):document['metadata']['title']=section
        elif not stripped:flush()
        elif re.match(r'^\s*(?:[-*•]\s+|\d+(?:\.\d+)*[.)]\s+)',line):
            flush();buffer.append(line.strip());flush()
        else:buffer.append(line.rstrip())
    flush();document['text_preview']=text[:16000]


def _zip_guard(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        items=archive.infolist()
        if len(items)>1000 or sum(item.file_size for item in items)>50*1024**2:raise ValueError('Слишком большой распакованный DOCX')
        if any(item.flag_bits&1 for item in items):raise ValueError('Зашифрованный DOCX не поддерживается')
        if any(item.compress_size and item.file_size/item.compress_size>2000 for item in items):raise ValueError('Недопустимая степень сжатия DOCX')


def _docx(document,content):
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    _zip_guard(content);doc=Document(io.BytesIO(content));section='';index=0;previews=[]
    for child in doc.element.body:
        if child.tag==qn('w:p'):
            para=Paragraph(child,doc);text=para.text.strip()
            if not text:continue
            index+=1;previews.append(text)
            if para.style and ('Heading' in para.style.name or 'Заголовок' in para.style.name):section=text
            else:_append(document,text,section,index)
        elif child.tag==qn('w:tbl'):
            table=Table(child,doc)
            for row in table.rows:
                index+=1;text=' | '.join(cell.text for cell in row.cells);previews.append(text);_append(document,text,section,index)
    document['text_preview']='\n'.join(previews)[:16000]


def _xlsx(document,content):
    from app_services.xlsx import extract
    parsed=extract(content)
    document['warnings'].extend(parsed['warnings'])
    evidence_size=0
    for row in parsed['rows']:
        first=len(document['clauses'])
        section='Лист «'+row['sheet']+'», строка '+str(row['row'])
        if row['hidden']:section+=' (скрытая)'
        _append(document,row['text'],section,row['row'],provenance='xlsx_xml')
        spans=[];position=0;cursor=0
        for cell in row['cells']:
            end=position+len('['+cell['ref']+'] '+cell['text'])
            spans.append((position,end,cell));position=end+3
        for clause in document['clauses'][first:]:
            start=row['text'].find(clause['text'],cursor);end=start+len(clause['text']);cursor=end
            if start<0:raise ValueError('Не удалось сохранить точные координаты фрагмента XLSX')
            cells=[cell for left,right,cell in spans if left<end and right>start]
            evidence_size+=len(json.dumps(cells,ensure_ascii=False))
            if evidence_size>8*1024**2:raise ValueError('Слишком большой объём координат и значений XLSX')
            details=dict(sheet=row['sheet'],row=row['row'],cell_refs=[cell['ref'] for cell in cells],spreadsheet_cells=cells,
                         requires_review=row['requires_review'],formulas_evaluated=False)
            clause.update(details);clause['evidence'].update(details)
    document['text_preview']='\n'.join(row['text'] for row in parsed['rows'])[:16000]
    document['extraction']=dict(method='xlsx_xml',sheets=parsed['sheets'],formula_cells=parsed['formula_cells'],
        formulas_evaluated=False,cached_values_may_be_stale=True,external_links_not_followed=parsed['external_links_not_followed'],
        external_links_detected=parsed['external_links_detected'],
        unsupported_visual_content=parsed['unsupported_visual_content'],requires_review=parsed['unsupported_visual_content'])
    document['metadata']['extraction_provenance']='xlsx_xml'


def _page_needs_ocr(page,words):
    if not words:return True
    text=' '.join(word.get('text','') for word in words)
    if text.count('�')>max(1,len(text)*.05):return True
    # A text header/page number must not conceal an image-only policy body.
    for image in page.images:
        box=(image.get('x0',0),image.get('top',0),image.get('x1',0),image.get('bottom',0))
        area=max(0,box[2]-box[0])*max(0,box[3]-box[1])
        if area<float(page.width)*float(page.height)*.35:
            continue
        inside=sum(box[0]<=(word['x0']+word['x1'])/2<=box[2] and box[1]<=(word['top']+word['bottom'])/2<=box[3] for word in words)
        if inside<3:return True
    return False


def _ocr_words(page):
    """Validate a callback result independently; coordinates remain source based."""
    width=float(page.get('width',0));height=float(page.get('height',0));output=[]
    if not all(math.isfinite(v) and v>0 for v in [width,height]):raise ValueError('Некорректный размер страницы OCR.')
    if len(page.get('lines',[]))>30000:raise ValueError('Слишком много элементов OCR.')
    for raw in page.get('lines',[]):
        text=str(raw.get('text','')).strip();box=raw.get('bbox');confidence=raw.get('confidence',0)
        if not text:continue
        if not isinstance(box,(tuple,list)) or len(box)!=4:raise ValueError('Отсутствуют координаты слова OCR.')
        try:x0,top,x1,bottom=map(float,box);confidence=float(confidence)
        except (ValueError,TypeError):raise ValueError('Некорректная геометрия или оценка OCR.')
        if not all(math.isfinite(v) for v in [x0,top,x1,bottom,confidence]) or not (0<=x0<x1<=width+1 and 0<=top<bottom<=height+1) or not 0<=confidence<=1:raise ValueError('Недостоверные координаты или оценка OCR.')
        word=dict(text=text,x0=x0,top=top,x1=x1,bottom=bottom,provenance='ocr',confidence=confidence)
        # TSV line numbers restart in every paragraph/block. Preserve the full
        # identity: glyph ascenders/descenders make top coordinates differ even
        # inside one correctly detected line (especially Cyrillic lower case).
        identity=tuple(raw.get(key) for key in ('block','paragraph','line'))
        if all(isinstance(value,int) and not isinstance(value,bool) and value>0 for value in identity):
            word['_ocr_line']=identity
        output.append(word)
    return output


def _reading_lines(words):
    """Use recognizer line identity, then a scale-independent geometry fallback.

    Comparing only word tops splits 'сухого' from 'Перед' because their ink has
    different heights. Overlapping vertical ink bands describe a line better;
    checking a bounded number of recent bands keeps dense inputs inexpensive.
    Separate recognizer blocks/paragraphs are never combined into one line.
    OCR still requires original-source review, including multi-column layouts.
    """
    explicit={};fallback=[]
    for word in words:
        identity=word.get('_ocr_line')
        if identity is None:fallback.append(word)
        else:explicit.setdefault(identity,[]).append(word)
    bands=[]
    for word in sorted(fallback,key=lambda w:(w['top'],w['x0'])):
        height=word['bottom']-word['top'];best=None;best_overlap=0.
        for band in bands[-8:]:
            # Running ink bounds do not depend on one unusually short glyph.
            top=band['top']/band['count'];bottom=band['bottom']/band['count']
            overlap=min(bottom,word['bottom'])-max(top,word['top'])
            minimum=min(height,bottom-top)
            center_distance=abs((top+bottom-word['top']-word['bottom'])/2)
            if minimum>0 and overlap>=minimum*.5 and center_distance<=max(height,bottom-top)*.6:
                relative=overlap/minimum
                if relative>best_overlap:best=band;best_overlap=relative
        if best is None:bands.append(dict(words=[word],top=word['top'],bottom=word['bottom'],count=1))
        else:
            best['words'].append(word);best['top']+=word['top'];best['bottom']+=word['bottom'];best['count']+=1
    groups=[band['words'] for band in bands]
    groups.extend(explicit.values())
    return sorted(groups,key=lambda group:(min(word['top'] for word in group),min(word['x0'] for word in group)))


def _consume_page(document,words,number,width,height,index,section,coordinate_system='pdf_points'):
    groups=_reading_lines(words);previews=[]
    buffer=[];box=None;last_bottom=0;buffer_words=[]
    def flush():
        nonlocal index,box
        if not buffer:return
        index+=1;ocr=[word for word in buffer_words if word.get('provenance')=='ocr'];native=[word for word in buffer_words if word.get('provenance')!='ocr']
        provenance='pdf_text+ocr' if ocr and native else 'ocr' if ocr else 'pdf_text'
        details=[dict(text=word['text'],confidence=round(word['confidence'],4),bbox=[round(word[k],2) for k in ['x0','top','x1','bottom']],numeric=bool(re.search(r'\d',word['text']))) for word in ocr]
        confidence=round(min((word['confidence'] for word in ocr),default=1.),4) if ocr else None
        _append(document,'\n'.join(buffer),section,index,number,box,provenance,confidence,details,width,height,coordinate_system)
        buffer.clear();buffer_words.clear();box=None
    for group in groups:
        group.sort(key=lambda word:word['x0']);text=' '.join(word['text'] for word in group)
        top=min(word['top'] for word in group);bottom=max(word['bottom'] for word in group)
        numbered=bool(re.match(r'^\d+(?:\.\d+)*[.)]?\s+',text))
        heading=bool(re.match(r'^\d+(?:\.\d+)*\.?\s+[А-ЯA-Z]',text)) and len(text)<140
        if numbered or (buffer and top-last_bottom>max(8,(bottom-top)*.9)):flush()
        if heading:section=text
        buffer.append(text);previews.append(text);buffer_words.extend(group)
        rect=[min(word['x0'] for word in group),top,max(word['x1'] for word in group),bottom]
        box=rect if box is None else [min(box[0],rect[0]),min(box[1],rect[1]),max(box[2],rect[2]),max(box[3],rect[3])]
        last_bottom=bottom
    flush();return index,section,previews


def _pdf(document,content,ocr_extractor=None,item=None):
    import pdfplumber
    pages=[];needed=[];previews=[];index=0;section=''
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        if len(pdf.pages)>MAX_PAGES:raise ValueError('Не более 150 страниц PDF')
        for number,page in enumerate(pdf.pages,1):
            words=page.extract_words(x_tolerance=2,y_tolerance=3) or []
            if len(words)>30000:raise ValueError('Слишком много элементов на странице PDF')
            needs=_page_needs_ocr(page,words)
            if needs:needed.append(number)
            pages.append(dict(page=number,width=float(page.width),height=float(page.height),words=[dict(word,provenance='pdf_text',confidence=1.) for word in words],needs_ocr=needs))
    raw_pages={};engine=None
    if needed and callable(ocr_extractor):
        if len(needed)>20:document['warnings'].append('Нужно распознать более 20 страниц: OCR не запущен; разделите документ.')
        else:
            try:
                raw=ocr_extractor(item,page_numbers=needed)
                if not isinstance(raw,dict) or not isinstance(raw.get('pages'),list):raise ValueError('Некорректный ответ OCR callback.')
                engine=str(raw.get('engine') or 'local OCR')
                for page in raw['pages']:
                    number=page.get('page')
                    if number not in needed or number in raw_pages:raise ValueError('OCR вернул неверный или повторный номер страницы.')
                    raw_pages[number]=page
            except Exception as error:
                raw_pages={};document['warnings'].append('Локальный OCR недоступен или не завершён: '+str(error)[:220])
    text_pages=[];ocr_pages=[];unreadable=[]
    for page in pages:
        number=page['page'];native=page['words'];words=list(native)
        if native:text_pages.append(number)
        recognized=[]
        if number in raw_pages:
            try:
                raw_page=raw_pages[number]
                if abs(float(raw_page.get('width',0))-page['width'])>1 or abs(float(raw_page.get('height',0))-page['height'])>1:raise ValueError('OCR изменил размеры исходной PDF-страницы.')
                recognized=_ocr_words(raw_page)
                if len(recognized)*len(native)>2_000_000:raise ValueError('Слишком плотная смешанная геометрия; требуется ручная проверка страницы.')
                # Keep precise native text once. OCR may also see the same header.
                recognized=[word for word in recognized if not any(n['x0']-2<=(word['x0']+word['x1'])/2<=n['x1']+2 and n['top']-2<=(word['top']+word['bottom'])/2<=n['bottom']+2 for n in native)]
                if recognized:words.extend(recognized);ocr_pages.append(number)
            except Exception as error:document['warnings'].append(f'Страница {number}: результат OCR отклонён: '+str(error)[:180])
        if page['needs_ocr'] and not recognized:
            unreadable.append(number);document['warnings'].append(f'Страница {number} не подтверждена текстовым слоем/OCR; требуется ручная проверка оригинала.')
        index,section,part=_consume_page(document,words,number,page['width'],page['height'],index,section)
        previews.extend(part)
    if ocr_pages:document['warnings'].append('OCR выполнен локально для страниц '+', '.join(map(str,ocr_pages))+'. Все распознанные слова, числа и условия требуют подтверждения по оригиналу.')
    method='pdf_text+ocr' if text_pages and ocr_pages else 'ocr' if ocr_pages else 'pdf_text'
    document['extraction']=dict(method=method,text_pages=text_pages,ocr_pages=ocr_pages,unreadable_pages=unreadable,engine=engine,requires_review=bool(ocr_pages or unreadable))
    document['metadata']['extraction_provenance']=method
    document['text_preview']='\n'.join(previews)[:16000]


def _image(document,item,ocr_extractor=None):
    if not callable(ocr_extractor):raise ValueError('Изображение требует локального OCR; распознавание не подключено.')
    try:
        raw=ocr_extractor(item,page_numbers=[1])
        if not isinstance(raw,dict) or not isinstance(raw.get('pages'),list) or len(raw['pages'])!=1 or raw['pages'][0].get('page')!=1:raise ValueError('Изображение должно возвращать одну страницу OCR.')
        page=raw['pages'][0];words=_ocr_words(page)
        _,_,previews=_consume_page(document,words,1,float(page['width']),float(page['height']),0,'','image_pixels')
        document['text_preview']='\n'.join(previews)[:16000]
        document['extraction']=dict(method='ocr',text_pages=[],ocr_pages=[1] if words else [],unreadable_pages=[] if words else [1],engine=raw.get('engine'),requires_review=True)
        document['metadata']['extraction_provenance']='ocr'
        document['warnings'].append('Изображение распознано локальным OCR. Все слова, числа и условия требуют подтверждения по оригиналу.')
    except Exception as error:
        document['warnings'].append('Изображение не распознано: '+str(error)[:220]);document['extraction']=dict(method='ocr_unavailable',text_pages=[],ocr_pages=[],unreadable_pages=[1],requires_review=True)

def ingest(item,ocr_extractor=None):
    if isinstance(item,dict):item=DocumentInput(**item)
    if not isinstance(item,DocumentInput) or not isinstance(item.content,bytes):raise TypeError('Ожидается DocumentInput с bytes content')
    if not isinstance(item.metadata,(dict,type(None))):raise ValueError('Metadata должна быть объектом')
    name=Path(str(item.name).replace('\\','/')).name.replace('\x00','')[:200] or 'document'
    document=dict(id=str(item.id),name=name,sha256=hashlib.sha256(item.content).hexdigest(),metadata=dict(item.metadata or {}),status='parsed',clauses=[],warnings=[],text_preview='')
    try:
        if len(item.content)>MAX_BYTES:raise ValueError('Файл превышает 20 МБ')
        if not item.content:raise ValueError('Пустой документ')
        ext=Path(name).suffix.lower()
        if ext in {'.txt','.md'}:_text(document,decode(item.content))
        elif ext=='.pdf':_pdf(document,item.content,ocr_extractor,item)
        elif ext=='.docx':_docx(document,item.content)
        elif ext=='.xlsx':_xlsx(document,item.content)
        elif ext in {'.png','.jpg','.jpeg'}:_image(document,item,ocr_extractor)
        else:raise ValueError('Поддерживаются TXT, MD, PDF, DOCX и XLSX')
        from .document_metadata import infer_metadata_from_explicit_text
        trusted_text='\n\n'.join(c['text'] for c in document['clauses'] if not c.get('unsafe') and not c.get('requires_review'))
        meta,proof=infer_metadata_from_explicit_text(trusted_text,document['metadata'])
        proof['source_representation']='trusted_extracted_clauses_joined_with_two_newlines'
        proof['source_sha256']=hashlib.sha256(trusted_text.encode()).hexdigest()
        document['metadata']=meta;document['metadata_provenance']=proof
        if proof.get('ambiguities'):document['warnings'].append('В тексте неоднозначно заданы метаданные документа; применимость требует уточнения.')
        meta.setdefault('title',name)
        for field in ['effective_from','effective_to']:
            if meta.get(field):meta[field]=iso_date(meta[field])
        if meta.get('effective_from') and meta.get('effective_to') and meta['effective_from']>meta['effective_to']:raise ValueError('Дата начала действия позже даты окончания')
        for field in ['roles','supersedes']:meta[field]=list_value(meta.get(field))
        if not document['clauses']:document['warnings'].append('Не извлечено ни одного пункта; ответ по документу невозможен.')
        if document['warnings']:document['status']='review'
    except Exception as error:
        document['status']='unreadable';document['clauses']=[];document['warnings'].append('Документ не прочитан безопасно: '+str(error)[:250])
    document['clause_count']=len(document['clauses'])
    return document
