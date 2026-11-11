"""Local regulation navigator. Originals, corpus snapshots and answers stay on disk."""
from __future__ import annotations
import asyncio,copy,csv,hashlib,html,io,json,mimetypes,os,re,shutil,threading,time,uuid,zipfile
from datetime import date,datetime
from pathlib import Path
from typing import Annotated,Any
from zoneinfo import ZoneInfo
from fastapi import FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT=Path(__file__).resolve().parent
EXTRACTION_REVISION=hashlib.sha256(b''.join((ROOT/'navigator_engine'/name).read_bytes() for name in ['models.py','parsing.py','text.py','document_metadata.py'])+(ROOT/'app_services/xlsx.py').read_bytes()).hexdigest()
ENGINE_REVISION=hashlib.sha256(b''.join(path.name.encode()+path.read_bytes() for path in sorted((ROOT/'navigator_engine').glob('*.py')))).hexdigest()
DATA=Path(os.environ.get('NAVIGATOR_DATA_DIR',ROOT/'data')).resolve()
UPLOADS=DATA/'uploads';LIBRARY=DATA/'documents';ANSWERS=DATA/'answers'
for folder in [DATA,UPLOADS,LIBRARY,ANSWERS]:folder.mkdir(parents=True,exist_ok=True,mode=0o700)
IDENT=re.compile(r'^[a-f0-9]{32}$')
BASE_FORMATS={'.txt','.md','.pdf','.docx','.xlsx'}
IMAGE_FORMATS={'.png','.jpg','.jpeg'}
ALLOWED=BASE_FORMATS|IMAGE_FORMATS
MAX_BYTES=20*1024**2;MAX_TOTAL=40*1024**2;MAX_FILES=20;MAX_LIBRARY=50
MUTEX=threading.RLock();SLOTS=asyncio.Semaphore(1)
PUBLIC_DEMO=os.environ.get('NAVIGATOR_PUBLIC_DEMO','0')=='1'
ALLOWED_HOSTS=['localhost','127.0.0.1','testserver']
if PUBLIC_DEMO:
    for host in [os.environ.get('SPACE_HOST',''),*os.environ.get('NAVIGATOR_ALLOWED_HOSTS','').split(',')]:
        host=host.strip().lower()
        if not host:continue
        if len(host)>253 or not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?',host) or '..' in host:
            raise ValueError('Укажите точное имя хоста публичного демо, без wildcard, схемы или порта.')
        if host not in ALLOWED_HOSTS:ALLOWED_HOSTS.append(host)
DEMO_ANSWERS={}
app=FastAPI(title='Порядок · SPN',version='0.4.0',description='Локальный навигатор по регламентам: действия и проверяемые источники.')
app.add_middleware(TrustedHostMiddleware,allowed_hosts=ALLOWED_HOSTS)

@app.middleware('http')
async def local_headers(request:Request,call_next):
    origin=request.headers.get('origin')
    if request.method not in {'GET','HEAD','OPTIONS'} and origin:
        from urllib.parse import urlparse
        parsed=urlparse(origin)
        if parsed.hostname not in ALLOWED_HOSTS or parsed.netloc!=request.headers.get('host'):
            return JSONResponse({'detail':'Запрос разрешён только из локального интерфейса.'},status_code=403)
    response=await call_next(request)
    response.headers['Cache-Control']='no-store'
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; frame-src 'self' blob:; object-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'"
    return response

def today():return datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat()

def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8');temporary.replace(path)

def safe_name(value):
    name=str(value or 'document').replace('\\','/').split('/')[-1]
    return ''.join(char for char in name if ord(char)>=32 and char!='\x7f')[:180] or 'document'

def validate_file(name,content):
    suffix=Path(name).suffix.lower()
    if suffix not in ALLOWED:raise HTTPException(415,'Поддерживаются TXT, MD, PDF, DOCX, XLSX и изображения с локальным OCR.')
    if suffix in IMAGE_FORMATS:
        from app_services.ocr import status
        if not status().get('available'):raise HTTPException(422,'Локальное распознавание недоступно. Загрузите текстовый оригинал.')
    if not content:raise HTTPException(422,'Пустой документ: '+name)
    if len(content)>MAX_BYTES:raise HTTPException(413,'Документ превышает20МБ: '+name)
    if suffix=='.docx':
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members=archive.infolist()
                if len(members)>1500 or sum(item.file_size for item in members)>40*1024**2:raise HTTPException(413,'Слишком большой распакованный DOCX.')
                if any(item.flag_bits&1 for item in members):raise HTTPException(422,'Зашифрованный DOCX не поддерживается.')
        except zipfile.BadZipFile:raise HTTPException(422,'Повреждённый DOCX.')
    return suffix

META_KEYS={'title','doc_number','version','effective_from','effective_to','supersedes','roles','topic','category','synthetic','authority_level'}
def validate_metadata(raw):
    if not isinstance(raw,dict) or set(raw)-META_KEYS:raise HTTPException(422,'Неизвестные поля метаданных.')
    metadata=dict(raw)
    for key in ['effective_from','effective_to']:
        if metadata.get(key):
            try:date.fromisoformat(metadata[key])
            except (ValueError,TypeError):raise HTTPException(422,'Дата должна иметь формат ГГГГ-ММ-ДД.')
        elif key in metadata:metadata.pop(key)
    if metadata.get('effective_from') and metadata.get('effective_to') and metadata['effective_from']>metadata['effective_to']:raise HTTPException(422,'Начало действия позже окончания.')
    for key in ['supersedes','roles']:
        if key in metadata:
            if not isinstance(metadata[key],list) or len(metadata[key])>50 or any(not isinstance(value,str) or len(value)>120 for value in metadata[key]):raise HTTPException(422,'Поле '+key+' должно быть списком строк.')
    for key in ['title','doc_number','version','topic','category','authority_level']:
        if key in metadata and (not isinstance(metadata[key],str) or len(metadata[key])>300):raise HTTPException(422,'Некорректное поле '+key+'.')
    if 'synthetic' in metadata and not isinstance(metadata['synthetic'],bool):raise HTTPException(422,'synthetic должно быть true или false.')
    return metadata

def corpus():
    documents=[]
    for path in sorted(LIBRARY.glob('*.json')):
        try:documents.append(json.loads(path.read_text()))
        except (OSError,ValueError):continue
    if PUBLIC_DEMO and any(d.get('metadata',{}).get('synthetic') is not True for d in documents):
        raise HTTPException(503,'Для публичного демо нужна отдельная база вымышленных документов.')
    return documents

def persist_document(name,content,metadata,identifier=None):
    from navigator_engine import DocumentInput,ingest
    name=safe_name(name);suffix=validate_file(name,content);metadata=validate_metadata(metadata)
    digest=hashlib.sha256(content).hexdigest()
    fingerprint=hashlib.sha256((digest+json.dumps(metadata,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()
    with MUTEX:
        existing=None
        for document in corpus():
            if document.get('fingerprint')==fingerprint:
                existing=document
                if document.get('extraction_revision')==EXTRACTION_REVISION:return document
                identifier=document['id'];break
        if not existing and len(corpus())>=MAX_LIBRARY:raise HTTPException(413,'В локальной базе разрешено не более50документов.')
        if shutil.disk_usage(UPLOADS).free-len(content)<5*1024**3:raise HTTPException(507,'Нужно сохранить минимум5ГБ свободного диска.')
        identifier=identifier or uuid.uuid4().hex
        from app_services.ocr import status as ocr_status,extract_pages
        callback=extract_pages if suffix in {'.pdf'}|IMAGE_FORMATS and ocr_status().get('available') else None
        parsed=ingest(DocumentInput(identifier,name,content,metadata),ocr_extractor=callback)
        if not isinstance(parsed,dict):raise ValueError('Парсер вернул некорректный результат.')
        parsed.update(id=identifier,name=name,created_at=existing.get('created_at',time.time()) if existing else time.time(),updated_at=time.time(),suffix=suffix,sha256=digest,fingerprint=fingerprint,extraction_revision=EXTRACTION_REVISION,size=len(content),url='/api/files/'+identifier)
        (UPLOADS/(identifier+suffix)).write_bytes(content)
        atomic(LIBRARY/(identifier+'.json'),parsed)
        return parsed

def document_record(identifier):
    if not IDENT.fullmatch(identifier):raise HTTPException(404,'Документ не найден.')
    path=LIBRARY/(identifier+'.json')
    if not path.is_file():raise HTTPException(404,'Документ не найден.')
    record=json.loads(path.read_text())
    if PUBLIC_DEMO and record.get('metadata',{}).get('synthetic') is not True:
        raise HTTPException(404,'Документ не найден.')
    return record

def persist_batch(pending,metadata):
    """Capacity failures never leave a partially imported batch."""
    with MUTEX:
        before=corpus();previous={document['id'] for document in before}
        known={document.get('fingerprint') for document in before}
        fingerprints=set()
        for name,content in pending:
            digest=hashlib.sha256(content).hexdigest()
            fingerprints.add(hashlib.sha256((digest+json.dumps(metadata,sort_keys=True,ensure_ascii=False)).encode()).hexdigest())
        if len(previous)+len(fingerprints-known)>MAX_LIBRARY:raise HTTPException(413,'Комплект превышает лимит базы50документов.')
        if shutil.disk_usage(UPLOADS).free-sum(len(content) for name,content in pending)<5*1024**3:raise HTTPException(507,'Нужно сохранить минимум5ГБ свободного диска.')
        try:return [persist_document(name,content,metadata) for name,content in pending]
        except Exception:
            for document in corpus():
                if document['id'] not in previous:
                    (LIBRARY/(document['id']+'.json')).unlink(missing_ok=True)
                    (UPLOADS/(document['id']+document['suffix'])).unlink(missing_ok=True)
            raise

def stored_answer(identifier):
    if not IDENT.fullmatch(identifier):raise HTTPException(404,'Ответ не найден.')
    if PUBLIC_DEMO:
        with MUTEX:
            value=DEMO_ANSWERS.get(identifier)
            if value is None or time.time()-value['created_at']>1800:
                DEMO_ANSWERS.pop(identifier,None)
                raise HTTPException(404,'Временный ответ демо истёк. Задайте вопрос заново.')
            return copy.deepcopy(value)
    path=ANSWERS/(identifier+'.json')
    if not path.is_file():raise HTTPException(404,'Ответ не найден.')
    return json.loads(path.read_text())

class AskRequest(BaseModel):
    question:str=Field(min_length=3,max_length=1500)
    as_of:str|None=None
    role:str=Field(default='',max_length=100)
    parent_answer_id:str|None=Field(default=None,pattern=r'^[a-f0-9]{32}$')
    @field_validator('question')
    @classmethod
    def meaningful_question(cls,value):
        if len(value.strip())<3:raise ValueError('Введите вопрос.')
        return value.strip()
    @field_validator('as_of')
    @classmethod
    def valid_date(cls,value):
        if value is None or value=='':return today()
        date.fromisoformat(value);return value

@app.get('/')
def index():return FileResponse(ROOT/'frontend/index.html')

@app.get('/api/health')
def health():
    from app_services.semantic import status
    from app_services.ocr import status as ocr_status
    ocr=ocr_status()
    supported=BASE_FORMATS|IMAGE_FORMATS if ocr.get('available') else BASE_FORMATS
    return dict(status='ok',version=app.version,processing='demo_server' if PUBLIC_DEMO else 'local',features={'clarification_continue':True,'local_reasoner':False,'public_demo':PUBLIC_DEMO,'uploads_enabled':not PUBLIC_DEMO,'shared_history':False},models={'semantic':status(),'ocr':ocr},supported_formats=sorted(supported),max_file_mb=MAX_BYTES//1024**2,max_files=20,max_documents=MAX_LIBRARY,today=today(),official_data=False)

@app.get('/api/documents')
def documents():
    with MUTEX:return corpus()

@app.post('/api/documents')
async def add_documents(files:Annotated[list[UploadFile],File()],metadata:Annotated[str,Form()]='{}'):
    if PUBLIC_DEMO:raise HTTPException(403,'В веб-демо используется только вымышленная база. Для своих документов запустите локальную версию.')
    if not files or len(files)>MAX_FILES:raise HTTPException(413,'Загрузите от1до20документов.')
    try:meta=validate_metadata(json.loads(metadata))
    except (ValueError,TypeError):raise HTTPException(422,'Некорректный JSON метаданных.')
    total=0;pending=[]
    for upload in files:
        content=await upload.read(MAX_BYTES+1);total+=len(content);name=safe_name(upload.filename)
        if total>MAX_TOTAL:raise HTTPException(413,'Общий размер превышает40МБ.')
        validate_file(name,content);pending.append((name,content))
    async with SLOTS:
        try:
            records=await run_in_threadpool(persist_batch,pending,meta)
        except (ValueError,ArithmeticError) as error:raise HTTPException(422,str(error)[:400])
    return dict(documents=records,document_count=len(corpus()))

@app.post('/api/demo')
async def load_demo():
    from demo.fixtures import demo_manifest,demo_examples
    if PUBLIC_DEMO:return dict(documents=corpus(),examples=demo_examples(),synthetic=True)
    entries=demo_manifest()
    idmap={item['id']:uuid.uuid5(uuid.NAMESPACE_URL,'spn-poryadok-demo:'+item['id']).hex for item in entries}
    def seed():
        for item in entries:
            path=Path(item['path']);metadata=dict(item['metadata']);metadata['synthetic']=True
            if 'supersedes' in metadata:metadata['supersedes']=[idmap.get(value,value) for value in metadata['supersedes']]
            persist_document(item.get('name') or path.name,path.read_bytes(),metadata,idmap[item['id']])
        return dict(documents=corpus(),examples=demo_examples(),synthetic=True)
    async with SLOTS:
        try:return await run_in_threadpool(seed)
        except (ValueError,ArithmeticError) as error:raise HTTPException(422,str(error)[:400])

@app.get('/api/examples')
def examples():
    from demo.fixtures import demo_examples
    return demo_examples()

def verify_grounding(result,documents):
    """No citation or action leaves the API without an exact source span."""
    if result.get('status') not in {'answered','not_found','conflict','needs_clarification'}:raise ValueError('Неизвестный статус ответа.')
    normalize=lambda value:' '.join(str(value or '').split())
    clauses={str(c['id']):(d,c) for d in documents for c in d.get('clauses',[]) if 'id' in c}
    citations={}
    for citation in result.get('citations',[]):
        identity=str(citation.get('id',''));clause_id=str(citation.get('clause_id',''))
        if not identity or identity in citations or clause_id not in clauses:raise ValueError('Не удалось проверить происхождение цитаты.')
        document,clause=clauses[clause_id];quote=normalize(citation.get('quote'))
        if citation.get('document_id')!=document['id'] or not quote or quote not in normalize(clause.get('text')):raise ValueError('Цитата не совпадает с загруженным источником.')
        for key in ['provenance','ocr_confidence','requires_review','numeric_uncertainty','page_width','page_height','coordinate_system','sheet','row','cell_refs','spreadsheet_cells']:
            value=clause.get(key,clause.get('evidence',{}).get(key))
            if value is not None:citation[key]=value
        requires_review=clause.get('requires_review') or clause.get('provenance') in {'ocr','pdf_text+ocr'} or bool(document.get('extraction',{}).get('unreadable_pages')) or bool(document.get('metadata_provenance',{}).get('ambiguities'))
        if requires_review and result.get('status')=='answered':raise ValueError('Распознанный или неполный источник требует проверки; уверенный ответ запрещён.')
        citations[identity]=quote
    steps=result.get('steps',[]);facts=result.get('facts',[])
    if not isinstance(steps,list) or not isinstance(facts,list):raise ValueError('Некорректный список подтверждённых фрагментов.')
    for step in steps+facts:
        references=step.get('evidence_ids',[])
        if not references or any(ref not in citations for ref in references):raise ValueError('Действие не подтверждено источником.')
        if normalize(step.get('text')) not in ' '.join(citations[ref] for ref in references):raise ValueError('Текст действия не совпадает с процитированным пунктом.')
    if result.get('status')=='answered' and not (steps or facts):raise ValueError('Для ответа нет подтверждённых действий или фактов.')
    if result.get('status')=='answered':
        expected='\n\n'.join(str(index)+'. '+step['text'] for index,step in enumerate(steps+facts,1))
        if normalize(result.get('answer'))!=normalize(expected):raise ValueError('Текст ответа не подтверждён проверенными действиями.')
    return True

def execute_question(payload):
    from navigator_engine import answer
    from app_services.semantic import status,similarity_matrix
    with MUTEX:
        started=time.monotonic();documents=corpus();options={}
        snapshot=[dict(id=d['id'],sha256=d['sha256'],metadata=d.get('metadata',{}),extraction_revision=d.get('extraction_revision')) for d in documents]
        question=payload.question;as_of=payload.as_of or today();role=payload.role;parent=None
        if payload.parent_answer_id:
            from navigator_engine.core import role_key
            parent=stored_answer(payload.parent_answer_id)
            if parent.get('status')!='needs_clarification':raise HTTPException(409,'Этот ответ не требует уточнения. Задайте новый вопрос.')
            if parent.get('corpus_snapshot')!=snapshot:raise HTTPException(409,'Документы изменились после предыдущего ответа. Задайте вопрос заново по актуальной базе.')
            if 'as_of' not in payload.model_fields_set:as_of=parent['as_of']
            if as_of!=parent['as_of']:raise HTTPException(409,'Для другой даты задайте новый вопрос.')
            if 'role' not in payload.model_fields_set:role=parent.get('role') or ''
            if parent.get('role') and role_key(role)!=role_key(parent['role']):raise HTTPException(409,'Для другой роли задайте новый вопрос.')
            if int(parent.get('clarification_depth',0))>=4:raise HTTPException(409,'Сформулируйте новый вопрос с накопленными условиями.')
            question=parent['question']+'\n\n'+payload.question
            if len(question)>1800:raise HTTPException(422,'Вопрос с уточнениями слишком длинный. Сформулируйте его заново вместе с нужными условиями.')
        if status().get('available'):options['semantic_similarity']=similarity_matrix
        result=answer(question,copy.deepcopy(documents),as_of=as_of,role=role or None,options=options)
        if not isinstance(result,dict):raise ValueError('Некорректная схема ответа.')
        result['grounding_verified']=verify_grounding(result,documents)
        identifier=uuid.uuid4().hex
        result.update(id=identifier,created_at=time.time(),question=question,query=question,as_of=as_of,role=role,synthetic_data=bool(documents) and all(d.get('metadata',{}).get('synthetic') for d in documents))
        result['corpus_snapshot']=snapshot
        if parent:
            result.update(parent_answer_id=parent['id'],clarification_depth=int(parent.get('clarification_depth',0))+1,clarification_text=payload.question,initial_question=parent.get('initial_question') or parent['question'])
        result['implementation']=dict(api_version=app.version,engine_source_sha256=ENGINE_REVISION)
        result['metrics']=dict(processing_seconds=round(time.monotonic()-started,3),processing_location='demo_server' if PUBLIC_DEMO else 'local',external_requests=0,document_count=len(documents),semantic_available=status().get('available'))
        for citation in result.get('citations',[]):
            doc_id=citation.get('document_id')
            if doc_id and IDENT.fullmatch(str(doc_id)):citation['url']='/api/files/'+doc_id
        if PUBLIC_DEMO:
            for key in list(DEMO_ANSWERS):
                if time.time()-DEMO_ANSWERS[key]['created_at']>1800:DEMO_ANSWERS.pop(key,None)
            while len(DEMO_ANSWERS)>=128:DEMO_ANSWERS.pop(next(iter(DEMO_ANSWERS)))
            DEMO_ANSWERS[identifier]=copy.deepcopy(result)
        else:atomic(ANSWERS/(identifier+'.json'),result)
        return result

@app.post('/api/ask')
async def ask(payload:AskRequest):
    async with SLOTS:
        try:return await run_in_threadpool(execute_question,payload)
        except (ValueError,ArithmeticError) as error:raise HTTPException(422,str(error)[:400])

@app.get('/api/history')
def history():
    if PUBLIC_DEMO:return []
    result=[]
    for path in sorted(ANSWERS.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]:
        try:
            value=json.loads(path.read_text());result.append({key:value.get(key) for key in ['id','question','query','status','created_at','as_of','role','answer','score']})
        except (OSError,ValueError):continue
    return result

@app.get('/api/answers/{identifier}')
def get_answer(identifier:str):return stored_answer(identifier)

@app.get('/api/files/{identifier}')
def original(identifier:str):
    record=document_record(identifier);path=UPLOADS/(identifier+record['suffix'])
    if not path.is_file():raise HTTPException(404,'Оригинал не найден.')
    return FileResponse(path,media_type=mimetypes.guess_type(record['name'])[0] or 'application/octet-stream',filename=record['name'],content_disposition_type='inline')

def source_pdf(identifier):
    record=document_record(identifier)
    if record['suffix']!='.pdf':raise HTTPException(415,'Просмотр страниц доступен для PDF.')
    return UPLOADS/(identifier+record['suffix'])

def source_image(identifier):
    """Use the same EXIF orientation as OCR so source overlays stay aligned."""
    from PIL import Image,ImageOps
    record=document_record(identifier)
    if record['suffix'] not in IMAGE_FORMATS:raise HTTPException(415,'Источник не является изображением.')
    with Image.open(UPLOADS/(identifier+record['suffix'])) as original:
        if original.width*original.height>30_000_000:raise HTTPException(413,'Изображение слишком большое для просмотра.')
        return ImageOps.exif_transpose(original).convert('RGB')

@app.get('/api/files/{identifier}/info')
def pdf_info(identifier:str,page:int=1):
    import pypdfium2 as pdfium
    try:
        with MUTEX:
            record=document_record(identifier)
            if record['suffix'] in IMAGE_FORMATS:
                if page!=1:raise HTTPException(422,'Изображение содержит одну страницу.')
                with source_image(identifier) as image:
                    return dict(page_count=1,page=1,width=image.width,height=image.height,coordinate_system='image_pixels',bbox_convention='top-left oriented image pixels: x0,y0,x1,y1')
            with pdfium.PdfDocument(str(source_pdf(identifier))) as document:
                if not 1<=page<=len(document):raise HTTPException(422,'Страница вне диапазона.')
                selected=document[page-1]
                try:width,height=selected.get_size()
                finally:selected.close()
                return dict(page_count=len(document),page=page,width=width,height=height,coordinate_system='pdf_points',bbox_convention='top-left PDF points: x0,y0,x1,y1')
    except HTTPException:raise
    except Exception:raise HTTPException(422,'PDF не удалось открыть.')

@app.get('/api/files/{identifier}/preview')
def pdf_preview(identifier:str,page:int=1):
    import pypdfium2 as pdfium
    try:
        with MUTEX:
            record=document_record(identifier)
            if record['suffix'] in IMAGE_FORMATS:
                if page!=1:raise HTTPException(422,'Изображение содержит одну страницу.')
                with source_image(identifier) as image:
                    image.thumbnail((1800,1800));buffer=io.BytesIO();image.save(buffer,format='PNG')
                    return Response(buffer.getvalue(),media_type='image/png')
            with pdfium.PdfDocument(str(source_pdf(identifier))) as document:
                if not 1<=page<=min(len(document),150):raise HTTPException(422,'Страница вне диапазона.')
                selected=document[page-1]
                try:
                    width,height=selected.get_size();bitmap=selected.render(scale=min(1.8,1800/max(width,height)))
                    try:
                        buffer=io.BytesIO();bitmap.to_pil().save(buffer,format='PNG')
                    finally:bitmap.close()
                finally:selected.close()
                return Response(buffer.getvalue(),media_type='image/png')
    except HTTPException:raise
    except Exception:raise HTTPException(422,'PDF не удалось отобразить.')

@app.get('/api/answers/{identifier}/export')
def export(identifier:str,format:str='json'):
    result=stored_answer(identifier);name='poryadok-'+identifier[:8]
    if format=='json':return Response(json.dumps(result,ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':f'attachment; filename="{name}.json"'})
    if format!='html':raise HTTPException(422,'Выберите json или html.')
    esc=lambda v:html.escape(str(v or ''))
    status_label={'answered':'Найдены подтверждённые действия','not_found':'Нет подходящего основания','conflict':'Противоречие в правилах','needs_clarification':'Нужно уточнение'}.get(result.get('status'),'Требуется проверка')
    overview='' if result.get('steps') else '<p>'+esc(result.get('answer'))+'</p>'
    warnings=''.join('<p><strong>Обратите внимание.</strong> '+esc(message)+'</p>' for message in result.get('warnings',[]))
    steps=''.join('<li>'+esc(step.get('text'))+' <small>'+esc(', '.join(str(x) for x in step.get('evidence_ids',[])))+'</small></li>' for step in result.get('steps',[]))
    citations=''.join('<article><b>'+esc(c.get('id'))+' · '+esc(c.get('document_name'))+'</b><small> '+esc(c.get('section'))+' · '+esc(c.get('page') or c.get('paragraph'))+'</small><blockquote>'+esc(c.get('quote'))+'</blockquote></article>' for c in result.get('citations',[]))
    content='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Порядок · ответ с основаниями</title><style>body{font:16px system-ui;color:#17283a;max-width:1000px;margin:48px auto;padding:0 24px;line-height:1.6}h1{font-size:30px}small{color:#657385}li{margin:14px 0}article{border:1px solid #dbe2e8;padding:18px;margin:12px 0;break-inside:avoid}blockquote{white-space:pre-wrap;margin:12px 0}header{border-bottom:2px solid #17283a}footer{border-top:1px solid #dbe2e8;margin-top:32px;font-size:13px}@media print{body{margin:10mm}}</style><header><small>SPN / ПОРЯДОК · '+esc(result['id'])+'</small><h1>'+esc(result['question'])+'</h1><p>На дату '+esc(result['as_of'])+' · статус '+esc(status_label)+'</p></header>'+overview+warnings+'<ol>'+steps+'</ol><h2>Основания</h2>'+citations+'<footer>Ответ основан только на загруженных документах. Проверьте применимость регламента. '+('Использована синтетическая демонстрационная база. ' if result.get('synthetic_data') else '')+'Обработка локальная, внешние API не использовались.</footer></html>'
    return HTMLResponse(content,headers={'Content-Disposition':f'attachment; filename="{name}.html"'})

app.mount('/static',StaticFiles(directory=ROOT/'frontend'),name='static')
