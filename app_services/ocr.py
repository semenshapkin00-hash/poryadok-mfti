"""Bounded local Tesseract.js OCR with pinned weights and no runtime downloads.

The service returns positioned words only. Regulation parsing and review flags
are applied independently by navigator_engine.parsing; no text is corrected.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib,io,json,math,os,shutil,subprocess,tempfile,threading,time

ROOT=Path(__file__).resolve().parent.parent
LANG=ROOT/'models/tessdata'
BUNDLED=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
MAX_PAGES=20;TIMEOUT=90;DISK_RESERVE=5*1024**3;MAX_INPUT=20*1024**2
LOCK=threading.Lock()
EXPECTED={'rus.traineddata':'e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225','eng.traineddata':'7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2'}


def runtime():
    node=os.environ.get('NAVIGATOR_NODE_BINARY') or os.environ.get('AUDIT_NODE_BINARY') or shutil.which('node') or str(BUNDLED/'bin/node')
    modules=Path(os.environ.get('NAVIGATOR_NODE_MODULES') or os.environ.get('AUDIT_NODE_MODULES') or ROOT/'node_modules')
    if not (modules/'tesseract.js').is_dir():modules=BUNDLED/'node_modules'
    return str(node),modules


def _fingerprints():
    paths=[LANG/'manifest.json',*[LANG/name for name in EXPECTED]]
    return tuple((str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns) for path in paths)


@lru_cache(maxsize=4)
def _verify_weights(fingerprints):
    manifest=json.loads((LANG/'manifest.json').read_text())
    entries={item['name']:item for item in manifest.get('files',[])}
    if set(entries)!=set(EXPECTED):raise ValueError('Неверный состав локальных OCR-весов.')
    for name,digest in EXPECTED.items():
        item=entries[name];path=LANG/name
        if item.get('sha256')!=digest or path.stat().st_size!=item.get('bytes'):raise ValueError('Manifest OCR-весов не соответствует закреплённой версии.')
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Нарушена целостность локальных OCR-весов: '+name)
    return True


def status():
    node,modules=runtime();reason=None
    try:
        if not Path(node).is_file() or not os.access(node,os.X_OK):raise ValueError('Исполняемый Node.js не установлен.')
        if not (modules/'tesseract.js').is_dir():raise ValueError('Локальный пакет tesseract.js не установлен.')
        package=json.loads((modules/'tesseract.js/package.json').read_text())
        if str(package.get('version','')).split('.')[0]!='7':raise ValueError('Требуется проверенная локальная версия tesseract.js 7.')
        if not (ROOT/'tools/tesseract_ocr.cjs').is_file():raise ValueError('Локальный OCR runner отсутствует.')
        import pypdfium2
        from PIL import Image
        _verify_weights(_fingerprints())
    except Exception as error:reason=str(error)[:220]
    return dict(available=reason is None,backend='Tesseract.js 7 · WASM · rus+eng',offline=True,requires_review=True,max_pages=MAX_PAGES,timeout_seconds=TIMEOUT,reason=reason)


def _space(path,estimated=0):
    if shutil.disk_usage(path).free-max(0,estimated)<DISK_RESERVE:raise ValueError('Для OCR требуется сохранить не менее 5 ГБ свободного диска.')


def _deadline(started):
    remaining=TIMEOUT-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('Превышено время локального OCR (90 секунд).')
    return remaining


def _validate_result(raw,images):
    if not isinstance(raw,dict) or not isinstance(raw.get('pages'),list):raise ValueError('Некорректный ответ локального OCR.')
    expected={image['page']:image for image in images};seen=set()
    if len(raw['pages'])!=len(images):raise ValueError('OCR обработал не все запрошенные страницы.')
    for page in raw['pages']:
        number=page.get('page')
        if number not in expected or number in seen:raise ValueError('Некорректный номер страницы OCR.')
        seen.add(number);image=expected[number]
        if len(page.get('lines',[]))>30000:raise ValueError('Слишком много распознанных слов на странице.')
        page['coordinate_system']=image['coordinate_system'];page['width']=image['width'];page['height']=image['height']
        for word in page.get('lines',[]):
            box=word.get('bbox');confidence=word.get('confidence')
            if not isinstance(box,list) or len(box)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in box):raise ValueError('Некорректные координаты OCR.')
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=image['width']+1 and 0<=y0<y1<=image['height']+1):raise ValueError('Координаты OCR выходят за страницу.')
            if not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Некорректная оценка OCR.')
    raw['offline']=True;raw['requires_review']=True
    return raw


def extract_pages(item,page_numbers=None):
    """OCR selected 1-based PDF pages, or one PNG/JPEG; never OCR text pages.

    Coordinates are PDF points for PDFs and oriented image pixels for images.
    This function does not decide which pages need OCR: the parser supplies the
    exact unreadable-page selection, allowing mixed documents without duplicates.
    """
    from navigator_engine.models import DocumentInput
    if not isinstance(item,DocumentInput) or not isinstance(item.content,bytes):raise TypeError('Ожидается DocumentInput.')
    if len(item.content)>MAX_INPUT:raise ValueError('Файл OCR превышает 20 МБ.')
    info=status()
    if not info['available']:raise RuntimeError(info['reason'] or 'Локальный OCR недоступен.')
    if not LOCK.acquire(blocking=False):raise RuntimeError('Локальный OCR уже занят; повторите после окончания текущей обработки.')
    started=time.monotonic()
    try:
        node,modules=runtime();cache=ROOT/'.cache';cache.mkdir(exist_ok=True);_space(cache,64*1024**2)
        with tempfile.TemporaryDirectory(prefix='ocr-',dir=cache) as folder:
            directory=Path(folder);images=[];suffix=Path(item.name).suffix.lower()
            if suffix=='.pdf':
                import pypdfium2 as pdfium
                document=pdfium.PdfDocument(item.content)
                try:
                    numbers=list(range(1,len(document)+1)) if page_numbers is None else list(page_numbers)
                    if len(numbers)>MAX_PAGES:raise ValueError('Для OCR разрешено не больше 20 страниц за одну проверку.')
                    if len(set(numbers))!=len(numbers) or any(not isinstance(n,int) or isinstance(n,bool) or n<1 or n>len(document) for n in numbers):raise ValueError('Некорректный список страниц OCR.')
                    for number in sorted(numbers):
                        _deadline(started);_space(directory,64*1024**2);page=document[number-1]
                        try:
                            width,height=page.get_size()
                            if width<=0 or height<=0 or not all(math.isfinite(x) for x in [width,height]):raise ValueError('Некорректный размер страницы PDF.')
                            scale=min(3.,2800/max(width,height));bitmap=page.render(scale=scale)
                            try:
                                target=directory/f'page-{number}.png';bitmap.to_pil().save(target)
                            finally:bitmap.close()
                            images.append(dict(path=str(target),page=number,width=width,height=height,scale=scale,coordinate_system='pdf_points'))
                        finally:page.close()
                finally:document.close()
            elif suffix in {'.png','.jpg','.jpeg'}:
                if page_numbers is not None and list(page_numbers)!=[1]:raise ValueError('Изображение содержит одну страницу.')
                from PIL import Image,ImageOps
                with Image.open(io.BytesIO(item.content)) as source:
                    if source.width*source.height>30_000_000:raise ValueError('Изображение превышает 30 миллионов пикселей.')
                    image=ImageOps.exif_transpose(source).convert('RGB');original_width,original_height=image.size;image.thumbnail((2800,2800));target=directory/'image.png';image.save(target)
                    images.append(dict(path=str(target),page=1,width=original_width,height=original_height,scale=1.,scale_x=image.width/original_width,scale_y=image.height/original_height,coordinate_system='image_pixels'))
            else:raise ValueError('OCR поддерживает PDF, PNG и JPEG.')
            if not images:return dict(engine='Tesseract.js 7 / WASM / rus+eng',pages=[],offline=True,requires_review=True)
            manifest=directory/'input.json';manifest.write_text(json.dumps(dict(images=images,langPath=str(LANG.resolve()),cachePath=str(directory))))
            env={**os.environ,'NAVIGATOR_TESSERACT_MODULE':str(modules/'tesseract.js')}
            try:process=subprocess.run([node,str(ROOT/'tools/tesseract_ocr.cjs'),str(manifest)],env=env,capture_output=True,timeout=_deadline(started),check=False)
            except subprocess.TimeoutExpired:raise TimeoutError('Превышено время локального OCR (90 секунд).') from None
            if process.returncode:raise RuntimeError('Локальный OCR не завершён: '+process.stderr.decode(errors='replace')[:200])
            if len(process.stdout)>10*1024**2:raise ValueError('Слишком большой результат OCR.')
            _space(directory)
            return _validate_result(json.loads(process.stdout),images)
    finally:LOCK.release()
