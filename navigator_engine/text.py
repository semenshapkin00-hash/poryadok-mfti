"""Small deterministic Russian normalizer and BM25; no remote dictionaries."""
import re
import unicodedata
from math import log
from collections import Counter
from functools import lru_cache

STOP=set('за через течение надо может сейчас свой свою будет есть кто куда когда сколько каков размер получить подать подачи подается подать ли'.split())|set('и в на по с со к у из о об от до для при а но или же ли не это как что какой какая какие каким нужно нужен нужны необходимо мне нам пожалуйста объясни расскажи подскажи оформить оформлять порядок сделать делать сотрудник работник инструкция регламент положение правила документ документы мой моя мое'.split())
ALIASES={
 'командировка':'командиров','командировки':'командиров','командировку':'командиров','командировке':'командиров','командировочный':'командиров','поездка':'командиров','поездки':'командиров','поездку':'командиров',
 'дистанционная':'удален','дистанционный':'удален','удаленная':'удален','удаленный':'удален','удаленка':'удален',
 'отпуска':'отпуск','отпуске':'отпуск','отпускной':'отпуск',
 'закупка':'закуп','закупку':'закуп','закупки':'закуп','приобрести':'закуп','покупка':'закуп','покупки':'закуп',
 'возместить':'возмещ','возмещение':'возмещ','компенсация':'возмещ','компенсируют':'возмещ',
 'дней':'день','дня':'день','дни':'день','день':'день','сроки':'срок','срока':'срок',
 'заявку':'заявк','заявка':'заявк','заявки':'заявк','заявление':'заявл','заявления':'заявл','заявлению':'заявл',
 'согласование':'соглас','согласования':'соглас','согласовать':'соглас','согласует':'соглас','согласовывает':'соглас','согласован':'соглас',
 'авансовый':'аванс','авансового':'аванс','авансовом':'аванс','отчет':'отчет','отчета':'отчет','отчету':'отчет','отчеты':'отчет',
 'чеки':'чек','чека':'чек','чеков':'чек','квитанции':'квитанц','квитанцию':'квитанц',
 'рабочих':'рабоч','рабочие':'рабоч','рабочий':'рабоч','рабочего':'рабоч','рабочая':'рабоч',
 'календарных':'календар','календарные':'календар','календарный':'календар',
 'пароли':'парол','пароля':'парол','пароль':'парол','vpn':'vpn',
}
SUFFIXES=sorted('иями ями ами иями ениями ованного ованные ование ение ание ения ания овать ировать ирует ирования ыми ими ых их ого его ему ому ая яя ое ее ые ие ый ий ой ить ать ять ет ют ут ем им ах ях ов ев ам ям ом ем ы и а я у ю е о ь'.split(),key=len,reverse=True)

def normalize(value):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(value or '')).lower().replace('ё','е')).strip()

@lru_cache(maxsize=1)
def _morphology():
    # The dictionary ships with the application; neither construction nor word
    # analysis downloads data or sends document text to an external service.
    import pymorphy3
    return pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=16384)
def morphology(word):
    """Stable, bounded local Russian lexical analysis, separate from retrieval.

    Keep the original token stemmer below for ranking compatibility.  Lemmas
    serve scope checks, where a finite verb must not become a missing entity.
    """
    word=normalize(word)
    if not re.fullmatch(r'[а-я]+',word):return word,None,frozenset()
    parsed=_morphology().parse(word)[0]
    return normalize(parsed.normal_form),parsed.tag.POS,frozenset(parsed.tag.grammemes)

@lru_cache(maxsize=8192)
def nominal_cases(word):
    """Possible cases of the chosen noun lemma, including syncretic forms.

    A dictionary's first analysis cannot distinguish ``результат`` as subject
    or object. Conversely, unambiguously nominative ``служба`` is not the
    direct object of the preceding verb merely because it follows it.
    """
    lemma,pos,_=morphology(word)
    if pos!='NOUN':return frozenset()
    return frozenset(parsed.tag.case for parsed in _morphology().parse(normalize(word))
                     if parsed.tag.POS=='NOUN' and normalize(parsed.normal_form)==lemma and parsed.tag.case)

def lexical_key(word):
    return stem(morphology(word)[0])

def lexical_vocabulary(text):
    return {lexical_key(word) for word in re.findall(r'[a-zа-я]+',normalize(text)) if len(word)>1}

# These are grammatical heads of requests and measured values, not policy
# subjects.  Domain nouns (parking, maternity, equipment, account...) do not
# appear here.  Adjectives inherit the materiality of their noun phrase head.
_FRAME_NOUNS=frozenset('действие шаг этап порядок процедура правило инструкция регламент документ пункт редакция версия условие требование основание случай способ возможность срок время дата период количество размер лимит значение сведение информация получение согласование рассмотрение подача оформление исполнение компенсация возмещение день сутки час минута секунда неделя месяц год'.split())

def query_subjects(question,focus_only=False):
    """Extract material noun phrases rather than vetoing every unknown word.

    Actor nouns, verbal predicates, ordinals, pronouns and request heads are
    grammatical context.  A material noun and its descriptive adjectives remain
    binding constraints.  Scope validation uses the entire discourse: a final
    question about cost or an application must retain the previously named item.
    The optional focus view is for presentation only, never a scope relaxation.
    """
    original=normalize(question);text=original
    if focus_only:
        sentences=[part.strip() for part in re.split(r'[.!?;]+\s*',text) if part.strip()]
        text=sentences[-1] if sentences else text
    words=re.findall(r'[a-zа-я]+',text);analyses=[morphology(word) for word in words]
    result=[]
    for index,(word,(lemma,pos,tags)) in enumerate(zip(words,analyses)):
        if word in STOP or lemma in _FRAME_NOUNS:continue
        material=pos=='NOUN' and 'anim' not in tags
        # A named Latin product or an out-of-dictionary noun remains a subject.
        if pos is None:material=len(word)>=3
        if pos=='ADJF' and not tags&{'Apro','Anum'}:
            for next_lemma,next_pos,next_tags in analyses[index+1:index+4]:
                if next_pos=='ADJF':continue
                material=next_pos=='NOUN' and 'anim' not in next_tags and next_lemma not in _FRAME_NOUNS
                break
        if material:result.append((word,stem(lemma)))
    # 'Хочу оплатить парковку. Как это оформить?' retains parking as the
    # referent.  A generic final question cannot erase a material constraint.
    if focus_only and not result and text!=original:return query_subjects(original)
    return list(dict.fromkeys(result))

def query_terms(question):
    """Content-bearing lemmas for coverage, without interrogative scaffolding.

    BM25 still sees the original query.  Only its coverage denominator changes:
    adding 'какое первое обязательное действие' must not dilute a valid subject
    below the evidence threshold.  Conditions in earlier sentences remain in
    the original query and are handled independently of retrieval coverage.
    """
    text=normalize(question)
    words=re.findall(r'[a-zа-я]+',text);analyses=[morphology(word) for word in words]
    out={key for word,key in query_subjects(text)}
    for word,(lemma,pos,tags) in zip(words,analyses):
        if word in STOP or lemma in _FRAME_NOUNS:continue
        if pos in {'VERB','INFN','PRTF','PRTS'}:out.add(stem(lemma))
    return out or set(tokens(question))

def stem(word):
    word=normalize(word)
    if word in ALIASES:return ALIASES[word]
    for root in ['подключ','регистр','одобр','подпис']:
        if word.startswith(root):return root
    for prefix,root in [('командиров','командиров'),('суточ','суточ'),('соглас','соглас'),('закуп','закуп'),('покуп','закуп'),('программ','программ'),('отпуск','отпуск'),('удален','удален'),('инициир','начат'),('нача','начат'),('потер','утрат'),('утрат','утрат'),('утрач','утрат'),('приклад','прилож'),('прилож','прилож'),('прилага','прилож'),('указ','указ'),('исполн','исполн'),('превыш','превыш'),('предлож','предлож'),('рабоч','рабоч'),('систем','систем'),('свед','свед'),('направ','направ'),('созд','созд'),('выда','выда'),('провер','провер'),('реш','реш')]:
        if word.startswith(prefix):return root
    # Russian fleeting е: продавец/продавца, истец/истца, огурец/огурца.
    if word.endswith('ец') and len(word)>=4:return word[:-2]+'ц'
    if re.fullmatch(r'[a-z0-9]+',word):return word
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word)-len(suffix)>=4:return word[:-len(suffix)]
    return word

def tokens(text,stop=True):
    words=re.findall(r'[a-zа-я0-9]+',normalize(text))
    return [stem(word) for word in words if len(word)>1 and (not stop or word not in STOP) and not word.isdecimal()]

class BM25:
    def __init__(self,texts):
        self.docs=[Counter(tokens(text)) for text in texts];self.lengths=[sum(doc.values()) for doc in self.docs]
        self.avg=sum(self.lengths)/max(1,len(self.docs));self.df=Counter()
        for doc in self.docs:self.df.update(doc.keys())
    def scores(self,query):
        terms=set(tokens(query));out=[];n=len(self.docs)
        for doc,length in zip(self.docs,self.lengths):
            score=0.
            for term in terms:
                if term not in doc:continue
                frequency=doc[term];idf=log(1+(n-self.df[term]+.5)/(self.df[term]+.5))
                score+=idf*frequency*2.5/(frequency+1.5*(.25+.75*length/max(1.,self.avg)))
            out.append(score)
        return out
