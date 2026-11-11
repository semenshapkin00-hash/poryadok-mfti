"""Bounded three-valued reasoning over explicit Russian source conditions.

This module selects source clauses; it never writes an instruction or a number.
Mention, availability, events, categorical attributes and measured quantities
are different facts.  An absent fact is UNKNOWN, never implicitly false.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from decimal import Decimal,InvalidOperation
from functools import lru_cache
from types import MappingProxyType
import re
from .text import normalize,morphology,lexical_key,lexical_vocabulary,tokens

TRUE,FALSE,UNKNOWN=True,False,None
_GENERIC_ACTORS={'заявитель','сотрудник','работник','пользователь','человек'}
_EXISTS={'быть','иметься','наличие','присутствовать','существовать'}
_ABSENT={'нет','отсутствовать','отсутствие','утратить','потерять','утерять'}
_RECEIVE={'получить','получать','выдать','выдавать','дать','давать','предоставить','предоставлять'}
_GIVE=_RECEIVE-{'получить','получать'}
_REFUSE={'отказать','отказывать'}
_CREATE={'создать','создавать','зарегистрировать','регистрировать'}
_PAY={'платить','оплатить','оплачивать','заплатить'}
_UNKNOWN=re.compile(r'не\s+зна\w*|не\s+уверен\w*|неизвест\w*|не\s+установлен\w*|нет\s+сведений')
_REQUEST=re.compile(r'\b(?:как|какой|какая|какие|какое|каким|какому|каков|кто|кому|куда|когда|сколько|что|чем|почему|нужно|надо|можно|могу|можем|достаточно|разрешен(?:о|а|ы)?|считается\s+ли|блокирует\s+ли)\b')
_WORD=re.compile(r'[a-zа-я]+')
_NUMBER=re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:(?:рабоч|календар)\w*\s+)?(кг|килограмм\w*|грамм\w*|г|мг|рубл\w*|₽|час\w*|дн\w*|день|дня|дней)\b')


@dataclass(frozen=True)
class Word:
    text:str
    lemma:str
    key:str
    pos:str|None
    tags:frozenset


@lru_cache(maxsize=8192)
def words(text):
    out=[]
    for raw in _WORD.findall(normalize(text)):
        lemma,pos,tags=morphology(raw)
        out.append(Word(raw,lemma,lexical_key(raw),pos,tags))
    return tuple(out)


def event_key(lemma):
    if lemma in _PAY:return '@pay'
    if lemma in _GIVE:return '@give'
    if lemma in {'получить','получать'}:return '@get'
    return lexical_key(lemma)


def _negative(items,index):
    # Negation belongs to the predicate, not every noun in a four-word window.
    return any(w.text in {'не','ни','нет','без'} or w.key.startswith('невозмож') for w in items[max(0,index-2):index])


def _unavailable(item):return item.lemma in _ABSENT or item.key=='утрат' or item.text=='нет'


def _failed_receive(items,index):
    return _negative(items,index) or any(w.text=='не' for w in items[index+1:index+3])


def _nouns(items,actors=False):
    return {w.key for w in items if w.pos=='NOUN' and (actors or 'anim' not in w.tags) and w.lemma not in {'наличие','отсутствие'}}


def _instruments(items):
    result=set()
    for index,item in enumerate(items):
        if item.pos!='NOUN' or 'ablt' not in item.tags or 'anim' in item.tags:continue
        # A typed monetary means slot is not a procedure's material object.
        # This equivalence is used only inside an explicitly stated pay event.
        result.add('@funds' if item.lemma in {'деньги','деньга','средство'} else item.key)
        for previous in reversed(items[max(0,index-3):index]):
            if previous.pos!='ADJF':break
            result.add(previous.key)
    return frozenset(result)


def _instrument_entities(items):
    return {w.key for w in items if w.pos=='NOUN' and 'ablt' in w.tags and 'anim' not in w.tags}


def _head_group(items):
    """Noun head plus genitive dependents, bounded by a preposition/predicate."""
    result=[];started=False
    for item in items:
        if started and (item.pos in {'PREP','VERB','INFN','PRTS','ADJS'} or item.text in {'но','а'}):break
        if item.pos=='NOUN' and 'anim' not in item.tags:
            result.append(item.key);started=True
    return tuple(dict.fromkeys(result))


def _unit(raw):
    raw=normalize(raw)
    if raw=='кг' or raw.startswith('килограмм'):return 'mass',Decimal(1000)
    if raw=='мг':return 'mass',Decimal('.001')
    if raw=='г' or raw.startswith('грамм'):return 'mass',Decimal(1)
    if raw.startswith('руб') or raw=='₽':return 'money',Decimal(1)
    if raw.startswith('час'):return 'hours',Decimal(1)
    return 'days',Decimal(1)


def quantities(text):
    result=[]
    for match in _NUMBER.finditer(normalize(text)):
        try:value=Decimal(match.group(1).replace(',','.'))
        except InvalidOperation:continue
        dimension,factor=_unit(match.group(2))
        if dimension in {'days','hours'}:
            if re.search(r'рабоч',match.group(0)):dimension+=':working'
            elif re.search(r'календар',match.group(0)):dimension+=':calendar'
        result.append((dimension,value*factor))
    return tuple(result)


def factual_fragments(question):
    """Keep asserted facts, including facts appended after the original question."""
    out=[]
    for sentence in re.split(r'[.!?;\n]+',normalize(question)):
        if not sentence.strip():continue
        # An explicitly unknown state contributes no positive or negative fact.
        if _UNKNOWN.search(sentence):continue
        request=_REQUEST.search(sentence)
        factual=sentence[:request.start()] if request else sentence
        out.extend(part.strip() for part in re.split(r'(?<!\d),(?!\d|\s*(?:и\s+)?ни\b)',factual) if part.strip())
    return tuple(out)


def request_text(question):
    parts=[]
    for sentence in re.split(r'[.!?;\n]+',normalize(question)):
        match=_REQUEST.search(sentence)
        if match:parts.append(sentence[match.start():])
    return ' '.join(parts) or normalize(question)


@dataclass(frozen=True)
class Event:
    predicate:str
    entities:frozenset
    actors:frozenset
    positive:bool
    instrument:frozenset=frozenset()
    attributes:frozenset=frozenset()


@dataclass(frozen=True)
class Interval:
    lower:Decimal|None=None
    upper:Decimal|None=None
    lower_inclusive:bool=True
    upper_inclusive:bool=True

    @classmethod
    def from_operator(cls,value,operator):
        if operator=='<':return cls(upper=value,upper_inclusive=False)
        if operator=='<=':return cls(upper=value)
        if operator=='>':return cls(lower=value,lower_inclusive=False)
        if operator=='>=':return cls(lower=value)
        return cls(value,value)

    def compare(self,operator,value):
        if operator in {'>','>='}:
            opposite=self.compare('<=' if operator=='>' else '<',value)
            return None if opposite is None else not opposite
        if operator=='=':
            if self.lower==self.upper==value:return True
            if self.lower is not None and self.lower>value or self.upper is not None and self.upper<value:return False
            return None
        if operator=='<=':
            if self.upper is not None and self.upper<=value:return True
            if self.lower is not None and (self.lower>value or self.lower==value and not self.lower_inclusive):return False
        elif operator=='<':
            if self.upper is not None and (self.upper<value or self.upper==value and not self.upper_inclusive):return True
            if self.lower is not None and self.lower>=value:return False
        return None


@dataclass(frozen=True)
class Facts:
    availability:dict=field(default_factory=dict)
    assertions:dict=field(default_factory=dict)
    events:list=field(default_factory=list)
    measures:dict=field(default_factory=dict)
    mentioned:frozenset=frozenset()
    unknown:bool=False

    def add(self,mapping,key,value):mapping.setdefault(key,set()).add(value)


@lru_cache(maxsize=128)
def question_facts(question):
    facts=Facts(mentioned=frozenset(lexical_vocabulary(question)),unknown=bool(_UNKNOWN.search(normalize(question))))
    for fragment in factual_fragments(question):
        items=words(fragment)
        numeric=_numeric_atom(fragment)
        for dimension,value in quantities(fragment):
            interval=Interval.from_operator(value,numeric.operator if numeric else '=')
            facts.measures.setdefault(dimension,set()).add(interval)
        for index,item in enumerate(items):
            if item.pos in {'NOUN','ADJF','ADJS','PRTF','PRTS'}:
                facts.add(facts.assertions,item.key,not _negative(items,index))
        all_entities=frozenset(_nouns(items))
        for index,item in enumerate(items):
            neg=_negative(items,index)
            if _unavailable(item):
                # 'нет X' looks forward; 'X утрачен/отсутствует' looks back.
                group=(_head_group(items[index+1:]) or _head_group(items[:index])) if item.text=='нет' or item.lemma=='отсутствие' else _head_group(items[:index]) or _head_group(items[index+1:])
                for key in group:facts.add(facts.availability,key,neg)
            if item.lemma in _RECEIVE:
                before=[w for w in items[:index] if w.pos=='NOUN' and 'anim' not in w.tags]
                after=_head_group(items[index+1:])
                passive=item.pos in {'PRTS','PRTF'}
                group=tuple(w.key for w in before) if passive and before else after or tuple(w.key for w in before[-1:])
                for key in group:facts.add(facts.availability,key,not _failed_receive(items,index))
                if not passive and item.lemma in {'получить','получать'}:
                    for key in _head_group(items[:index]):facts.add(facts.availability,key,not _failed_receive(items,index))
            if item.lemma in _CREATE:
                group=_head_group(items[:index]) if item.pos in {'PRTF','PRTS'} else _head_group(items[index+1:])
                for key in group:facts.add(facts.availability,key,not neg)
            if item.lemma in _EXISTS or item.text=='есть':
                group=_head_group(items[:index]) or _head_group(items[index+1:])
                for key in group:facts.add(facts.availability,key,not neg)
            if item.pos in {'VERB','INFN','PRTF','PRTS'}:
                actors=frozenset(w.key for w in items if w.pos=='NOUN' and 'anim' in w.tags and ('ablt' in w.tags if item.pos in {'PRTF','PRTS'} else 'nomn' in w.tags))
                attributes=frozenset(w.key for w in items if w.pos in {'ADJF','ADVB'} and not _negative(items,items.index(w)))
                facts.events.append(Event(event_key(item.lemma),all_entities-frozenset(_instrument_entities(items)),actors,not neg,_instruments(items),attributes))
    # Auditors share the cache, but cannot mutate a later engine decision.
    return Facts(availability=MappingProxyType({key:frozenset(value) for key,value in facts.availability.items()}),
        assertions=MappingProxyType({key:frozenset(value) for key,value in facts.assertions.items()}),
        events=tuple(facts.events),measures=MappingProxyType({key:frozenset(value) for key,value in facts.measures.items()}),
        mentioned=facts.mentioned,unknown=facts.unknown)


@dataclass(frozen=True)
class Atom:
    kind:str
    keys:tuple=()
    positive:bool=True
    predicate:str=''
    instrument:frozenset=frozenset()
    operator:str=''
    dimension:str=''
    number:Decimal=Decimal(0)
    actor_keys:tuple=()
    attributes:tuple=()

    def evaluate(self,facts):
        if self.kind=='numeric':
            values=facts.measures.get(self.dimension,set())
            if len(values)!=1:return UNKNOWN
            return next(iter(values)).compare(self.operator,self.number)
        if self.kind in {'available','assertion'}:
            table=facts.availability if self.kind=='available' else facts.assertions
            decisions=[]
            for key in self.keys:
                values=table.get(key,set())
                if self.kind=='assertion' and len(facts.availability.get(key,set()))==1:values=facts.availability[key]
                decisions.append(next(iter(values))==self.positive if len(values)==1 else UNKNOWN)
            return conjunction(decisions)
        if self.kind in {'event','actor_event','nominal_event'}:
            if self.positive and any(facts.availability.get(key)=={False} for key in self.keys):return FALSE
            matches=[]
            for event in facts.events:
                same_event=self.predicate==event.predicate or self.kind=='nominal_event' and _same_event_family(self.predicate,event.predicate)
                refusal=self.kind=='actor_event' and not self.positive and any(event.predicate==event_key(lemma) for lemma in _REFUSE)
                if not same_event and not refusal:continue
                if self.actor_keys and not set(self.actor_keys)<=set(event.actors):continue
                if self.attributes and not set(self.attributes)<=set(event.attributes):continue
                # No event with a different explicit object is borrowed.
                if self.keys and not set(self.keys)<=set(event.entities):
                    # An implicit argument may use a unique previously named
                    # source subject, while the instrument still matches.
                    implicit_refusal=refusal and not event.entities and set(self.keys)<=facts.mentioned and all(facts.availability.get(key)=={False} for key in self.keys)
                    if not implicit_refusal and (not self.instrument or not set(self.keys)<=facts.mentioned):continue
                if self.instrument:
                    if not event.instrument:continue
                    if not self.instrument<=event.instrument:matches.append(FALSE);continue
                matches.append(True if refusal else event.positive==self.positive)
            if not matches and self.kind=='event' and self.predicate in {event_key('пройти'),event_key('проходить')}:
                # A successful named check is a completed check, whereas mere
                # исправность or an unchecked mention does not prove success.
                for head in self.keys:
                    others=set(self.keys)-{head}
                    for event in facts.events:
                        if _same_event_family(head,event.predicate) and others<=set(event.entities) and lexical_key('успешно') in event.attributes and event.positive:
                            matches.append(self.positive)
            if not matches:return UNKNOWN
            return matches[0] if all(value==matches[0] for value in matches) else UNKNOWN
        return UNKNOWN


def conjunction(values):
    values=list(values)
    if any(value is FALSE for value in values):return FALSE
    return TRUE if values and all(value is TRUE for value in values) else UNKNOWN


def disjunction(values):
    values=list(values)
    if any(value is TRUE for value in values):return TRUE
    return FALSE if values and all(value is FALSE for value in values) else UNKNOWN


@dataclass(frozen=True)
class Condition:
    text:str
    groups:tuple=()  # OR of AND groups, preserving Boolean scope.
    kind:str='case'

    def evaluate(self,facts):
        return disjunction(conjunction(atom.evaluate(facts) for atom in group) for group in self.groups)

    @property
    def subject_keys(self):return frozenset(key for group in self.groups for atom in group for key in atom.keys)


def _numeric_atom(text):
    found=quantities(text)
    if len(found)!=1:return None
    dimension,value=found[0]
    for pattern,operator in [(r'не\s+(?:более|больше|дороже|выше|превыша\w*|превыс\w*)|до\s+\d','<='),(r'не\s+(?:менее|меньше|ниже)|от\s+\d','>='),(r'свыше|более|больше|дороже|выше|превыша\w*|превыс\w*','>'),(r'менее|меньше|дешевле|ниже','<'),(r'ровно|составля\w*|равн\w*','=')]:
        if re.search(pattern,text):return Atom('numeric',operator=operator,dimension=dimension,number=value)
    return None


def _atoms(text,nominal=False):
    numeric=_numeric_atom(text)
    if numeric:return (numeric,)
    items=words(text)
    absent=next((i for i,w in enumerate(items) if _unavailable(w)),None)
    if absent is not None:
        item=items[absent];group=_head_group(items[absent+1:]) if item.lemma=='отсутствие' or item.text=='нет' else _head_group(items[:absent]) or _head_group(items[absent+1:])
        return (Atom('available',group,_negative(items,absent)),) if group else ()
    receive=next((i for i,w in enumerate(items) if w.lemma in _RECEIVE),None)
    if receive is not None:
        before=_head_group(items[:receive]);after=_head_group(items[receive+1:])
        group=before if items[receive].pos in {'PRTF','PRTS'} and before else after or before
        atoms=[Atom('available',group,not _failed_receive(items,receive))] if group else []
        actors=tuple(w.key for w in items if w.pos=='NOUN' and 'anim' in w.tags and w.lemma not in _GENERIC_ACTORS)
        if actors and group:atoms.append(Atom('actor_event',group,not _failed_receive(items,receive),event_key(items[receive].lemma),actor_keys=actors))
        # Explicit descriptors (written approval, named amount...) stay binding.
        descriptors=tuple(w.key for w in items if w.pos=='ADJF' and not w.tags&{'Apro','Anum'})
        with_index=next((i for i,w in enumerate(items) if w.text=='с' and i>receive),None)
        if with_index is not None:descriptors+=tuple(w.key for w in items[with_index+1:] if w.pos=='NOUN' and 'anim' not in w.tags)
        if descriptors:atoms.append(Atom('assertion',descriptors))
        return tuple(atoms)
    predicates=[(i,w) for i,w in enumerate(items) if w.pos in {'VERB','INFN','PRTF','PRTS'}]
    if predicates:
        index,predicate=predicates[0];instrument=_instruments(items)
        entities=tuple(sorted(_nouns(items)-_instrument_entities(items)))
        actors=tuple(w.key for w in items if w.pos=='NOUN' and 'anim' in w.tags and w.lemma not in _GENERIC_ACTORS)
        return (Atom('event',entities,not _negative(items,index),event_key(predicate.lemma),instrument,actor_keys=actors),)
    if nominal:
        head=next((w for w in items if w.pos=='NOUN' and 'anim' not in w.tags),None)
        if head:
            arguments=tuple(sorted(_nouns(items)-{head.key}))
            attributes=tuple(w.key for w in items if w.pos=='ADJF' and not w.tags&{'Apro','Anum'})
            if arguments:return (Atom('nominal_event',arguments,predicate=head.key,attributes=attributes),)
    # A categorical condition ('заявитель штатный работник') is a signed
    # assertion, not availability of an object or the user's selected API role.
    required=tuple(w.key for w in items if w.pos in {'NOUN','ADJF','ADJS'} and w.lemma not in _GENERIC_ACTORS and w.lemma not in {'наличие','отсутствие'})
    if not required:return ()
    return tuple(Atom('assertion',(w.key,),not _negative(items,i)) for i,w in enumerate(items) if w.key in required)


@lru_cache(maxsize=8192)
def parse_condition(text):
    clean=re.sub(r'^\s*(?:(?:пункт|правило)\s*)?\d+(?:\.\d+)*[.)]?\s*','',normalize(text))
    match=re.match(r'(?:если|при условии(?: что)?|в случае(?: если)?)\s+(.+)',clean)
    kind='case'
    if match:
        parts=re.split(r'(?<!\d),(?!\d)',match.group(1));condition=parts[0]
        if len(parts)>2 and re.match(r'\s*(?:но|а|и)\b',parts[1]):condition+=', '+parts[1]
    else:
        match=re.match(r'при\s+(.+)',clean)
        if not match:return None
        items=words(match.group(1));end=None
        for index,item in enumerate(items):
            subject=index and item.tags&{'nomn','accs'} and not any(w.pos=='PREP' for w in items[max(0,index-2):index])
            if item.pos in {'VERB','INFN','PRTS','ADJS'} or subject and (item.pos=='NOUN' or 'Subx' in item.tags):end=index;break
        if end is None:return None
        if end and items[end-1].text=='не':end-=1
        spans=list(_WORD.finditer(match.group(1)))
        condition=match.group(1)[:spans[end].start()].strip() if end<len(spans) else match.group(1).strip()
        kind='nominal'
    if '(' in condition or ')' in condition:
        return Condition(condition,(),kind='unsupported_grouping')
    # The scope of absence already covers coordinated nouns; do not turn
    # 'отсутствии A и B' into an unrelated positive assertion about B.
    first=next(iter(words(condition)),None)
    if first and _unavailable(first) and not re.search(r'\bили\b',condition):groups=(_atoms(condition),)
    else:
        groups=[]
        for alternative in re.split(r'\s+или\s+',condition):
            atoms=[]
            for term in re.split(r'\s+(?:и|но)\s+',alternative):atoms.extend(_atoms(term,nominal=kind=='nominal'))
            groups.append(tuple(atoms))
        groups=tuple(groups)
    return Condition(condition,groups,kind) if condition else None


def is_restriction(text):return bool(re.search(r'запрещ|нельзя|не\s+допуска|не\s+разреш|не\s+применя',normalize(text)))


def validated_context_terms(question,source_items):
    """Return only source-validated typed context spans, never object aliases.

    A pay event with an entailed personal/corporate instrument may use the
    monetary heads деньги/средства. Unknown components, expenses and vehicles
    remain material constraints. Callers may keep these records for audit.
    """
    facts=question_facts(question);allowed=set();evidence=[]
    query_words=words(request_text(question))
    for index,item in enumerate(query_words):
        # A genitive event after 'срок' may differ from its source verb only
        # by nominalisation (передачи/передаётся). Require a long lexical stem
        # match to an actual source predicate; material noun phrases stay hard.
        if not index or item.pos!='NOUN' or 'gent' not in item.tags or query_words[index-1].lemma!='срок':continue
        for doc,clause in source_items:
            for predicate in words(clause['text']):
                if predicate.pos not in {'VERB','INFN'}:continue
                common=0
                for a,b in zip(item.key,predicate.key):
                    if a!=b:break
                    common+=1
                if common>=5 and common>=min(len(item.key),len(predicate.key))-1:
                    allowed.add(item.key);evidence.append(dict(kind='event_nominalization',query_span=item.text,clause_id=clause['id'],document_id=doc['id'],source_predicate=predicate.text,validated_terms=[item.key]))
    for doc,clause in source_items:
        condition=parse_condition(clause['text'])
        if not condition:continue
        for group in condition.groups:
            for atom in group:
                if atom.kind!='event' or atom.predicate!='@pay' or '@funds' not in atom.instrument or atom.evaluate(facts) is not TRUE:continue
                for fragment in factual_fragments(question):
                    items=words(fragment)
                    if not any(w.lemma in _PAY for w in items) or not atom.instrument<=_instruments(items):continue
                    span_terms={w.key for w in items if w.lemma in {'деньги','деньга','средство'} and 'ablt' in w.tags}
                    if span_terms:
                        allowed|=span_terms;evidence.append(dict(kind='payment_instrument',query_span=fragment,clause_id=clause['id'],document_id=doc['id'],condition=condition.text,validated_terms=sorted(span_terms)))
    return allowed,evidence


def administrative(text,section=''):
    value=normalize(text);heading=normalize(section)
    return bool(re.search(r'редакци\w*.{0,45}(?:вступа\w* в силу|заменя\w* редакц)|ответственн\w* за хранение подлинника редакц',value))


def procedural_spans(text):
    """Remove only wholly administrative units, preserving mixed duties.

    Every returned string is an exact contiguous substring of the input. If a
    sentence mixes a replacement notice with a real duty, retain that sentence
    rather than deleting or paraphrasing the duty.
    """
    units=[];start=0
    for match in re.finditer(r'[.!?;](?:\s+|$)|\n+',text):
        end=match.end();units.append((start,end,text[start:end]));start=end
    if start<len(text):units.append((start,len(text),text[start:]))
    if not any(administrative(unit) for _,_,unit in units):return [text]
    administrative_verbs={'вступать','заменять','назначить','назначать','хранить'}
    kept=[]
    for left,right,unit in units:
        if not unit.strip():continue
        if re.fullmatch(r'\s*\d+(?:\.\d+)*[.)]?\s*',unit):continue
        mixed=any(w.pos in {'VERB','INFN'} and w.lemma not in administrative_verbs for w in words(unit))
        if administrative(unit) and not mixed:continue
        kept.append((left,right))
    ranges=[]
    for left,right in kept:
        if ranges and ranges[-1][1]==left:ranges[-1]=(ranges[-1][0],right)
        else:ranges.append((left,right))
    return [text[left:right].strip() for left,right in ranges if text[left:right].strip()]


def completed_creation(text,facts):
    items=words(text)
    predicate=next(((i,w) for i,w in enumerate(items) if w.pos in {'VERB','INFN','PRTF','PRTS'}),None)
    if not predicate or predicate[1].lemma not in _CREATE or _negative(items,predicate[0]):return False
    objects=_head_group(items[predicate[0]+1:])
    return bool(objects and facts.availability.get(objects[0])=={True})


def prerequisites(text):
    """Literal nominal prerequisites ('после подписи', 'после отметки')."""
    result=[]
    for match in re.finditer(r'\bпосле\s+([^.;]+)',normalize(text)):
        items=words(match.group(1));end=len(items)
        for index,item in enumerate(items):
            if item.pos in {'VERB','INFN','PRTS','ADJS'} or item.pos=='NOUN' and 'anim' in item.tags and 'nomn' in item.tags:
                end=index
                while end and items[end-1].pos=='ADJF':end-=1
                break
        condition=items[:end]
        head=next((w for w in condition if w.pos=='NOUN' and 'anim' not in w.tags),None)
        if head:result.append((head.key,frozenset(_nouns(condition,actors=True)-{head.key})))
    return tuple(result)


def _same_event_family(noun,verb):
    if min(len(noun),len(verb))<4:return False
    if noun.startswith(verb) or verb.startswith(noun):return True
    # Nominal/verb palatalisation in source pairs: отметка/отмечает, подпись/подписывает.
    folded=lambda value:value.replace('ч','т').replace('щ','ст')
    a,b=folded(noun),folded(verb)
    return a[:4]==b[:4]


def prerequisite_producers(clause,source_clauses):
    """Only uniquely supported source producers, never fabricated plan edges."""
    result=[];consumer_entities=_nouns(words(clause['text']),actors=True)
    for head,arguments in prerequisites(clause['text']):
        matches=[]
        for candidate in source_clauses:
            if candidate['id']==clause['id']:continue
            candidate_condition=parse_condition(candidate['text'])
            body=normalize(candidate['text']).replace(candidate_condition.text,'') if candidate_condition else candidate['text']
            candidate_words=words(body)
            verbs={w.key for w in candidate_words if w.pos in {'VERB','INFN'} and not _negative(candidate_words,candidate_words.index(w))}
            first_verb=next((i for i,w in enumerate(candidate_words) if w.pos in {'VERB','INFN'}),len(candidate_words))
            nominal_result=any(w.key==head and i>first_verb for i,w in enumerate(candidate_words))
            if not any(_same_event_family(head,verb) for verb in verbs) and not nominal_result:continue
            entities=_nouns(candidate_words,actors=True)
            if arguments and not arguments&entities:continue
            if not (entities&consumer_entities)-{head}:continue
            matches.append(candidate)
        if len(matches)==1:result.append(matches[0])
    # A rule about calendar/working time can refer to a numerical limit stated
    # in another source clause. Keep its exact value and unit beside the rule.
    clean=normalize(clause['text'])
    if not quantities(clean) and re.search(r'календар\w*|рабоч\w*',clean) and re.search(r'срок|выходн',clean):
        qualifiers={w.key for w in words(clean) if w.lemma.startswith(('календар','рабоч'))}
        for candidate in source_clauses:
            if candidate['id']==clause['id'] or not quantities(candidate['text']):continue
            candidate_words=words(candidate['text'])
            if qualifiers&{w.key for w in candidate_words} and len(consumer_entities&_nouns(candidate_words))>=1:result.append(candidate)
    return list({candidate['id']:candidate for candidate in result}.values())


def _content(text):
    return {item.key for item in words(text) if item.pos in {'NOUN','VERB','INFN','PRTF','PRTS','ADJF','ADJS'} and item.lemma not in {'действие','первый','правило','порядок','условие','документ','срок'}}


def _overlap(left,right):
    return bool(left&right) or any(min(len(a),len(b))>=4 and (a.startswith(b) or b.startswith(a)) for a in left for b in right)


def select_candidates(ranked,all_items,question,parallel_exclusions=()):
    """Select confirmed cases, retain exact constraints and expose missing facts.

    The distinction between explaining a whole conditional policy and deciding
    an individual case is explicit. Unmentioned independent alternatives are
    not mistaken for contradictory facts or mandatory extra requirements.
    """
    facts=question_facts(question);request=request_text(question);request_keys=_content(request)
    parallel_exclusions=frozenset(parallel_exclusions);seed_ids={item[1]['id'] for item in ranked}
    factual=bool(facts.availability or facts.events or facts.assertions or facts.measures or facts.unknown)
    explicit_whole=bool(re.search(r'весь|полный|целиком|все\s+этап|общий\s+порядок',request))
    decision_request=bool(re.search(r'можно|могу|разреш|однознач|единствен|достаточно|нужн|требу|какие.{0,20}(?:основан|документ)',request))
    permission=bool(re.search(r'можно|могу|разреш|однознач|единствен|достаточно|нужно\s+ли',request))
    whole=explicit_whole or not factual and not permission
    conditions={item[1]['id']:parse_condition(item[1]['text']) for item in all_items}
    states={identifier:condition.evaluate(facts) if condition else UNKNOWN for identifier,condition in conditions.items()}
    selected_docs={item[0]['id'] for item in ranked}
    by_doc={}
    by_id={item[1]['id']:item for item in all_items}
    for doc,clause,score in all_items:by_doc.setdefault(doc['id'],[]).append(clause)
    # An explicit target action can retrieve its prerequisite producers even
    # when a lexical score favours the earlier check or the later prohibition.
    goal_verbs={w.key for w in words(request) if w.pos in {'VERB','INFN'} and w.lemma not in {'делать','действовать','быть','мочь'}}
    goal_names={w.key for w in words(request) if w.pos=='NOUN'} if re.search(r'дальше|следующ',request) else set()
    finish_request=bool(re.search(r'заверш|законч|довест|дальше|следующ',request))
    augmented=list(ranked)
    for item in all_items:
        if item[1]['id'] in parallel_exclusions:continue
        if item[0]['id'] not in selected_docs or is_restriction(item[1]['text']):continue
        clause_verbs={w.key for w in words(item[1]['text']) if w.pos in {'VERB','INFN'}}
        terminal_gate=finish_request and 'только после' in normalize(item[1]['text']) and len(_nouns(words(item[1]['text']))&facts.mentioned)>=2
        if goal_verbs&clause_verbs or goal_names&clause_verbs or terminal_gate:
            if not any(entry[1]['id']==item[1]['id'] for entry in augmented):augmented.append(item)
    next_stage=bool(re.search(r'дальше|следующ\w*\s+(?:шаг|этап)|после этого',request)) and not whole
    completed={item[1]['id'] for item in all_items if next_stage and completed_creation(item[1]['text'],facts)}
    for identifier in completed:
        item=by_id[identifier];siblings=by_doc[item[0]['id']];index=next(i for i,c in enumerate(siblings) if c['id']==identifier)
        if index+1<len(siblings):
            following=by_id[siblings[index+1]['id']]
            if _nouns(words(following[1]['text']))&facts.mentioned and not is_restriction(following[1]['text']):
                if not any(entry[1]['id']==following[1]['id'] for entry in augmented):augmented.append(following)
    ranked=augmented
    dependency_ids=set();dependencies={}
    queue=[item[1]['id'] for item in ranked]
    while queue and len(dependencies)<48:
        identifier=queue.pop(0)
        if identifier in dependencies:continue
        item=by_id[identifier]
        producers=prerequisite_producers(item[1],by_doc[item[0]['id']])
        dependencies[identifier]=[clause['id'] for clause in producers]
        for producer in producers:
            dependency_ids.add(producer['id']);queue.append(producer['id'])
    confirmed_docs={item[0]['id'] for item in all_items if item[0]['id'] in selected_docs and states[item[1]['id']] is TRUE}
    blocked=set();hard_blocked=set();missing=[];unknown_restrictions=[];conditional_constraints=[];result=[]
    for item in all_items:
        identifier=item[1]['id'];condition=conditions[identifier]
        if identifier in parallel_exclusions and identifier not in seed_ids|dependency_ids:
            blocked.add(identifier);continue
        numeric_decision=bool(condition and decision_request and not explicit_whole and any(atom.kind=='numeric' for group in condition.groups for atom in group))
        if identifier in completed:blocked.add(identifier);hard_blocked.add(identifier)
        elif condition and states[identifier] is FALSE:blocked.add(identifier);hard_blocked.add(identifier)
        elif condition and states[identifier] is UNKNOWN and (not whole or numeric_decision):
            body=normalize(item[1]['text']).replace(condition.text,'')
            relevant=_overlap(_content(body),request_keys) or facts.unknown and bool(condition.subject_keys&facts.mentioned) or numeric_decision
            if is_restriction(item[1]['text']):
                # A conditional prohibition may be quoted as a constraint. It
                # never licenses a positive action under an unproved premise.
                if relevant:
                    conditional_constraints.append(identifier)
                    if permission:unknown_restrictions.append(identifier)
                else:blocked.add(identifier)
            elif identifier in dependency_ids:
                if permission:missing.append(identifier)
            elif relevant:missing.append(identifier)
            else:blocked.add(identifier)
    # False producer conditions cannot license their dependent target action.
    for _ in range(min(48,len(dependencies))):
        newly={identifier for identifier,producers in dependencies.items() if any(producer in hard_blocked for producer in producers)}-hard_blocked
        if not newly:break
        blocked|=newly;hard_blocked|=newly
    for item in ranked:
        if item[1]['id'] not in blocked:result.append(item)
    for item in all_items:
        identifier=item[1]['id']
        unconditional_base=not conditions[identifier] and bool(re.search(r'\bвсегда\b|в любом случае|независимо от суммы|при люб(?:ой|ом|ых)\s+\w+',normalize(item[1]['text'])))
        if item[0]['id'] in selected_docs and (states[identifier] is TRUE or identifier in missing or identifier in dependency_ids or identifier in conditional_constraints or unconditional_base) and identifier not in blocked and not any(entry[1]['id']==identifier for entry in result):result.append(item)
    for identifier in unknown_restrictions:
        restriction=next((item for item in all_items if item[1]['id']==identifier),None)
        if not restriction:continue
        condition=conditions[identifier]
        body=normalize(restriction[1]['text']).replace(condition.text,'')
        prohibited_verbs={w.key for w in words(body) if w.pos in {'VERB','INFN'}}
        positive=[item for item in result if item[0]['id']==restriction[0]['id'] and not is_restriction(item[1]['text']) and prohibited_verbs&{w.key for w in words(item[1]['text']) if w.pos in {'VERB','INFN'}}]
        if positive:missing.append(identifier)
    # A true exception dominates its scoped general rule. A true prohibition
    # dominates an allowance for the same action/object, rather than an earlier
    # independent check or an explicit exception to that prohibition.
    restrictions=[item for item in result if states[item[1]['id']] is TRUE and is_restriction(item[1]['text'])]
    suppressed=set()
    for restriction in restrictions:
        rid=restriction[1]['id'];condition=conditions[rid]
        body=normalize(restriction[1]['text']).replace(condition.text,'') if condition else normalize(restriction[1]['text'])
        body_keys=_content(body)
        body_verbs={w.key for w in words(body) if w.pos in {'VERB','INFN','PRTF','PRTS'}}
        exception=bool(re.search(r'не\s+применя',body))
        for item in result:
            identifier=item[1]['id']
            if identifier==rid or item[0]['id']!=restriction[0]['id'] or is_restriction(item[1]['text']):continue
            value=normalize(item[1]['text'])
            if re.search(r'исключение\s+к\s+запрет',value):continue
            keys=_content(value);verbs={w.key for w in words(value) if w.pos in {'VERB','INFN','PRTF','PRTS'}}
            same_section=item[1].get('section')==restriction[1].get('section')
            if exception and same_section and body_keys&keys:suppressed.add(identifier)
            elif same_section and (body_verbs&verbs or len(body_keys&keys)>=2 and len(body_keys&keys)/max(1,len(body_keys))>=.5):suppressed.add(identifier)
    blocked|=suppressed;hard_blocked|=suppressed
    result=[item for item in result if item[1]['id'] not in suppressed]
    missing=[identifier for identifier in dict.fromkeys(missing) if any(item[1]['id']==identifier for item in result)]
    return result,[item for item in all_items if item[1]['id'] not in hard_blocked],missing,blocked
