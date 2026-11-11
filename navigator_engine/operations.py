"""Ground the requested operation separately from its objects and past facts.

Semantic proximity of whole sentences cannot prove that creating an object is
an instruction for deleting it. This module requires a source action, explicit
procedure context or a narrowly typed transfer/acquisition relation. It selects
source evidence only and never generates a factual answer.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import re
from .text import normalize,morphology,stem,nominal_cases
from .conditions import words,request_text,parse_condition,procedural_spans,question_facts


# Closed linguistic frames, not policy domains or a list of forbidden goals.
_FRAME_VERBS={'быть','есть','считаться','разрешаться','допускаться','мочь','следовать','хотеть','планировать','собираться','намереваться','нуждаться','требоваться','потребоваться','рассказать','объяснить','описать','показать','узнать','уточнить','отличаться','отличить','сравнить','сравнивать','различаться','оформить','оформлять','сделать','делать','действовать','выполнить','выполнять','осуществить','осуществлять','начать','начинать','закончить','заканчивать','завершить','завершать','инициировать','приступить'}
_FRAME_NOUNS={'действие','шаг','этап','порядок','процедура','инструкция','правило','сведения','сведение','информация','условие','основание','способ','возможность','вопрос','срок','время','дата','размер','количество','лимит'}
_TRANSFER={'передать','передавать','направить','направлять','отправить','отправлять','послать','посылать','выслать','высылать','сдать','сдавать','отдать','отдавать'}
_ACTIVATE={'включить','включать','запустить','запускать','активировать','запуск','включение','активация'}
_APPROVE={'согласовать','согласовывать','утвердить','утверждать','одобрить','одобрять'}
_RECORD={'указать','указывать','записать','записывать','написать','писать','внести','вносить','занести','заносить','вписать','вписывать'}
_TRANSPORT={'переместить','перемещать','перевезти','перевозить','перенести','переносить','доставить','доставлять'}
_COMMUNICATE={'сообщить','сообщать','уведомить','уведомлять','информировать'}
_GET={'получить','получать'}
_PROVIDE={'выдать','выдавать','предоставить','предоставлять','дать','давать'}
_CREATE={'создать','создавать','зарегистрировать','регистрировать'}
_PAY={'возместить','возмещать','компенсировать','выплатить','выплачивать'}
_REQUEST_ARTIFACTS={'заявка','заявление','обращение','запрос'}
_EVENT_NOUNS={'передача','отправка','пересылка','сдача','выдача','подача'}
_PERFORM={'выполнить','выполнять','провести','проводить','осуществить','осуществлять','делать','сделать'}
_MESSAGES={'уведомление','сообщение','письмо'}
_DATA_OBJECTS={'значение','запись','строка','поле','данные','номер','результат'}
_NORM_HEADS={'запрет','ограничение','правило','условие','норма','положение','регламент','требование'}
_INTENT=re.compile(r'\b(?:нужн\w*|необходим\w*|хочу|хотим|планиру\w*|собира\w*|намерен\w*|требуется)\b')
_WORD=re.compile(r'[a-zа-я]+')


@lru_cache(maxsize=8192)
def operation_forms(value):
    """Inflection/aspect/nominalisation forms without dropping verb prefixes.

    Prefixes distinguish, for example, установить/восстановить. A long common
    character prefix or an embedding similarity alone is not equivalence.
    """
    lemma,pos,tags=morphology(value);lemma=normalize(lemma)
    forms={lemma,stem(lemma)}
    bare=re.sub(r'(?:ся|сь)$','',lemma)
    if pos in {'VERB','INFN','PRTF','PRTS'}:
        base=bare[:-2] if bare.endswith('ть') else bare
        variants={base}
        if base.endswith(('ыва','ива','ева','ава','ява')):variants.add(base[:-2])
        if base.endswith('ва'):variants.add(base[:-2])
        for variant in variants:
            forms.add(variant)
            if variant.endswith(('а','я','и','е','ы','о')):forms.add(variant[:-1])
    elif pos=='NOUN':
        for suffix in ('ирование','яция','ование','ение','ание','ние','тие','ция','ача','яча'):
            if bare.endswith(suffix) and len(bare)-len(suffix)>=3:forms.add(bare[:-len(suffix)])
        if bare.endswith(('а','я','е','о')) and len(bare)>4:forms.add(bare[:-1])
    if bare in _TRANSFER:forms.add('@transfer')
    if bare in _ACTIVATE:forms.add('@activation')
    if bare in _APPROVE:forms.add('@approval')
    if bare in _RECORD:forms.add('@record')
    if bare in _TRANSPORT:forms.add('@transport')
    if bare in _COMMUNICATE:forms.add('@communicate')
    if bare in {'войти','входить','вход'}:forms.add('@ingress')
    # Productive смотр/сматр aspect alternation and three irregular aspect
    # stems. These are grammatical correspondences, not policy-specific goals.
    forms|={form.replace('смотр','сматр') for form in forms}
    irregular={'отозвать':'отзыв','пройти':'проход','войти':'вход'}
    if lemma in irregular:forms.add(irregular[lemma])
    if lemma.startswith('с') and lemma.endswith('ировать'):
        forms|=set(operation_forms(lemma[1:]))
    return frozenset(form for form in forms if len(form)>=3)


def equivalent(left,right):
    if left&right:return True
    # Imperfective -лять / perfective -ить (направлять/направить).
    return any(min(len(a),len(b))>=4 and (a==b+'л' or b==a+'л') for a in left for b in right)


@dataclass(frozen=True)
class RequestedOperation:
    text:str
    lemma:str
    mode:str  # perform, about, acquire
    forms:frozenset
    object_key:str=''
    object_lemma:str=''
    object_modifiers:frozenset=frozenset()


def _resolve_pronoun(pronoun,context):
    context=normalize(context);items=words(context);positions=list(_WORD.finditer(context))
    candidates=[word for word in items if word.pos=='NOUN' and 'anim' not in word.tags and word.lemma not in _FRAME_NOUNS]
    allowed={'его':{'masc','neut'},'него':{'masc','neut'},'ее':{'femn'},'нее':{'femn'}}.get(pronoun,set())
    if allowed:candidates=[word for word in candidates if word.tags&allowed]
    if pronoun=='их':candidates=[word for word in candidates if 'plur' in word.tags]
    # A completed event supplies discourse focus. A subsequent static contrast
    # ("шнур порвался, а корпус цел") does not silently change its participant.
    # This preference is deliberately limited to explicit perfective past
    # events, with a nominative/accusative participant in their local segment.
    for event_index in reversed(range(len(items))):
        event=items[event_index]
        if event.pos!='VERB' or not {'past','perf'}<=event.tags:continue
        start,end=positions[event_index].span()
        left=max(context.rfind('.',0,start),context.rfind(';',0,start),context.rfind(',',0,start))
        right=min([value for value in [context.find('.',end),context.find(';',end),context.find(',',end)] if value>=0] or [len(context)])
        local=[(abs(positions[index].start()-start),word) for index,word in enumerate(items) if word in candidates and left<positions[index].start()<right and nominal_cases(word.text)&{'nomn','accs'} and not (index and items[index-1].pos=='PREP')]
        if local:return min(local,key=lambda pair:pair[0])[1]
    return candidates[-1] if candidates else None


def _context_modifiers(argument,context):
    items=words(context);variants=[]
    for index,item in enumerate(items):
        if item.pos!='NOUN' or item.key!=argument.key:continue
        modifiers=[]
        for previous in reversed(items[:index]):
            if previous.pos not in {'ADJF','PRTF'}:break
            modifiers.append(previous.key)
        if modifiers:variants.append(frozenset(modifiers))
    unique=set(variants)
    return next(iter(unique)) if len(unique)==1 else frozenset()


def _argument(items,index,context='',fronted=False):
    for position,word in enumerate(items[index+1:],index+1):
        if word.pos=='PREP':break
        if word.text in {'его','ее','их','это'}:
            resolved=_resolve_pronoun(word.text,context)
            return (resolved,_context_modifiers(resolved,context)) if resolved else (None,frozenset())
        if word.pos=='NOUN' and 'anim' not in word.tags:
            cases=nominal_cases(word.text)
            # In a reflexive passive the nominative subject is the semantic
            # patient ("подаётся заявка"). In an active clause the nominative
            # following a verb is an actor, not its direct object.
            passive=items[index].lemma.endswith(('ся','сь'))
            if 'accs' not in cases and not (passive and 'nomn' in cases):
                if cases=={'nomn'}:break
                continue
            adjectives=[]
            for previous in reversed(items[index+1:position]):
                if previous.pos not in {'ADJF','PRTF'}:break
                adjectives.append(previous.key)
            return word,frozenset(adjectives) or _context_modifiers(word,context)
    if fronted:
        for position,word in enumerate(items[:index]):
            if word.pos=='NOUN' and 'anim' not in word.tags and word.tags&{'nomn','accs'}:
                modifiers=frozenset(previous.key for previous in items[max(0,position-2):position] if previous.pos in {'ADJF','PRTF'})
                return word,modifiers or _context_modifiers(word,context)
    return None,frozenset()


def _negated_goal(items,index):
    before=items[max(0,index-4):index]
    contrast=[i for i,word in enumerate(before) if word.text in {'а','но','зато'}]
    if contrast:before=before[contrast[-1]+1:]
    return any(word.text=='не' for word in before) and (index and items[index-1].text=='не' or any(word.lemma in {'нужный','нужно','надо','хотеть','необходимый','требоваться'} for word in before))


def excluded_task_without_goal(question,goals):
    return not goals and bool(re.search(r'\bне\s+(?:нужно|надо|хочу|необходимо|требуется)\b',normalize(question)))


def _nominal_goals(text,mode='about'):
    items=words(text);out=[]
    for index,item in enumerate(items):
        if item.pos!='NOUN' or item.lemma in _FRAME_NOUNS:continue
        before=items[max(0,index-2):index]
        if item.lemma in _NORM_HEADS or any(w.lemma in _NORM_HEADS for w in before):continue
        if any(w.text in {'после','до','без','если'} for w in before):continue
        derivational=item.lemma.endswith(('ние','тие','ция'))
        governed=any(w.lemma in _FRAME_NOUNS or w.text in {'при','для'} for w in before)
        if any(w.text in {'по','в','на','о','об','к','с','из','от'} for w in before) and not governed:continue
        if derivational or governed and index and (items[index-1].lemma in {'порядок','процедура','инструкция','срок'} or any(w.text=='при' for w in before) and item.lemma not in _REQUEST_ARTIFACTS):
            out.append(RequestedOperation(item.text,item.lemma,mode,operation_forms(item.text)))
    return out


def requested_operations(question):
    questions=[match.group(1).strip() for match in re.finditer(r'([^.!?;]+)\?',normalize(question))]
    request=' '.join(questions) if questions else request_text(question)
    # Explicit goals precede subordinate conditions and temporal prerequisites.
    target=re.split(r'\b(?:если|после|до|без)\b',request,maxsplit=1)[0]
    items=words(target);goals=[]
    temporal=bool(re.search(r'срок|когда|сколько|день|дн[ея]|час',request))
    for index,item in enumerate(items):
        if item.pos not in {'VERB','INFN'} or item.lemma in _FRAME_VERBS:continue
        if temporal and item.lemma=='занимать':continue
        if _negated_goal(items,index):continue
        # A question about whether an existing norm applies has a normative
        # subject, not an instruction to execute the finite predicate. Explicit
        # infinitives ("как отменить запрет") remain independently grounded.
        if item.pos=='VERB' and any(w.text=='ли' for w in items) and any(w.pos=='NOUN' and w.lemma in _NORM_HEADS and 'nomn' in nominal_cases(w.text) for w in items):continue
        # A wish is not a historical predicate. Factual preceding sentences are
        # already excluded by request_text; do not re-import them as goals.
        if item.pos=='VERB' and 'past' in item.tags:continue
        mode='acquire' if item.lemma in _GET and re.search(r'\bкак\b|что.{0,20}(?:нужн|необходим)',request) else 'requirement' if item.pos=='VERB' and re.search(r'\bчто\b',request) else 'about' if temporal else 'perform'
        forms=operation_forms(item.text)
        if re.search(r'срок|когда|сколько|день|дн[ея]|час',request):forms=forms-{'@approval'}
        following,modifiers=_argument(items,index,normalize(question)[:normalize(question).find(item.text)])
        if following and following.lemma in {'срок','количество','размер','стоимость','порядок','способ'}:following=None
        if following and mode!='acquire' and following.lemma not in _REQUEST_ARTIFACTS|_DATA_OBJECTS|_MESSAGES and (following.lemma.endswith(('ние','тие','ция')) or following.lemma in _EVENT_NOUNS):following=None
        if following and following.lemma in {'дата','время','срок'}:forms=forms-{'@transport'}
        goals.append(RequestedOperation(item.text,item.lemma,mode,forms,following.key if following else '',following.lemma if following else '',modifiers))
    if goals:
        for nominal in _nominal_goals(target):
            if re.search(r'\b(?:при|для)\s+(?:[а-я]+\s+){0,2}'+re.escape(nominal.text)+r'\b',target):goals.append(nominal)
    if not goals:
        procedure=bool(re.search(r'\bкак\b|действи|порядок|процедур|инструкц',target))
        detail=bool(re.search(r'сведени|документ|срок|сколько|когда|размер|лимит',target))
        goals=_nominal_goals(target,'perform' if procedure and not detail else 'about')
    if not goals:
        # An earlier explicit wish still binds a final generic 'Какие действия?'.
        # A completed past event does not become a new task.
        for sentence in re.split(r'[.!?;\n]+|,\s*(?=(?:а|но)\b)',normalize(question)):
            marker=_INTENT.search(sentence)
            if not marker:continue
            if re.search(r'\b(?:как|какие|какая|какое|какой|что|сколько|когда|кто)\b',sentence[:marker.start()]):continue
            if re.search(r'\bне\s*$',sentence[:marker.start()]):continue
            tail=sentence[marker.end():]
            tail_words=words(tail)
            for index,item in enumerate(tail_words):
                if item.pos=='INFN' and item.lemma not in _FRAME_VERBS and not _negated_goal(tail_words,index):
                    argument,modifiers=_argument(tail_words,index,normalize(question)[:normalize(question).find(item.text)])
                    goals.append(RequestedOperation(item.text,item.lemma,'perform',operation_forms(item.text),argument.key if argument else '',argument.lemma if argument else '',modifiers))
            if not goals:
                nominal=next((item for item in words(tail) if item.pos=='NOUN' and item.lemma not in _FRAME_NOUNS),None)
                if nominal:goals.append(RequestedOperation(nominal.text,nominal.lemma,'perform',operation_forms(nominal.text)))
    return tuple({(goal.lemma,goal.mode):goal for goal in goals}.values())


def _without_temporal_prerequisites(text):
    """Remove completed prerequisites from operation witnesses, not sources."""
    value=normalize(text)
    ranges=[]
    for match in re.finditer(r'\b(?:после|до|без)\s+',value):
        tail=value[match.end():];spans=list(_WORD.finditer(tail));end=len(tail)
        comma=tail.find(',')
        if comma>=0:end=comma
        for index,span in enumerate(spans):
            lemma,pos,tags=morphology(span.group())
            direct_case=index and pos=='NOUN' and bool(tags&{'nomn','accs'})
            if pos in {'VERB','INFN','PRTS','ADJS'} or direct_case:
                end=min(end,span.start());break
        ranges.append((match.start(),match.end()+end))
    for left,right in reversed(ranges):value=value[:left]+' '+value[right:]
    return value


def _action_bindings(clause,document=None):
    """Predicate-local direct arguments, never another sentence's reference."""
    records=[]
    for raw in re.split(r'(?<=[.!?;])\s+|\n+',clause['text']):
        if not raw.strip() or re.fullmatch(r'\s*\d+(?:\.\d+)*[.)]?\s*',raw):continue
        condition=parse_condition(raw);body=normalize(raw)
        if condition and condition.text in body:body=body.replace(condition.text,' ',1)
        body=_without_temporal_prerequisites(body);items=words(body)
        for index,predicate in enumerate(items):
            if predicate.pos not in {'VERB','INFN'}:continue
            context=(document.get('metadata',{}).get('title','')+' ' if document else '')+normalize(raw)[:normalize(raw).find(predicate.text)]
            argument,modifiers=_argument(items,index,context,fronted=True)
            prefix=' '.join(word.text for word in items[:index])
            purpose_keys=frozenset(word.key for word in items[:index] if word.pos=='NOUN') if re.search(r'\bдля\b',prefix) else frozenset()
            purpose_start=next((i for i,w in enumerate(items[:index]) if w.text=='для'),None)
            purpose_head=next((w for w in items[purpose_start+1:index] if w.pos=='NOUN'),None) if purpose_start is not None else None
            effects=[]
            if re.sub(r'(?:ся|сь)$','',predicate.lemma) in _CREATE and argument and argument.lemma in _DATA_OBJECTS:
                effects=[word for word in items[index+1:] if word.pos=='PRTF' and {'actv','pres'}<=word.tags]
            records.append(dict(forms=operation_forms(predicate.text),lemma=predicate.lemma,object_key=argument.key if argument else '',object_lemma=argument.lemma if argument else '',modifiers=modifiers,purpose_keys=purpose_keys,purpose_head=purpose_head,effects=tuple(effects),text=raw.strip()))
    return records


def _bound_to_goal(goal,record):
    if not goal.object_key:return True
    if goal.object_key!=record['object_key']:return False
    return not goal.object_modifiers or goal.object_modifiers<=record['modifiers']


def _entry_restriction(text):
    """Bind the access patient to its first local predicative.

    A different ban later in the paragraph does not prohibit entry. Only the
    actual access predicate supplies this capability relation; another finite
    predicate or short adjective before the ban ends that interpretation.
    """
    for unit in re.split(r'[.!?;]+',normalize(text)):
        items=words(unit)
        subject=next(((i,w) for i,w in enumerate(items) if w.pos=='NOUN'),None)
        if not subject or subject[1].lemma!='доступ':continue
        predicate=next(((i,w) for i,w in enumerate(items[subject[0]+1:],subject[0]+1) if w.pos in {'VERB','INFN','PRTS','ADJS','PRED'}),None)
        if not predicate:continue
        index,word=predicate;negative=any(w.text=='не' for w in items[max(subject[0]+1,index-2):index])
        if word.lemma=='запретить' and not negative:return True
        if word.lemma in {'разрешить','разрешать','разрешаться','допускать','допускаться'} and negative:return True
    return False


def _source_operations(clause,include_administrative=False,document=None):
    """Yield (forms, lemma, evidence role) from actual operational source text."""
    for span in [clause['text']] if include_administrative else procedural_spans(clause['text']):
        condition=parse_condition(span)
        body=normalize(span)
        # A noun reference is useful for a question about required checks; it
        # does not become an instruction to perform a completed prerequisite.
        for word in words(body):
            if word.pos=='NOUN':yield operation_forms(word.text),word.lemma,'reference'
        if condition and condition.text in body:
            if condition.kind=='nominal':
                head=next((word for word in words(condition.text) if word.pos=='NOUN'),None)
                if head and head.lemma not in {'наличие','отсутствие'}:yield operation_forms(head.text),head.lemma,'procedure_context'
            body=body.replace(condition.text,' ',1)
        body=_without_temporal_prerequisites(body);items=words(body)
        first_noun=next((word for word in items if word.pos=='NOUN'),None)
        for index,word in enumerate(items):
            if word.pos in {'VERB','INFN'}:yield operation_forms(word.text),word.lemma,'action'
            elif word.pos=='NOUN':
                previous=items[max(0,index-2):index]
                purpose=any(w.text in {'для','при','перед'} or w.lemma in {'порядок','процедура','инструкция','срок'} for w in previous)
                light_complement=any(w.lemma in _PERFORM for w in previous)
                normative=any(w.lemma in {'запретить','разрешить','допускаться','предусмотреть'} for w in items[index+1:index+4])
                if purpose or word==first_noun or normative or light_complement:yield operation_forms(word.text),word.lemma,'procedure_context'
                yield operation_forms(word.text),word.lemma,'about'
        # An explicit restriction on access grounds an entry-permission
        # question. This is a typed capability restriction, not a procedure
        # for any arbitrary action that happens to mention the same place.
        if _entry_restriction(body):
            yield frozenset({'@ingress'}),'доступ','capability_restriction'
        # A literal source heading can identify the operation of terse actions.
        # Metadata titles are deliberately not treated as factual source text.
        for word in words(_without_temporal_prerequisites(clause.get('section',''))):
            if word.pos=='NOUN' and word.lemma.endswith(('ние','тие','ция')):yield operation_forms(word.text),word.lemma,'section_context'
        if document:
            title=normalize(document.get('metadata',{}).get('title',''))
            preview=normalize(document.get('text_preview',''))
            if title and re.search(r'(?:^|\s)#\s*'+re.escape(title)+r'(?:\s|$)',preview):
                for word in words(_without_temporal_prerequisites(title)):
                    if word.pos=='NOUN' and word.lemma.endswith(('ние','тие','ция')):yield operation_forms(word.text),word.lemma,'document_context'


def _verified_context(document,clause):
    context=clause.get('section','');title=normalize(document.get('metadata',{}).get('title',''))
    preview=normalize(document.get('text_preview',''))
    if title and re.search(r'(?:^|\s)#\s*'+re.escape(title)+r'(?:\s|$)',preview):context+=' '+title
    return context


def operation_witnesses(goal,clauses,question='',include_administrative=False):
    witnesses=[];exact=[];facts=question_facts(question) if question else None
    for doc,clause in clauses:
        condition=parse_condition(clause['text'])
        clause_words=words(clause['text']+' '+_verified_context(doc,clause))
        if goal.object_key and not any(goal.object_key==word.key for word in clause_words):continue
        bindings=_action_bindings(clause,doc)
        source_operations=list(_source_operations(clause,include_administrative,doc))
        reference_only=any(role=='reference' and equivalent(goal.forms,forms) for forms,lemma,role in source_operations) and not any(role in {'action','procedure_context'} and equivalent(goal.forms,forms) for forms,lemma,role in source_operations)
        for forms,lemma,role in source_operations:
            permitted_role=role not in {'about','reference'} or role=='about' and goal.mode in {'about','requirement'} or role=='reference' and goal.mode=='requirement'
            if role in {'section_context','document_context'} and reference_only and goal.mode!='requirement':permitted_role=False
            supported=equivalent(goal.forms,forms) and permitted_role
            lexical=equivalent(frozenset(f for f in goal.forms if not f.startswith('@')),frozenset(f for f in forms if not f.startswith('@')))
            if supported and not lexical and '@approval' in goal.forms&forms and not condition:supported=False
            matching=[record for record in bindings if record['lemma']==lemma and equivalent(record['forms'],forms)]
            bound=[record for record in matching if _bound_to_goal(goal,record)]
            if role=='action' and supported and not bound:supported=False
            # A weaker heading/reference must not override an explicit action
            # on a different object in the same source clause.
            same_predicates=[record for record in bindings if equivalent(goal.forms,record['forms'])]
            if goal.object_key and role in {'section_context','document_context'} and same_predicates and not any(_bound_to_goal(goal,record) for record in same_predicates):supported=False
            if goal.mode=='acquire' and role=='action':
                result_forms=operation_forms(goal.object_lemma) if goal.object_lemma else frozenset()
                produced=re.sub(r'(?:ся|сь)$','',lemma) in _CREATE|_GET|_PROVIDE|_PAY and bool(bound)
                explicit_purpose=False
                for record in matching:
                    head=record['purpose_head']
                    acquisition_head=bool(head and any(equivalent(operation_forms(head.text),operation_forms(verb)) for verb in _GET|_PROVIDE|_CREATE))
                    if head and goal.object_key and (head.key==goal.object_key or acquisition_head and goal.object_key in record['purpose_keys']):explicit_purpose=True
                named_result=any(equivalent(result_forms,record['forms']) for record in matching) if result_forms else False
                supported=produced or explicit_purpose or named_result
            if goal.lemma in {'подать','подавать','подаваться'} and goal.object_lemma in _REQUEST_ARTIFACTS and role=='action' and re.sub(r'(?:ся|сь)$','',lemma) in _CREATE and bound:supported=True
            if '@communicate' in goal.forms and role=='action' and lemma in _TRANSFER and any(record['object_lemma'] in _MESSAGES for record in matching):supported=True
            if role=='action' and goal.object_lemma in _DATA_OBJECTS:
                for record in matching:
                    if any(equivalent(goal.forms,operation_forms(effect.text)) for effect in record['effects']) and goal.object_key in {word.key for word in words(record['text'])}:supported=True
            if supported:
                witness=dict(document_id=doc['id'],clause_id=clause['id'],requested=goal.text,source_operation=lemma,evidence_role=role,quote=clause['text'])
                witnesses.append(witness)
                if equivalent(frozenset(f for f in goal.forms if not f.startswith('@')),frozenset(f for f in forms if not f.startswith('@'))):exact.append(witness)
                break
    return witnesses


@dataclass(frozen=True)
class OperationScope:
    goals:tuple
    witnesses:tuple
    missing:tuple

    @property
    def clause_ids(self):return frozenset(item['clause_id'] for item in self.witnesses)

    @property
    def document_ids(self):return frozenset(item['document_id'] for item in self.witnesses)

    def diagnostic(self):
        return dict(requested=[dict(text=goal.text,lemma=goal.lemma,mode=goal.mode) for goal in self.goals],missing=[goal.text for goal in self.missing],witnesses=[{key:value for key,value in item.items() if key!='quote'} for item in self.witnesses])


def ground_operations(question,clauses,include_administrative=False):
    goals=requested_operations(question);witnesses=[];missing=[]
    for goal in goals:
        support=operation_witnesses(goal,clauses,question,include_administrative)
        if support:witnesses.extend(support)
        else:missing.append(goal)
    return OperationScope(goals,tuple(witnesses),tuple(missing))


def parallel_operation_clauses(scope,clauses):
    """Same predicate on another patient cannot broaden an explicit goal.

    These clauses remain in the corpus for genuine source prerequisites and
    literal references. They are excluded only from automatic topic/verb
    expansion, which would otherwise undo the predicate/argument check.
    """
    excluded=set()
    for document,clause in clauses:
        if clause['id'] in scope.clause_ids:continue
        bindings=_action_bindings(clause,document)
        for goal in scope.goals:
            if not goal.object_key:continue
            witnessed_forms=[operation_forms(item['source_operation']) for item in scope.witnesses if item['requested']==goal.text]
            matches=[record for record in bindings if equivalent(goal.forms,record['forms']) or any(equivalent(forms,record['forms']) for forms in witnessed_forms)]
            if matches and not any(_bound_to_goal(goal,record) for record in matches):
                excluded.add(clause['id']);break
    return frozenset(excluded)
