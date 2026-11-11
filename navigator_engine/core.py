"""Date-aware retrieval and extractive answers; source content is never code."""
from __future__ import annotations
from datetime import date
from collections import defaultdict
from math import isfinite
import re
from .parsing import iso_date
from .spreadsheet_answers import answer_spreadsheet
from .conditions import select_candidates,parse_condition,administrative,request_text,validated_context_terms,prerequisite_producers,procedural_spans
from .operations import ground_operations,excluded_task_without_goal,parallel_operation_clauses
from .text import BM25,tokens,normalize,stem,STOP,query_subjects,query_terms,lexical_vocabulary,morphology,lexical_key

ROLE_ALIASES={'сотрудник':'employee','работник':'employee','employee':'employee','руководитель':'manager','начальник':'manager','manager':'manager','бухгалтер':'accountant','accountant':'accountant','студент':'student','student':'student','все':'all','all':'all','*':'all'}

def role_key(value):return ROLE_ALIASES.get(normalize(value),normalize(value))

def _base(status='not_found',question='',as_of='',role=None):
    return dict(schema_version='1.0',status=status,question=question,as_of=as_of,role=role,answer='',steps=[],citations=[],sources=[],excluded=[],warnings=[],followups=[],score=0.,confidence=0.,score_label='Оценка близости, не вероятность правильности')

def _profiles(text):
    clean=re.sub(r'^\s*\d+(?:\.\d+)*[.)]?\s+','',normalize(text))
    values=[]
    for match in re.finditer(r'(\d+(?:[.,]\d+)?)\s*(рабоч\w*|календар\w*)?\s*(дн\w*|день|дня|дней|час\w*|недел\w*|месяц\w*|руб\w*|₽|тысяч\w*)',clean):
        number=match.group(1).replace(',','.');qualifier=match.group(2) or '';unit=match.group(3)
        unit='день' if re.match(r'дн|день|дня|дней',unit) else 'час' if unit.startswith('час') else 'руб' if unit.startswith('руб') or unit=='₽' else unit[:5]
        values.append((number,qualifier[:6],unit))
    return tuple(values)

def _rule_kind(text):
    text=normalize(text)
    for pattern,kind in [
        (r'после возвращения|отчет.{0,60}(?:направ|предостав|сда)|авансов.{0,30}отчет','reporting'),
        (r'срок согласования|согласу[ею]|согласовыва|рассматрива','approval'),
        (r'не позднее чем за|до (?:выезда|начала)|пода[ею]т|созда[ею]т.{0,30}заяв|подать','submission'),
        (r'исполня|исполнени','execution'),(r'суточн','daily_allowance'),(r'гостиниц|проживан|отел','lodging'),(r'такси','taxi'),
        (r'переезд','relocation'),(r'парол','password'),(r'доступ','access')]:
        if re.search(pattern,text):return kind
    if re.search(r'переда[ею]т(?:ся)?',text):return 'transfer'
    return None

def _conflicts(items):
    conflicts=[]
    banned=re.compile(r'запрещ|нельзя|не\s+допуска|не\s+разреш|не\s+вправе')
    allowed=re.compile(r'(?<!не )разреша|(?<!не )допуска|можно|вправе')
    action_words={'подать','подает','подач','соглас','предостав','направ','оформ','передав','оплат','выплат','утвержд','подпис','заявк','заявл','закуп','отчет','отпуск','удален','командиров','расход','доступ'}
    for index,left in enumerate(items):
        for right in items[index+1:]:
            ld,lc,ls=left;rd,rc,rs=right
            if lc['id']==rc['id']:continue
            lt=normalize(ld['metadata'].get('topic'));rt=normalize(rd['metadata'].get('topic'))
            if lt and rt and lt!=rt:continue
            a,b=lc['text'],rc['text'];ta,tb=set(tokens(a)),set(tokens(b));common=ta&tb
            overlap=len(common)/max(1,min(len(ta),len(tb)))
            same_event=overlap>=.82 and len(common)>=4
            known_action=any(any(t.startswith(w) for w in action_words) for t in common)
            if overlap<(.45 if lt and rt else .65) or (not known_action and not same_event):continue
            # Explicitly different conditional scopes are not contradictions.
            ca,cb=_condition_signature(a),_condition_signature(b)
            if (ca or cb) and ca!=cb:continue
            ka,kb=_rule_kind(a),_rule_kind(b)
            if ka!=kb and ((ka is not None and kb is not None) or not same_event):continue
            if ka is None and lc.get('section')!=rc.get('section') and not same_event:continue
            pa,pb=_profiles(a),_profiles(b);reason=None
            if pa and pb and pa!=pb:reason='Различаются явно указанные значения или единицы срока/лимита.'
            elif (banned.search(normalize(a)) and allowed.search(normalize(b))) or (banned.search(normalize(b)) and allowed.search(normalize(a))):reason='Один пункт запрещает действие, другой разрешает.'
            if reason:conflicts.append(dict(left=lc['id'],right=rc['id'],reason=reason,document_ids=[ld['id'],rd['id']]))
    return conflicts


def _condition_signature(text):
    """Compare antecedents, not the different consequences following a comma.

    Numbers and negation stay in the condition.  This is deliberately narrower
    than semantic equivalence: unknown paraphrases are not declared conflicts.
    """
    clean=re.sub(r'^\s*\d+(?:\.\d+)*[.)]?\s*','',normalize(text))
    match=re.search(r'\b(?:если|при условии(?: что)?|в случае(?: если)?)\s+([^,;.!?]+)',clean)
    if match:return tuple(tokens(match.group(1),stop=False))+tuple(re.findall(r'\d+(?:[.,]\d+)?',match.group(1)))
    match=re.search(r'\b(без|после|до)\s+((?:письменн\w*\s+|предварительн\w*\s+)?(?:согласован\w*|одобрен\w*|утвержден\w*|подтвержден\w*))',clean)
    if match:return (match.group(1),)+tuple(tokens(match.group(2),stop=False))
    return ()


def _version_graph(active,all_documents):
    """Resolve explicit replacements without silently accepting cyclic metadata."""
    identifiers={str(doc['id']) for doc in active}
    known={str(doc['id']) for doc in all_documents}
    graph={identifier:[] for identifier in identifiers}
    issues=[];invalid=set()
    by_id={str(doc['id']):doc for doc in active}
    for doc in active:
        source=str(doc['id'])
        for reference in doc.get('metadata',{}).get('supersedes',[]):
            target=str(reference)
            if target not in known:
                issues.append(dict(code='missing_replacement_target',document_ids=[source],target_id=target,
                    message=doc['name']+': заменяемый документ отсутствует в загруженной базе; связь редакций не проверена.'))
            if target not in identifiers:continue
            left=normalize(doc.get('metadata',{}).get('topic'));right=normalize(by_id[target].get('metadata',{}).get('topic'))
            if left and right and left!=right:
                invalid.update([source,target]);issues.append(dict(code='replacement_scope_mismatch',document_ids=[source,target],
                    message=doc['name']+': связь замены соединяет документы с разными явно заданными темами.'))
                continue
            graph[source].append(target)
    # Tarjan's SCC pass is linear in documents plus declared replacement edges.
    cyclic=set();indices={};low={};stack=[];on_stack=set()
    def visit(node):
        indices[node]=low[node]=len(indices);stack.append(node);on_stack.add(node)
        for target in graph[node]:
            if target not in indices:visit(target);low[node]=min(low[node],low[target])
            elif target in on_stack:low[node]=min(low[node],indices[target])
        if low[node]==indices[node]:
            component=[]
            while True:
                target=stack.pop();on_stack.remove(target);component.append(target)
                if target==node:break
            if len(component)>1 or node in graph[node]:cyclic.update(component)
    for origin in graph:
        if origin not in indices:visit(origin)
    if cyclic:
        invalid.update(cyclic)
        issues.append(dict(code='replacement_cycle',document_ids=sorted(cyclic),
            message='Обнаружен цикл явной замены редакций: '+', '.join(by_id[identifier]['name'] for identifier in sorted(cyclic))+'. Нельзя определить применимую редакцию до исправления метаданных.'))
    replaced={target:source for source,targets in graph.items() for target in targets if source not in invalid and target not in invalid}
    return replaced,issues,invalid


_CONTEXT_PREFIX=re.compile(r'^\s*(?:[-*•]\s+|\d+(?:\.\d+)*[.)]?\s*)?(?:если|иначе|в этом случае|в противном случае|при этом|однако|кроме|за исключением|исключение|только при|при условии|при отсутствии|при наличии|запрещ|нельзя|не допуска|не разреш)',re.I)
_DEICTIC_PREFIX=re.compile(r'^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:иначе|в этом случае|в противном случае|при этом|после этого)',re.I)


def _conditional_text(text):
    """Read an explicit source antecedent, never invent an implicit one."""
    clean=re.sub(r'^\s*\d+(?:\.\d+)*[.)]?\s*','',normalize(text))
    match=re.match(r'(?:если|при условии(?: что)?|в случае(?: если)?)\s+(.+)',clean)
    if match:
        pieces=match.group(1).split(',')
        condition=pieces[0]
        if len(pieces)>2 and re.match(r'\s*(?:но|и|а)\b',pieces[1]):condition+=', '+pieces[1]
        return condition
    match=re.match(r'при\s+(.+)',clean)
    if match:
        words=re.findall(r'[а-яa-z]+',match.group(1));end=None
        for index,word in enumerate(words):
            lemma,pos,tags=morphology(word)
            ungoverned_subject=index and tags&{'nomn','accs'} and not any(morphology(w)[1]=='PREP' for w in words[max(0,index-2):index])
            if pos in {'VERB','INFN','PRTS','ADJS'} or (ungoverned_subject and (pos=='NOUN' or 'Subx' in tags)):end=index;break
        if end and words[end-1]=='не':end-=1
        if end is not None and end>0:return ' '.join(words[:end])
    return ''


def _literal_states(text):
    """Only explicit noun presence/negation; no world-knowledge inference.

    This deliberately cannot equate 'исправно' with a completed inspection.
    Missing predicates remain unknown, and exact source conditions are retained.
    """
    states=defaultdict(set)
    for phrase in re.split(r'[.;!?]|,(?!\s*(?:и\s+)?ни\b)',normalize(text)):
        words=re.findall(r'[а-яa-z]+',phrase)
        for index,word in enumerate(words):
            lemma,pos,tags=morphology(word)
            if pos!='NOUN' or 'anim' in tags or lemma in {'наличие','отсутствие'}:continue
            before=words[max(0,index-4):index];after=words[index+1:index+4]
            # Negation of a receipt does not negate the goods named after 'на'.
            # Preserve 'без' itself, which directly governs a missing object.
            boundaries=[i for i,w in enumerate(before) if morphology(w)[1]=='PREP' and w!='без']
            if boundaries:before=before[boundaries[-1]+1:]
            negative=any(w in {'нет','без','не','ни'} or morphology(w)[0] in {'отсутствие','отсутствовать'} for w in before)
            negative=negative or any(w in {'не','нет'} for w in after)
            states[stem(lemma)].add(not negative)
    return states


def _condition_state(condition,question_states):
    if not condition:return None
    states=_literal_states(condition);matched=0
    for key,values in states.items():
        found=question_states.get(key)
        if not found:continue
        # A contradictory or mixed user statement cannot prove a branch.
        if len(found)>1:return None
        if values.isdisjoint(found):return False
        matched+=1
    return True if states and matched==len(states) else None


def _applicable_candidates(ranked,all_items,question,parallel_exclusions=()):
    """Compatibility wrapper around the typed, three-valued condition planner."""
    return select_candidates(ranked,all_items,question,parallel_exclusions)


def _source_dependencies(seeds,all_items,limit=10,parallel_exclusions=()):
    """Return a bounded closure of source prerequisites and nearby exceptions.

    Selected instructions, their explicit numbered prerequisites, and local
    conditional continuations are never independently truncated.  An unresolved
    reference or a closure above the display budget becomes an explicit partial
    answer rather than a definitive instruction with its exception omitted.
    """
    lookup={clause['id']:(doc,clause,score) for doc,clause,score in all_items}
    selected={clause['id']:(doc,clause,score) for doc,clause,score in seeds}
    ordered={};numbered={};missing=[]
    for item in all_items:
        doc,clause,score=item;ordered.setdefault((doc['id'],clause.get('section','')),[]).append(item)
        match=re.match(r'^\s*(\d+(?:\.\d+)+)[.)]?\s+',clause['text'])
        if match:numbered.setdefault((doc['id'],match.group(1)),[]).append(item)
    for items in ordered.values():items.sort(key=lambda item:(int(item[1].get('paragraph') or 0),item[1]['id']))
    document_clauses={}
    for doc,clause,score in all_items:document_clauses.setdefault(doc['id'],[]).append(clause)
    queue=list(selected);visited=set()
    while queue and len(visited)<48:
        identifier=queue.pop(0)
        if identifier in visited:continue
        visited.add(identifier);doc,clause,score=selected[identifier]
        additions=[]
        for producer in prerequisite_producers(clause,document_clauses[doc['id']]):additions.append(lookup[producer['id']])
        reference_groups=re.findall(r'(?i)\b(?:пункт[а-я]*|п\.)\s*((?:\d+(?:\.\d+)+)(?:\s*(?:,|и|или)\s*\d+(?:\.\d+)+)*)',clause['text'])
        references=[reference for group in reference_groups for reference in re.findall(r'\d+(?:\.\d+)+',group)]
        for reference in references:
            candidates=numbered.get((doc['id'],reference),[])
            if len(candidates)==1:additions.extend(candidates)
            else:missing.append(dict(document_id=doc['id'],clause_id=identifier,reference=reference))
        explicit_additions={item[1]['id'] for item in additions}
        siblings=ordered[(doc['id'],clause.get('section',''))]
        index=next(i for i,item in enumerate(siblings) if item[1]['id']==identifier)
        # Local context, not every conditional clause from a long entire policy.
        for neighbour in siblings[max(0,index-2):min(len(siblings),index+4)]:
            text=neighbour[1]['text']
            if _CONTEXT_PREFIX.search(text):additions.append(neighbour)
        if _DEICTIC_PREFIX.search(clause['text']) and index:additions.append(siblings[index-1])
        if index and siblings[index-1][1]['text'].rstrip().endswith(':'):additions.append(siblings[index-1])
        if clause['text'].rstrip().endswith(':'):
            for neighbour in siblings[index+1:min(len(siblings),index+7)]:
                if not re.match(r'^\s*(?:[-*•]\s+|\d+(?:\.\d+)*[.)]\s+)',neighbour[1]['text']):break
                additions.append(neighbour)
        for addition in additions:
            key=addition[1]['id']
            if key in parallel_exclusions and key not in explicit_additions:continue
            if key not in selected:selected[key]=addition;queue.append(key)
    items=list(selected.values())
    return items[:limit],missing,len(items)>limit or bool(queue)

def answer(question,documents,as_of=None,role=None,options=None):
    options=dict(options or {})
    if not isinstance(question,str):raise ValueError('Вопрос должен быть строкой')
    question=question.strip()
    if len(question)>2000:raise ValueError('Вопрос не должен превышать 2000 символов')
    as_of=iso_date(as_of or date.today().isoformat());result=_base(question=question,as_of=as_of,role=role)
    original_question=question
    injection=re.search(r'(?i)(?:игнорируй|ignore).{0,70}(?:документ|регламент|инструкц|правил|previous|system)',question)
    if injection:
        question=question[:injection.start()].strip(' .;')
        result['warnings'].append('Указание игнорировать источники не использовано; проверяется только предметный вопрос.')
    if not options.get('_single_facet'):
        facets=_question_facets(question)
        if len(facets)>1:return _answer_facets(original_question,facets,documents,as_of,role,options)
    if not question:
        result.update(status='needs_clarification',answer='Опишите действие, для которого нужен порядок оформления.',followups=['Что вы хотите сделать: оформить командировку, отпуск, закупку или другое действие?']);return result
    if len(documents)>50:raise ValueError('Не более 50 документов в базе')
    if len({str(d.get('id')) for d in documents})!=len(documents):raise ValueError('Идентификаторы документов должны быть уникальными')
    from .document_metadata import resolve_document_relationships
    documents=resolve_document_relationships(documents)
    active=[];seen_hash={};doc_roles={};selected_role=role_key(role) if role else None
    for doc in documents:
        metadata=doc.get('metadata') or {};reason=None
        if doc.get('status')=='unreadable' or not doc.get('clauses'):reason='unreadable'
        try:
            start=iso_date(metadata.get('effective_from'));end=iso_date(metadata.get('effective_to'))
            if start and start>as_of:reason='not_yet_effective'
            elif end and end<as_of:reason='expired'
        except ValueError:reason='invalid_metadata'
        roles=[role_key(value) for value in metadata.get('roles',[])];doc_roles[doc['id']]=roles
        if selected_role and roles and 'all' not in roles and selected_role not in roles:reason='role_mismatch'
        digest=doc.get('sha256')
        if not reason and digest and digest in seen_hash and metadata==seen_hash[digest].get('metadata'):reason='duplicate'
        if reason:result['excluded'].append(dict(document_id=doc.get('id'),document_name=doc.get('name'),reason=reason))
        else:
            active.append(doc)
            if digest:seen_hash[digest]=doc
    replaced,version_issues,invalid_versions=_version_graph(active,documents)
    invalid_versions=set(invalid_versions)|{d['id'] for d in active if d.get('metadata_provenance',{}).get('ambiguities')}
    if version_issues:result['version_issues']=version_issues
    for doc in list(active):
        if doc['id'] in replaced:
            active.remove(doc);result['excluded'].append(dict(document_id=doc['id'],document_name=doc['name'],reason='superseded',replaced_by=replaced[doc['id']]))
    spreadsheet = answer_spreadsheet(question,active,result,invalid_versions,version_issues)
    if spreadsheet is not None:return spreadsheet
    clauses=[]
    administration_question=bool(re.search(r'(?i)кто.{0,35}(?:назначен|ответствен)|хранени\w* подлинник|когда.{0,25}вступ|что.{0,25}измен|кака\w* (?:редакц|верси)|сравн.{0,35}редакц',question))
    for doc in active:
        for clause in doc.get('clauses',[]):
            if clause.get('kind')=='notice':continue
            if administrative(clause['text'],clause.get('section','')) and not administration_question and not procedural_spans(clause['text']):continue
            if clause.get('unsafe'):
                result['warnings'].append('Из поиска исключён пункт с признаками инструкции для модели: '+clause['id']);continue
            clauses.append((doc,clause))
    if len(clauses)>5000:raise ValueError('Не более 5000 пунктов в активной базе')
    query_tokens=query_terms(question)
    if not query_tokens:
        result.update(status='needs_clarification',answer='Уточните предмет вопроса: в нём пока нет достаточно конкретного действия.',followups=['О каком действии или процедуре идёт речь?']);return result
    if query_tokens<= {'заявк','заявл','соглас','срок','день'}:
        result.update(status='needs_clarification',answer='Уточните, для какой процедуры нужна заявка: разные регламенты устанавливают разные действия.',followups=['Заявка на командировку, отпуск, закупку или другое действие?']);return result
    if not clauses:
        result['answer']='В базе нет доступных действующих пунктов для ответа на выбранную дату и роль.';result['followups']=['Добавьте действующий документ или проверьте дату и роль.'];return result
    corpus_text=' '.join(clause['text'] for _,clause in clauses)
    qualifiers=re.findall(r'\b(?:декрет\w*|беремен\w*|учебн\w*|больнич\w*|загран\w*|виз\w*|пенсион\w*)',normalize(question))
    missing_qualifiers=[word for word in qualifiers if not any(token.startswith(word[:5]) for token in tokens(corpus_text))]
    if missing_qualifiers:
        result['answer']='В действующих документах не найдено правило для уточнения «'+', '.join(missing_qualifiers)+'». Общий порядок нельзя автоматически применять к этому случаю.'
        result['followups']=['Загрузите документ, который прямо регулирует этот случай.'];return result
    # Explicit subject qualifiers restrict the search before lexical/semantic
    # scoring. A question about trip applications must not inherit the shared
    # words 'application' and 'days' from purchase approval.
    anchor_groups=[{'суточ'},{'чек','квитанц'},{'аванс'},{'отпуск'},{'удален','vpn','доступ'},{'переезд'},{'закуп'},{'командиров'}]
    anchors=next((group&query_tokens for group in anchor_groups if group&query_tokens),set())
    if anchors:
        doc_subjects={doc['id']:set(tokens(str(doc['metadata'].get('title',''))+' '+' '.join(c['text'] for c in doc.get('clauses',[])))) for doc in active}
        clauses=[(doc,clause) for doc,clause in clauses if anchors&doc_subjects[doc['id']]]
    if not clauses:
        result['answer']='В действующих источниках нет пункта, который прямо относится к указанной процедуре.';result['followups']=['Загрузите регламент для этой процедуры.'];return result
    clauses=_specific_document_scope(question,clauses)
    if not clauses:
        result.update(status='needs_clarification',answer='Не найдено источника, который одновременно подтверждает указанные особенности процедуры.',followups=['Разделите разные процедуры на отдельные вопросы или добавьте документ с нужными условиями.']);return result
    vocabulary=lexical_vocabulary(' '.join(str(doc['metadata'].get('title',''))+' '+clause['text'] for doc,clause in clauses))
    context_terms,context_evidence=validated_context_terms(question,clauses)
    unsupported=_unsupported_entities(question,vocabulary,context_terms)
    if unsupported:
        result['answer']='В действующих источниках не найдено основание именно для «'+', '.join(unsupported)+'». Близкое правило по другой процедуре применять автоматически нельзя.'
        result['followups']=['Уточните термин или загрузите документ, который прямо регулирует этот случай.'];return result
    operation_scope=ground_operations(question,clauses,include_administrative=administration_question)
    if excluded_task_without_goal(question,operation_scope.goals):
        result.update(status='needs_clarification',answer='Вы указали действие, которое не нужно выполнять. Уточните, какое действие требуется вместо него.',followups=['Какую операцию вы хотите выполнить?']);return result
    if operation_scope.goals:result['operation_scope']=operation_scope.diagnostic()
    if operation_scope.missing and not operation_scope.witnesses:
        missing=', '.join('«'+goal.text+'»' for goal in operation_scope.missing)
        result['answer']='В действующих источниках не найден порядок для запрошенной операции '+missing+'. Совпадение предмета с другой процедурой недостаточно.'
        result['followups']=['Добавьте пункт, который прямо описывает это действие, или уточните, какую операцию нужно выполнить.'];return result
    if operation_scope.missing:
        # A partly supported compound request is not an absent procedure.
        # Continue collecting its genuine evidence, then withhold operative
        # steps unless every requested action is supported. A known prohibition
        # remains visible without inventing a procedure for the other action.
        result['warnings'].append('Источники подтверждают только часть запрошенных действий. Не подтверждены: '+', '.join(goal.text for goal in operation_scope.missing)+'.')
    if operation_scope.goals:clauses=[(doc,clause) for doc,clause in clauses if doc['id'] in operation_scope.document_ids]
    parallel_exclusions=parallel_operation_clauses(operation_scope,clauses)
    texts=[str(doc['metadata'].get('title',''))+' '+str(doc['metadata'].get('topic',''))+' '+str(clause.get('section',''))+' '+clause['text'] for doc,clause in clauses]
    model=BM25(texts);lexical=model.scores(question);rank=[]
    for i,((doc,clause),raw) in enumerate(zip(clauses,lexical)):
        content=lexical_vocabulary(texts[i]);coverage=len(query_tokens&content)/max(1,len(query_tokens))
        body_coverage=len(query_tokens&lexical_vocabulary(clause['text']))/max(1,len(query_tokens))
        if operation_scope.goals and clause['id'] in operation_scope.clause_ids:
            # Verified predicate/argument evidence carries the explicit goal;
            # negative wishes and earlier completed facts do not dilute it.
            supported={witness['requested'] for witness in operation_scope.witnesses if witness['clause_id']==clause['id']}
            goal_coverage=sum(goal.text in supported for goal in operation_scope.goals)/len(operation_scope.goals)
            coverage=max(coverage,goal_coverage*.75)
            body_coverage=max(body_coverage,goal_coverage*.75)
        score=.55*body_coverage+.30*coverage+.15*raw/(raw+4.)
        rank.append([score,i,coverage,0.])
    # Semantic retrieval is restricted to a bounded shortlist for predictable CPU.
    semantic=options.get('semantic_similarity');shortlist=sorted(rank,reverse=True)[:min(80,max(1,int(options.get('_semantic_limit',80))),len(rank))]
    if callable(semantic):
        try:
            matrix=semantic([question],[texts[item[1]][:1800] for item in shortlist])
            for k,item in enumerate(shortlist):
                sem=float(matrix[0][k])
                if isfinite(sem):
                    sem=max(0.,min(1.,sem));item[3]=sem
                    item[0]=max(item[0],.65*sem+.35*item[2])
        except Exception:result['warnings'].append('Семантический поиск недоступен; использован локальный BM25.')
    rank.sort(reverse=True);best=rank[0][0]
    relevant=[item for item in rank if item[0]>=max(.38,best*.62) and (item[2]>=.25 or item[3]>=.70)]
    if not relevant:
        result['answer']='В действующих документах не найдено достаточного основания для ответа. Я не буду добавлять сроки или действия без источника.'
        result['followups']=['Уточните название процедуры или загрузите регламент по этому вопросу.'];return result
    question_kind=None
    if re.search(r'(?i)срок|сколько|когда|дн[еёя]|день',question):
        # The requested process head determines the stage. A participle such
        # as 'согласованного' describes the cargo; it is not an approval query.
        for word in re.findall(r'[а-яa-z]+',request_text(question)):
            lemma,pos,tags=morphology(word)
            if pos not in {'NOUN','VERB','INFN'}:continue
            if lemma.startswith('исполн'):question_kind='execution'
            elif lemma.startswith(('соглас','рассмотр','рассматр')):question_kind='approval'
            elif lemma.startswith('переда'):question_kind='transfer'
            elif lemma.startswith(('подать','подач','подат')):question_kind='submission'
            elif lemma.startswith(('отчет','возвращ')):question_kind='reporting'
            if question_kind:break
        if not question_kind and re.search(r'(?i)заявк|заявлен',request_text(question)):question_kind='submission'
    if question_kind:
        focused=[item for item in rank if _rule_kind(clauses[item[1]][1]['text'])==question_kind and (item[2]>=.15 or item[3]>=.55)]
        if focused:
            focus_best=max(item[0] for item in focused)
            relevant=[item for item in focused if item[0]>=max(.28,focus_best*.62)]
    if operation_scope.goals:
        relevant=[item for item in relevant if clauses[item[1]][1]['id'] in operation_scope.clause_ids]
        if not relevant:
            relevant=[item for item in rank if clauses[item[1]][1]['id'] in operation_scope.clause_ids and item[0]>=.2 and (item[2]>=.1 or item[3]>=.7)]
        if not relevant:
            result['answer']='Найдено упоминание запрошенной операции, но недостаточно подтверждённых действий для ответа.'
            result['followups']=['Уточните этап процедуры или добавьте её полный порядок.'];return result
    # Detect relevant role forks before combining instructions.
    if not selected_role:
        role_sets=[set(doc_roles.get(clauses[item[1]][0]['id'],[])) for item in relevant[:8]]
        restricted=[r for r in role_sets if r and 'all' not in r]
        roles=set().union(*restricted) if restricted else set()
        if len(roles)>1 and restricted and not set.intersection(*restricted):
            result.update(status='needs_clarification',answer='Порядок зависит от роли. Укажите свою роль, чтобы не смешивать разные инструкции.',followups=['Ваша роль: '+', '.join(sorted(roles))+'?']);return result
    ranked=[(*clauses[item[1]],item[0]) for item in relevant[:24]]
    # A broad how-to question expands the strongest procedural document in its
    # original order. Cost schedules and related policies are separate answers.
    howto=bool(re.search(r'(?i)как.{0,35}(?:оформ|получ|подключ|нача|инициир)|порядок (?:оформ|получ)|весь порядок|полный порядок|все этапы|процедуру целиком',question))
    if howto and not re.search(r'(?i)срок|сколько|когда|за .*дн|расход|суточ|лимит|чек',question):
        procedural=[]
        for doc in active:
            candidates=[item for item in ranked if item[0]['id']==doc['id']]
            if not candidates:continue
            title=normalize(doc['metadata'].get('title',''))
            bonus=.18 if re.search(r'порядок|оформлен|инициирован|положение о командиров',title) else 0.
            procedural.append((max(item[2] for item in candidates)+bonus,doc))
        if procedural:
            chosen=max(procedural,key=lambda x:x[0])[1]
            ranked=[(doc,clause,max(item[2] for item in ranked if item[0]['id']==chosen['id'])) for doc,clause in clauses if doc['id']==chosen['id']]

    all_source_items=[(doc,clause,best*.65) for doc,clause in clauses]
    ranked=[item for item in ranked if item[1]['id'] not in parallel_exclusions]
    ranked,all_source_items,ambiguous_conditions,blocked_conditions=_applicable_candidates(ranked,all_source_items,question,parallel_exclusions)
    if not ranked:
        result.update(status='needs_clarification',answer='Найденные действия зависят от условий, которые противоречат описанной ситуации.',followups=['Уточните исходное состояние или добавьте пункт для этого случая.']);return result
    was_many=len(ranked)>48
    ranked,unresolved_refs,candidate_truncated=_source_dependencies(ranked[:48],all_source_items,limit=48,parallel_exclusions=parallel_exclusions)
    ranked,all_source_items,closure_ambiguity,_=_applicable_candidates(ranked,all_source_items,question,parallel_exclusions)
    ambiguous_conditions=list(dict.fromkeys(ambiguous_conditions+closure_ambiguity))
    if was_many or candidate_truncated:
        result['warnings'].append('Найдено много подходящих пунктов; анализ конфликтов ограничен 48 кандидатами. Уточните раздел, чтобы проверить всю процедуру.')
    conflicts=_conflicts(ranked)
    if conflicts:
        conflicting_ids={value for conflict in conflicts for value in [conflict['left'],conflict['right']]}
        selected=[item for item in ranked if item[1]['id'] in conflicting_ids]
        result['status']='conflict';result['answer']='В действующих источниках обнаружено расхождение. Без подтверждения владельца регламента нельзя выбрать один из вариантов.'
        result['answer']+='\n\n'+'\n\n'.join(doc['name']+': «'+clause['text']+'»' for doc,clause,_ in selected)
        result['conflicts']=conflicts;result['warnings'].extend(sorted({conflict['reason'] for conflict in conflicts}))
        result['followups']=['Попросите владельца документов определить применимый пункт или опубликовать заменяющую редакцию.']
    else:
        result['status']='answered'
        max_steps=max(1,min(10,int(options.get('max_steps',10 if howto else 6))))
        selected,missing_refs,selection_truncated=_source_dependencies(ranked[:max_steps],all_source_items,limit=10,parallel_exclusions=parallel_exclusions)
        selected,_,selected_ambiguity,_=_applicable_candidates(selected,all_source_items,question,parallel_exclusions)
        ambiguous_conditions=list(dict.fromkeys(ambiguous_conditions+selected_ambiguity))
        if len(selected)>10:selected=selected[:10];selection_truncated=True
        omitted_procedure=bool(howto and {clause['id'] for doc,clause,score in ranked}-{clause['id'] for doc,clause,score in selected})
        # This diagnostic means a real source dependency was not resolved, not
        # that the model is merely uncertain about a similarity score.
        if missing_refs or selection_truncated or omitted_procedure or was_many or candidate_truncated or ambiguous_conditions:
            result['status']='needs_clarification'
            result['coverage']=dict(complete=False,unresolved_references=missing_refs,truncated=bool(selection_truncated or omitted_procedure or was_many or candidate_truncated),unconfirmed_conditions=ambiguous_conditions)
            result['answer']='Найдены подтверждённые фрагменты, но полный порядок с необходимыми условиями не удалось восстановить.'
            if missing_refs:result['warnings'].append('В выбранном пункте есть ссылка на отсутствующий или неоднозначный пункт. Нельзя пропускать эту зависимость.')
            if selection_truncated or omitted_procedure:result['warnings'].append('Полная процедура с условиями не помещается в краткий ответ; уточните её этап.')
            if ambiguous_conditions:
                result['warnings'].append('Для выбора действий не хватает явно подтверждённых условий. Неизвестное условие нельзя считать выполненным или невыполненным.')
                for identifier in ambiguous_conditions[:3]:
                    candidate=next((clause for doc,clause,score in all_source_items if clause['id']==identifier),None)
                    condition=parse_condition(candidate['text']) if candidate else None
                    if condition:result['followups'].append('Уточните, выполнено ли условие: «'+condition.text+'».')
            result['followups'].append('Уточните этап процедуры или добавьте недостающий пункт исходного документа.')
        # Preserve original order within each document; no invented chronology.
        order={doc['id']:i for i,doc in enumerate(active)}
        selected.sort(key=lambda item:(order[item[0]['id']],int(item[1].get('paragraph') or 0)))
    seen_sources=set()
    for index,(doc,clause,score) in enumerate(selected,1):
        cid='cite-'+str(index)
        citation=dict(id=cid,clause_id=clause['id'],document_id=doc['id'],document_name=doc['name'],page=clause.get('page'),section=clause.get('section',''),paragraph=clause.get('paragraph'),quote=clause['text'],bbox=clause.get('bbox'),version=doc['metadata'].get('version'),effective_from=doc['metadata'].get('effective_from'),effective_to=doc['metadata'].get('effective_to'))
        evidence=clause.get('evidence') or {}
        for key in ['provenance','ocr_confidence','requires_review','numeric_uncertainty','page_width','page_height','coordinate_system','sheet','row','cell_refs','spreadsheet_cells','formulas_evaluated']:
            if key in clause or key in evidence:citation[key]=clause.get(key,evidence.get(key))
        result['citations'].append(citation)
        if result['status']=='answered':
            spans=[clause['text']] if administration_question else procedural_spans(clause['text'])
            for span in spans:result['steps'].append(dict(id='step-'+str(len(result['steps'])+1),text=span,evidence_ids=[cid]))
        if doc['id'] not in seen_sources:
            seen_sources.add(doc['id']);result['sources'].append(dict(document_id=doc['id'],document_name=doc['name'],title=doc['metadata'].get('title',doc['name']),version=doc['metadata'].get('version'),effective_from=doc['metadata'].get('effective_from'),effective_to=doc['metadata'].get('effective_to'),score=round(score,4)))
            if doc.get('status')=='review':result['warnings'].append(doc['name']+': извлечение документа требует проверки; часть пунктов могла быть не прочитана.')
            if not doc['metadata'].get('effective_from') and not doc['metadata'].get('effective_to'):result['warnings'].append(doc['name']+': даты действия редакции не заданы; актуальность необходимо подтвердить.')
    if result['status']=='answered':result['answer']='\n\n'.join(str(i)+'. '+step['text'] for i,step in enumerate(result['steps'],1))
    selected_ids={doc['id'] for doc,clause,score in selected}
    if result['status']=='answered' and len(operation_scope.goals)>1:
        selected_scope=ground_operations(question,[(doc,clause) for doc,clause,score in selected],include_administrative=administration_question)
        if selected_scope.missing:
            result.update(status='needs_clarification',steps=[],answer='Найденные фрагменты не подтверждают все запрошенные операции. Уточните действие или добавьте недостающий порядок.')
            result['coverage']=dict(complete=False,missing_operations=[goal.text for goal in selected_scope.missing])
            result['followups'].append('Не подтверждены операции: '+', '.join(goal.text for goal in selected_scope.missing)+'.')
    applicable_issues=[issue for issue in version_issues if selected_ids&set(issue['document_ids'])]
    result['warnings'].extend(issue['message'] for issue in applicable_issues)
    ambiguous_version=bool(selected_ids&invalid_versions)
    explicit_current=bool(re.search(r'(?i)действующ\w*|актуальн\w*|текущ\w* редакц|\bсегодня\b|\bсейчас\b',request_text(question)))
    missing_validity=explicit_current and any(not doc['metadata'].get('effective_from') and not doc['metadata'].get('effective_to') for doc,clause,score in selected)
    uncertain_source=any(clause.get('requires_review') or clause.get('numeric_uncertainty') or clause.get('provenance') in {'ocr','pdf_text+ocr'} or (clause.get('evidence') or {}).get('requires_review') or (clause.get('evidence') or {}).get('numeric_uncertainty') or doc.get('extraction',{}).get('unreadable_pages') or doc.get('extraction',{}).get('requires_review') for doc,clause,score in selected)
    if (ambiguous_version or uncertain_source or missing_validity) and result['status']!='conflict':
        result['status']='needs_clarification';result['steps']=[]
        result['coverage']=dict(complete=False,version_ambiguity=ambiguous_version,source_requires_review=bool(uncertain_source),unknown_effective_dates=bool(missing_validity))
        result['answer']='Найдены возможные основания, но применять их как подтверждённый порядок пока нельзя.'
        if ambiguous_version:result['followups'].append('Проверьте связи замены редакций и укажите применимый документ.')
        if missing_validity:result['followups'].append('Подтвердите даты действия редакции: без них нельзя объявить норму действующей на выбранную дату.')
        if uncertain_source:
            result['warnings'].append('Выбранный источник распознан OCR или прочитан не полностью. Требуется проверка исходной страницы и чисел.')
            result['followups'].append('Подтвердите распознанные пункты по оригиналу или загрузите полный текстовый документ.')
    result['score']=result['confidence']=round(max(item[2] for item in selected),4) if selected else 0.
    result['warnings']=list(dict.fromkeys(result['warnings']))
    return result


GENERIC_SCOPE_TERMS={'заявк','заявл','день','рабоч','календар','срок','соглас','исполн','рассмотр','руководител','сотрудник','работник','времен','врем','пункт','редакц','верси','правил','поряд','инструк','документ','систем','данн','свед','получ','указ','внутренн','служебн'}

def _specific_document_scope(question,clauses):
    """Require the question's rare supported subject in every eligible source.

    Global vocabulary support alone is insufficient: a rare procedure qualifier
    may exist in one upload while unrelated regulations share 'request/days'.
    Subject frequencies are calculated per document and this gate runs separately
    for each explicit question facet, before semantic similarity can lift a hit.
    """
    doc_terms={}
    for doc,clause in clauses:
        doc_terms.setdefault(doc['id'],set()).update(lexical_vocabulary(str(doc['metadata'].get('title',''))+' '+str(clause.get('section',''))+' '+clause['text']))
    if len(doc_terms)<2:return clauses
    candidates={}
    for word,key in query_subjects(question):
        if len(key)<4 or key in GENERIC_SCOPE_TERMS:continue
        supporting={identifier for identifier,vocabulary in doc_terms.items() if any(key==value or (min(len(key),len(value))>=4 and (key.startswith(value) or value.startswith(key))) for value in vocabulary)}
        if supporting:candidates[key]=supporting
    if not candidates:return clauses
    rarest=min(len(supporting) for supporting in candidates.values())
    if rarest==len(doc_terms):return clauses
    anchors=[supporting for supporting in candidates.values() if len(supporting)==rarest]
    # Requiring all equally specific qualifiers avoids silently dropping one
    # half of a single procedure name. Explicit multiple topics use facets.
    allowed=set.intersection(*anchors)
    return [(doc,clause) for doc,clause in clauses if doc['id'] in allowed]

def _unsupported_entities(question,vocabulary,validated_context=()):
    """Fail closed on an unsupported specific entity, not a curated topic list.

    Functional query words and verbs may differ from the source. An unmatched
    noun/adjective naming a subject cannot borrow a number from the nearest
    generic policy (e.g. a parking request cannot use a hotel allowance).
    """
    absent=[]
    # Every explicit material referent in the discourse binds the answer.
    # A later noun such as 'стоимость' or 'заявка' cannot replace the object
    # introduced in a previous sentence. Actors and predicates are filtered by
    # grammatical analysis, rather than by discarding preceding sentences.
    for word,key in query_subjects(question):
        if key in vocabulary or key in GENERIC_SCOPE_TERMS or key in validated_context or len(key)<4:continue
        if not any(key.startswith(value) or value.startswith(key) for value in vocabulary if len(value)>=4):absent.append(word)
    return list(dict.fromkeys(absent))[:5]

def _question_facets(question):
    clean=normalize(question)
    # Reuse the interrogative frame for coordinated nominal stages, so a
    # second facet is not misread as a newly asserted user fact.
    deadlines=re.match(r'^(.*?)(?:(?:какие|каковы)\s+|чем\s+отличаются\s+)?сроки\s+(.+)$',clean)
    deadline_words=re.findall(r'[а-яa-z]+',deadlines.group(2)) if deadlines else []
    if deadlines and deadline_words and morphology(deadline_words[0])[1]!='INFN' and not re.search(r'\bесли\b|при условии',deadlines.group(2)):
        tails=re.split(r'\s+и\s+',deadlines.group(2),maxsplit=2)
        if len(tails)>1 and all(any(morphology(w)[1]=='NOUN' for w in re.findall(r'[а-яa-z]+',tail)) for tail in tails):
            return [(deadlines.group(1)+'Какие сроки '+tail.rstrip(' ?')+'?').strip() for tail in tails]
    # Multiple explicit objects of one requested action are separate facets.
    # Conditional conjunctions are never split into independent permissions.
    spans=list(re.finditer(r'[а-яa-z]+',clean))
    infinitives=[span for span in spans if morphology(span.group())[1]=='INFN']
    if infinitives:
        verb=infinitives[-1];tail=clean[verb.end():].strip(' ?')
        objects=re.split(r'\s+и\s+',tail,maxsplit=2)
        if len(objects)>1 and not re.search(r'\bесли\b|при условии',tail) and all(any(morphology(w)[1]=='NOUN' for w in re.findall(r'[а-яa-z]+',part)) and not any(morphology(w)[1] in {'VERB','INFN'} for w in re.findall(r'[а-яa-z]+',part)) for part in objects):
            return [(clean[:verb.end()]+' '+part.rstrip(' ?')+'?').strip() for part in objects]
    parts=re.split(r'\s+и\s+',question,maxsplit=2,flags=re.I)
    if len(parts)<2:return [question]
    comparison=bool(re.search(r'(?i)чем.{0,30}отлич|сравн|различ',question))
    if not comparison and not all(re.search(r'(?i)^(?:как|како|когда|за сколько|что|сколько|нужно ли)',part.strip()) for part in parts[1:]):return [question]
    if any(len(tokens(part))<1 for part in parts):return [question]
    contexts=[word for word in re.findall(r'(?i)командиров\w*|отпуск\w*|закуп\w*|доступ\w*',parts[0])]
    output=[]
    for index,part in enumerate(parts):
        if index and contexts and set(tokens(part))<= {'заявк','заявл','срок','день','подат','подать'}:part+=' '+contexts[0]
        output.append(part.strip())
    return output[:3]

def _answer_facets(question,facets,documents,as_of,role,options):
    result=_base(question=question,as_of=as_of,role=role)
    child_options={**options,'_single_facet':True,'_semantic_limit':60}
    children=[answer(facet,documents,as_of,role,child_options) for facet in facets]
    result['facets']=[dict(question=facet,status=child['status'],answer=child['answer']) for facet,child in zip(facets,children)]
    if any(child['status']=='conflict' for child in children):result['status']='conflict'
    elif all(child['status']=='answered' for child in children):result['status']='answered'
    elif any(child['status']=='answered' for child in children):result['status']='needs_clarification'
    elif any(child['status']=='needs_clarification' for child in children):result['status']='needs_clarification'
    else:result['status']='not_found'
    citation_map={};source_ids=set();excluded_ids=set();step_texts=set()
    for child in children:
        remap={}
        for citation in child['citations']:
            key=citation['clause_id']
            if key not in citation_map:
                value={**citation,'id':'cite-'+str(len(citation_map)+1)};citation_map[key]=value;result['citations'].append(value)
            remap[citation['id']]=citation_map[key]['id']
        if result['status']!='conflict':
            for step in child['steps']:
                if step['text'] not in step_texts:
                    step_texts.add(step['text']);result['steps'].append(dict(id='step-'+str(len(result['steps'])+1),text=step['text'],evidence_ids=[remap[e] for e in step['evidence_ids']]))
            for fact in child.get('facts',[]):
                if fact['text'] not in step_texts:
                    step_texts.add(fact['text']);result.setdefault('facts',[]).append(dict(text=fact['text'],evidence_ids=[remap[e] for e in fact['evidence_ids']]))
        for source in child['sources']:
            if source['document_id'] not in source_ids:source_ids.add(source['document_id']);result['sources'].append(source)
        for excluded in child['excluded']:
            if excluded['document_id'] not in excluded_ids:excluded_ids.add(excluded['document_id']);result['excluded'].append(excluded)
        result['warnings'].extend(child['warnings']);result['followups'].extend(child['followups'])
    if result['status']=='answered':result['answer']='\n\n'.join(str(i)+'. '+item['text'] for i,item in enumerate(result['steps']+result.get('facts',[]),1))
    elif result['status']=='conflict':result['answer']='\n\n'.join(child['answer'] for child in children if child['status']=='conflict')
    else:
        result['answer']='Не для всех частей вопроса найдено подтверждённое правило. Уточните недостающую часть или добавьте источник.'
        result['warnings'].append('Нельзя считать найденные пункты полным ответом на составной вопрос.')
    scores=[child['score'] for child in children if child['citations']]
    result['score']=result['confidence']=round(min(scores),4) if scores else 0.
    result['warnings']=list(dict.fromkeys(result['warnings']));result['followups']=list(dict.fromkeys(result['followups']))
    return result
