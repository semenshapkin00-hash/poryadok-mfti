"""Experimental, local clause-ID selector guard. Not enabled by the web app.

This module imports no model, performs no network I/O, and accepts no generated
factual prose. Eligibility and explicit references come from parsed source
metadata/text, never benchmark answers or model-provided dependency labels.
"""
from __future__ import annotations
import copy,json,re
from datetime import date

STATUSES={'answered','not_found','conflict','needs_clarification'}
SCHEMA_VERSION='local-selector-1'
MAX_CLAUSES=5000
MAX_SELECTED=128
ROLE_ALIASES={'сотрудник':'employee','работник':'employee','employee':'employee','руководитель':'manager','manager':'manager','финансы':'finance','finance':'finance','все':'all','all':'all'}
REFERENCE=re.compile(r'(?i)(?:\bпункт[а-яё]*|\bпп?\.)\s*((?:\d+(?:\.\d+)*)(?:\s*(?:,|и)\s*\d+(?:\.\d+)*)*)')
NUMBER=re.compile(r'^\s*(\d+(?:\.\d+)*)[.)]?\s+')


def _role(value):return ROLE_ALIASES.get(str(value or '').strip().lower(),str(value or '').strip().lower())

def _date(value):
    if value in (None,''):return None
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):raise ValueError('Date must be explicit ISO YYYY-MM-DD.')
    return date.fromisoformat(value)


def _list(value):
    if value in (None,''):return []
    return value if isinstance(value,list) else [value]


def build_envelope(question,documents,as_of=None,role=None):
    """Serialize complete parsed procedures; do not rank or hide competing rules."""
    if not isinstance(question,str) or not question.strip() or len(question)>2000:raise ValueError('Question must contain1..2000 characters.')
    if not isinstance(documents,list) or not 1<=len(documents)<=50:raise ValueError('Use1..50 parsed documents.')
    ids=[str(doc.get('id')) for doc in documents]
    if len(set(ids))!=len(ids):raise ValueError('Document IDs must be unique.')
    selected_date=_date(as_of);selected_role=_role(role);doc_rows=[];by_id={};clauses=[]
    for source in documents:
        identifier=str(source['id']);metadata=copy.deepcopy(source.get('metadata') or {});excluded=[];review=[]
        try:
            start=_date(metadata.get('effective_from'));end=_date(metadata.get('effective_to'))
            if start and end and end<start:raise ValueError('Inverted validity interval')
            if selected_date and start and selected_date<start:excluded.append('not_yet_effective')
            if selected_date and end and selected_date>end:excluded.append('expired')
            if selected_date is None:review.append('as_of_not_specified')
            if start is None:review.append('effective_from_not_specified')
        except ValueError:excluded.append('invalid_effective_date')
        roles=[_role(value) for value in _list(metadata.get('roles'))]
        if selected_role and roles and 'all' not in roles and selected_role not in roles:excluded.append('role_mismatch')
        elif not selected_role and roles and 'all' not in roles:review.append('role_not_specified')
        if source.get('status')=='unreadable' or not source.get('clauses'):excluded.append('unreadable')
        extraction=source.get('extraction') or {}
        if extraction.get('unreadable_pages'):review.append('unreadable_pages')
        row=dict(id=identifier,name=source.get('name',identifier),metadata=metadata,eligible=not excluded,exclusion_reasons=excluded,review_reasons=review)
        doc_rows.append(row);by_id[identifier]=row
    # Only an active, explicitly declared replacement may replace another doc.
    graph={row['id']:[] for row in doc_rows if row['eligible']}
    for row in doc_rows:
        if not row['eligible']:continue
        for target in _list(row['metadata'].get('supersedes')):
            target=str(target)
            if target not in by_id:row['review_reasons'].append('missing_replacement_target:'+target);continue
            other=by_id[target]
            left=str(row['metadata'].get('topic') or '').casefold();right=str(other['metadata'].get('topic') or '').casefold()
            if left and right and left!=right:
                row['review_reasons'].append('replacement_scope_mismatch:'+target);other['review_reasons'].append('replacement_scope_mismatch:'+row['id']);continue
            if other['eligible']:graph[row['id']].append(target)
    cyclic=set();colors={};stack=[]
    def visit(node):
        if colors.get(node)==1:cyclic.update(stack[stack.index(node):]);return
        if colors.get(node)==2:return
        colors[node]=1;stack.append(node)
        for target in graph.get(node,[]):visit(target)
        stack.pop();colors[node]=2
    for node in graph:visit(node)
    for node in cyclic:by_id[node]['review_reasons'].append('replacement_cycle')
    for origin,targets in graph.items():
        if origin in cyclic or by_id[origin]['review_reasons']:continue
        for target in targets:
            if target not in cyclic:
                by_id[target]['eligible']=False;by_id[target]['exclusion_reasons'].append('superseded_by:'+origin)
    for source in documents:
        doc=by_id[str(source['id'])]
        for original in source.get('clauses',[]):
            if len(clauses)>=MAX_CLAUSES:raise ValueError('Whole-procedure context exceeds5000 clauses; no truncation allowed.')
            text=original.get('text')
            if not isinstance(text,str) or not text or len(text)>12000:raise ValueError('Invalid or oversized source clause.')
            excluded=list(doc['exclusion_reasons'])
            if original.get('unsafe'):excluded.append('source_prompt_injection')
            if original.get('kind')=='notice':excluded.append('non_normative_notice')
            evidence=original.get('evidence') or {}
            ocr=original.get('requires_review') or evidence.get('requires_review') or original.get('provenance') in {'ocr','pdf_text+ocr'}
            clauses.append(dict(id=str(original['id']),document_id=doc['id'],text=text,section=original.get('section'),paragraph=original.get('paragraph'),page=original.get('page'),
                eligible=not excluded,exclusion_reasons=excluded,requires_review=bool(ocr or doc['review_reasons']),review_reasons=list(doc['review_reasons'])+(['unverified_ocr'] if ocr else []),requires=[],unresolved_references=[]))
    lookup={clause['id']:clause for clause in clauses}
    if len(lookup)!=len(clauses):raise ValueError('Clause IDs must be globally unique.')
    numbered={}
    for clause in clauses:
        match=NUMBER.match(clause['text'])
        if match:numbered.setdefault((clause['document_id'],match.group(1)),[]).append(clause['id'])
    for clause in clauses:
        references=[]
        for group in REFERENCE.findall(clause['text']):references.extend(re.findall(r'\d+(?:\.\d+)*',group))
        for reference in dict.fromkeys(references):
            candidates=numbered.get((clause['document_id'],reference),[])
            if len(candidates)==1:
                if candidates[0]!=clause['id']:clause['requires'].append(candidates[0])
            else:clause['unresolved_references'].append(reference)
    return dict(schema_version=SCHEMA_VERSION,question=question,context=dict(as_of=as_of,role=role),documents=doc_rows,clauses=clauses)


def parse_selection(raw):
    def unique(pairs):
        output={}
        for key,value in pairs:
            if key in output:raise ValueError('Duplicate JSON key: '+key)
            output[key]=value
        return output
    parsed=json.loads(raw,object_pairs_hook=unique)
    if not isinstance(parsed,dict) or set(parsed)!={'status','clause_ids'}:raise ValueError('Only status and clause_ids are accepted.')
    if not isinstance(parsed['status'],str) or parsed['status'] not in STATUSES:raise ValueError('Invalid status.')
    ids=parsed['clause_ids']
    if not isinstance(ids,list) or len(ids)>MAX_SELECTED or any(not isinstance(value,str) for value in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid clause IDs.')
    if parsed['status']=='answered' and not ids:raise ValueError('Answered requires source clauses.')
    if parsed['status']=='not_found' and ids:raise ValueError('not_found must not select clauses.')
    return parsed


def validate_selection(selection,envelope):
    """Reject invalid scope/dependencies independently of the model decision."""
    lookup={clause['id']:clause for clause in envelope['clauses']};selected=set(selection['clause_ids'])
    for identifier in selected:
        clause=lookup.get(identifier)
        if clause is None:raise ValueError('Unknown clause ID: '+identifier)
        if not clause['eligible']:raise ValueError('Ineligible source: '+identifier+' ('+', '.join(clause['exclusion_reasons'])+')')
        missing=set(clause['requires'])-selected
        if missing:raise ValueError('Missing explicit reference dependencies: '+', '.join(sorted(missing)))
        if selection['status']=='answered' and clause['unresolved_references']:raise ValueError('Unresolved source reference: '+identifier)
        if selection['status']=='answered' and clause['requires_review']:raise ValueError('Unverified date, role, version, OCR or completeness: '+identifier+' ('+', '.join(clause['review_reasons'])+')')
    return selection


def render_selection(selection,envelope,parsed_documents):
    """Exact source renderer only. Caller must validate selection first.

    This experimental result is NOT a proof of semantic completeness and is not
    routed to the production API; the existing independent API guard is intact.
    """
    status=selection['status'];lookup={clause['id']:(doc,clause) for doc in parsed_documents for clause in doc.get('clauses',[])}
    citations=[];steps=[];warnings=[]
    for identifier in selection['clause_ids']:
        if identifier not in lookup:raise ValueError('Cannot render unknown source: '+identifier)
        doc,clause=lookup[identifier];evidence=copy.deepcopy(clause.get('evidence') or {})
        cid='q'+str(len(citations)+1)
        evidence.update(id=cid,clause_id=identifier,document_id=doc['id'],document_name=doc['name'],quote=clause['text'])
        for key in ('page','section','paragraph','bbox','provenance','requires_review','ocr_confidence','numeric_uncertainty','coordinate_system','word_confidences'):
            if key in clause:evidence[key]=copy.deepcopy(clause[key])
        citations.append(evidence);steps.append(dict(id='s'+str(len(steps)+1),text=clause['text'],evidence_ids=[cid]))
    texts={'answered':'Выбраны дословные пункты источников.','not_found':'В предоставленных источниках не найден подтверждённый порядок для предмета вопроса.','conflict':'Выбраны несовместимые требования; нужно уточнить применимый порядок.','needs_clarification':'Нужно уточнить условия, применимость или полноту источников; не считайте выбранные пункты готовой инструкцией.'}
    return dict(status=status,answer=texts[status],steps=steps,citations=citations,sources=[dict(document_id=doc['id'],document_name=doc['name']) for doc in parsed_documents if any(c['document_id']==doc['id'] for c in citations)],
        excluded=[dict(document_id=doc['id'],document_name=doc['name'],reason='; '.join(doc['exclusion_reasons'])) for doc in envelope['documents'] if not doc['eligible']],warnings=warnings,followups=[],experimental=True,semantic_completeness_verified=False)
