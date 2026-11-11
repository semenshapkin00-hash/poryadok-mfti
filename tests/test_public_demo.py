"""Public-demo isolation and independent factual grounding; synthetic data only."""
import copy
import importlib
import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def demo(tmp_path,monkeypatch):
    api=importlib.import_module('app')
    for name,folder in [('DATA',tmp_path),('UPLOADS',tmp_path/'uploads'),
                        ('LIBRARY',tmp_path/'documents'),('ANSWERS',tmp_path/'answers')]:
        folder.mkdir(parents=True,exist_ok=True);monkeypatch.setattr(api,name,folder)
    monkeypatch.setattr(api,'PUBLIC_DEMO',True)
    monkeypatch.setattr(api,'DEMO_ANSWERS',{})
    semantic=importlib.import_module('app_services.semantic')
    monkeypatch.setattr(semantic,'status',lambda:dict(available=False))
    with TestClient(api.app) as client:yield client,api


def test_public_demo_blocks_upload_and_exposes_mode(demo):
    client,api=demo
    health=client.get('/api/health').json()
    assert health['features']['public_demo'] and not health['features']['uploads_enabled']
    assert health['processing']=='demo_server' and '.xlsx' in health['supported_formats']
    r=client.post('/api/documents',files={'files':('example.md',b'fictional','text/markdown')})
    assert r.status_code==403 and not list(api.LIBRARY.iterdir())


def test_public_demo_never_reads_disk_answer_history(demo):
    client,api=demo;identifier='a'*32
    (api.ANSWERS/(identifier+'.json')).write_text(json.dumps({'question':'private synthetic sentinel'}))
    assert client.get('/api/history').json()==[]
    assert client.get('/api/answers/'+identifier).status_code==404


def test_public_demo_rejects_accidental_nonsynthetic_corpus(demo):
    client,api=demo;identifier='b'*32
    (api.LIBRARY/(identifier+'.json')).write_text(json.dumps({'id':identifier,'metadata':{}}))
    assert client.get('/api/documents').status_code==503
    assert client.get('/api/files/'+identifier).status_code==404


def fact_fixture(api):
    from navigator_engine import ingest,DocumentInput
    doc=ingest(DocumentInput('c'*32,'fictional.md','Код учебного подразделения: 041.'.encode(),{'synthetic':True}))
    clause=doc['clauses'][0]
    citation={'id':'cite-1','clause_id':clause['id'],'document_id':doc['id'],
              'document_name':doc['name'],'quote':clause['text']}
    result={'status':'answered','answer':'1. '+clause['text'],'steps':[],
            'facts':[{'text':clause['text'],'evidence_ids':['cite-1']}],
            'citations':[citation],'sources':[],'warnings':[],'followups':[]}
    return doc,result


def test_literal_facts_grounded_without_invented_actions(demo):
    _,api=demo;doc,result=fact_fixture(api)
    assert api.verify_grounding(result,[doc])
    altered=copy.deepcopy(result);altered['facts'][0]['text']='Код: 999.'
    altered['answer']='1. Код: 999.'
    with pytest.raises(ValueError):api.verify_grounding(altered,[doc])
    altered=copy.deepcopy(result);altered['answer']='1. Любой неподтверждённый вывод.'
    with pytest.raises(ValueError):api.verify_grounding(altered,[doc])


def test_demo_answers_ephemeral_bounded_and_expire(demo,monkeypatch):
    client,api=demo;doc,result=fact_fixture(api)
    (api.LIBRARY/(doc['id']+'.json')).write_text(json.dumps(doc,ensure_ascii=False))
    engine=importlib.import_module('navigator_engine')
    monkeypatch.setattr(engine,'answer',lambda *a,**kw:copy.deepcopy(result))
    identifiers=[]
    for _ in range(129):
        response=client.post('/api/ask',json={'question':'Какой код учебного подразделения?'})
        assert response.status_code==200,response.text
        identifiers.append(response.json()['id'])
    assert len(api.DEMO_ANSWERS)==128 and not list(api.ANSWERS.iterdir())
    assert client.get('/api/answers/'+identifiers[0]).status_code==404
    assert client.get('/api/answers/'+identifiers[-1]+'/export?format=json').status_code==200
    api.DEMO_ANSWERS[identifiers[-1]]['created_at']=time.time()-1801
    assert client.get('/api/answers/'+identifiers[-1]).status_code==404
    assert client.get('/api/history').json()==[]
