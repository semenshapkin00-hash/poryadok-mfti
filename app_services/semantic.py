"""Offline ONNX multilingual sentence embeddings; never downloads at inference."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import hashlib,importlib.util,json,threading
import numpy as np

ROOT=Path(__file__).resolve().parent.parent/'models/semantic-minilm'
LOCK=threading.Lock();_session=None;_tokenizer=None
EXPECTED={'model_qint8_arm64.onnx':'783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672','tokenizer.json':'2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8'}

def status():
    available=all((ROOT/f).is_file() for f in ['model_qint8_arm64.onnx','tokenizer.json','config.json','manifest.json']) and all(importlib.util.find_spec(module) is not None for module in ['onnxruntime','tokenizers'])
    return dict(available=available,name='Multilingual MiniLM-L12 · ONNX int8',backend='CPUExecutionProvider',offline=True,loaded=_session is not None,license='Apache-2.0')

def load():
    global _session,_tokenizer
    if _session is not None:return
    if not status()['available']:raise RuntimeError('Локальная семантическая модель не установлена.')
    for name,digest in EXPECTED.items():
        with (ROOT/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise RuntimeError('Нарушена целостность локальной модели: '+name)
    import onnxruntime as ort
    ort.disable_telemetry_events()
    from tokenizers import Tokenizer
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session=ort.InferenceSession(str(ROOT/'model_qint8_arm64.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    _tokenizer=Tokenizer.from_file(str(ROOT/'tokenizer.json'))
    cfg=json.loads((ROOT/'config.json').read_text());pad_id=int(cfg.get('pad_token_id',1));pad=_tokenizer.id_to_token(pad_id)
    _tokenizer.enable_truncation(max_length=128);_tokenizer.enable_padding(pad_id=pad_id,pad_token=pad)

@lru_cache(maxsize=2048)
def embedding(text):
    with LOCK:
        load();item=_tokenizer.encode(str(text)[:2000]);ids=np.asarray([item.ids],dtype=np.int64);mask=np.asarray([item.attention_mask],dtype=np.int64)
        available={'input_ids':ids,'attention_mask':mask,'token_type_ids':np.asarray([item.type_ids],dtype=np.int64)}
        inputs={x.name:available[x.name] for x in _session.get_inputs()}
        output=_session.run(None,inputs)[0]
        if output.ndim==3:vector=(output[0]*mask[0,:,None]).sum(0)/max(mask.sum(),1)
        else:vector=output[0]
        vector=np.asarray(vector,dtype=np.float32);vector=vector/max(float(np.linalg.norm(vector)),1e-12)
        if not np.isfinite(vector).all():raise ValueError('Некорректный вектор локальной модели.')
        return vector

def similarity_matrix(left,right):
    if not left or not right:return np.zeros((len(left),len(right)),dtype=np.float32)
    a=np.stack([embedding(str(t)) for t in left]);b=np.stack([embedding(str(t)) for t in right]);return np.clip(a@b.T,-1.,1.)
