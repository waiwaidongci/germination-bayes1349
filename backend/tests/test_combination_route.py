"""合计路由端到端测试：SQLite临时库 + 冻结历史运行，覆盖成功、重复、容量、未完成与未登录。"""
import os
os.environ['DATABASE_URL']='sqlite:////tmp/combination-route-test.db'
os.environ['SEED_DEMO']='false'
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.config.database import Base,engine,SessionLocal
from src.models import User,BayesRun
from src.utils.security import hash_password
from src.services.analysis_service import analyze_germination
from src.main import app

def build_run():
    rng=np.random.default_rng(320);ns=rng.integers(20,181,size=28);ps=rng.beta(8,4,size=28);ks=rng.binomial(ns,ps)
    lots=[{'id':i+1,'study_id':1,'code':f'DC-01-L{i+1:02}','name':f'独立种子批次 {i+1:02}','origin':'测试'} for i in range(28)]
    counts=[{'id':i+1,'lot_id':i+1,'n':int(ns[i]),'k':int(ks[i]),'note':''} for i in range(28)]
    snapshot={'study':{'id':1,'code':'DC-01','name':'历史研究','description':'','source':''},'lots':lots,'counts':counts}
    return snapshot,analyze_germination(snapshot,.7,50)

@pytest.fixture(scope='module')
def client():
    Base.metadata.create_all(engine)
    snapshot,result=build_run()
    with SessionLocal() as db:
        if not db.query(User).filter_by(email='analyst@duocen.demo').first():db.add(User(email='analyst@duocen.demo',name='许禾 · 批次分析员',role='analyst',password_hash=hash_password('Duocen2026!')))
        if not db.get(BayesRun,1):db.add(BayesRun(id=1,name='历史冻结研究',study_id=1,threshold=.7,future_n=50,input_snapshot=snapshot,status='completed',result=result,owner='许禾 · 批次分析员'))
        if not db.get(BayesRun,2):db.add(BayesRun(id=2,name='排队中研究',study_id=1,threshold=.7,future_n=50,input_snapshot=snapshot,status='queued',result={},owner='许禾 · 批次分析员'))
        db.commit()
    with TestClient(app) as c:
        token=c.post('/auth/login',json={'email':'analyst@duocen.demo','password':'Duocen2026!'}).json()['token']
        c.headers['Authorization']='Bearer '+token
        yield c

def body(pairs,target=100):return {'selections':[{'lot_id':l,'future_n':m} for l,m in pairs],'target':target}

def test_combine_success(client):
    r=client.post('/runs/1/combinations',json=body([(1,50),(2,60),(3,40)]))
    assert r.status_code==200,r.text
    out=r.json();total=out['total']
    assert out['run_id']==1 and len(out['components'])==3 and total['future_n']==150
    assert len(total['pmf'])==151 and 0<=total['lower']<=total['upper']<=150
    assert 0<total['prob_reach_target']<1
    assert out['fit_hash'] and out['selection_hash']

def test_combine_duplicate_rejected(client):
    r=client.post('/runs/1/combinations',json=body([(1,50),(1,60)]))
    assert r.status_code==422 and '重复' in r.json()['detail']

def test_combine_capacity_rejected(client):
    r=client.post('/runs/1/combinations',json=body([(i+1,200) for i in range(11)]))
    assert r.status_code==422 and '容量超限' in r.json()['detail']

def test_combine_unknown_lot_rejected(client):
    r=client.post('/runs/1/combinations',json=body([(1,50),(999,60)]))
    assert r.status_code==422 and '冻结结果' in r.json()['detail']

def test_combine_unfinished_run_rejected(client):
    r=client.post('/runs/2/combinations',json=body([(1,50),(2,60)]))
    assert r.status_code==409

def test_combine_requires_login():
    with TestClient(app) as anon:
        assert anon.post('/runs/1/combinations',json=body([(1,50),(2,60)])).status_code==401

def test_combine_schema_validation(client):
    assert client.post('/runs/1/combinations',json=body([(1,50)])).status_code==422
    assert client.post('/runs/1/combinations',json=body([(1,50),(2,60)],target=-1)).status_code==422
    assert client.post('/runs/1/combinations',json=body([(1,50),(2,201)])).status_code==422
