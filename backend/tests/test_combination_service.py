"""合计服务测试：历史研究沿用冻结结果、重复批次与容量超限拒绝、边界夹具稳健。"""
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from src.services.analysis_service import analyze_germination
from src.services.combination_service import combine_frozen_run
from src.schemas.run import CombinationInput,CombinationSelection

def make_snapshot(mode='random'):
    rng=np.random.default_rng(320);ns=rng.integers(20,181,size=28);ps=rng.beta(8,4,size=28);ks=rng.binomial(ns,ps)
    if mode=='flat':ns[:]=100;ks[:]=70
    lots=[{'id':i+1,'study_id':1,'code':f'DC-01-L{i+1:02}','name':f'独立种子批次 {i+1:02}','origin':'铎岑种质试验室 / 合成批次'} for i in range(28)]
    counts=[{'id':i+1,'lot_id':i+1,'n':int(ns[i]),'k':int(ks[i]),'note':'历史冻结计数'} for i in range(28)]
    return {'study':{'id':1,'code':'DC-01','name':'异质批次 · 萌发观察','description':'历史研究夹具','source':'duocen-germination-lots-202611.csv'},'lots':lots,'counts':counts}

def frozen_run(mode='random',status='completed'):
    return SimpleNamespace(id=7,name='历史冻结研究',status=status,future_n=50,result=analyze_germination(make_snapshot(mode),.7,50) if status=='completed' else {})

def select(*pairs):return [CombinationSelection(lot_id=l,future_n=m) for l,m in pairs]

def test_frozen_result_immune_to_source_change():
    run=frozen_run();selections=select((1,50),(2,60),(3,40))
    first=combine_frozen_run(run,selections,100)
    changed=make_snapshot();changed['counts'][0]['k']=changed['counts'][0]['n'];changed['counts'][1]['k']=0
    refit=analyze_germination(changed,.7,50)
    assert refit['lots'][0]['post_alpha']!=run.result['lots'][0]['post_alpha'],"夹具前提：来源计数确实改动"
    second=combine_frozen_run(run,selections,100)
    assert second['selection_hash']==first['selection_hash']
    assert second['total']==first['total']
    assert second['components'][0]['post_alpha']==run.result['lots'][0]['post_alpha']
    assert second['fit_hash']==run.result['fit_hash'],"已有研究必须沿用冻结结果，不受来源改动影响"

def test_duplicate_lot_rejected():
    with pytest.raises(HTTPException) as e:combine_frozen_run(frozen_run(),select((1,50),(1,60)),50)
    assert e.value.status_code==422 and '重复' in str(e.value.detail)

def test_unknown_lot_rejected():
    with pytest.raises(HTTPException) as e:combine_frozen_run(frozen_run(),select((1,50),(999,60)),50)
    assert e.value.status_code==422 and '冻结结果' in str(e.value.detail)

def test_capacity_overflow_rejected():
    with pytest.raises(HTTPException) as e:combine_frozen_run(frozen_run(),select(*[(i+1,200) for i in range(11)]),100)
    assert e.value.status_code==422 and '容量超限' in str(e.value.detail)

def test_unfinished_run_rejected():
    with pytest.raises(HTTPException) as e:combine_frozen_run(frozen_run(status='queued'),select((1,50),(2,60)),50)
    assert e.value.status_code==409

def test_target_beyond_total_rejected():
    with pytest.raises(HTTPException) as e:combine_frozen_run(frozen_run(),select((1,50),(2,60)),111)
    assert e.value.status_code==422 and '目标萌发数' in str(e.value.detail)

def test_response_hashes_components_and_shares():
    run=frozen_run();out=combine_frozen_run(run,select((3,50),(1,60),(2,40)),90)
    assert out['fit_hash']==run.result['fit_hash'] and out['input_hash']==run.result['input_hash']
    assert [c['lot_id'] for c in out['components']]==[3,1,2],"保持选择顺序"
    assert sum(c['mean_share'] for c in out['components'])==pytest.approx(1.,abs=1e-12)
    assert out['total']['mean']==pytest.approx(sum(c['mean'] for c in out['components']),rel=1e-15)
    again=combine_frozen_run(run,select((3,50),(1,60),(2,40)),90)
    assert again['selection_hash']==out['selection_hash'],"同一冻结结果与选择必须可复现"

def test_boundary_fixture_still_combines():
    out=combine_frozen_run(frozen_run(mode='flat'),select((1,50),(2,50),(3,50)),120)
    assert out['components'][0]['post_alpha']>5000,"DC-02夹具α贴上界，合计仍须数值稳健"
    total=out['total'];assert total['normalization_error']<=1e-7
    assert 0<=total['lower']<=total['upper']<=150
    assert sum(p['probability'] for p in total['pmf'])==pytest.approx(1.,abs=1e-12)

def test_schema_limits_and_strict_types():
    with pytest.raises(ValidationError):CombinationInput(selections=[{'lot_id':1,'future_n':50}],target=10)
    with pytest.raises(ValidationError):CombinationInput(selections=[{'lot_id':i,'future_n':50} for i in range(1,42)],target=10)
    with pytest.raises(ValidationError):CombinationInput(selections=[{'lot_id':1,'future_n':201},{'lot_id':2,'future_n':50}],target=10)
    with pytest.raises(ValidationError):CombinationInput(selections=[{'lot_id':1,'future_n':True},{'lot_id':2,'future_n':50}],target=10)
    with pytest.raises(ValidationError):CombinationInput(selections=[{'lot_id':1,'future_n':50},{'lot_id':2,'future_n':50}],target=-1)
    ok=CombinationInput(selections=[{'lot_id':1,'future_n':50},{'lot_id':2,'future_n':60}],target=0)
    assert len(ok.selections)==2
