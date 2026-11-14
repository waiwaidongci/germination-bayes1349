"""合计分布卷积算法测试：暴力枚举对照、矩与归一、等尾的区间口径、极小尾概率log空间可信。"""
import numpy as np
import pytest
from scipy.special import betaln,logsumexp
from src.services.combination_algorithm import beta_binomial_logpmf,convolve_logpmfs,combine_distributions,MAX_COMPONENTS,MAX_TOTAL_FUTURE

def component(m,a,b):return {'future_n':m,'post_alpha':a,'post_beta':b}

def test_convolution_matches_brute_force():
    specs=[(3,2.,3.),(4,1.5,4.),(2,3.,1.)];logpmfs=[beta_binomial_logpmf(m,a,b)[0] for m,a,b in specs]
    brute={}
    for i,li in enumerate(logpmfs[0]):
        for j,lj in enumerate(logpmfs[1]):
            for k,lk in enumerate(logpmfs[2]):brute[i+j+k]=np.logaddexp(brute.get(i+j+k,-np.inf),li+lj+lk)
    combined=convolve_logpmfs(logpmfs)
    assert len(combined)==sum(m for m,_,_ in specs)+1
    for s,lv in brute.items():assert abs(combined[s]-lv)<1e-12,f"s={s}处卷积与暴力枚举不一致"

def test_total_moments_and_normalization():
    comps=[component(50,9.,3.5),component(60,8.2,4.1),component(40,12.,2.8)]
    out=combine_distributions(comps,80);total=out['total'];pmf=total['pmf']
    assert total['future_n']==150
    assert total['mean']==pytest.approx(sum(c['mean'] for c in out['components']),rel=1e-15)
    assert total['variance']==pytest.approx(sum(c['variance'] for c in out['components']),rel=1e-15)
    assert sum(p['probability'] for p in pmf)==pytest.approx(1.,abs=1e-12)
    assert total['normalization_error']<=1e-7
    k=np.array([p['k'] for p in pmf]);p=np.array([x['probability'] for x in pmf])
    assert float(k@p)==pytest.approx(total['mean'],abs=1e-9)
    assert float((k**2)@p-total['mean']**2)==pytest.approx(total['variance'],abs=1e-6)

def test_equal_tail_interval_from_combined_cdf():
    out=combine_distributions([component(50,9.,3.5),component(60,8.2,4.1)],70)['total']
    cdf=np.cumsum([p['probability'] for p in out['pmf']]);lo,hi=out['lower'],out['upper']
    assert 0<=lo<=hi<=out['future_n']
    if lo>0:assert cdf[lo-1]<.025<=cdf[lo]
    if hi>0:assert cdf[hi-1]<.975<=cdf[hi]

def test_interval_not_sum_of_single_batch_intervals():
    specs=[(50,9.,3.5),(60,8.2,4.1),(40,12.,2.8)]
    def single_width(m,a,b):
        lp,_=beta_binomial_logpmf(m,a,b);logcdf=np.logaddexp.accumulate(lp)
        return int(np.searchsorted(logcdf,np.log(.975)))-int(np.searchsorted(logcdf,np.log(.025)))
    naive=sum(single_width(m,a,b) for m,a,b in specs)
    total=combine_distributions([component(*s) for s in specs],0)['total']
    assert total['upper']-total['lower']<naive,"合计等尾区间必须实算，不能相加单批区间"

def test_extreme_tail_credible_in_log_space():
    m,a,b,n_comps=200,.05,50.,6
    single=betaln(a+m,b)-betaln(a,b);expected=n_comps*single
    assert expected<-745,"前提：线性空间双精度必然下溢为0"
    assert float(np.exp(expected))==0.
    out=combine_distributions([component(m,a,b)]*n_comps,n_comps*m)['total']
    top=out['pmf'][-1]
    assert top['k']==n_comps*m and top['probability']==0.
    assert top['log_probability']==pytest.approx(expected,rel=1e-12)
    assert out['log_prob_reach_target']==pytest.approx(expected,rel=1e-12)
    assert out['prob_reach_target']==0.

def test_tail_complement_identity():
    out=combine_distributions([component(50,9.,3.5),component(60,8.2,4.1),component(30,6.,6.)],0)['total']
    logpmf=np.array([p['log_probability'] for p in out['pmf']])
    for t in [1,40,80,120,140]:
        reach=float(np.exp(logsumexp(logpmf[t:])));below=float(np.exp(logsumexp(logpmf[:t])))
        assert reach+below==pytest.approx(1.,abs=1e-10)

def test_validation_rejects_bad_input():
    with pytest.raises(ValueError,match="2–40"):combine_distributions([component(10,2.,2.)],5)
    with pytest.raises(ValueError,match="2–40"):combine_distributions([component(10,2.,2.)]*(MAX_COMPONENTS+1),5)
    with pytest.raises(ValueError,match="1–200"):combine_distributions([component(0,2.,2.),component(10,2.,2.)],5)
    with pytest.raises(ValueError,match="1–200"):combine_distributions([component(201,2.,2.),component(10,2.,2.)],5)
    with pytest.raises(ValueError,match="有限正数"):combine_distributions([component(10,0.,2.),component(10,2.,2.)],5)
    with pytest.raises(ValueError,match="有限正数"):combine_distributions([component(10,float('nan'),2.),component(10,2.,2.)],5)
    with pytest.raises(ValueError,match="容量超限"):
        combine_distributions([component(200,2.,2.)]*(MAX_TOTAL_FUTURE//200+1),5)
    with pytest.raises(ValueError,match="目标萌发数"):combine_distributions([component(50,2.,2.),component(50,2.,2.)],101)
    with pytest.raises(ValueError,match="目标萌发数"):combine_distributions([component(50,2.,2.),component(50,2.,2.)],-1)
    with pytest.raises(ValueError,match="整数"):combine_distributions([component(50,2.,2.),component(50,2.,2.)],10.5)

def test_target_zero_reaches_with_probability_one():
    total=combine_distributions([component(50,9.,3.5),component(60,8.2,4.1)],0)['total']
    assert total['prob_reach_target']==1. and total['log_prob_reach_target']==0.
