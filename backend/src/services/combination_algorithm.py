"""Conditional-independent convolution of per-lot Beta-Binomial future counts.

各批未来萌发计数条件于同一冻结点估αβ相互独立，合计分布在对数空间逐批卷积；
极小尾概率以log概率保存与报告，不在线性空间下溢。等尾区间由合计CDF实算，
不能由单批区间相加得到。
"""
import numpy as np
from scipy.special import betaln,gammaln,logsumexp
VERSION='conditional-independent-convolution-v1'
MAX_COMPONENTS=40
MAX_TOTAL_FUTURE=2000
INTERVAL_LEVEL=.95

def beta_binomial_logpmf(m,a,b):
    if type(m) is not int or not 1<=m<=200:raise ValueError("每批试种数须为1–200整数")
    if isinstance(a,bool) or isinstance(b,bool) or not np.isfinite([a,b]).all() or min(a,b)<=0:raise ValueError("条件后验形状须为有限正数")
    j=np.arange(m+1);logp=gammaln(m+1)-gammaln(j+1)-gammaln(m-j+1)+betaln(a+j,b+m-j)-betaln(a,b);normalizer=float(logsumexp(logp));mass=float(np.exp(normalizer))
    if not np.isfinite(mass) or abs(mass-1)>1e-7:raise ValueError("单批预测PMF归一偏差过大")
    return logp-normalizer,abs(mass-1)

def convolve_logpmfs(logpmfs):
    total=np.zeros(1)
    for lp in logpmfs:
        lp=np.asarray(lp,dtype=float);width=len(lp);out=np.full(len(total)+width-1,-np.inf)
        for j in range(width):out[j:j+len(total)]=np.logaddexp(out[j:j+len(total)],total+lp[j])
        total=out
    return total

def combine_distributions(components,target):
    if not 2<=len(components)<=MAX_COMPONENTS:raise ValueError(f"合并批次须为2–{MAX_COMPONENTS}个独立批次")
    if type(target) is not int or isinstance(target,bool):raise ValueError("目标萌发数须为整数")
    logpmfs=[];rows=[]
    for c in components:
        m,a,b=c['future_n'],c['post_alpha'],c['post_beta'];lp,err=beta_binomial_logpmf(m,a,b);logpmfs.append(lp);mu=a/(a+b)
        rows.append({'future_n':m,'post_alpha':float(a),'post_beta':float(b),'mean':float(m*mu),'variance':float(m*mu*(1-mu)*(a+b+m)/(a+b+1)),'normalization_error':err})
    total_n=sum(r['future_n'] for r in rows)
    if total_n>MAX_TOTAL_FUTURE:raise ValueError(f"合计试种容量超限：{total_n}粒超过{MAX_TOTAL_FUTURE}粒上限")
    if not 0<=target<=total_n:raise ValueError(f"目标萌发数须为0到合计试种数{total_n}之间的整数")
    logpmf=convolve_logpmfs(logpmfs);mass=float(np.exp(logsumexp(logpmf)))
    if not np.isfinite(mass) or abs(mass-1)>1e-7:raise ValueError("合计PMF归一偏差过大")
    logcdf=np.logaddexp.accumulate(logpmf);tail=(1-INTERVAL_LEVEL)/2
    lower=min(int(np.searchsorted(logcdf,np.log(tail),side='left')),total_n);upper=min(int(np.searchsorted(logcdf,np.log(1-tail),side='left')),total_n)
    log_reach=float(logsumexp(logpmf[target:])) if target>0 else 0.
    return {'components':rows,'total':{'future_n':total_n,'mean':float(sum(r['mean'] for r in rows)),'variance':float(sum(r['variance'] for r in rows)),'lower':lower,'upper':upper,'level':INTERVAL_LEVEL,'target':target,'prob_reach_target':float(np.exp(log_reach)),'log_prob_reach_target':log_reach,'pmf':[{'k':int(i),'probability':float(np.exp(v)),'log_probability':float(v)} for i,v in enumerate(logpmf)],'normalization_error':abs(mass-1)}}
