"""合计补种分布：只读已完成运行的冻结结果，不回写、不受来源计数后续改动影响。"""
from fastapi import HTTPException
from .combination_algorithm import combine_distributions,VERSION
from ..utils.fingerprint import fingerprint

def combine_frozen_run(run,selections,target):
    if run.status!='completed' or not run.result:raise HTTPException(409,"仅已完成的冻结研究可合并计算合计分布")
    frozen={x['lot']['id']:x for x in run.result.get('lots',[])};seen=set();components=[]
    for sel in selections:
        lid=sel.lot_id
        if lid in seen:raise HTTPException(422,f"批次{lid}重复选择；合计卷积要求每个独立批次至多一条")
        seen.add(lid)
        row=frozen.get(lid)
        if not row:raise HTTPException(422,f"批次{lid}不在运行#{run.id}的冻结结果中，不能并入合计")
        components.append({'lot_id':lid,'code':row['lot']['code'],'future_n':sel.future_n,'post_alpha':row['post_alpha'],'post_beta':row['post_beta']})
    try:out=combine_distributions(components,target)
    except ValueError as e:raise HTTPException(422,str(e))
    total=out['total'];mean=total['mean']
    for c,row in zip(components,out['components']):c.update({'mean':row['mean'],'variance':row['variance'],'mean_share':row['mean']/mean if mean>0 else 0.})
    return {'run_id':run.id,'run_name':run.name,'algorithm_version':VERSION,'fit_hash':run.result['fit_hash'],'input_hash':run.result['input_hash'],'selection_hash':fingerprint({'method':VERSION,'fit_hash':run.result['fit_hash'],'selections':[[c['lot_id'],c['future_n']] for c in components],'target':target}),'components':components,'total':total,'source':f"运行#{run.id}冻结结果；来源计数后续改动不影响本合计。",'note':'合计分布按条件独立假设逐批卷积：各批未来计数条件于同一冻结点估αβ相互独立。等尾区间由合计CDF实算，不能由单批区间相加得到；极小尾概率在log空间计算。不含超参数估计误差，不是种子合格保证或自动发货依据。'}
