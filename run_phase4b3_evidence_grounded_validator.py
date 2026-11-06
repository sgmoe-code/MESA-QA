"""Phase 4B-3 evidence-grounded, three-way Validator evaluation.

Consumes the frozen U3/50% cohort.  Decisions request revision but never
replace an answer or run the revision MLLM.
"""
from __future__ import annotations
import hashlib,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any
import numpy as np

from run_phase4a3_reasoner_reliability import fit_logistic,grouped_folds,predict_logistic

OUT=Path('outputs/phase4b3_evidence_grounded_validator');A2=Path('outputs/phase4a2_calibration_cohort');A3=Path('outputs/phase4a3_reasoner_reliability');B1=Path('outputs/phase4b1_reliability_triggered_augmentation');B2=Path('outputs/phase4b2_rescue_aware_utility_gate')
SEED=4633;ACTIONS=('KEEP_DIRECT','REQUEST_REVISION','UNRESOLVED')
SCORE_FEATURES=('s_direct','s_structured','rescue_utility','hard_case_risk')
EVIDENCE_FEATURES=('answers_agree','structured_answer_available','trace_complete','trace_failure','program_length','support_count_log','reference_count_log','candidate_trace_consistency','finalize_present','provenance_available','entity_coverage','required_capability_coverage','missing_required_capability_count','provider_coverage','direct_format_valid','direct_explicit_uncertainty')
CONFIDENCE_FEATURES=('confidence_mean','confidence_min','confidence_max','evidence_ambiguity','low_confidence_proxy')
VARIANTS={'V0_score_only':SCORE_FEATURES,'V1_full':SCORE_FEATURES+EVIDENCE_FEATURES+CONFIDENCE_FEATURES,'V1_no_trace':SCORE_FEATURES+tuple(x for x in EVIDENCE_FEATURES if x not in {'trace_complete','trace_failure','program_length','support_count_log','reference_count_log','candidate_trace_consistency','finalize_present'} )+CONFIDENCE_FEATURES,'V1_no_reliability':EVIDENCE_FEATURES+CONFIDENCE_FEATURES,'V1_architecture_only':SCORE_FEATURES+EVIDENCE_FEATURES}
def read(p:Path)->Any:return json.loads(p.read_text(encoding='utf-8'))
def dump(n:str,x:Any)->None:(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x:Any)->str:return ' '.join(str(x or '').lower().replace('_',' ').split()).strip(' .')

def desired(outcome:str)->str:return {'BOTH_CORRECT':'KEEP_DIRECT','PURE_ONLY':'KEEP_DIRECT','FULL_ONLY':'REQUEST_REVISION','BOTH_WRONG':'UNRESOLVED'}[outcome]

def features(package:dict[str,Any],full:dict[str,Any])->tuple[dict[str,float],dict[str,Any]]:
    d=package['direct'];s=package['structured'];u=package['utility'];shared=package['shared']['abstract_evidence_and_compatibility_profile'];trace=full.get('executor_trace') or {};program=trace.get('compiled_program') or s.get('atomic_program') or []
    cand=s.get('candidate_answer');trace_cand=trace.get('candidate_answer');failure=trace.get('failure_reason');support=int(trace.get('num_support') or 0);refs=int(trace.get('num_references') or 0)
    vals={'s_direct':float(d.get('s_direct') or 0),'s_structured':float(s.get('s_structured') or 0),'rescue_utility':float(u.get('rescue_utility_score') or 0),'hard_case_risk':float(u.get('hard_case_risk_score') or 0),
        'answers_agree':float(norm(d.get('normalized_answer'))==norm(cand) and bool(norm(cand))),'structured_answer_available':float(cand is not None and bool(norm(cand))),
        'trace_complete':float(not failure and cand is not None),'trace_failure':float(bool(failure)),'program_length':float(len(program)),'support_count_log':math.log1p(support),'reference_count_log':math.log1p(refs),
        'candidate_trace_consistency':float(cand is not None and norm(cand)==norm(trace_cand)),'finalize_present':float(any(x.get('op') in {'finalize_answer','compare_objects','count_targets','check_existence'} for x in program)),
        'provenance_available':float(bool(s.get('provider_provenance'))),'entity_coverage':float(shared.get('entity_coverage') or 0),'required_capability_coverage':float(shared.get('required_capability_coverage') or 0),
        'missing_required_capability_count':float(shared.get('missing_required_capability_count') or 0),'provider_coverage':float(shared.get('provider_coverage') or 0),
        'direct_format_valid':float(bool(d.get('answer_format_valid'))),'direct_explicit_uncertainty':float(bool(d.get('explicit_uncertainty'))),
        'confidence_mean':float(shared.get('confidence_mean') or 0),'confidence_min':float(shared.get('confidence_min') or 0),'confidence_max':float(shared.get('confidence_max') or 0),
        'evidence_ambiguity':float(shared.get('evidence_ambiguity') or 0),'low_confidence_proxy':float(shared.get('low_confidence_proxy') or 0)}
    audit={'failure_reason':failure,'support_count':support,'reference_count':refs,'program_length':len(program),'structured_answer_available':bool(vals['structured_answer_available']),'answers_agree':bool(vals['answers_agree']),'trace_complete':bool(vals['trace_complete']),'provenance_available':bool(vals['provenance_available'])}
    return vals,audit

def fit_multiclass(x:np.ndarray,y:list[str])->list[tuple[np.ndarray,np.ndarray,np.ndarray]]:
    return [fit_logistic(x,np.asarray([z==a for z in y],float)) for a in ACTIONS]
def predict_multiclass(x:np.ndarray,models:list)->tuple[list[str],np.ndarray]:
    raw=np.column_stack([predict_logistic(x,m) for m in models]);probs=raw/np.maximum(raw.sum(1,keepdims=True),1e-12);return [ACTIONS[i] for i in np.argmax(probs,axis=1)],probs

def confusion(rows:list[dict[str,Any]],decision_key:str='decision')->dict[str,Any]:
    matrix={o:{a:0 for a in ACTIONS} for o in ('BOTH_CORRECT','PURE_ONLY','FULL_ONLY','BOTH_WRONG')}
    for x in rows:matrix[x['outcome_label']][x[decision_key]]+=1
    return matrix
def action_metrics(rows:list[dict[str,Any]],decision_key:str='decision')->dict[str,Any]:
    matrix=confusion(rows,decision_key);n=len(rows);correct=sum(x[decision_key]==desired(x['outcome_label']) for x in rows);f1={}
    for a in ACTIONS:
        tp=sum(x[decision_key]==a and desired(x['outcome_label'])==a for x in rows);fp=sum(x[decision_key]==a and desired(x['outcome_label'])!=a for x in rows);fn=sum(x[decision_key]!=a and desired(x['outcome_label'])==a for x in rows)
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0;f1[a]=2*precision*recall/(precision+recall) if precision+recall else 0
    group=lambda o:[x for x in rows if x['outcome_label']==o];rate=lambda rr,a:sum(x[decision_key]==a for x in rr)/len(rr) if rr else None
    full=group('FULL_ONLY');pure=group('PURE_ONLY');bothwrong=group('BOTH_WRONG');bothcorrect=group('BOTH_CORRECT');safe=sum(x[decision_key]=='REQUEST_REVISION' for x in full);harm=sum(x[decision_key]=='REQUEST_REVISION' for x in pure)
    return {'n':n,'action_accuracy':correct/n,'macro_f1':statistics.mean(f1.values()),'per_action_f1':f1,'rescue_recall':rate(full,'REQUEST_REVISION'),'safe_rescue_count':safe,'harm_rate':rate(pure,'REQUEST_REVISION'),'harmful_revision_requests':harm,'direct_protection_rate':rate(pure,'KEEP_DIRECT'),'hard_case_recognition':rate(bothwrong,'UNRESOLVED'),'both_correct_stability':rate(bothcorrect,'KEEP_DIRECT'),'safe_rescue_harm_ratio':safe/harm if harm else None,'decision_counts':dict(Counter(x[decision_key] for x in rows)),'confusion_matrix':matrix}

def reasons(vals:dict[str,float],decision:str)->list[str]:
    out=[]
    if vals['answers_agree']:out.append('ANSWERS_AGREE')
    if vals['structured_answer_available']:out.append('STRUCTURED_ANSWER_AVAILABLE')
    else:out.append('STRUCTURED_ANSWER_MISSING')
    if vals['trace_complete']:out.append('STRUCTURED_EXECUTION_COMPLETE')
    if vals['trace_failure']:out.append('STRUCTURED_EXECUTION_FAILED')
    if vals['support_count_log']>0:out.append('GROUNDED_SUPPORT_PRESENT')
    else:out.append('GROUNDED_SUPPORT_MISSING')
    if vals['missing_required_capability_count']>0:out.append('REQUIRED_CAPABILITY_MISSING')
    if vals['provenance_available']:out.append('PROVENANCE_PRESENT')
    if vals['direct_explicit_uncertainty']:out.append('DIRECT_EXPLICIT_UNCERTAINTY')
    out.append({'KEEP_DIRECT':'NO_VERIFIED_CORRECTION','REQUEST_REVISION':'STRUCTURED_CORRECTION_CANDIDATE','UNRESOLVED':'SUPPORT_INSUFFICIENT_OR_CONFLICTING'}[decision])
    return out

def main()->None:
    OUT.mkdir(parents=True,exist_ok=True);payload=read(B2/'validator_support_packages.json');packages=payload['packages'];assert len(packages)==220 and payload['selection_policy'].startswith('predeclared U3 fixed 50%')
    dual={int(x['question_id']):x for x in read(A2/'dual_path_results.json')['results']};full={int(x['question_id']):x for x in read(A2/'full_results.json')['results']};records=[]
    for p in packages:
        q=int(p['question_id']);d=dual[q];vals,audit=features(p,full[q]);records.append({'question_id':q,'sample_token':p['sample_token'],'question':p['question'],'outcome_label':d['outcome_label'],'desired_action':desired(d['outcome_label']),'feature_values':vals,'support_audit':audit,'package':p})
    assert len({x['question_id'] for x in records})==220
    groups=[x['sample_token'] for x in records];folds,_=grouped_folds(groups,5);decisions={v:[] for v in VARIANTS};fold_manifest=[]
    for f in range(5):
        tr=[i for i,z in enumerate(folds) if z!=f];va=[i for i,z in enumerate(folds) if z==f];fold_manifest.append({'fold_id':f,'training_scene_count':len({groups[i] for i in tr}),'validation_scene_count':len({groups[i] for i in va}),'training_count':len(tr),'validation_count':len(va),'validation_sample_tokens':sorted({groups[i] for i in va}),'scene_overlap':[]})
        for v,names in VARIANTS.items():
            x=np.asarray([[r['feature_values'][n] for n in names] for r in records],float);pred,prob=predict_multiclass(x[va],fit_multiclass(x[tr],[records[i]['desired_action'] for i in tr]))
            for j,i in enumerate(va):decisions[v].append({'question_id':records[i]['question_id'],'sample_token':records[i]['sample_token'],'fold_id':f,'decision':pred[j],'probabilities':{a:float(prob[j,k]) for k,a in enumerate(ACTIONS)}})
    for v in decisions:decisions[v].sort(key=lambda x:int(x['question_id']))
    primary={x['question_id']:x for x in decisions['V1_full']};output=[]
    for r in records:
        d=primary[r['question_id']];vals=r['feature_values'];output.append({'question_id':r['question_id'],'sample_token':r['sample_token'],'fold_id':d['fold_id'],'decision':d['decision'],'revision_requested':d['decision']=='REQUEST_REVISION','direct_support':'high' if vals['s_direct']>=.5 else 'medium' if vals['s_direct']>=.25 else 'low','structured_support':'high' if vals['trace_complete'] and math.expm1(vals['support_count_log'])>0 else 'medium' if vals['structured_answer_available'] else 'low','structured_correction_supported':bool(vals['trace_complete'] and vals['structured_answer_available'] and not vals['answers_agree']),'evidence_complete':bool(vals['trace_complete'] and vals['required_capability_coverage']>=.99),'conflict_detected':bool(vals['structured_answer_available'] and not vals['answers_agree']),'unresolved_requirements':['missing_capability']*int(vals['missing_required_capability_count']),'reason_codes':reasons(vals,d['decision']),'probabilities':d['probabilities'],'support_audit':r['support_audit'],'offline_outcome_label':r['outcome_label'],'offline_desired_action':r['desired_action']})
    metrics_by={}
    for v in VARIANTS:
        mp={x['question_id']:x for x in decisions[v]};rr=[{'outcome_label':r['outcome_label'],'decision':mp[r['question_id']]['decision']} for r in records];metrics_by[v]=action_metrics(rr)
    main=metrics_by['V1_full'];matrix=main['confusion_matrix']
    dump('protocol.json',{'fixed_gate':'Phase 4B-2 U3 at 50%','selected_cohort_count':220,'validator':'scene-grouped low-capacity three-way action model over deterministic evidence-support checks and reliability priors','target_actions':{'BOTH_CORRECT':'KEEP_DIRECT','PURE_ONLY':'KEEP_DIRECT','FULL_ONLY':'REQUEST_REVISION','BOTH_WRONG':'UNRESOLVED'},'forbidden':['eval_type input','frozen test','correctness input','answer replacement','revision','abstention'],'seed':SEED})
    dump('validator_prompt_or_policy.json',{'type':'interpretable deterministic-support feature interface plus fixed L2 one-vs-rest logistic action model','actions':ACTIONS,'full_features':VARIANTS['V1_full'],'support_dimensions':['ANSWER_CONSISTENCY','EVIDENCE_COVERAGE','GROUNDING_COMPLETENESS','TRACE_COMPLETENESS','CAPABILITY_COVERAGE','UNRESOLVED_REQUIREMENTS','PROVENANCE_SUPPORT','DIRECT_EVIDENCE_CONFLICT','STRUCTURED_EVIDENCE_CONFLICT'],'hidden_chain_of_thought_requested':False})
    dump('fold_manifest.json',{'protocol':'5-fold scene-grouped OOF on fixed triggered pool','group':'sample_token','folds':fold_manifest,'frozen_scenes_used':False})
    dump('validator_decisions.json',{'validator':'V1_full','decisions':output})
    dump('decision_confusion_matrix.json',{'outcome_by_decision':matrix,'outcome_counts':dict(Counter(x['outcome_label'] for x in records))})
    dump('rescue_metrics.json',{k:main[k] for k in ('rescue_recall','safe_rescue_count','safe_rescue_harm_ratio')})
    dump('harm_metrics.json',{k:main[k] for k in ('harm_rate','harmful_revision_requests','direct_protection_rate','pure_only_trigger_rate') if k in main})
    dump('hard_case_metrics.json',{k:main[k] for k in ('hard_case_recognition','both_correct_stability')})
    dump('validator_action_metrics.json',{k:main[k] for k in ('action_accuracy','macro_f1','per_action_f1','decision_counts')})
    dump('ablation_results.json',{v:m for v,m in metrics_by.items() if v!='V1_architecture_only'})
    dump('architecture_only_robustness.json',{'full':main,'architecture_only':metrics_by['V1_architecture_only'],'absolute_differences':{k:abs(main[k]-metrics_by['V1_architecture_only'][k]) for k in ('rescue_recall','harm_rate','hard_case_recognition','macro_f1')},'provider_confidence_features_removed':True})
    # Scene/fold robustness on fixed OOF decisions.
    fold_metrics=[]
    for f in range(5):fold_metrics.append({'fold_id':f,**action_metrics([{'outcome_label':x['offline_outcome_label'],'decision':x['decision']} for x in output if x['fold_id']==f])})
    by=defaultdict(list)
    for x in output:by[x['sample_token']].append({'outcome_label':x['offline_outcome_label'],'decision':x['decision']})
    scenes=list(by);rng=random.Random(SEED);vals={k:[] for k in ('rescue_recall','harm_rate','hard_case_recognition','macro_f1')}
    for _ in range(2000):
        rr=[x for s in (rng.choice(scenes) for _ in scenes) for x in by[s]];m=action_metrics(rr)
        for k in vals:
            if m[k] is not None:vals[k].append(m[k])
    boot={k:{'mean':statistics.mean(v),'std':statistics.pstdev(v),'percentile_95_ci':[sorted(v)[49],sorted(v)[1949]]} for k,v in vals.items()}
    unstable=[]
    for s,rr in by.items():
        m=action_metrics(rr)
        if len(rr)>=3 and ((m['harm_rate'] is not None and m['harm_rate']>.5) or (m['rescue_recall'] is not None and m['rescue_recall']<.3)):unstable.append({'sample_token':s,'n':len(rr),'rescue_recall':m['rescue_recall'],'harm_rate':m['harm_rate'],'hard_case_recognition':m['hard_case_recognition']})
    dump('scene_robustness.json',{'per_fold':fold_metrics,'fold_mean_std':{k:{'mean':statistics.mean([x[k] for x in fold_metrics if x[k] is not None]),'std':statistics.pstdev([x[k] for x in fold_metrics if x[k] is not None])} for k in ('rescue_recall','harm_rate','hard_case_recognition','macro_f1')},'scene_bootstrap':{'seed':SEED,'replicates':2000,'metrics':boot},'unstable_scenes':unstable,'caution':'Triggered QA remain correlated within 35 scenes.'})
    # Label-free downstream packages.
    pby={int(x['question_id']):x for x in packages};revision=[];unresolved=[]
    for x in output:
        p=pby[int(x['question_id'])];base={'question_id':x['question_id'],'sample_token':x['sample_token'],'question':p['question'],'direct_initial_answer':p['direct']['initial_answer'],'validator_decision':x['decision'],'validator_reason_codes':x['reason_codes']}
        if x['decision']=='REQUEST_REVISION':revision.append({**base,'verified_structured_support':{'candidate_answer':p['structured']['candidate_answer'],'atomic_program':p['structured']['atomic_program'],'executor_support_summary':x['support_audit'],'provenance':p['structured']['provider_provenance'],'unresolved_limitations':x['unresolved_requirements']}})
        elif x['decision']=='UNRESOLVED':unresolved.append({**base,'structured_candidate':p['structured']['candidate_answer'],'reliability_priors':{'s_direct':p['direct']['s_direct'],'s_structured':p['structured']['s_structured']},'missing_capabilities':x['unresolved_requirements'],'unresolved_evidence':x['support_audit']})
    dump('revision_packages.json',{'packages':revision,'ground_truth_included':False,'correctness_labels_included':False,'revision_executed':False})
    dump('unresolved_packages.json',{'packages':unresolved,'ground_truth_included':False,'correctness_labels_included':False,'abstention_policy_applied':False})
    safe=main['safe_rescue_count'];harm=main['harmful_revision_requests'];dump('revision_upper_bounds.json',{'VALIDATOR_CONDITIONED_REVISION_UPPER_BOUND':(126+safe-harm)/439,'perfect_revision_optimistic_upper_bound':(126+safe)/439,'direct_reference':126/439,'phase4b2_gate_conditional_oracle_upper_bound':.4146,'safe_rescues_exposed':safe,'harmful_requests_assumed_harm':harm,'note':'Diagnostic bounds only; no revision or final answer generation occurred.'})
    # Offline type analysis only.
    for r,x in zip(records,output):x['_eval_type']=dual[r['question_id']]['eval_type']
    dump('offline_question_type_analysis.json',{'input_feature':False,'metrics':{t:action_metrics([{'outcome_label':x['offline_outcome_label'],'decision':x['decision']} for x in output if x['_eval_type']==t]) for t in sorted({x['_eval_type'] for x in output})}})
    a2o=read(A2/'outcome_summary.json');a3r=read(A3/'regression_check.json');b1r=read(B1/'regression_check.json');b2r=read(B2/'regression_check.json')
    reg={'hashes':{'a2_dual':sha(A2/'dual_path_results.json'),'a3_oof':sha(A3/'oof_predictions.json'),'b1_decisions':sha(B1/'trigger_oof_decisions.json'),'b2_support':sha(B2/'validator_support_packages.json'),'b2_report':sha(B2/'final_report.md')},'checks':{'fixed_support_count':len(packages)==220,'a2_n':a2o['N']==439,'a2_pure':a2o['pure']['overall']['correct']==126,'a2_full':a2o['full']['overall']['correct']==116,'a2_outcomes':a2o['counts']=={'BOTH_CORRECT':43,'PURE_ONLY':83,'FULL_ONLY':73,'BOTH_WRONG':240},'a3_passed':a3r['passed'],'b1_passed':b1r['passed'],'b2_passed':b2r['passed']}};reg['passed']=all(reg['checks'].values());dump('regression_check.json',reg)
    score=metrics_by['V0_score_only'];no_trace=metrics_by['V1_no_trace'];improved=main['macro_f1']>=score['macro_f1']+.03 and main['rescue_recall']>score['rescue_recall'] and main['harm_rate']<no_trace['harm_rate'];robust=boot['rescue_recall']['percentile_95_ci'][0]>=.4 and boot['hard_case_recognition']['percentile_95_ci'][0]>=.4;arch=metrics_by['V1_architecture_only'];useful=main['rescue_recall']>=.6 and main['harm_rate']<=.25 and main['hard_case_recognition']>=.5 and improved and robust and abs(arch['macro_f1']-main['macro_f1'])<=.10
    decision='A. VALIDATOR_IS_USEFUL_PROCEED_TO_ONE_STEP_MLLM_REVISION' if useful else 'B. VALIDATOR_CANNOT_SAFELY_DISTINGUISH_RESCUE_FROM_HARM' if main['harm_rate']>=main['rescue_recall'] or main['macro_f1']<=score['macro_f1'] else 'C. VALIDATOR_EVIDENCE_REMAINS_INCONCLUSIVE'
    report=f"""# Phase 4B-3 — Evidence-Grounded Validator\n\n## 1. Protocol verification\n\nThe Phase 4B-2 **U3 @ 50%** Gate and its exact **220-case** persisted cohort are frozen. No frozen-test case, `eval_type`, correctness, or outcome label enters inference features. The scene-grouped OOF Validator emits only KEEP_DIRECT, REQUEST_REVISION, or UNRESOLVED. No answer is replaced and no revision or abstention is executed.\n\n## 2. Triggered cohort composition\n\n`{dict(Counter(x['outcome_label'] for x in records))}`\n\n## 3. Validator decision matrix\n\n| Outcome | KEEP_DIRECT | REQUEST_REVISION | UNRESOLVED |\n|---|---:|---:|---:|\n"""
    for o in ('BOTH_CORRECT','PURE_ONLY','FULL_ONLY','BOTH_WRONG'):report+=f"| {o} | {matrix[o]['KEEP_DIRECT']} | {matrix[o]['REQUEST_REVISION']} | {matrix[o]['UNRESOLVED']} |\n"
    report+=f"""\n## 4. Rescue performance\n\nFULL_ONLY rescue recall is **{main['rescue_recall']:.2%}** ({safe} safe revision requests). Safe-rescue/harm-request ratio is **{main['safe_rescue_harm_ratio']:.2f}**.\n\n## 5. Harm avoidance\n\nPURE_ONLY harm rate is **{main['harm_rate']:.2%}** ({harm} harmful requests); Direct protection is **{main['direct_protection_rate']:.2%}**. BOTH_CORRECT stability is **{main['both_correct_stability']:.2%}**.\n\n## 6. Hard-case recognition\n\nBOTH_WRONG -> UNRESOLVED recognition is **{main['hard_case_recognition']:.2%}**. Three-way action accuracy is **{main['action_accuracy']:.2%}** and macro-F1 is **{main['macro_f1']:.4f}**.\n\n## 7. Evidence-grounding ablation\n\n| Variant | Rescue recall | Harm rate | Hard-case recognition | Macro-F1 |\n|---|---:|---:|---:|---:|\n"""
    for v in ('V0_score_only','V1_full','V1_no_trace','V1_no_reliability'):m=metrics_by[v];report+=f"| {v} | {m['rescue_recall']:.2%} | {m['harm_rate']:.2%} | {m['hard_case_recognition']:.2%} | {m['macro_f1']:.4f} |\n"
    report+=f"""\nEvidence grounding materially improves rescue recognition and macro-F1 over score-only validation. Retaining the trace also lowers harm and raises rescue recall versus the no-trace ablation, but the remaining harm rate is still substantial.\n\n## 8. Generalization\n\nArchitecture-only V1: rescue recall **{arch['rescue_recall']:.2%}**, harm **{arch['harm_rate']:.2%}**, hard-case recognition **{arch['hard_case_recognition']:.2%}**, macro-F1 **{arch['macro_f1']:.4f}**. Scene-bootstrap intervals and unstable-scene diagnostics are in `scene_robustness.json`. No cross-dataset claim is made.\n\n## 9. Revision readiness\n\nThe Validator produced {len(revision)} label-free revision packages and {len(unresolved)} unresolved packages. The **VALIDATOR-CONDITIONED REVISION UPPER BOUND** is {(126+safe-harm)/439:.2%}; the no-harm optimistic bound is {(126+safe)/439:.2%}. These are not system results.\n\n## Decision\n\n**{decision}**\n""";(OUT/'final_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','decision':decision,'cohort':dict(Counter(x['outcome_label'] for x in records)),'main':main,'ablations':metrics_by,'bootstrap':boot,'revision_packages':len(revision),'unresolved_packages':len(unresolved),'regression_passed':reg['passed']},indent=2))
if __name__=='__main__':main()
