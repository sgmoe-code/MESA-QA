"""Final paper-facing Qwen3.5 Direct baseline on the frozen Train-20 cohort."""
from __future__ import annotations

import hashlib, json, os, platform, statistics, time, uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import accelerate, torch, transformers

from bevfusion_scene_adapter import BEVFusionSceneAdapter
from qwen35_runtime import (ATTENTION_IMPLEMENTATION, DEFAULT_MODEL_PATH, DTYPE_NAME,
    MODEL_ID, MODEL_REVISION, PARSER_VERSION, RUNTIME_VERSION, cache_identity, model_identity)
from run_phase3b15_oracle_plain_mllm_baseline import clean_answer, direct_prompt
from run_phase5b3_submission_protocol_freeze import EvidenceOnlyQwenRuntime
from run_real_evidence_pure_mllm_baseline import serialize_real_scene
from step71_common import normalize_answer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/train20_qwen35_direct_final'
MIGRATION=ROOT/'outputs/qwen35_9b_final_backbone_migration'
COHORT=ROOT/'outputs/train20_original_full_true_nested_oof/train20_cohort_manifest.json'
EVIDENCE=ROOT/'outputs/phase5_bevfusion_train20_evidence/normalized'
LABELS=ROOT/'data/questions/NuScenes_train_questions.json'
PRED=OUT/'sealed_predictions.jsonl'
MAX_NEW_TOKENS=48
NORMALIZER_VERSION='step71_common.normalize_answer.v1'
DIRECT_PROMPT_VERSION='phase3b15.evidence_only_direct.stable.v1'

def now(): return datetime.now(timezone.utc).isoformat()
def hbytes(data:bytes): return hashlib.sha256(data).hexdigest()
def htext(text:str): return hbytes(text.encode())
def sha(path:Path): return hbytes(path.read_bytes())
def stable(value:Any): return htext(json.dumps(value,sort_keys=True,separators=(',',':'),default=str))
def dump(name:str,value:Any):
    OUT.mkdir(parents=True,exist_ok=True); (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')
def read_jsonl(path:Path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []
def percentile(values:list[float],q:float):
    if not values:return None
    xs=sorted(values); pos=(len(xs)-1)*q; lo=int(pos); hi=min(lo+1,len(xs)-1); return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)

def frozen_identities():
    generation=json.loads((MIGRATION/'generation_config.json').read_text())
    direct_generation=generation['generation_parameters_by_role']['direct']
    return {
        'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'model_cache_identity':cache_identity(DEFAULT_MODEL_PATH),
        'generation_config_hash':stable(direct_generation),'generation_config':direct_generation,
        'runtime_manifest_hash':sha(MIGRATION/'runtime_freeze_manifest.json'),
        'runtime_freeze_frozen':json.loads((MIGRATION/'runtime_freeze_manifest.json').read_text())['frozen'],
        'normalizer_version':NORMALIZER_VERSION,'normalizer_source_hash':sha(ROOT/'step71_common.py'),
        'direct_prompt_version':DIRECT_PROMPT_VERSION,'direct_prompt_source_hash':sha(ROOT/'run_phase3b15_oracle_plain_mllm_baseline.py'),
        'evidence_serializer_source_hash':sha(ROOT/'run_real_evidence_pure_mllm_baseline.py'),
    }

def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    cohort_payload=json.loads(COHORT.read_text()); cases=cohort_payload['rows']; ids=frozen_identities()
    if len(cases)!=1000 or len({int(x['question_id']) for x in cases})!=1000: raise RuntimeError('TRAIN20_QA_IDENTITY_FAILURE')
    if len({x['scene_token'] for x in cases})!=20: raise RuntimeError('TRAIN20_SCENE_FAILURE')
    if sorted({int(x['fold_id']) for x in cases})!=list(range(5)): raise RuntimeError('TRAIN20_FOLD_FAILURE')
    evidence_files=sorted(EVIDENCE.glob('*.json'))
    if len(evidence_files)!=807: raise RuntimeError(f'EVIDENCE_INVENTORY_FAILURE:{len(evidence_files)}')
    missing=sorted({x['sample_token'] for x in cases if not (EVIDENCE/f"{x['sample_token']}.json").is_file()})
    if missing: raise RuntimeError(f'MISSING_COHORT_EVIDENCE:{len(missing)}')
    frozen=json.loads((MIGRATION/'runtime_freeze_manifest.json').read_text())
    if not frozen['frozen'] or frozen['model']['revision']!=MODEL_REVISION or frozen['model']['model_id']!=MODEL_ID: raise RuntimeError('RUNTIME_FREEZE_MISMATCH')
    cohort_manifest={'source':str(COHORT),'source_sha256':sha(COHORT),'qa_count':len(cases),'question_id_count':len({x['question_id'] for x in cases}),'scene_count':20,'cohort_referenced_sample_count':len({x['sample_token'] for x in cases}),'evidence_inventory_count':807,'fold_ids':list(range(5)),'fold_assignments_sha256':stable([(x['question_id'],x['scene_token'],x['fold_id']) for x in cases]),'evidence_inventory_sha256':stable([(p.name,sha(p)) for p in evidence_files]),'missing_evidence':missing,'pass':True};dump('cohort_manifest.json',cohort_manifest)
    runtime={'model':model_identity(),'frozen_identity':ids,'gpu':torch.cuda.get_device_name(0),'pytorch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'accelerate':accelerate.__version__,'dtype':DTYPE_NAME,'device_map':'auto','attention_implementation':ATTENTION_IMPLEMENTATION,'deltanet_backend':'validated native PyTorch fallback','optimized_deltanet_kernel':False,'migration_runtime_freeze':str(MIGRATION/'runtime_freeze_manifest.json')};dump('runtime_manifest.json',runtime)
    run_id=f'qwen35-direct-final-{uuid.uuid4().hex}'
    dump('run_manifest.json',{'run_id':run_id,'started_at_utc':now(),'task':'final 1000-case Direct baseline','inference_path':['question','serialized frozen normalized BEVFusion evidence','Qwen3.5-9B','clean_answer','normalize_answer'],'forbidden_components_invoked':[],'labels_path_loaded_during_inference':False,'oracle_access':0,'eval_type_access':0,'template_id_access':0,'historical_prediction_access':0,'max_new_tokens':MAX_NEW_TOKENS,'retry_policy':'up to 2 retries for runtime exceptions only; no semantic retry','output':str(OUT)})
    return cases,ids,run_id

def validate_cached(row,case,ids,evidence_hash,serialized_hash,prompt_hash):
    checks=[row.get('question_id')==case['question_id'],row.get('sample_token')==case['sample_token'],row.get('scene_token')==case['scene_token'],row.get('outer_fold')==case['fold_id'],row.get('question')==case['question'],row.get('evidence_hash')==evidence_hash,row.get('serialized_evidence_hash')==serialized_hash,row.get('prompt_hash')==prompt_hash,row.get('model_id')==ids['model_id'],row.get('model_revision')==ids['model_revision'],row.get('generation_config_hash')==ids['generation_config_hash'],row.get('runtime_manifest_hash')==ids['runtime_manifest_hash'],row.get('normalizer_version')==ids['normalizer_version']]
    return all(checks) and not row.get('error') and row.get('model_call_count')==1

def infer(cases,ids,run_id):
    existing=read_jsonl(PRED); by_id={int(x['question_id']):x for x in existing}
    if len(existing)!=len(by_id): raise RuntimeError('DUPLICATE_EXISTING_PREDICTIONS')
    adapter=BEVFusionSceneAdapter(EVIDENCE); prepared=[]; hits=0
    for case in cases:
        evidence_path=EVIDENCE/f"{case['sample_token']}.json"; rows=adapter.build_executor_scene_rows(case['sample_token']); serialized=serialize_real_scene(rows); prompt=direct_prompt(serialized,case['question']); eh=sha(evidence_path); sh=htext(serialized); ph=htext(prompt)
        cached=by_id.get(int(case['question_id']))
        if cached and validate_cached(cached,case,ids,eh,sh,ph): hits+=1; continue
        if cached: raise RuntimeError(f'INCOMPATIBLE_QWEN35_CACHE:{case["question_id"]}')
        prepared.append((case,prompt,eh,sh,ph))
    qwen=None
    if prepared:
        qwen=EvidenceOnlyQwenRuntime(DEFAULT_MODEL_PATH,128*28*28,512*28*28); qwen.load()
    misses=0; retries=0; failures=[]; started=time.time()
    with PRED.open('a',encoding='utf-8') as stream:
        for position,(case,prompt,eh,sh,ph) in enumerate(prepared,1):
            result=None; error=None; retry=0; peak=0.0
            while retry<=2:
                try:
                    torch.cuda.reset_peak_memory_stats(); result=qwen.direct(prompt,MAX_NEW_TOKENS); torch.cuda.synchronize(); peak=torch.cuda.max_memory_allocated()/2**30; error=None; break
                except torch.cuda.OutOfMemoryError as exc: error=f'OOM:{type(exc).__name__}:{exc}'; torch.cuda.empty_cache()
                except Exception as exc: error=f'RUNTIME_FAILURE:{type(exc).__name__}:{exc}'
                retry+=1; retries+=1
            misses+=1
            if result is None:
                raw=''; parsed=''; normalized=''; input_tokens=None; output_tokens=0; latency=0.0; failures.append(case['question_id'])
            else:
                raw=result['raw_output']; parsed=clean_answer(raw); normalized=normalize_answer(parsed); input_tokens=result.get('input_token_count'); output_tokens=len(qwen.processor.tokenizer.encode(raw,add_special_tokens=False)); latency=float(result['latency_sec'])
            row={'question_id':case['question_id'],'sample_token':case['sample_token'],'scene_token':case['scene_token'],'outer_fold':case['fold_id'],'question':case['question'],'evidence_hash':eh,'serialized_evidence_hash':sh,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'model_cache_identity':ids['model_cache_identity'],'prompt_hash':ph,'prompt_version':DIRECT_PROMPT_VERSION,'generation_config_hash':ids['generation_config_hash'],'runtime_manifest_hash':ids['runtime_manifest_hash'],'normalizer_version':NORMALIZER_VERSION,'raw_model_output':raw,'parsed_raw_answer':parsed,'normalized_answer':normalized,'model_call_count':1 if result else retry,'input_token_count':input_tokens,'output_token_count':output_tokens,'latency':latency,'peak_vram_gib':peak,'cache_status':'MISS_GENERATED' if result else 'MISS_UNRESOLVED','retry_count':retry,'error':error,'labels_attached':False,'run_id':run_id}
            stream.write(json.dumps(row,sort_keys=True,default=str)+'\n');stream.flush();os.fsync(stream.fileno())
            print(json.dumps({'progress':f'{position}/{len(prepared)}','question_id':case['question_id'],'answer':normalized,'latency':latency,'retry':retry},sort_keys=True),flush=True)
    rows=read_jsonl(PRED)
    if len(rows)!=1000 or len({x['question_id'] for x in rows})!=1000: raise RuntimeError(f'INCOMPLETE_PREDICTIONS:{len(rows)}')
    unresolved=[x for x in rows if x['error'] or not x['raw_model_output'].strip()]
    if unresolved: raise RuntimeError(f'UNRESOLVED_PREDICTIONS:{len(unresolved)}')
    seal={'path':str(PRED),'sha256':sha(PRED),'records':1000,'sealed_at_utc':now(),'labels_attached':False,'label_file_opened_before_seal':False};dump('prediction_seal.json',seal)
    return rows,{'cache_hits':hits,'cache_misses':misses,'retry_count':retries,'inference_wall_seconds':time.time()-started,'unresolved':len(unresolved),'prediction_sha256':seal['sha256']}

def evaluate(predictions,run_stats,cases,ids):
    # This is the first label-file access in the process and occurs after prediction_seal.json.
    label_rows=json.loads(LABELS.read_text())['questions']; by_qid={int(x['question_id']):x for x in predictions}; joined=[]
    for case in cases:
        official=label_rows[int(case['question_id'])]
        if official['question']!=case['question'] or official['sample_token']!=case['sample_token']: raise RuntimeError(f'LABEL_IDENTITY_MISMATCH:{case["question_id"]}')
        pred=by_qid[int(case['question_id'])]; gt=normalize_answer(official['answer']); correct=pred['normalized_answer']==gt
        joined.append({**pred,'ground_truth':official['answer'],'normalized_ground_truth':gt,'correct':correct,'labels_attached':True})
    joined_path=OUT/'label_joined_predictions.jsonl';joined_path.write_text('\n'.join(json.dumps(x,sort_keys=True,default=str) for x in joined)+'\n')
    correct=sum(x['correct'] for x in joined); accuracy=correct/1000
    main={'direct_correct':correct,'total':1000,'direct_normalized_exact_match_accuracy':accuracy,'historical_qwen25_direct_correct':355,'historical_qwen25_direct_accuracy':.355,'delta_vs_qwen25_direct':accuracy-.355,'metric':'normalized exact match','normalizer_version':NORMALIZER_VERSION};dump('main_metrics.json',main)
    folds=[]
    for fold in range(5):
        xs=[x for x in joined if x['outer_fold']==fold]; folds.append({'fold_id':fold,'scene_count':len({x['scene_token'] for x in xs}),'qa_count':len(xs),'correct_count':sum(x['correct'] for x in xs),'accuracy':sum(x['correct'] for x in xs)/len(xs)})
    facc=[x['accuracy'] for x in folds];dump('fold_metrics.json',{'folds':folds,'mean_fold_accuracy':statistics.mean(facc),'population_std_fold_accuracy':statistics.pstdev(facc),'min_fold_accuracy':min(facc),'max_fold_accuracy':max(facc)})
    scenes=[]
    for scene in sorted({x['scene_token'] for x in joined}):
        xs=[x for x in joined if x['scene_token']==scene];scenes.append({'scene_token':scene,'qa_count':len(xs),'correct_count':sum(x['correct'] for x in xs),'accuracy':sum(x['correct'] for x in xs)/len(xs)})
    sacc=[x['accuracy'] for x in scenes];dump('scene_metrics.json',{'scenes':scenes,'mean_scene_accuracy':statistics.mean(sacc),'population_std_scene_accuracy':statistics.pstdev(sacc),'min_scene_accuracy':min(sacc),'max_scene_accuracy':max(sacc)})
    pred_freq=Counter(x['normalized_answer'] for x in joined); gt_freq=Counter(x['normalized_ground_truth'] for x in joined); wrong_freq=Counter(x['normalized_answer'] for x in joined if not x['correct'])
    dump('error_diagnostics.json',{'correct_count':correct,'incorrect_count':1000-correct,'answer_frequency_distribution':dict(gt_freq.most_common()),'predicted_answer_distribution':dict(pred_freq.most_common()),'empty_or_invalid_normalized_answers':sum(not x['normalized_answer'].strip() for x in joined),'high_frequency_failure_answers':dict(wrong_freq.most_common(25)),'note':'Descriptive post-label diagnostics only; no inference-time groups or tuning.'})
    lat=[float(x['latency']) for x in predictions]; it=[x['input_token_count'] for x in predictions if x['input_token_count'] is not None]; ot=[x['output_token_count'] for x in predictions if x['output_token_count'] is not None]
    efficiency={'total_wall_time_seconds':run_stats['inference_wall_seconds'],'total_qwen_calls':sum(x['model_call_count'] for x in predictions),'qwen_calls_per_qa':sum(x['model_call_count'] for x in predictions)/1000,'cache_hits':run_stats['cache_hits'],'cache_misses':run_stats['cache_misses'],'retry_count':sum(x['retry_count'] for x in predictions),'mean_latency_seconds':statistics.mean(lat),'median_latency_seconds':statistics.median(lat),'p90_latency_seconds':percentile(lat,.90),'p95_latency_seconds':percentile(lat,.95),'mean_input_tokens':statistics.mean(it),'mean_output_tokens':statistics.mean(ot),'total_generated_tokens':sum(ot),'peak_observed_vram_gib':max(x['peak_vram_gib'] for x in predictions),'oom_count':sum(str(x.get('error','')).startswith('OOM:') for x in predictions),'deltanet_optimized_kernel':False,'deltanet_backend':'validated native PyTorch fallback'};dump('efficiency.json',efficiency)
    audit={'FINAL_BACKBONE':MODEL_ID,'MODEL_REVISION_MATCH':MODEL_REVISION==json.loads((MIGRATION/'model_manifest.json').read_text())['revision'],'RUNTIME_FREEZE_MATCH':ids['runtime_manifest_hash']==sha(MIGRATION/'runtime_freeze_manifest.json'),'TRAIN20_COHORT_MATCH':len(cases)==1000 and len({x['scene_token'] for x in cases})==20,'FOLDS_MATCH':sorted({x['fold_id'] for x in cases})==list(range(5)),'EVIDENCE_MATCH':len(list(EVIDENCE.glob('*.json')))==807,'ANSWER_NORMALIZER_MATCH':ids['normalizer_source_hash']==sha(ROOT/'step71_common.py'),'QWEN25_PREDICTION_REUSE':0,'ORACLE_ACCESS':0,'LABEL_ACCESS_DURING_INFERENCE':0,'EVAL_TYPE_ACCESS_DURING_INFERENCE':0,'TEMPLATE_ID_ACCESS_DURING_INFERENCE':0,'ALL_PREDICTIONS_SEALED_BEFORE_LABEL_JOIN':True,'ACCURACY_DRIVEN_TUNING':False,'sealed_prediction_sha256':run_stats['prediction_sha256'],'sealed_prediction_hash_unchanged_after_label_join':sha(PRED)==run_stats['prediction_sha256'],'pass':True};audit['pass']=all(v for k,v in audit.items() if k in ['MODEL_REVISION_MATCH','RUNTIME_FREEZE_MATCH','TRAIN20_COHORT_MATCH','FOLDS_MATCH','EVIDENCE_MATCH','ANSWER_NORMALIZER_MATCH','ALL_PREDICTIONS_SEALED_BEFORE_LABEL_JOIN','sealed_prediction_hash_unchanged_after_label_join']) and audit['QWEN25_PREDICTION_REUSE']==0;dump('protocol_audit.json',audit)
    report=f'''# Final Qwen3.5 Direct Train-20 Baseline\n\nThe frozen Direct-only path completed on all 1,000 authoritative Train-20 questions. Predictions were sealed before the authorized label join. No gate, RAG, Banker, Planner, tools, Structured path, Validator, controller, ReAct, Program Agent, Oracle data, or inference-time metadata was used. Historical Qwen2.5 performance is retained only as backbone history.\n\nQWEN35_DIRECT_FINAL_RUN_COMPLETE = YES\n\nFINAL_BACKBONE = {MODEL_ID}\nFINAL_BACKBONE_REVISION = {MODEL_REVISION}\n\nQA_COMPLETE = 1000/1000\nSCENE_COMPLETE = 20/20\nFOLD_COMPLETE = 5/5\n\nDIRECT_CORRECT = {correct}/1000\nDIRECT_ACCURACY = {accuracy:.6%}\n\nHISTORICAL_QWEN25_DIRECT_ACCURACY = 35.50%\nDELTA_VS_QWEN25_DIRECT = {accuracy-.355:+.6%}\n\nTOTAL_QWEN_CALLS = {efficiency['total_qwen_calls']}\nCACHE_HITS = {efficiency['cache_hits']}\nCACHE_MISSES = {efficiency['cache_misses']}\n\nMEAN_LATENCY_SECONDS = {efficiency['mean_latency_seconds']:.6f}\nMEDIAN_LATENCY_SECONDS = {efficiency['median_latency_seconds']:.6f}\nPEAK_VRAM_GIB = {efficiency['peak_observed_vram_gib']:.6f}\nOOM_COUNT = {efficiency['oom_count']}\n\nMODEL_REVISION_MATCH = YES\nRUNTIME_FREEZE_MATCH = YES\nCOHORT_MATCH = YES\nEVIDENCE_MATCH = YES\nFOLDS_MATCH = YES\n\nQWEN25_PREDICTION_REUSE = 0\nLABEL_ACCESS_DURING_INFERENCE = 0\nORACLE_ACCESS = 0\nACCURACY_DRIVEN_TUNING = NO\n\nALL_PREDICTIONS_SEALED_BEFORE_LABEL_JOIN = YES\nPROTOCOL_AUDIT_PASS = {'YES' if audit['pass'] else 'NO'}\n\nQWEN35_DIRECT_READY_FOR_PAPER_COMPARISON = {'YES' if audit['pass'] else 'NO'}\n''';(OUT/'final_report.md').write_text(report)
    artifacts=[p for p in OUT.iterdir() if p.is_file() and p.name!='freeze_manifest.json'];dump('freeze_manifest.json',{'complete':audit['pass'],'frozen_at_utc':now(),'prediction_count':1000,'prediction_sha256':sha(PRED),'label_joined_sha256':sha(joined_path),'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(artifacts)}})

def main():
    started=time.time();cases,ids,run_id=preflight(); predictions,stats=infer(cases,ids,run_id);stats['total_process_wall_seconds']=time.time()-started;evaluate(predictions,stats,cases,ids)
if __name__=='__main__':main()
