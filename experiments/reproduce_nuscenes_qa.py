#!/usr/bin/env python3
"""CPU-only reproduction of frozen P2 and selector results.

Requires access to the external, access-controlled NuScenes-QA artifact root.
The root checkout is the default in this development workspace; cloned users
must provide --artifact-root explicitly.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path

RELEASE=Path(__file__).resolve().parents[1]
for rel in ('core','domains/nuscenes_qa','selection','experiments/source_snapshot'):
    sys.path.insert(0,str(RELEASE/rel))
import run_p2_restoration_fivefold as restoration
from run_simple_reasoner_selection import OOF_SEED,fit_predict

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def readl(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifact-root',type=Path,default=RELEASE.parent);a=ap.parse_args();root=a.artifact_root.resolve()
    def p(rel):
        q=root/rel
        if not q.is_file():raise FileNotFoundError(f'missing required external artifact: {rel}')
        return q
    restoration.S0_SEALED=p('outputs/train20_qwen35_structured_final/sealed_predictions.jsonl')
    restoration.S0_LABELS=p('outputs/train20_qwen35_structured_final/label_joined_predictions.jsonl')
    restoration.DIRECT=p('outputs/train20_qwen35_direct_final/label_joined_predictions.jsonl')
    restoration.HIST_P2=p('outputs/rase_phase3_constraint_resolution/paired_evaluation.csv')
    restoration.EVIDENCE=root/'outputs/phase5_bevfusion_train20_evidence/normalized'
    if not restoration.EVIDENCE.is_dir():raise FileNotFoundError('missing required external artifact directory: outputs/phase5_bevfusion_train20_evidence/normalized')
    candidates,reproduction=restoration.replay()
    if reproduction.get('stage_a')!='PASS_EXACT':raise RuntimeError(f'P2 replay was not exact: {reproduction}')
    direct_n=sum(int(x['direct_correct']) for x in candidates);p2_n=sum(int(x['p2_correct']) for x in candidates)
    if (len(candidates),len({x['question_id'] for x in candidates}),direct_n,p2_n)!=(1000,1000,396,311):
        raise RuntimeError(f'Frozen D0/P2 counts or cohort differ: n={len(candidates)}, D={direct_n}, P2={p2_n}')
    rows=restoration.selector_rows(candidates)
    old_folds=readj(p('outputs/simple_reasoner_selection/scene_oof_split_manifest.json'))
    frozen_folds=readj(p('outputs/p2_restoration_fivefold/fivefold_manifest.json'))
    predictions=[]
    for fold in sorted(map(int,old_folds)):
        hold=set(old_folds[str(fold)]['heldout_scenes'])
        if sorted(hold)!=sorted(frozen_folds[str(fold)]['heldout_scenes']):raise RuntimeError(f'fold mismatch {fold}')
        train=[x for x in rows if x['scene_id'] not in hold]
        test=[{**x,'outer_fold_source':fold} for x in rows if x['scene_id'] in hold]
        if (len(train),len(test))!=(800,200):raise RuntimeError(f'wrong fold sizes: {fold}')
        predicted,_=fit_predict(train,test,OOF_SEED);predictions.extend(predicted)
    predictions.sort(key=lambda x:int(x['question_id']))
    frozen=readl(p('outputs/p2_restoration_fivefold/fivefold_predictions.jsonl'))
    fm={int(x['question_id']):x for x in frozen};pm={int(x['question_id']):x for x in predictions}
    if set(fm)!=set(pm) or len(fm)!=1000:raise RuntimeError('selector OOF prediction coverage mismatch')
    route_mismatch=sum(fm[q]['selected_path']!=pm[q]['selected_path'] for q in pm)
    correct_mismatch=sum(int(fm[q]['selector_correct'])!=int(pm[q]['selector_correct']) for q in pm)
    max_probability_delta=max(abs(float(fm[q]['probabilities'][k])-float(pm[q]['probabilities'][k])) for q in pm for k in ('direct_only','tie','structured_only'))
    selector_n=sum(int(x['selector_correct']) for x in predictions)
    if (route_mismatch,correct_mismatch,selector_n)!=(0,0,428) or max_probability_delta>1e-10:
        raise RuntimeError(f'Selector reproduction mismatch: route={route_mismatch}, correctness={correct_mismatch}, count={selector_n}, max_prob_delta={max_probability_delta}')
    metrics=readj(p('outputs/p2_restoration_fivefold/fivefold_metrics.json'))
    p2_metric=metrics['methods'].get('P2',metrics['methods'].get('S1_P2'))
    if metrics['methods']['D0']['correct']!=396 or p2_metric['correct']!=311 or metrics['methods']['H1']['correct']!=428:
        raise RuntimeError('Frozen main metrics disagree with independent recomputation')
    manifest=readj(p('outputs/p2_restoration_fivefold/frozen_input_manifest.json'))
    verified={}
    for rel,expected in manifest['inputs'].items():
        actual=sha(p(rel))
        if actual!=expected:raise RuntimeError(f'frozen input hash mismatch: {rel}')
        verified[rel]=actual
    freeze=readj(p('outputs/train20_qwen35_structured_final/freeze_manifest.json'))
    freeze_checks={}
    outdir=root/'outputs/train20_qwen35_structured_final'
    for name,rec in freeze['artifacts'].items():
        fp=outdir/name
        if not fp.is_file() or fp.stat().st_size!=rec['bytes'] or sha(fp)!=rec['sha256']:
            raise RuntimeError(f'structured frozen artifact hash/size mismatch: {name}')
        freeze_checks[name]=rec['sha256']
    direct_rt=readj(p('outputs/train20_qwen35_direct_final/runtime_manifest.json'))
    structured_rt=readj(p('outputs/train20_qwen35_structured_final/runtime_manifest.json'))
    if (direct_rt['model']['model_id'],direct_rt['model']['revision'])!=(structured_rt['model']['model_id'],structured_rt['model']['revision']):raise RuntimeError('Direct/Structured frozen model identities differ')
    result={'status':'PASS','gpu_or_external_inference':False,'p2_exact_reproduction':reproduction,'independent_counts':{'qa':1000,'scenes':20,'samples':581,'direct':direct_n,'structured_p2':p2_n,'p2_postselection':selector_n},'selector_reproduction':{'route_mismatches':route_mismatch,'correctness_mismatches':correct_mismatch,'maximum_probability_delta':max_probability_delta,'folds':5},'verified_p2_input_hashes':verified,'verified_structured_freeze_hashes':freeze_checks,'model':{'id':structured_rt['model']['model_id'],'revision':structured_rt['model']['revision']}}
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
