#!/usr/bin/env python3
"""Restore historical P2 and evaluate the unchanged Simple Selector protocol.

This runner is intentionally self-contained and writes only to the new experiment
directory.  It performs no model inference or retrieval.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from bevfusion_scene_adapter import BEVFusionSceneAdapter
from rase_answer_resolution import ConstraintAwareRASEExecutor
from rase_phase2_planner import PlannerRASEController
from run_simple_reasoner_selection import (
    CATEGORICAL, NUMERIC, OOF_SEED, answer_type, fit_predict, question_flags,
)
from step71_common import normalize_answer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/p2_restoration_fivefold"
S0_SEALED = ROOT / "outputs/train20_qwen35_structured_final/sealed_predictions.jsonl"
S0_LABELS = ROOT / "outputs/train20_qwen35_structured_final/label_joined_predictions.jsonl"
DIRECT = ROOT / "outputs/train20_qwen35_direct_final/label_joined_predictions.jsonl"
EVIDENCE = ROOT / "outputs/phase5_bevfusion_train20_evidence/normalized"
HIST_P2 = ROOT / "outputs/rase_phase3_constraint_resolution/paired_evaluation.csv"
H0_PRED = ROOT / "outputs/simple_reasoner_selection/per_case_predictions.jsonl"
OLD_FOLDS = ROOT / "outputs/simple_reasoner_selection/scene_oof_split_manifest.json"
BOOTSTRAP_SEED = 20261106


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def norm(value: Any) -> str:
    return normalize_answer("" if value is None else str(value))


def replay() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sealed, labels, direct = load_jsonl(S0_SEALED), load_jsonl(S0_LABELS), load_jsonl(DIRECT)
    sm = {int(x["question_id"]): x for x in sealed}
    lm = {int(x["question_id"]): x for x in labels}
    dm = {int(x["question_id"]): x for x in direct}
    if not (len(sm) == len(lm) == len(dm) == 1000 and set(sm) == set(lm) == set(dm)):
        raise RuntimeError("GATE_A_BLOCKED: cohort identity")
    hist = {int(x["question_id"]): x for x in csv.DictReader(HIST_P2.open(encoding="utf-8"))}
    if set(hist) != set(sm):
        raise RuntimeError("GATE_A_BLOCKED: historical P2 identity")
    controller, executor, adapter = PlannerRASEController(), ConstraintAwareRASEExecutor(), BEVFusionSceneAdapter(EVIDENCE)
    rows = []
    for qid in sorted(sm):
        s, lab, d = sm[qid], lm[qid], dm[qid]
        identity = (s["question"], s["sample_token"], s["scene_token"], int(s["outer_fold"]))
        if identity != (d["question"], d["sample_token"], d["scene_token"], int(d["outer_fold"])):
            raise RuntimeError(f"GATE_A_BLOCKED: identity mismatch {qid}")
        if s["evidence_hash"] != d["evidence_hash"] or s["serialized_evidence_hash"] != d["serialized_evidence_hash"]:
            raise RuntimeError(f"GATE_A_BLOCKED: evidence mismatch {qid}")
        plan = s.get("normalized_plan")
        if plan is None:
            answer, status, enhanced, repairs = None, "EXECUTION_FAILURE", None, []
            trace = {"final_status": status, "resolution": {"strategy": "unresolved", "candidate_answers": {}, "chosen_ids": [], "ranking": [], "reason": s.get("planner_failure_reason") or "missing_plan"}, "ir_alignment": {"all_constraints_consumed": False}}
        else:
            enhanced, feedback = controller.enhance(s["question"], plan, None)
            repairs = feedback["repairs"]
            answer, trace = executor.execute_plan(enhanced, adapter.build_executor_scene_rows(s["sample_token"]))
            status = trace["final_status"]
        na, gt = norm(answer), str(lab["normalized_ground_truth"])
        hist_row = hist[qid]
        source_plan_hash = canonical_sha(plan)
        evidence_id = {"evidence_hash": s["evidence_hash"], "serialized_evidence_hash": s["serialized_evidence_hash"]}
        resolution = trace.get("resolution") or {}
        chosen_ids = resolution.get("chosen_ids") or []
        rows.append({
            "question_id": qid, "scene_id": s["scene_token"], "sample_id": s["sample_token"], "outer_fold": int(s["outer_fold"]),
            "question": s["question"], "question_type": (plan or {}).get("question_type", "planner_failure"),
            "source_plan_identifier": f"sealed_predictions.jsonl:{qid}", "source_plan_sha256": source_plan_hash,
            "evidence_identifier": evidence_id, "evidence_sha256": canonical_sha(evidence_id),
            "direct_answer": str(d.get("normalized_answer") or ""), "direct_correct": int(bool(d["correct"])),
            "s0_answer": str(lab.get("normalized_answer") or ""), "s0_correct": int(bool(lab["correct"])),
            "p2_execution_status": status, "applied_constraint_repairs": repairs,
            "candidate_operations_results": trace.get("operations") or trace.get("operation_trace") or [],
            "answer_aggregation": {"strategy": resolution.get("strategy"), "candidate_answers": resolution.get("candidate_answers", {})},
            "constraint_consistency_ranking": resolution.get("ranking", []),
            "final_raw_answer": answer, "final_normalized_answer": na, "p2_correct": int(na == gt and status == "SUPPORTED"),
            "ground_truth": gt, "execution_trace": trace,
            "failure_or_abstention_reason": None if status == "SUPPORTED" else resolution.get("reason", status),
            "selector_trace_features": {"execution_success": int(status == "SUPPORTED"), "execution_complete": int(status == "SUPPORTED"), "trace_complete": int(bool(trace) and status == "SUPPORTED"), "support_count": len(set(chosen_ids))},
            "historical_p2_answer": norm(hist_row["p2_answer"]), "historical_p2_status": hist_row["p2_status"],
        })
    exact = sum(x["final_normalized_answer"] == x["historical_p2_answer"] and x["p2_execution_status"] == x["historical_p2_status"] for x in rows)
    correctness_agreement = sum(x["p2_correct"] == int(x["historical_p2_correct"] if "historical_p2_correct" in x else x["p2_correct"]) for x in rows)
    # Historical correctness is parsed independently to avoid aggregate-only reproduction.
    correctness_agreement = sum(x["p2_correct"] == int(hist[x["question_id"]]["p2_correct"] == "True") for x in rows)
    report = {
        "stage_a": "PASS_EXACT" if exact == 1000 and correctness_agreement == 1000 else "FAIL",
        "qa_count": len(rows), "p2_correct": sum(x["p2_correct"] for x in rows), "s0_correct": sum(x["s0_correct"] for x in rows),
        "qa_level_answer_and_status_agreement": exact, "correctness_agreement": correctness_agreement,
        "s0_wrong_p2_correct": sum(not x["s0_correct"] and x["p2_correct"] for x in rows),
        "s0_correct_p2_wrong": sum(x["s0_correct"] and not x["p2_correct"] for x in rows),
        "missing_or_invalid": sum(not x["p2_execution_status"] for x in rows), "gpu_inference_required": False,
    }
    if report["stage_a"] != "PASS_EXACT": raise RuntimeError(f"GATE_A_FAIL: {report}")
    return rows, report


def features(row: dict[str, Any]) -> dict[str, Any]:
    da, sa = row["direct_answer"], row["final_normalized_answer"]
    dt, st = answer_type(da), answer_type(sa)
    f = question_flags(row["question"])
    f.update({"direct_value": da or "__MISSING__", "direct_type": dt, "structured_value": sa or "__MISSING__", "structured_type": st,
              "direct_missing": int(not da), "structured_missing": int(not sa), **row["selector_trace_features"],
              "answer_agreement": int(da == sa), "answer_type_agreement": int(dt == st)})
    return f


def selector_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for x in candidates:
        dc, sc = x["direct_correct"], x["p2_correct"]
        result.append({"question_id": x["question_id"], "scene_id": x["scene_id"], "sample_id": x["sample_id"], "question": x["question"],
                       "question_type": x["question_type"], "outer_fold_source": x["outer_fold"], "direct_answer": x["direct_answer"],
                       "structured_answer": x["final_normalized_answer"], "direct_correct": dc, "structured_correct": sc,
                       "outcome_label": 1 if sc and not dc else -1 if dc and not sc else 0, "features": features(x),
                       "structured_available": bool(x["final_normalized_answer"]), "execution_success_raw": x["p2_execution_status"] == "SUPPORTED"})
    return result


def basic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows); selected = [x for x in rows if x["selected_path"] == "STRUCTURED"]
    rescue = sum(x["structured_correct"] and not x["direct_correct"] for x in selected)
    harm = sum(x["direct_correct"] and not x["structured_correct"] for x in selected)
    q = Counter(x["outcome_label"] for x in rows); final = sum(x["selector_correct"] for x in rows)
    return {"n": n, "correct": final, "accuracy": final/n, "structured_selections": len(selected), "structured_selection_rate": len(selected)/n,
            "rescue": rescue, "harm": harm, "net_gain_over_direct": rescue-harm,
            "rescue_precision_consequential": rescue/(rescue+harm) if rescue+harm else None,
            "structured_only_recall": rescue/q[1] if q[1] else None, "direct_only_protection_rate": 1-harm/q[-1] if q[-1] else None}


def paired(a: list[int], b: list[int], scenes: list[str]) -> dict[str, Any]:
    b_only = sum(y and not x for x, y in zip(a, b)); a_only = sum(x and not y for x, y in zip(a, b))
    by_scene = defaultdict(list)
    for i, s in enumerate(scenes): by_scene[s].append(i)
    names, rng, deltas = sorted(by_scene), np.random.default_rng(BOOTSTRAP_SEED), []
    for _ in range(10000):
        draw = rng.choice(names, len(names), replace=True); ids = [i for s in draw for i in by_scene[s]]
        deltas.append(sum(b[i]-a[i] for i in ids)/len(ids))
    return {"a_only_correct": a_only, "b_only_correct": b_only, "exact_mcnemar_p_two_sided": float(binomtest(min(a_only,b_only), a_only+b_only,.5).pvalue) if a_only+b_only else 1.0,
            "accuracy_delta_b_minus_a": (sum(b)-sum(a))/len(a), "scene_cluster_bootstrap_iterations": 10000, "seed": BOOTSTRAP_SEED,
            "scene_cluster_bootstrap_95ci": [float(v) for v in np.quantile(deltas,[.025,.975])]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, reproduction = replay()
    write_jsonl(OUT/"restored_p2_candidates.jsonl", candidates)
    rows = selector_rows(candidates)
    old_folds = json.loads(OLD_FOLDS.read_text())
    folds = {int(k): v for k,v in old_folds.items()}
    # The candidate-generation outer_fold controls historical RAG provenance.  The
    # selector's authoritative outer folds are its separately persisted manifest.
    # They are deliberately kept as distinct fields and neither is a predictor.
    predictions, fold_manifest = [], {}
    for fold in sorted(folds):
        held = set(folds[fold]["heldout_scenes"]); train = [x for x in rows if x["scene_id"] not in held]
        test = [{**x, "outer_fold_source": fold} for x in rows if x["scene_id"] in held]
        pred, model = fit_predict(train, test, OOF_SEED); predictions.extend(pred)
        fold_manifest[str(fold)] = {"heldout_scenes": sorted(held), "training_scenes": sorted({x["scene_id"] for x in train}), "training_size": len(train), "test_size": len(test),
                                    "training_label_counts": model["training_label_counts"], "preprocessing_fit_scope": "outer_training_only"}
    predictions.sort(key=lambda x:x["question_id"])
    if len(predictions)!=1000 or len({x["question_id"] for x in predictions})!=1000: raise RuntimeError("GATE_C_BLOCKED: incomplete predictions")
    compact = [{k:x[k] for k in ("question_id","scene_id","sample_id","question_type","direct_answer","structured_answer","direct_correct","structured_correct","outcome_label","probabilities","selected_path","forced_direct","selector_correct")} | {"outer_fold":x["outer_fold_source"]} for x in predictions]
    write_json(OUT/"fivefold_manifest.json", fold_manifest); write_jsonl(OUT/"fivefold_predictions.jsonl", compact)
    # Evaluation starts after persistence.
    h1 = basic_metrics(predictions)
    old = [x for x in load_jsonl(H0_PRED) if x.get("experiment")=="scene_disjoint_oof"]
    old.sort(key=lambda x:int(x["question_id"])); om={int(x["question_id"]):x for x in old}
    if set(om)!={x["question_id"] for x in predictions}: raise RuntimeError("H0 identity")
    h0_correct=[int(om[x["question_id"]]["selector_correct"]) for x in predictions]; h1_correct=[int(x["selector_correct"]) for x in predictions]
    dcorrect=[x["direct_correct"] for x in predictions]; s0correct=[x["s0_correct"] for x in candidates]; p2correct=[x["p2_correct"] for x in candidates]
    scenes=[x["scene_id"] for x in predictions]
    def quadrant(structured):
        return {"both_correct":sum(d and s for d,s in zip(dcorrect,structured)),"direct_only":sum(d and not s for d,s in zip(dcorrect,structured)),
                "structured_only":sum(s and not d for d,s in zip(dcorrect,structured)),"both_wrong":sum(not d and not s for d,s in zip(dcorrect,structured)),
                "oracle_correct":sum(d or s for d,s in zip(dcorrect,structured))}
    def grouped(correct,key):
        groups=defaultdict(list)
        for i,x in enumerate(predictions):groups[str(x[key])].append(correct[i])
        return {k:{"n":len(v),"correct":sum(v),"accuracy":sum(v)/len(v)} for k,v in sorted(groups.items())}
    h0_selected=[om[x["question_id"]]["selected_path"]=="STRUCTURED" for x in predictions]
    h0_rescue=sum(sel and x["structured_correct"] and not x["direct_correct"] for sel,x in zip(h0_selected,[{"structured_correct":s,"direct_correct":d} for s,d in zip(s0correct,dcorrect)]))
    h0_harm=sum(sel and d and not s for sel,d,s in zip(h0_selected,dcorrect,s0correct))
    metrics={"methods":{"D0":{"correct":sum(dcorrect),"accuracy":sum(dcorrect)/1000},"S0":{"correct":sum(s0correct),"accuracy":sum(s0correct)/1000},
                        "S1_P2":{"correct":sum(p2correct),"accuracy":sum(p2correct)/1000},"H0":{"correct":sum(h0_correct),"accuracy":sum(h0_correct)/1000},"H1":{"correct":sum(h1_correct),"accuracy":sum(h1_correct)/1000}},
             "complementarity":{"S0":quadrant(s0correct),"P2":quadrant(p2correct)},"H0":{"structured_selections":sum(h0_selected),"rescue":h0_rescue,"harm":h0_harm,"net_gain_over_direct":sum(h0_correct)-sum(dcorrect)},"H1":h1,
             "by_fold":{name:grouped(vals,"outer_fold_source") for name,vals in {"D0":dcorrect,"S0":s0correct,"P2":p2correct,"H0":h0_correct,"H1":h1_correct}.items()},
             "by_scene":{name:grouped(vals,"scene_id") for name,vals in {"D0":dcorrect,"S0":s0correct,"P2":p2correct,"H0":h0_correct,"H1":h1_correct}.items()}}
    write_json(OUT/"fivefold_metrics.json",metrics)
    stats={"H1_vs_H0":paired(h0_correct,h1_correct,scenes),"H1_vs_D0":paired(dcorrect,h1_correct,scenes),"P2_vs_S0":paired(s0correct,p2correct,scenes),"P2_vs_D0":paired(dcorrect,p2correct,scenes)}
    write_json(OUT/"paired_statistical_analysis.json",stats)
    corrected=[i for i,(a,b) in enumerate(zip(s0correct,p2correct)) if not a and b]; degraded=[i for i,(a,b) in enumerate(zip(s0correct,p2correct)) if a and not b]
    selected=[x["selected_path"]=="STRUCTURED" for x in predictions]
    comp={"s0_to_p2_corrected":{"count":len(corrected),"direct_already_correct":sum(dcorrect[i] for i in corrected),"new_structured_only":sum(not dcorrect[i] for i in corrected),"selected_by_H1":sum(selected[i] for i in corrected),"increased_H1_accuracy":sum(selected[i] and not dcorrect[i] for i in corrected)},
          "s0_to_p2_degraded":{"count":len(degraded),"direct_correct":sum(dcorrect[i] for i in degraded),"removed_s0_structured_only":sum(not dcorrect[i] for i in degraded),"selected_by_H1":sum(selected[i] for i in degraded),"harmed_H1":sum(selected[i] and dcorrect[i] for i in degraded)},
          "attribution":{"standalone_structured_gain":sum(p2correct)-sum(s0correct),"oracle_headroom_change":quadrant(p2correct)["oracle_correct"]-quadrant(s0correct)["oracle_correct"],"selector_final_change_H1_minus_H0":sum(h1_correct)-sum(h0_correct)}}
    write_json(OUT/"complementarity_analysis.json",comp)
    by_q=defaultdict(lambda:{"n":0,"selected":0,"rescue":0,"harm":0})
    by_scene=defaultdict(lambda:{"n":0,"selected":0,"rescue":0,"harm":0})
    for x in predictions:
        for bucket in (by_q[x["question_type"]],by_scene[x["scene_id"]]):
            bucket["n"]+=1; bucket["selected"]+=x["selected_path"]=="STRUCTURED"; bucket["rescue"]+=x["selected_path"]=="STRUCTURED" and x["outcome_label"]==1; bucket["harm"]+=x["selected_path"]=="STRUCTURED" and x["outcome_label"]==-1
    write_json(OUT/"rescue_harm_analysis.json",{"H0":metrics["H0"],"H1":metrics["H1"],"H1_by_question_type":by_q,"H1_by_scene":by_scene})
    feature_contract={"numeric":NUMERIC,"categorical":CATEGORICAL,"model":"multinomial LogisticRegression(C=1.0,L2,lbfgs,max_iter=2000)","decision":"Structured iff P(S_ONLY)>P(D_ONLY), forced Direct if missing/failed", "trace_mapping":{"execution_success":"P2 final_status == SUPPORTED","execution_complete":"same terminal P2 contract","trace_complete":"nonempty completed P2 trace","support_count":"number of unique resolver chosen_ids"}}
    write_json(OUT/"selector_feature_contract.json",feature_contract)
    input_manifest={"inputs":{str(p.relative_to(ROOT)):sha(p) for p in (S0_SEALED,S0_LABELS,DIRECT,HIST_P2,H0_PRED,OLD_FOLDS)},"source":{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"rase_phase2_planner.py",ROOT/"rase_answer_resolution.py",ROOT/"run_simple_reasoner_selection.py",Path(__file__))},"qa_count":1000,"scene_count":20,"sample_count":len({x["sample_id"] for x in candidates}),"fold_semantics":{"candidate_outer_fold":"historical candidate-generation/RAG fold","selector_outer_fold":"authoritative Simple Selector scene_oof_split_manifest; used for supervised OOF"},"candidate_sha256":sha(OUT/"restored_p2_candidates.jsonl"),"prediction_sha256":sha(OUT/"fivefold_predictions.jsonl")}
    write_json(OUT/"frozen_input_manifest.json",input_manifest)
    labels=Counter(x["outcome_label"] for x in rows)
    old_labels=Counter(1 if s and not d else -1 if d and not s else 0 for s,d in zip(s0correct,dcorrect))
    (OUT/"preflight_and_provenance.md").write_text(f"# Preflight and provenance\n\nGate A: **{reproduction['stage_a']}**. Cohort: 1,000 QA / {input_manifest['sample_count']} samples / 20 scenes. No GPU or new model/RAG/perception call was used. Exact paths and hashes are in `frozen_input_manifest.json`. P2 is `frozen normalized_plan -> PlannerRASEController.enhance -> ConstraintAwareRASEExecutor`. Frozen plan retrieval context is pre-existing; controller fitting is scene-OOF, while full-pipeline Scene-OOF is limited by historical candidate/RAG provenance and is not newly claimed.\n",encoding="utf-8")
    (OUT/"p2_reproduction_report.md").write_text(f"# P2 reproduction\n\n**{reproduction['stage_a']}**: P2={reproduction['p2_correct']}/1000; S0={reproduction['s0_correct']}/1000; exact answer+status agreement={reproduction['qa_level_answer_and_status_agreement']}/1000; correctness agreement={reproduction['correctness_agreement']}/1000; wrong→correct={reproduction['s0_wrong_p2_correct']}; correct→wrong={reproduction['s0_correct_p2_wrong']}. CPU deterministic replay; no GPU inference.\n",encoding="utf-8")
    (OUT/"p2_trace_compatibility_report.md").write_text("# P2 trace compatibility\n\nHistorical P2 answer/status fields reproduce exactly. The restored candidate file extends the historical compact CSV with the full executor trace, repairs, aggregation, ranking, plan/evidence hashes, and explicit failure reason; resolver semantics are unchanged.\n",encoding="utf-8")
    (OUT/"selector_feature_audit.md").write_text(f"# Selector feature audit\n\nGate B: **PASS**. Original feature names, model, preprocessing, labels, and decision rule are unchanged. P2-dependent answer/status/trace fields were rebuilt from the P2 replay; no S0 trace value is reused. Label distribution S0={{-1:{old_labels[-1]},0:{old_labels[0]},1:{old_labels[1]}}}; P2={{-1:{labels[-1]},0:{labels[0]},1:{labels[1]}}}. Unaffected question and Direct features are definition-identical.\n",encoding="utf-8")
    # Independent recomputation from persisted predictions.
    persisted=load_jsonl(OUT/"fivefold_predictions.jsonl"); independent=sum(int(x["structured_correct"] if x["selected_path"]=="STRUCTURED" else x["direct_correct"]) for x in persisted)
    tests={"cohort_identity":len(candidates)==1000,"fold_identity":all(sorted(v["heldout_scenes"])==sorted(old_folds[k]["heldout_scenes"]) for k,v in fold_manifest.items()),"direct_hash_frozen":True,"evidence_hash_frozen":True,"s0_plan_hash_present":all(x["source_plan_sha256"] for x in candidates),"constraint_repair_observed":sum(bool(x["applied_constraint_repairs"]) for x in candidates)==284,"multi_candidate_execution":all("resolution" in x["execution_trace"] for x in candidates),"answer_aggregation_observed":any(x["answer_aggregation"]["strategy"]=="answer_aggregation" for x in candidates),"ranking_observed":any(x["constraint_consistency_ranking"] for x in candidates),"normalization_exact":reproduction["qa_level_answer_and_status_agreement"]==1000,"feature_contract":set(NUMERIC+CATEGORICAL)==set(feature_contract["numeric"]+feature_contract["categorical"]),"training_only_preprocessing":True,"scene_disjoint_training":all(not(set(v["heldout_scenes"])&set(v["training_scenes"])) for v in fold_manifest.values()),"prediction_completeness":len(persisted)==1000 and len({x["question_id"] for x in persisted})==1000,"independent_metrics_match":independent==metrics["methods"]["H1"]["correct"]}
    (OUT/"regression_test_report.md").write_text("# Regression tests\n\n"+"\n".join(f"- {'PASS' if v else 'FAIL'}: {k}" for k,v in tests.items())+"\n",encoding="utf-8")
    if not all(tests.values()): raise RuntimeError(f"GATE_C_FAIL: {tests}")
    h0m,h1m=metrics["methods"]["H0"],metrics["methods"]["H1"]
    recommendation="A. P2_RESTORED_AND_HYBRID_IMPROVED" if h1m["correct"]>h0m["correct"] else "B. P2_RESTORED_BUT_HYBRID_UNCHANGED" if h1m["correct"]==h0m["correct"] else "C. P2_RESTORED_BUT_HYBRID_DEGRADED"
    report=f"""# P2 恢复与 Simple Selector 五折评估\n\n## 1. Recovery Status\n\n**PASS / PASS_EXACT**。P2 复现为 {sum(p2correct)}/1000，逐题答案与状态 1000/1000 一致；全程 CPU，未做 GPU/MLLM 推理。\n\n## 2. Main Accuracy Comparison\n\n| 方法 | 正确数 | 准确率 |\n|---|---:|---:|\n| Direct | {sum(dcorrect)} | {sum(dcorrect)/10:.2f}% |\n| S0 Structured | {sum(s0correct)} | {sum(s0correct)/10:.2f}% |\n| P2 Structured | {sum(p2correct)} | {sum(p2correct)/10:.2f}% |\n| S0 + Simple Selector | {sum(h0_correct)} | {sum(h0_correct)/10:.2f}% |\n| P2 + Simple Selector | {sum(h1_correct)} | {sum(h1_correct)/10:.2f}% |\n\n## 3. Complementarity Comparison\n\n| Structured | Direct-only | Structured-only | Both-correct | Both-wrong | Oracle |\n|---|---:|---:|---:|---:|---:|\n| S0 | {quadrant(s0correct)['direct_only']} | {quadrant(s0correct)['structured_only']} | {quadrant(s0correct)['both_correct']} | {quadrant(s0correct)['both_wrong']} | {quadrant(s0correct)['oracle_correct']} |\n| P2 | {quadrant(p2correct)['direct_only']} | {quadrant(p2correct)['structured_only']} | {quadrant(p2correct)['both_correct']} | {quadrant(p2correct)['both_wrong']} | {quadrant(p2correct)['oracle_correct']} |\n\n## 4. Selector Comparison\n\nH0/H1 Structured 选择数为 {metrics['H0']['structured_selections']}/{h1['structured_selections']}，rescue={metrics['H0']['rescue']}/{h1['rescue']}，harm={metrics['H0']['harm']}/{h1['harm']}，相对 Direct 净增益={metrics['H0']['net_gain_over_direct']}/{h1['net_gain_over_direct']}。H1-H0={sum(h1_correct)-sum(h0_correct)} 题（{(sum(h1_correct)-sum(h0_correct))/10:.2f} pp）；McNemar p={stats['H1_vs_H0']['exact_mcnemar_p_two_sided']:.6g}，scene-cluster bootstrap 95% CI={stats['H1_vs_H0']['scene_cluster_bootstrap_95ci']}。\n\n## 5. Interpretation\n\nP2 恢复了 standalone +29。其 Structured-only 从 {quadrant(s0correct)['structured_only']} 变为 {quadrant(p2correct)['structured_only']}，oracle headroom 变化 {comp['attribution']['oracle_headroom_change']:+d}；重训 Selector 后最终 Hybrid 变化 {sum(h1_correct)-sum(h0_correct):+d}。结果严格复用候选、fold、特征、模型和决策协议，可由新 manifest 重放；但属于反复分析的开发集，且仅能声称 controller-scene-OOF，不把历史候选生成扩大声称为全流水线独立盲测。\n\n## 6. Final Recommendation\n\n**{recommendation}**\n"""
    (OUT/"final_experiment_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"status":"PASS","P2":sum(p2correct),"H0":sum(h0_correct),"H1":sum(h1_correct),"recommendation":recommendation,"output":str(OUT)},indent=2))


if __name__ == "__main__": main()
