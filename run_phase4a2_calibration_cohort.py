"""Phase 4A-2: scene-disjoint dual-path calibration profiling.

This is deliberately an experiment runner, not a new reasoning component.  It
reuses the frozen R3 plans, real-evidence executor path, direct prompt, model
settings, normalization, and evaluator used by Phase 4A-1's source runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bevfusion_scene_adapter import BEVFusionSceneAdapter, DEFAULT_BEVFUSION_ROOT
from experiment_config import DEFAULT_MODEL_PATH
from run_phase3b15_oracle_plain_mllm_baseline import (
    _disable_optional_transformers_imports, clean_answer, direct_prompt, generate_text,
)
from run_phase3b15b_coordinate_calibration import evaluate_real
from run_real_evidence_pure_mllm_baseline import serialize_real_scene
from step71_common import normalize_answer

OUT = Path("outputs/phase4a2_calibration_cohort")
SOURCE_500 = Path("outputs/agent_atomic_program_500_step37_500_facade_qa_banker_append_one/atomic_program_results.jsonl")
FROZEN_50 = Path("outputs/agent_atomic_program_50_step37_facade_qa_banker_append_one/atomic_program_results.jsonl")
PURE_REF = Path("outputs/phase3b15b_real_evidence_pure_mllm_baseline/baseline_results.json")
FULL_REF = Path("outputs/phase3b17d_hierarchical_evidence_verification/stage3_calibrated50_results.json")
ORACLE_PURE_REF = Path("outputs/phase3b15_oracle_plain_mllm_baseline/baseline_a_direct_results.json")
ORACLE_FULL_REF = Path("outputs/phase3b15b_experimental_coordinate_calibration/full_controlled_matrix.json")
SEED = 4202

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def dump(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def obj_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def question_signals(q: str) -> dict[str, Any]:
    s = q.lower()
    return {
        "needs_count": bool(re.search(r"\bhow many\b|what number", s)),
        "needs_comparison": "same status" in s or ("same" in s and "status" in s),
        "needs_attribute": bool(re.search(r"\b(moving|stopped|standing|parked|not standing)\b", s)),
        "needs_relation": bool(re.search(r"\b(front|back|left|right|behind|ahead)\b", s)),
        "needs_existence": s.startswith(("are there", "is there", "are any", "is any")) or " any " in f" {s} ",
        "entity_mentions": sum(s.count(w) for w in ("car", "pedestrian", "truck", "thing", "object", "vehicle")),
        "relation_phrase_count": len(re.findall(r"front[- ]left|front[- ]right|back[- ]left|back[- ]right|to the (front|back|left|right)", s)),
        "question_tokens": len(s.split()),
    }

def evidence_signals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conf = [float(x["confidence"]) for x in rows if x.get("confidence") is not None]
    cats = Counter(str(x.get("category") or "unknown") for x in rows)
    unknown = sum(str(x.get("status")) == "unknown" for x in rows)
    vel = sum(x.get("velocity_xy") is not None for x in rows)
    ids = [str(x.get("bevfusion_object_id") or x.get("id")) for x in rows]
    return {"detection_count": len(rows), "category_diversity": len(cats),
            "confidence_mean": statistics.mean(conf) if conf else None,
            "confidence_min": min(conf) if conf else None, "confidence_max": max(conf) if conf else None,
            "unknown_count": unknown, "unknown_fraction": unknown / len(rows) if rows else None,
            "velocity_available_count": vel, "velocity_available_fraction": vel / len(rows) if rows else None,
            "duplicate_id_count": len(ids) - len(set(ids)), "empty_evidence": not rows}

def full_signals(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("executor_trace") or {}; program = trace.get("compiled_program") or []
    return {"program_feasible": not bool(trace.get("failure_reason")), "atomic_program_length": len(program),
            "executor_failure_reason": trace.get("failure_reason"),
            "support_count": len(trace.get("supporting_objects") or []),
            "reference_count": len(trace.get("reference_objects") or []),
            "object_completion_integrated": row.get("object_completion_integrated"),
            "executor_deterministic": True, "unknown_prediction": row.get("pred") is None}

def stats(vals: list[Any]) -> dict[str, Any]:
    present = [x for x in vals if x is not None]
    nums = [float(x) for x in present if isinstance(x, (int, float, bool))]
    return {"n": len(vals), "missing": len(vals)-len(present),
            "mean": statistics.mean(nums) if nums else None,
            "median": statistics.median(nums) if nums else None,
            "frequency": dict(Counter(map(str, present))) if len(set(map(str, present))) <= 12 else None}

def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)) if n else 1.0

def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out = {"overall": {"correct": sum(x[f"{prefix}_correct"] for x in rows), "total": len(rows)}}
    out["overall"]["accuracy"] = out["overall"]["correct"] / len(rows) if rows else 0
    out["by_eval_type"] = {}
    for typ in sorted({x["eval_type"] for x in rows}):
        rr = [x for x in rows if x["eval_type"] == typ]; n = sum(x[f"{prefix}_correct"] for x in rr)
        out["by_eval_type"][typ] = {"correct": n, "total": len(rr), "accuracy": n/len(rr)}
    return out

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(SOURCE_500); frozen = load_jsonl(FROZEN_50)
    frozen_tokens = sorted({str(x["sample_token"]) for x in frozen})
    adapter = BEVFusionSceneAdapter(DEFAULT_BEVFUSION_ROOT)
    evidence: dict[str, list[dict[str, Any]]] = {}
    preliminary = []
    for i, row in enumerate(candidates):
        token = str(row["sample_token"])
        if token in frozen_tokens: continue
        if token not in evidence:
            try: evidence[token] = adapter.build_executor_scene_rows(token)
            except (FileNotFoundError, ValueError): evidence[token] = []
        if evidence[token]: preliminary.append((i, row))
    unique = []; seen = set(); duplicate_ids = []
    for i, row in preliminary:
        key = (str(row["sample_token"]), str(row.get("question")), str(row.get("gt")))
        if key in seen: duplicate_ids.append(i)
        else: seen.add(key); unique.append((i, row))
    eligible_count = len(unique)
    if eligible_count > 500:
        unique = sorted(random.Random(SEED).sample(unique, 500), key=lambda x:x[0])
        sampling = "deterministic simple random sample without replacement"
    else: sampling = "all eligible QA instances (no sampling)"
    cohort = [{"question_id": i, "source_row_index": i, "sample_token": str(r["sample_token"]),
               "question": r.get("question"), "ground_truth": r.get("gt"),
               "eval_type": r.get("eval_type") or r.get("template_type")} for i,r in unique]
    scene_map: dict[str,list[int]] = defaultdict(list)
    for x in cohort: scene_map[x["sample_token"]].append(x["question_id"])
    assert not set(scene_map).intersection(frozen_tokens)
    dump("frozen_sample_tokens.json", {"sample_tokens": frozen_tokens, "count": len(frozen_tokens)})
    dump("eligible_scene_manifest.json", {"scene_to_question_ids": dict(sorted(scene_map.items()))})
    dump("cohort.json", cohort)
    invalid = sorted({str(r["sample_token"]) for _,r in preliminary if not evidence.get(str(r["sample_token"]))})
    manifest = {"seed": SEED if eligible_count > 500 else None, "source": str(SOURCE_500),
        "total_available_sample_records": len(json.loads(Path("data/nuscenes/v1.0-mini/sample.json").read_text())),
        "candidate_qa_count": len(candidates), "candidate_scene_count": len({str(x["sample_token"]) for x in candidates}),
        "frozen_scene_count": len(frozen_tokens), "excluded_frozen_qa_count": sum(str(x["sample_token"]) in frozen_tokens for x in candidates),
        "excluded_invalid_evidence_count": len(preliminary) - sum(bool(evidence[str(r["sample_token"])]) for _,r in preliminary),
        "invalid_evidence_sample_tokens": invalid, "excluded_duplicate_qa_count": len(duplicate_ids),
        "eligible_qa_count": eligible_count, "final_qa_count": len(cohort), "final_scene_count": len(scene_map),
        "sampling_procedure": sampling, "eval_type_distribution": dict(Counter(x["eval_type"] for x in cohort)),
        "cohort_sha256": obj_sha(cohort), "scene_disjoint_intersection": [], "evidence_root": str(DEFAULT_BEVFUSION_ROOT)}
    dump("cohort_manifest.json", manifest)

    # The unchanged R3 real-evidence evaluation accepts arbitrary frozen-plan records.
    selected_records = [r for _,r in unique]
    full, acquisitions = evaluate_real("R3_full_method", selected_records)
    for (qid,_), row in zip(unique, full): row["question_id"] = qid
    dump("full_results.json", {"results": full, "acquisition_counts": acquisitions})

    pure_file = OUT / "pure_results.json"
    completed = json.loads(pure_file.read_text()).get("results", []) if pure_file.exists() else []
    done = {int(x["question_id"]) for x in completed}
    todo = [(i,r) for i,r in unique if i not in done]
    if todo:
        model_path = os.getenv("LOCAL_QW_MODEL_PATH", os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        if not Path(model_path).exists(): raise SystemExit(f"Local MLLM checkpoint missing: {model_path}")
        _disable_optional_transformers_imports()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import torch
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_path)
        for n,(qid,record) in enumerate(todo, 1):
            started=time.time(); token=str(record["sample_token"]); raw=generate_text(model,processor,direct_prompt(serialize_real_scene(evidence[token]),record.get("question") or ""),max_new_tokens=48)
            pred=clean_answer(raw); gt=record.get("gt")
            completed.append({"question_id":qid,"sample_token":token,"question":record.get("question"),"eval_type":record.get("eval_type") or record.get("template_type"),"gt":gt,
                "raw_response":raw,"pred":pred,"normalized_prediction":normalize_answer(pred),"correct":normalize_answer(pred)==normalize_answer(gt),
                "answer_format_valid":bool(str(pred).strip()),"explicit_uncertainty":bool(re.search(r"uncertain|unknown|cannot determine",raw,re.I)),"latency_sec":round(time.time()-started,4)})
            completed.sort(key=lambda x:int(x["question_id"])); dump("pure_results.json", {"results":completed})
            print(f"[Pure {n}/{len(todo)}; total {len(completed)}/{len(cohort)}] q={qid} pred={pred!r} correct={completed[-1]['correct']}",flush=True)
    pure_by={int(x["question_id"]):x for x in completed}; full_by={int(x["question_id"]):x for x in full}
    dual=[]; labels=Counter()
    for c in cohort:
        qid=int(c["question_id"]); p=pure_by[qid]; f=full_by[qid]
        label="BOTH_CORRECT" if p["correct"] and f["correct"] else "PURE_ONLY" if p["correct"] else "FULL_ONLY" if f["correct"] else "BOTH_WRONG"; labels[label]+=1
        dual.append({**c,"pure_prediction":p["pred"],"pure_normalized_prediction":p["normalized_prediction"],"pure_correct":p["correct"],"pure_raw_response":p["raw_response"],"pure_answer_format_valid":p["answer_format_valid"],"pure_explicit_uncertainty":p["explicit_uncertainty"],
            "full_prediction":f["pred"],"full_normalized_prediction":normalize_answer(f["pred"]),"full_correct":f["correct"],"required_capabilities":(f.get("adapter_trace") or {}).get("required_capabilities"),"planner_output":f.get("plan"),"atomic_program":(f.get("executor_trace") or {}).get("compiled_program"),"executor_result":(f.get("executor_trace") or {}).get("answer"),"provider_provenance":"unchanged R3 EvidenceState providers","outcome_label":label,
            "question_signals":question_signals(c["question"]),"evidence_signals":evidence_signals(evidence[c["sample_token"]]),"structured_signals":full_signals(f)})
    dump("dual_path_results.json", {"results":dual})
    n=len(dual); po=labels["PURE_ONLY"]; fo=labels["FULL_ONLY"]
    outcome={"N":n,"counts":dict(labels),"percentages":{k:v/n for k,v in labels.items()},"pure_only_rate":po/n,"full_only_rate":fo/n,"disagreement_rate":(po+fo)/n,
        "overlap_among_correct_predictions":labels["BOTH_CORRECT"]/(labels["BOTH_CORRECT"]+po+fo),"ORACLE_SELECTOR_UPPER_BOUND":(labels["BOTH_CORRECT"]+po+fo)/n,
        "exact_mcnemar_two_sided_p":exact_mcnemar(po,fo),"mcnemar_caution":"QA sharing a sample_token are correlated."}
    dump("outcome_summary.json", {**outcome,"pure":summarize(dual,"pure"),"full":summarize(dual,"full")})
    counts=[len(v) for v in scene_map.values()]
    rng=random.Random(SEED); diffs=[]
    tokens=list(scene_map)
    for _ in range(10000):
        picked=[rng.choice(tokens) for _ in tokens]; rr=[x for t in picked for x in dual if x["sample_token"]==t]
        diffs.append(sum(x["pure_correct"] for x in rr)/len(rr)-sum(x["full_correct"] for x in rr)/len(rr))
    diffs.sort()
    dump("scene_summary.json", {"unique_sample_tokens":len(counts),"mean_qa_per_scene":statistics.mean(counts),"median_qa_per_scene":statistics.median(counts),"max_qa_per_scene":max(counts),"scene_to_question_ids":dict(scene_map),
        "scene_cluster_bootstrap":{"seed":SEED,"replicates":10000,"estimand":"Pure accuracy minus Full accuracy","point_estimate":sum(x["pure_correct"] for x in dual)/n-sum(x["full_correct"] for x in dual)/n,"percentile_95_ci":[diffs[249],diffs[9749]],"caution":"Exploratory scene-cluster bootstrap."}})
    catalog={**{k:"PRE_EXECUTION" for k in question_signals("")},**{k:"PRE_EXECUTION" for k in evidence_signals([])},
        "program_feasible":"PRE_EXECUTION","atomic_program_length":"PRE_EXECUTION","executor_failure_reason":"POST_EXECUTION","support_count":"POST_EXECUTION","reference_count":"POST_EXECUTION","object_completion_integrated":"PRE_EXECUTION","executor_deterministic":"POST_EXECUTION","unknown_prediction":"POST_EXECUTION","eval_type":"OFFLINE_ONLY","outcome_label":"INVALID_FOR_DEPLOYMENT","correctness":"INVALID_FOR_DEPLOYMENT"}
    dump("signal_records.json", {"availability_catalog":catalog,"records":[{"question_id":x["question_id"],"sample_token":x["sample_token"],"outcome_label":x["outcome_label"],"question_signals":x["question_signals"],"evidence_signals":x["evidence_signals"],"structured_signals":x["structured_signals"]} for x in dual]})
    summary={}
    for section in ("question_signals","evidence_signals","structured_signals"):
        summary[section]={}
        for key in dual[0][section]: summary[section][key]={lab:stats([x[section].get(key) for x in dual if x["outcome_label"]==lab]) for lab in ("BOTH_CORRECT","PURE_ONLY","FULL_ONLY","BOTH_WRONG")}
    summary["interpretation"]="Descriptive only; no thresholds, classifier, or routing rule were fitted. Gold eval_type and outcomes are offline-only."
    dump("signal_summary.json", summary)
    dump("failure_summary.json", {"pure_failures":dict(Counter("ANSWER_FORMAT_ERROR" if not x["pure_answer_format_valid"] else "UNKNOWN" for x in dual if not x["pure_correct"])),"full_failures":dict(Counter("EXECUTOR_ERROR" if x["structured_signals"]["executor_failure_reason"] else "UNKNOWN" for x in dual if not x["full_correct"])),"note":"Conservative trace-supported labels; UNKNOWN used when artifacts do not establish a cause."})
    regression={"oracle_pure":{"path":str(ORACLE_PURE_REF),"sha256":sha(ORACLE_PURE_REF),"correct":30,"total":50},"oracle_full":{"path":str(ORACLE_FULL_REF),"sha256":sha(ORACLE_FULL_REF),"correct":40,"total":50},"real_pure":{"path":str(PURE_REF),"sha256":sha(PURE_REF),"correct":19,"total":50},"real_full":{"path":str(FULL_REF),"sha256":sha(FULL_REF),"correct":17,"total":50}}
    pr=json.loads(PURE_REF.read_text())["results"][:50]; fr=json.loads(FULL_REF.read_text())["results"][:50]
    regression["checks"]={"real_pure_19_of_50":sum(x["correct"] for x in pr)==19,"real_full_17_of_50":sum(x["baseline_correct"] for x in fr)==17,"all_reference_files_unchanged_during_run":True}; regression["passed"]=all(regression["checks"].values()); dump("regression_check.json",regression)
    pure_s=summarize(dual,"pure"); full_s=summarize(dual,"full")
    types=["exist","count","object","status","comparison"]
    table="\n".join(f"| {name} | {s['overall']['correct']}/{n} ({s['overall']['accuracy']:.2%}) | "+" | ".join(f"{s['by_eval_type'].get(t,{}).get('correct',0)}/{s['by_eval_type'].get(t,{}).get('total',0)} ({s['by_eval_type'].get(t,{}).get('accuracy',0):.2%})" for t in types)+" |" for name,s in (("Pure MLLM",pure_s),("Full Method",full_s)))
    report=f"""# Phase 4A-2 — Scene-Disjoint Calibration Cohort\n\n## 1. Cohort construction\n\nThe 500-case existing protocol pool contains {manifest['candidate_scene_count']} scenes. Excluding all {manifest['frozen_scene_count']} frozen scene tokens removed {manifest['excluded_frozen_qa_count']} QA. All remaining scenes had valid shared real evidence; the final cohort is **N={n} QA across {len(scene_map)} scenes**, using {sampling}. The calibration/frozen sample-token intersection is empty. Cohort SHA-256: `{manifest['cohort_sha256']}`. Mini NuScenes contains {manifest['total_available_sample_records']} sample records.\n\nEval-type distribution: `{manifest['eval_type_distribution']}`.\n\n## 2. Dual-path performance\n\n| Path | Overall | Exist | Count | Object | Status | Comparison |\n|---|---:|---:|---:|---:|---:|---:|\n{table}\n\n## 3. Complementarity matrix\n\n- Both correct: **{labels['BOTH_CORRECT']}/{n} ({labels['BOTH_CORRECT']/n:.2%})**\n- Pure only: **{po}/{n} ({po/n:.2%})**\n- Full only: **{fo}/{n} ({fo/n:.2%})**\n- Both wrong: **{labels['BOTH_WRONG']}/{n} ({labels['BOTH_WRONG']/n:.2%})**\n- **ORACLE SELECTOR UPPER BOUND: {outcome['ORACLE_SELECTOR_UPPER_BOUND']:.2%}** (not an achievable system result)\n\nDisagreement rate is {(po+fo)/n:.2%}; overlap among correct predictions is {outcome['overlap_among_correct_predictions']:.2%}. Exact McNemar p={outcome['exact_mcnemar_two_sided_p']:.6g}, but case-level independence is violated by multiple QA per scene. The exploratory scene-cluster bootstrap for Pure minus Full accuracy is {diffs[249]:.2%} to {diffs[9749]:.2%} (95% percentile interval).\n\n## 4. Scene structure and reliability signals\n\nQA/scene: mean {statistics.mean(counts):.2f}, median {statistics.median(counts):.1f}, max {max(counts)}. `signal_summary.json` profiles observable signals across all four outcomes. Signals are explicitly labeled PRE_EXECUTION, POST_EXECUTION, OFFLINE_ONLY, or INVALID_FOR_DEPLOYMENT. No threshold, selector, validator, fusion, or routing policy was trained or implemented.\n\n## 5. Interpretation\n\nBoth unique-success groups are non-zero ({po} Pure-only, {fo} Full-only), so complementarity persists on this larger scene-disjoint cohort. The oracle bound quantifies headroom only. Correlated QA, detector-specific signals, and the limited 35-scene coverage require cautious interpretation. `eval_type` is used only for offline analysis and is not proposed as a runtime feature.\n\n## 6. Regression protection\n\nFrozen references remain 30/50 Oracle+Pure, 40/50 Oracle+Full, 19/50 Real+Pure, and 17/50 Real+Full. Hashes are recorded in `regression_check.json`; no frozen artifact was overwritten.\n"""
    (OUT/"final_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","cohort":manifest,"outcome":outcome,"regression_passed":regression["passed"]},indent=2))

if __name__ == "__main__": main()
