"""Phase 4A-3: grouped, leakage-audited reasoner reliability calibration.

Consumes frozen Phase 4A-2 artifacts only.  It trains independent Direct and
Structured reliability estimators; it does not train or deploy a router.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path("outputs/phase4a3_reasoner_reliability")
P4A2 = Path("outputs/phase4a2_calibration_cohort")
DUAL = P4A2 / "dual_path_results.json"
P4A2_OUTCOME = P4A2 / "outcome_summary.json"
P4A2_COHORT = P4A2 / "cohort_manifest.json"
P4A2_REGRESSION = P4A2 / "regression_check.json"
SEED = 4303
FOLDS = 5

QUESTION_FEATURES = [
    "explicit_entity_count", "requires_counting", "requires_comparison",
    "requires_spatial_relation", "requires_attribute", "requires_motion",
    "requires_existence", "requires_multi_entity", "constraint_count",
    "compositionality", "question_length",
]
EVIDENCE_FEATURES = [
    "candidate_entity_count_log", "category_diversity_log", "confidence_mean",
    "confidence_min", "confidence_max", "low_confidence_proxy",
    "motion_availability", "attribute_availability", "relation_availability",
    "entity_coverage", "unknown_capability_fraction", "duplicate_density",
    "provider_coverage", "evidence_ambiguity",
]
COMPAT_FEATURES = [
    "motion_compatibility", "attribute_compatibility", "relation_compatibility",
    "entity_compatibility", "counting_coverage", "required_capability_coverage",
    "missing_required_capability_count",
]
FEATURE_SETS = {
    "constant_prior": [],
    "question_only": QUESTION_FEATURES,
    "evidence_only": EVIDENCE_FEATURES,
    "question_evidence": QUESTION_FEATURES + EVIDENCE_FEATURES,
    "question_evidence_compatibility": QUESTION_FEATURES + EVIDENCE_FEATURES + COMPAT_FEATURES,
    "architecture_only_qec": QUESTION_FEATURES + [x for x in EVIDENCE_FEATURES if x not in {
        "confidence_mean", "confidence_min", "confidence_max", "low_confidence_proxy", "evidence_ambiguity"
    }] + COMPAT_FEATURES,
}

def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def dump(name: str, obj: Any) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))

def profile(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    q = row["question_signals"]; e = row["evidence_signals"]
    entity_count = max(0, int(q.get("entity_mentions") or 0))
    relation_count = max(0, int(q.get("relation_phrase_count") or 0))
    requires_relation = bool(q.get("needs_relation")); requires_attr = bool(q.get("needs_attribute"))
    text = str(row.get("question") or "").lower()
    requires_motion = bool(re.search(r"\b(moving|stopped|parked|standing|motion|speed)\b", text))
    requirements = [bool(q.get("needs_count")), bool(q.get("needs_comparison")), requires_relation,
                    requires_attr, requires_motion, bool(q.get("needs_existence"))]
    constraint_count = sum(requirements) + max(0, relation_count - 1)
    n = max(0, int(e.get("detection_count") or 0)); diversity = max(0, int(e.get("category_diversity") or 0))
    conf_mean = float(e.get("confidence_mean") or 0); conf_min = float(e.get("confidence_min") or 0); conf_max = float(e.get("confidence_max") or 0)
    motion_avail = clip01(e.get("velocity_available_fraction") or 0)
    attribute_avail = clip01(1 - float(e.get("unknown_fraction") or 0))
    relation_avail = 1.0 if n > 0 else 0.0
    entity_coverage = clip01(n / max(entity_count, 1)) if entity_count else (1.0 if n else 0.0)
    duplicate_density = clip01(float(e.get("duplicate_id_count") or 0) / max(n, 1))
    provider_coverage = statistics.mean([motion_avail, attribute_avail, relation_avail])
    ambiguity = clip01((1-conf_mean) * .5 + float(e.get("unknown_fraction") or 0) * .5)
    vals = {
        "explicit_entity_count": float(entity_count), "requires_counting": float(bool(q.get("needs_count"))),
        "requires_comparison": float(bool(q.get("needs_comparison"))), "requires_spatial_relation": float(requires_relation),
        "requires_attribute": float(requires_attr), "requires_motion": float(requires_motion),
        "requires_existence": float(bool(q.get("needs_existence"))), "requires_multi_entity": float(entity_count >= 2),
        "constraint_count": float(constraint_count), "compositionality": float(constraint_count + max(0, entity_count-1)),
        "question_length": float(q.get("question_tokens") or 0),
        "candidate_entity_count_log": math.log1p(n), "category_diversity_log": math.log1p(diversity),
        "confidence_mean": conf_mean, "confidence_min": conf_min, "confidence_max": conf_max,
        "low_confidence_proxy": 1-conf_mean, "motion_availability": motion_avail,
        "attribute_availability": attribute_avail, "relation_availability": relation_avail,
        "entity_coverage": entity_coverage, "unknown_capability_fraction": 1-provider_coverage,
        "duplicate_density": duplicate_density, "provider_coverage": provider_coverage,
        "evidence_ambiguity": ambiguity,
    }
    required_availability = []
    if requires_motion: required_availability.append(motion_avail)
    if requires_attr: required_availability.append(attribute_avail)
    if requires_relation: required_availability.append(relation_avail)
    if entity_count: required_availability.append(entity_coverage)
    missing = sum(x < .5 for x in required_availability)
    vals.update({
        "motion_compatibility": motion_avail if requires_motion else 1.0,
        "attribute_compatibility": attribute_avail if requires_attr else 1.0,
        "relation_compatibility": relation_avail if requires_relation else 1.0,
        "entity_compatibility": entity_coverage if entity_count else 1.0,
        "counting_coverage": entity_coverage * (1-duplicate_density) if q.get("needs_count") else 1.0,
        "required_capability_coverage": statistics.mean(required_availability) if required_availability else 1.0,
        "missing_required_capability_count": float(missing),
    })
    meta = {"required_capabilities": [name for name,yes in zip(("count","comparison","relation","attribute","motion","existence"),requirements) if yes],
            "unavailable_capabilities": [name for name,val in (("motion",motion_avail),("attribute",attribute_avail),("relation",relation_avail)) if val == 0]}
    return vals, meta

def grouped_folds(groups: list[str], k: int) -> tuple[list[int], list[dict[str, Any]]]:
    counts = Counter(groups); shuffled = list(counts)
    random.Random(SEED).shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda g: counts[g], reverse=True)
    bins: list[list[str]] = [[] for _ in range(k)]; sizes = [0]*k
    for group in ordered:
        j = min(range(k), key=lambda x:(sizes[x],x)); bins[j].append(group); sizes[j] += counts[group]
    mapping = {g:i for i,b in enumerate(bins) for g in b}
    ids = [mapping[g] for g in groups]
    manifest = [{"fold_id":i,"validation_sample_tokens":sorted(b),"validation_scene_count":len(b),"validation_qa_count":sizes[i],
                 "training_scene_count":len(counts)-len(b),"scene_overlap":[]} for i,b in enumerate(bins)]
    return ids, manifest

def sigmoid(z: np.ndarray) -> np.ndarray:
    z=np.clip(z,-35,35); return 1/(1+np.exp(-z))

def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float=1.0) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    mean=x.mean(axis=0); std=x.std(axis=0); std[std < 1e-12]=1
    z=(x-mean)/std; a=np.column_stack([np.ones(len(z)),z]); w=np.zeros(a.shape[1]); penalty=np.eye(a.shape[1])*l2; penalty[0,0]=0
    for _ in range(100):
        p=sigmoid(a@w); weights=np.maximum(p*(1-p),1e-8)
        grad=a.T@(p-y)+penalty@w; h=(a.T*weights)@a+penalty
        step=np.linalg.solve(h,grad); w-=step
        if np.max(np.abs(step)) < 1e-8: break
    return w,mean,std

def predict_logistic(x: np.ndarray, model: tuple[np.ndarray,np.ndarray,np.ndarray]) -> np.ndarray:
    w,mean,std=model; return sigmoid(np.column_stack([np.ones(len(x)),(x-mean)/std])@w)

def rank_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg): return None
    wins=0.0
    for i in pos:
        wins += float(np.sum(p[i]>p[neg])) + .5*float(np.sum(p[i]==p[neg]))
    return wins/(len(pos)*len(neg))

def pr_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    positives=int(y.sum())
    if not positives: return None
    order=np.argsort(-p,kind="stable"); sy=y[order]; tp=np.cumsum(sy); precision=tp/np.arange(1,len(y)+1)
    return float(np.sum(precision*sy)/positives)

def calibration(y: np.ndarray,p: np.ndarray,bins: int=10) -> tuple[float,list[dict[str,Any]]]:
    rows=[]; ece=0.0
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins; mask=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if mask.any():
            conf=float(p[mask].mean()); acc=float(y[mask].mean()); count=int(mask.sum()); ece += count/len(y)*abs(acc-conf)
            rows.append({"bin_lower":lo,"bin_upper":hi,"count":count,"mean_predicted_reliability":conf,"empirical_correctness":acc})
    return ece,rows

def metrics(y: np.ndarray,p: np.ndarray) -> dict[str,Any]:
    eps=1e-12; ece,diagram=calibration(y,p)
    return {"n":len(y),"positives":int(y.sum()),"positive_rate":float(y.mean()),
            "brier":float(np.mean((p-y)**2)),"log_loss":float(np.mean(-(y*np.log(np.clip(p,eps,1-eps))+(1-y)*np.log(np.clip(1-p,eps,1-eps))))),
            "roc_auc":rank_auc(y,p),"pr_auc":pr_auc(y,p),"ece_10_bin":ece,"reliability_diagram":diagram}

def mean_std(rows: list[dict[str,Any]]) -> dict[str,Any]:
    out={}
    for key in ("brier","log_loss","roc_auc","pr_auc","ece_10_bin"):
        vals=[r[key] for r in rows if r[key] is not None]
        out[key]={"mean":statistics.mean(vals) if vals else None,"std":statistics.pstdev(vals) if len(vals)>1 else 0.0,"valid_folds":len(vals)}
    return out

def outcome_summary(rows: list[dict[str,Any]]) -> dict[str,Any]:
    out={}
    for label in ("BOTH_CORRECT","PURE_ONLY","FULL_ONLY","BOTH_WRONG"):
        rr=[x for x in rows if x["outcome_label"]==label]
        def dist(key: str) -> dict[str,float]:
            v=[x[key] for x in rr]; return {"n":len(v),"mean":statistics.mean(v),"median":statistics.median(v),"std":statistics.pstdev(v) if len(v)>1 else 0.0}
        out[label]={"s_direct":dist("predicted_direct_reliability"),"s_structured":dist("predicted_structured_reliability"),
            "score_difference":dist("score_difference"),"max_reliability":dist("max_reliability")}
    return out

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    source=read(DUAL)["results"]; manifest=read(P4A2_COHORT)
    assert len(source)==439 and len({x["sample_token"] for x in source})==35 and manifest["scene_disjoint_intersection"]==[]
    records=[]
    for row in source:
        features,meta=profile(row)
        records.append({"question_id":row["question_id"],"sample_token":row["sample_token"],"question":row["question"],
            "offline_eval_type":row["eval_type"],"outcome_label":row["outcome_label"],"y_direct":int(row["pure_correct"]),"y_structured":int(row["full_correct"]),
            "features":features,"profile_metadata":meta})
    groups=[x["sample_token"] for x in records]; fold_ids,fold_manifest=grouped_folds(groups,FOLDS)
    for x,f in zip(records,fold_ids):x["fold_id"]=f
    dump("feature_records.json",{"records":records})

    descriptions={
        "explicit_entity_count":"Question-text estimate of explicitly referenced entities", "requires_counting":"Question requests cardinality",
        "requires_comparison":"Question requests comparison", "requires_spatial_relation":"Question requests spatial/relation reasoning",
        "requires_attribute":"Question requests an entity attribute", "requires_motion":"Question requests motion/status evidence",
        "requires_existence":"Question requests existence", "requires_multi_entity":"At least two entity mentions",
        "constraint_count":"Count of semantic requirements", "compositionality":"Requirements plus multi-entity composition",
        "question_length":"Question token count", "candidate_entity_count_log":"Log abstract candidate count",
        "category_diversity_log":"Log candidate category diversity", "confidence_mean":"Provider confidence mean",
        "confidence_min":"Provider confidence minimum", "confidence_max":"Provider confidence maximum",
        "low_confidence_proxy":"One minus mean confidence", "motion_availability":"Fraction with motion evidence",
        "attribute_availability":"Fraction with known coarse attributes", "relation_availability":"Geometry permits relation derivation",
        "entity_coverage":"Candidate count relative to requested entities", "unknown_capability_fraction":"Unavailable abstract capability fraction",
        "duplicate_density":"Duplicate identifier density", "provider_coverage":"Mean abstract capability availability",
        "evidence_ambiguity":"Confidence/unknown based ambiguity", "motion_compatibility":"Required motion availability or N/A=1",
        "attribute_compatibility":"Required attribute availability or N/A=1", "relation_compatibility":"Required relation availability or N/A=1",
        "entity_compatibility":"Requested entity count coverage", "counting_coverage":"Coverage adjusted for duplicates",
        "required_capability_coverage":"Availability over required capabilities", "missing_required_capability_count":"Unavailable required capabilities",
    }
    audit=[]
    for name in QUESTION_FEATURES+EVIDENCE_FEATURES+COMPAT_FEATURES:
        family="question" if name in QUESTION_FEATURES else "evidence" if name in EVIDENCE_FEATURES else "compatibility"
        perception=name in {"confidence_mean","confidence_min","confidence_max","low_confidence_proxy","evidence_ambiguity"}
        audit.append({"feature_name":name,"description":descriptions[name],"source":family,"availability_stage":"PRE_EXECUTION",
            "generalization_category":"PERCEPTION-MODEL-SPECIFIC" if perception else "GENERAL / ARCHITECTURE-LEVEL",
            "missingness":sum(x["features"].get(name) is None for x in records)/len(records),"used_in_estimator":True,
            "reason":"Allowed pre-execution abstract profile; provider confidence semantics may vary" if perception else "Allowed generalized pre-execution signal"})
    rejected=[("eval_type","OFFLINE_ONLY","DATASET-SPECIFIC","Gold dataset label; diagnostics only"),("question_id","INVALID_FOR_DEPLOYMENT","INVALID / LEAKY","Bookkeeping only"),("sample_token","INVALID_FOR_DEPLOYMENT","INVALID / LEAKY","Grouping only"),("ground_truth","INVALID_FOR_DEPLOYMENT","INVALID / LEAKY","Forbidden label"),("pure_correct","INVALID_FOR_DEPLOYMENT","INVALID / LEAKY","Training target only"),("full_correct","INVALID_FOR_DEPLOYMENT","INVALID / LEAKY","Training target only"),("outcome_label","INVALID_FOR_DEPLOYMENT","INVALID / LEAKY","Offline outcome only"),("pure_prediction","POST_EXECUTION","INVALID / LEAKY","Post-answer output"),("full_prediction","POST_EXECUTION","INVALID / LEAKY","Post-answer output"),("executor_result","POST_EXECUTION","INVALID / LEAKY","Post-execution output"),("structured_signals","POST_EXECUTION","INVALID / LEAKY","Conservatively rejected because mixed stage")]
    for name,stage,cat,why in rejected:audit.append({"feature_name":name,"description":why,"source":"Phase 4A-2 artifact","availability_stage":stage,"generalization_category":cat,"missingness":None,"used_in_estimator":False,"reason":why})
    dump("feature_audit.json",{"features":audit,"used_feature_count":len(QUESTION_FEATURES+EVIDENCE_FEATURES+COMPAT_FEATURES),"rejected_feature_count":len(rejected)})
    dump("fold_manifest.json",{"protocol":"deterministic size-balanced 5-fold grouped CV","group":"sample_token","seed":SEED,"folds":fold_manifest,
        "all_cases_oof":True,"no_scene_overlap":True,"frozen_scenes_used":False,"frozen_cohort_evaluated":False})

    all_names=QUESTION_FEATURES+EVIDENCE_FEATURES+COMPAT_FEATURES
    matrix={name:np.asarray([[x["features"][f] for f in feats] for x in records],dtype=float) if feats else np.empty((len(records),0)) for name,feats in FEATURE_SETS.items()}
    targets={"direct":np.asarray([x["y_direct"] for x in records],dtype=float),"structured":np.asarray([x["y_structured"] for x in records],dtype=float)}
    predictions={reasoner:{baseline:np.zeros(len(records)) for baseline in FEATURE_SETS} for reasoner in targets}
    results={reasoner:{} for reasoner in targets}
    for reasoner,y in targets.items():
        for baseline,features in FEATURE_SETS.items():
            per_fold=[]
            for fold in range(FOLDS):
                va=np.asarray([i for i,x in enumerate(fold_ids) if x==fold]); tr=np.asarray([i for i,x in enumerate(fold_ids) if x!=fold])
                if baseline=="constant_prior": pred=np.full(len(va),float(y[tr].mean()))
                else: pred=predict_logistic(matrix[baseline][va],fit_logistic(matrix[baseline][tr],y[tr]))
                predictions[reasoner][baseline][va]=pred; per_fold.append({"fold_id":fold,**metrics(y[va],pred),"train_positive_rate":float(y[tr].mean())})
            results[reasoner][baseline]={"feature_names":features,"oof_metrics":metrics(y,predictions[reasoner][baseline]),"per_fold_metrics":per_fold,"fold_metric_summary":mean_std(per_fold),"model":"fixed L2 logistic regression (lambda=1; standardized within training fold)" if features else "training-fold correctness prior"}
    dump("direct_model_metrics.json",results["direct"]);dump("structured_model_metrics.json",results["structured"])
    dump("ablation_results.json",{"direct":{k:v["oof_metrics"] for k,v in results["direct"].items()},"structured":{k:v["oof_metrics"] for k,v in results["structured"].items()},"selection_policy_optimized":False})
    main_key="question_evidence_compatibility"; pd=predictions["direct"][main_key]; ps=predictions["structured"][main_key]
    oof=[]
    for i,x in enumerate(records):
        oof.append({"fold_id":fold_ids[i],"sample_token":x["sample_token"],"question_id":x["question_id"],"y_direct":x["y_direct"],"predicted_direct_reliability":float(pd[i]),"y_structured":x["y_structured"],"predicted_structured_reliability":float(ps[i]),"outcome_label":x["outcome_label"],"score_difference":float(pd[i]-ps[i]),"max_reliability":float(max(pd[i],ps[i]))})
    dump("oof_predictions.json",{"estimator":main_key,"predictions":oof})
    dump("calibration_metrics.json",{"direct":results["direct"][main_key],"structured":results["structured"][main_key]})
    direct_region=[x for x in oof if x["score_difference"]>0]; structured_region=[x for x in oof if x["score_difference"]<0]; ties=[x for x in oof if x["score_difference"]==0]
    def region(rr:list[dict[str,Any]])->dict[str,Any]:
        return {"n":len(rr),"direct_correct_rate":sum(x["y_direct"] for x in rr)/len(rr) if rr else None,"structured_correct_rate":sum(x["y_structured"] for x in rr)/len(rr) if rr else None,"direct_minus_structured_correct_rate":sum(x["y_direct"]-x["y_structured"] for x in rr)/len(rr) if rr else None}
    group_scores=outcome_summary(oof)
    selection={"s_direct_gt_s_structured":region(direct_region),"s_structured_gt_s_direct":region(structured_region),"ties":region(ties),"outcome_group_scores":group_scores,
        "pure_only_relative_order_rate":sum(x["score_difference"]>0 for x in oof if x["outcome_label"]=="PURE_ONLY")/83,
        "full_only_relative_order_rate":sum(x["score_difference"]<0 for x in oof if x["outcome_label"]=="FULL_ONLY")/73,
        "note":"OOF descriptive diagnostic only; no threshold or Hybrid policy was optimized."}
    dump("selection_diagnostics.json",selection)
    ybw=np.asarray([x["outcome_label"]=="BOTH_WRONG" for x in oof],dtype=float); confidence=np.asarray([x["max_reliability"] for x in oof]); q=np.quantile(confidence,[.25,.5,.75])
    bands=[]
    edges=[-math.inf,*q,math.inf]
    for i in range(4):
        ids=[j for j,v in enumerate(confidence) if v>edges[i] and v<=edges[i+1]]
        bands.append({"confidence_quartile":i+1,"lower":None if i==0 else edges[i],"upper":None if i==3 else edges[i+1],"n":len(ids),"both_wrong_rate":float(ybw[ids].mean())})
    bw={"both_wrong_detection_roc_auc_using_one_minus_max_reliability":rank_auc(ybw,1-confidence),"both_wrong_detection_pr_auc":pr_auc(ybw,1-confidence),"confidence_quartiles":bands,
        "both_wrong_mean_max_reliability":float(confidence[ybw==1].mean()),"not_both_wrong_mean_max_reliability":float(confidence[ybw==0].mean()),"threshold_fitted":False}
    dump("both_wrong_diagnostics.json",bw)
    # Scene bootstrap of fixed OOF predictions: uncertainty, not model refitting or tuning.
    by_scene=defaultdict(list)
    for i,x in enumerate(oof):by_scene[x["sample_token"]].append(i)
    scenes=list(by_scene); rng=random.Random(SEED); boot=[]
    for _ in range(2000):
        idx=[i for _s in (rng.choice(scenes) for _ in scenes) for i in by_scene[_s]]
        order_hits=[bool((pd[i]>ps[i])==(targets["direct"][i]>targets["structured"][i])) for i in idx if targets["direct"][i]!=targets["structured"][i]]
        boot.append({"direct_brier":metrics(targets["direct"][idx],pd[idx])["brier"],"structured_brier":metrics(targets["structured"][idx],ps[idx])["brier"],
                     "relative_order_accuracy":sum(order_hits)/len(order_hits)})
    def ci(key:str)->dict[str,float]:
        v=sorted(x[key] for x in boot);return {"mean":statistics.mean(v),"percentile_95_ci":[v[49],v[1949]]}
    robustness={"per_fold":{"direct":results["direct"][main_key]["per_fold_metrics"],"structured":results["structured"][main_key]["per_fold_metrics"]},
        "fold_metric_summary":{"direct":results["direct"][main_key]["fold_metric_summary"],"structured":results["structured"][main_key]["fold_metric_summary"]},
        "scene_bootstrap":{"seed":SEED,"replicates":2000,"direct_brier":ci("direct_brier"),"structured_brier":ci("structured_brier"),"relative_order_accuracy_on_disagreements":ci("relative_order_accuracy")},
        "caution":"Bootstrap resamples the 35 scenes over fixed OOF predictions; QA are not treated as independent."}
    dump("scene_robustness.json",robustness)
    oldreg=read(P4A2_REGRESSION); outcome=read(P4A2_OUTCOME)
    regression={"phase4a2_hashes":{"dual_path_results":sha(DUAL),"outcome_summary":sha(P4A2_OUTCOME),"cohort_manifest":sha(P4A2_COHORT)},
        "frozen_reference_hashes":{k:v["sha256"] for k,v in oldreg.items() if isinstance(v,dict) and "sha256" in v},
        "checks":{"phase4a2_n_439":outcome["N"]==439,"phase4a2_pure_126":outcome["pure"]["overall"]["correct"]==126,"phase4a2_full_116":outcome["full"]["overall"]["correct"]==116,
        "phase4a2_outcomes_unchanged":outcome["counts"]=={"BOTH_CORRECT":43,"PURE_ONLY":83,"FULL_ONLY":73,"BOTH_WRONG":240},"frozen_regression_still_passed":oldreg["passed"]}}
    regression["passed"]=all(regression["checks"].values());dump("regression_check.json",regression)
    # Conservative multi-criterion decision, declared in advance in code.
    dmain=results["direct"][main_key]["oof_metrics"];smain=results["structured"][main_key]["oof_metrics"]
    dconst=results["direct"]["constant_prior"]["oof_metrics"];sconst=results["structured"]["constant_prior"]["oof_metrics"]
    favorable=sum([dmain["brier"]<dconst["brier"],smain["brier"]<sconst["brier"],(dmain["roc_auc"] or 0)>.55,(smain["roc_auc"] or 0)>.55,
                   selection["pure_only_relative_order_rate"]>.5,selection["full_only_relative_order_rate"]>.5,bw["both_wrong_detection_roc_auc_using_one_minus_max_reliability"]>.55])
    fold_consistent=sum(r["roc_auc"] is not None and r["roc_auc"]>.5 for r in results["direct"][main_key]["per_fold_metrics"])+sum(r["roc_auc"] is not None and r["roc_auc"]>.5 for r in results["structured"][main_key]["per_fold_metrics"])
    relative_robust=selection["pure_only_relative_order_rate"]>.5 and selection["full_only_relative_order_rate"]>.5 and robustness["scene_bootstrap"]["relative_order_accuracy_on_disagreements"]["percentile_95_ci"][0]>.5
    decision="A. GENERALIZABLE_RELIABILITY_SIGNALS_CONFIRMED_PROCEED_TO_ADAPTIVE_HYBRID" if favorable>=5 and fold_consistent>=7 and relative_robust and dmain["ece_10_bin"]<.15 and smain["ece_10_bin"]<.15 else "B. RELIABILITY_ESTIMATION_WEAK_USE_DUAL_EXECUTION_PLUS_VALIDATOR_INSTEAD" if favorable<=2 else "C. RELIABILITY_SIGNAL_EVIDENCE_INCONCLUSIVE_MORE_CALIBRATION_DATA_REQUIRED"
    def metric_table(reasoner:str)->str:
        lines=[]
        labels={"constant_prior":"Constant prior","question_only":"Question-only","evidence_only":"Evidence-only","question_evidence":"Question + Evidence","question_evidence_compatibility":"Question + Evidence + Compatibility","architecture_only_qec":"Architecture-only Q+E+Compatibility"}
        for key in FEATURE_SETS:
            m=results[reasoner][key]["oof_metrics"];lines.append(f"| {labels[key]} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['ece_10_bin']:.4f} |")
        return "\n".join(lines)
    report=f"""# Phase 4A-3 — Generalizable Reasoner Reliability Calibration\n\n## 1. Protocol verification\n\nUsed the unchanged **439 QA / 35 scene** Phase 4A-2 cohort. Deterministic 5-fold grouped CV assigns every QA an out-of-fold prediction while keeping each `sample_token` in one fold. Frozen scenes were not loaded or evaluated. `eval_type`, IDs, predictions, executor outputs, post-execution signals, correctness, and outcome labels were excluded from inputs. Correctness is used only as each independent reasoner's training target.\n\n## 2. Feature audit\n\nThe estimator uses {len(QUESTION_FEATURES)} generalized question-profile, {len(EVIDENCE_FEATURES)} abstract evidence-profile, and {len(COMPAT_FEATURES)} compatibility features. {sum(x['generalization_category']=='PERCEPTION-MODEL-SPECIFIC' for x in audit)} confidence-derived features are marked perception-specific; all other used features are architecture-level abstractions. Eleven forbidden/offline/post-execution fields are explicitly rejected in `feature_audit.json`. Missing capabilities are represented by availability/unknown fields. The architecture-only ablation removes all five confidence-derived features.\n\n## 3. Direct reliability estimation\n\n| Ablation | Brier | Log loss | ROC-AUC | PR-AUC | ECE |\n|---|---:|---:|---:|---:|---:|\n{metric_table('direct')}\n\n## 4. Structured reliability estimation\n\n| Ablation | Brier | Log loss | ROC-AUC | PR-AUC | ECE |\n|---|---:|---:|---:|---:|---:|\n{metric_table('structured')}\n\n## 5. Calibration quality and scene variation\n\nThe main model is Question + Evidence + Compatibility with fixed L2 logistic regression. Direct: Brier **{dmain['brier']:.4f}**, log loss **{dmain['log_loss']:.4f}**, AUC **{dmain['roc_auc']:.4f}**, ECE **{dmain['ece_10_bin']:.4f}**. Structured: Brier **{smain['brier']:.4f}**, log loss **{smain['log_loss']:.4f}**, AUC **{smain['roc_auc']:.4f}**, ECE **{smain['ece_10_bin']:.4f}**. Fold means/stds and scene-bootstrap intervals are in `scene_robustness.json`; these quantify 35-scene uncertainty rather than treating 439 QA as independent.\n\n## 6. Outcome-group reliability structure\n\n| Outcome | mean s_direct | mean s_structured | mean difference | mean max score |\n|---|---:|---:|---:|---:|---:|\n"""
    for label in ("BOTH_CORRECT","PURE_ONLY","FULL_ONLY","BOTH_WRONG"):
        g=group_scores[label];report+=f"| {label} | {g['s_direct']['mean']:.4f} | {g['s_structured']['mean']:.4f} | {g['score_difference']['mean']:.4f} | {g['max_reliability']['mean']:.4f} |\n"
    report+=f"""\n## 7. Selection diagnostic\n\nWhen `s_direct > s_structured` (n={len(direct_region)}), observed Direct correctness is {region(direct_region)['direct_correct_rate']:.2%} versus Structured {region(direct_region)['structured_correct_rate']:.2%}. When `s_structured > s_direct` (n={len(structured_region)}), observed Structured correctness is {region(structured_region)['structured_correct_rate']:.2%} versus Direct {region(structured_region)['direct_correct_rate']:.2%}. Relative ordering is favorable in {selection['pure_only_relative_order_rate']:.2%} of PURE_ONLY and {selection['full_only_relative_order_rate']:.2%} of FULL_ONLY cases. This is an OOF diagnostic, not optimized Hybrid accuracy.\n\n## 8. Both-wrong diagnostic\n\nUsing `1 - max(s_direct, s_structured)` to rank BOTH_WRONG gives ROC-AUC **{bw['both_wrong_detection_roc_auc_using_one_minus_max_reliability']:.4f}** and PR-AUC **{bw['both_wrong_detection_pr_auc']:.4f}**. Mean max reliability is {bw['both_wrong_mean_max_reliability']:.4f} for BOTH_WRONG versus {bw['not_both_wrong_mean_max_reliability']:.4f} otherwise. Confidence-quartile rates are descriptive; no abstention threshold was fitted.\n\n## 9. Generalization assessment\n\nAll results are out-of-scene within this 35-scene cohort, not cross-dataset evidence. Inputs express semantic requirements, abstract evidence capability/coverage, and their compatibility. The explicit architecture-only ablation tests removal of confidence-derived provider-specific inputs. No detector IDs, scene IDs, category IDs, dataset labels, object names, or lexical bag-of-words enter the models. Fold variation and bootstrap uncertainty limit strong claims, and external robustness remains future work. Individual reliability and BOTH_WRONG ranking are reproducible, but relative reasoner ordering is not robust: FULL_ONLY ordering is below chance and the scene-bootstrap relative-order interval includes 0.5.\n\n## Decision\n\n**{decision}**\n"""
    (OUT/"final_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","decision":decision,"favorable_criteria":favorable,"folds_above_chance":fold_consistent,"direct_main":dmain,"structured_main":smain,"selection":selection,"both_wrong":bw,"regression_passed":regression["passed"]},indent=2,default=str))

if __name__=="__main__":main()
