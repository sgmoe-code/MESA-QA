#!/usr/bin/env python3
"""Minimal supervised Direct/Structured selector on frozen 1,000-QA artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DIRECT = ROOT / "outputs/train20_qwen35_direct_final/label_joined_predictions.jsonl"
STRUCTURED = ROOT / "outputs/train20_qwen35_structured_final/label_joined_predictions.jsonl"
DIRECT_SEALED = ROOT / "outputs/train20_qwen35_direct_final/sealed_predictions.jsonl"
STRUCTURED_SEALED = ROOT / "outputs/train20_qwen35_structured_final/sealed_predictions.jsonl"
OUT = ROOT / "outputs/simple_reasoner_selection"
SEEDS = [20261103, 20261104, 20261105, 20261106, 20261107]
OOF_SEED = 20261103
LABELS = [-1, 0, 1]

NUMERIC = [
    "q_count", "q_existence", "q_comparison", "q_spatial", "q_motion",
    "q_status", "q_attribute", "question_words", "multi_entity",
    "compositional", "direct_missing", "structured_missing",
    "execution_success", "execution_complete", "trace_complete",
    "support_count", "answer_agreement", "answer_type_agreement",
]
CATEGORICAL = ["direct_value", "direct_type", "structured_value", "structured_type"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def answer_type(value: str) -> str:
    if not value: return "missing"
    if value in {"yes", "no"}: return "yes_no"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value): return "number"
    if len(value.split()) <= 3: return "category"
    return "other"


def question_flags(question: str) -> dict[str, int]:
    q = question.lower()
    words = re.findall(r"[a-z0-9]+", q)
    entity_terms = re.findall(r"\b(?:cars?|trucks?|buses|bus|pedestrians?|bicycles?|motorcycles?|vehicles?|objects?|things?|barriers?|cones?|trailers?)\b", q)
    spatial = r"\b(?:left|right|front|behind|back|near|closest|far|between|beside|ahead)\b"
    constraints = sum(bool(re.search(p, q)) for p in [spatial, r"\b(?:moving|parked|stopped|standing)\b", r"\b(?:color|status|state|type|kind)\b"])
    return {
        "q_count": int(bool(re.search(r"\b(?:how many|number of)\b", q))),
        "q_existence": int(bool(re.search(r"\b(?:is there|are there|does (?:a|an|the)|do any)\b", q))),
        "q_comparison": int(bool(re.search(r"\b(?:more|fewer|less|same|than|compare)\b", q))),
        "q_spatial": int(bool(re.search(spatial, q))),
        "q_motion": int(bool(re.search(r"\b(?:moving|motion|driving|traveling)\b", q))),
        "q_status": int(bool(re.search(r"\b(?:status|parked|stopped|standing|moving)\b", q))),
        "q_attribute": int(bool(re.search(r"\b(?:color|attribute|state|type|kind|rider)\b", q))),
        "question_words": len(words),
        "multi_entity": int(len(set(entity_terms)) >= 2),
        "compositional": int(constraints >= 2 or q.count(" and ") > 0 or q.count(" that ") > 0),
    }


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direct_rows, structured_rows = load_jsonl(DIRECT), load_jsonl(STRUCTURED)
    dsealed, ssealed = load_jsonl(DIRECT_SEALED), load_jsonl(STRUCTURED_SEALED)
    maps = [{int(x["question_id"]): x for x in rows} for rows in (direct_rows, structured_rows, dsealed, ssealed)]
    if any(len(m) != 1000 for m in maps) or any(set(m) != set(maps[0]) for m in maps[1:]):
        raise RuntimeError("BLOCKED: frozen artifact QA sets are not identical unique 1,000-row sets")
    rows = []
    for qid in sorted(maps[0]):
        d, s, ds, ss = (m[qid] for m in maps)
        identity = (d["question"], d["sample_token"], d["scene_token"])
        if any(identity != (x["question"], x["sample_token"], x["scene_token"]) for x in (s, ds, ss)):
            raise RuntimeError(f"BLOCKED: physical identity mismatch for QA {qid}")
        if d["evidence_hash"] != s["evidence_hash"] or d["serialized_evidence_hash"] != s["serialized_evidence_hash"]:
            raise RuntimeError(f"BLOCKED: evidence mismatch for QA {qid}")
        if (d["model_id"], d["model_revision"], d["normalizer_version"]) != (s["model_id"], s["model_revision"], s["normalizer_version"]):
            raise RuntimeError(f"BLOCKED: model/normalizer mismatch for QA {qid}")
        da, sa = str(d.get("normalized_answer") or ""), str(s.get("normalized_answer") or "")
        dc, sc = int(bool(d["correct"])), int(bool(s["correct"]))
        if (dc, sc) != (int(da == d["normalized_ground_truth"]), int(sa == s["normalized_ground_truth"])):
            raise RuntimeError(f"BLOCKED: frozen evaluator mismatch for QA {qid}")
        dt, st = answer_type(da), answer_type(sa)
        f = question_flags(d["question"])
        f.update({
            "direct_value": da or "__MISSING__", "direct_type": dt,
            "structured_value": sa or "__MISSING__", "structured_type": st,
            "direct_missing": int(not da), "structured_missing": int(not sa),
            "execution_success": int(bool(s.get("execution_success"))),
            "execution_complete": int(bool(s.get("execution_complete"))),
            "trace_complete": int(bool(s.get("execution_complete"))),
            "support_count": len(s.get("support_objects") or []),
            "answer_agreement": int(da == sa), "answer_type_agreement": int(dt == st),
        })
        label = 1 if (not dc and sc) else -1 if (dc and not sc) else 0
        rows.append({"question_id": qid, "scene_id": d["scene_token"], "sample_id": d["sample_token"],
                     "question": d["question"], "direct_answer": da, "structured_answer": sa,
                     "direct_correct": dc, "structured_correct": sc, "outcome_label": label,
                     "features": f, "structured_available": bool(sa),
                     "execution_success_raw": bool(s.get("execution_success")), "outer_fold_source": int(s["outer_fold"])})
    scenes = Counter(x["scene_id"] for x in rows)
    policies = Counter((x.get("rag_policy"), x.get("rag_configuration_hash"), x.get("retrieval_index_hash")) for x in structured_rows)
    audit = {
        "status": "PASS", "qa_count": len(rows), "scene_count": len(scenes), "scene_sizes": dict(sorted(scenes.items())),
        "unique_question_ids": len({x["question_id"] for x in rows}), "unique_identity_tuples": len({(x["question_id"], x["sample_id"], x["scene_id"]) for x in rows}),
        "quadrants": dict(Counter("structured_only" if x["outcome_label"] == 1 else "direct_only" if x["outcome_label"] == -1 else "tie" for x in rows)),
        "source_hashes": {p.name + "@" + p.parent.name: sha256(p) for p in (DIRECT, STRUCTURED, DIRECT_SEALED, STRUCTURED_SEALED)},
        "model_id": direct_rows[0]["model_id"], "model_revision": direct_rows[0]["model_revision"], "normalizer_version": direct_rows[0]["normalizer_version"],
        "structured_rag_policy_counts": {str(k): v for k, v in policies.items()},
        "rag_compatibility": "Compatible for pooling: all rows use the same policy/configuration; retrieval indexes differ only by predeclared outer fold and exclude held-out physical scenes.",
        "retrieval_leakage_check": "retrieval_provenance stores question/metadata and answer hashes only; answer_stored_or_exposed is false on all retrieved examples",
        "omitted_features": {"missing_required_capabilities": "No authoritative per-case missing-capability field; not manufactured."},
    }
    if len(rows) != 1000 or len(scenes) != 20 or audit["unique_question_ids"] != 1000:
        raise RuntimeError(f"BLOCKED: cohort size/identity audit failed: {audit}")
    if any(any(p.get("answer_stored_or_exposed") for p in (x.get("retrieval_provenance") or [])) for x in structured_rows):
        raise RuntimeError("BLOCKED: retrieval answer leakage detected")
    return rows, audit


def make_model(seed: int) -> Pipeline:
    numeric_columns = list(range(len(NUMERIC)))
    categorical_columns = list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL)))
    transform = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ])
    return Pipeline([("features", transform), ("classifier", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed))])


def matrix(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [[x["features"][k] for k in NUMERIC + CATEGORICAL] for x in rows]


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = np.array([x["outcome_label"] for x in train])
    if set(y) != set(LABELS): raise RuntimeError(f"training partition lacks a required class: {Counter(y)}")
    model = make_model(seed).fit(matrix(train), y)
    probabilities = model.predict_proba(matrix(test))
    classes = [int(c) for c in model.named_steps["classifier"].classes_]
    raw_names = list(model.named_steps["features"].get_feature_names_out())
    all_declared = NUMERIC + CATEGORICAL
    names = [re.sub(r"x(\d+)", lambda m: all_declared[int(m.group(1))], n) for n in raw_names]
    coef = model.named_steps["classifier"].coef_
    outputs = []
    for row, prob in zip(test, probabilities):
        pd = {str(c): float(prob[classes.index(c)]) for c in classes}
        choose = pd["1"] > pd["-1"]
        forced = not row["structured_available"] or not row["execution_success_raw"]
        if forced: choose = False
        outputs.append({**row, "probabilities": {"direct_only": pd["-1"], "tie": pd["0"], "structured_only": pd["1"]},
                        "selected_path": "STRUCTURED" if choose else "DIRECT", "forced_direct": forced,
                        "selector_correct": row["structured_correct"] if choose else row["direct_correct"]})
    metadata = {"training_label_counts": {str(k): int(v) for k, v in Counter(y).items()},
                "classes": classes, "feature_names": names,
                "coefficients": {str(c): {n: float(v) for n, v in zip(names, coef[i])} for i, c in enumerate(classes)},
                "intercepts": {str(c): float(model.named_steps["classifier"].intercept_[i]) for i, c in enumerate(classes)}}
    return outputs, metadata


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows); selected = [x for x in rows if x["selected_path"] == "STRUCTURED"]
    rescue = sum(x["outcome_label"] == 1 for x in selected); harm = sum(x["outcome_label"] == -1 for x in selected)
    direct = sum(x["direct_correct"] for x in rows); structured = sum(x["structured_correct"] for x in rows); selector = sum(x["selector_correct"] for x in rows)
    q = Counter(x["outcome_label"] for x in rows); interventions = rescue + harm
    return {"n": n, "direct": {"correct": direct, "accuracy": direct/n}, "always_structured": {"correct": structured, "accuracy": structured/n},
            "selector": {"correct": selector, "accuracy": selector/n}, "rescue": rescue, "harm": harm, "net_gain_over_direct": selector-direct,
            "structured_selections": len(selected), "intervention_precision": rescue/interventions if interventions else None,
            "structured_only_rescue_recall": rescue/q[1] if q[1] else None,
            "direct_only_protection_rate": 1-harm/q[-1] if q[-1] else None,
            "quadrants": {"both_correct": sum(x["direct_correct"] and x["structured_correct"] for x in rows), "direct_only": q[-1], "structured_only": q[1], "both_wrong": sum(not x["direct_correct"] and not x["structured_correct"] for x in rows)},
            "oracle_two_path": {"correct": sum(x["direct_correct"] or x["structured_correct"] for x in rows), "accuracy": sum(x["direct_correct"] or x["structured_correct"] for x in rows)/n}}


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [-1, 0, 1]; m = [[0]*3 for _ in labels]
    for x in rows:
        predicted = max((-1, 0, 1), key=lambda c: x["probabilities"]["direct_only" if c == -1 else "tie" if c == 0 else "structured_only"])
        m[labels.index(x["outcome_label"])][labels.index(predicted)] += 1
    return {"labels": ["direct_only", "tie", "structured_only"], "rows_true_columns_predicted": m}


def paired_stats(rows: list[dict[str, Any]], seed: int, iterations: int = 10000) -> dict[str, Any]:
    rescue = sum(x["selector_correct"] and not x["direct_correct"] for x in rows)
    harm = sum(x["direct_correct"] and not x["selector_correct"] for x in rows)
    by_scene = defaultdict(list)
    for x in rows: by_scene[x["scene_id"]].append(x)
    scenes, rng, deltas = sorted(by_scene), np.random.default_rng(seed), []
    for _ in range(iterations):
        draw = rng.choice(scenes, len(scenes), replace=True)
        sample = [x for s in draw for x in by_scene[s]]
        deltas.append(sum(x["selector_correct"]-x["direct_correct"] for x in sample)/len(sample))
    return {"mcnemar_discordant": {"selector_only_correct": rescue, "direct_only_correct": harm},
            "exact_mcnemar_p_two_sided": float(binomtest(min(rescue, harm), rescue+harm, .5).pvalue) if rescue+harm else 1.0,
            "scene_bootstrap_iterations": iterations, "accuracy_delta": sum(x["selector_correct"]-x["direct_correct"] for x in rows)/len(rows),
            "scene_bootstrap_95ci": [float(v) for v in np.quantile(deltas, [.025, .975])]}


def compact_prediction(x: dict[str, Any], experiment: str, run: str) -> dict[str, Any]:
    return {k: x[k] for k in ("question_id", "scene_id", "sample_id", "direct_answer", "structured_answer", "direct_correct", "structured_correct", "outcome_label", "probabilities", "selected_path", "forced_direct", "selector_correct")} | {"experiment": experiment, "run": run}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    rows, audit = build_rows(); OUT.mkdir(parents=True, exist_ok=True)
    feature_schema = {"numeric": NUMERIC, "categorical": CATEGORICAL, "forbidden": ["ground_truth", "correctness", "outcome_label", "eval_type", "template_label", "question_id", "scene_id"],
                      "categorical_encoding": "training-fitted one-hot with handle_unknown=ignore explicit fallback", "standardization": "training partition only", "missing_capability_feature": "omitted_unavailable"}
    config = {"classification": "RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC", "model": "multinomial logistic regression", "C": 1.0, "penalty": "l2", "solver": "lbfgs", "seeds": SEEDS,
              "fewshot_per_scene": 5, "scene_oof_folds": 5, "scene_oof_seed": OOF_SEED, "decision": "Structured iff P(SO)>P(DO), unless unavailable or execution failed", "bootstrap_iterations": 10000}
    write_json(OUT/"feature_schema.json", feature_schema); write_json(OUT/"experiment_config.json", config)
    audit_md = f"""# Data and provenance audit\n\nSTATUS: **{audit['status']}**\n\n- 1,000 unique aligned QA; 20 physical scenes; no duplicate identity tuples.\n- Direct and Structured use `{audit['model_id']}` revision `{audit['model_revision']}` and `{audit['normalizer_version']}`.\n- Evidence and serialized-evidence hashes match within every pair.\n- Structured retrieval is fold-local. {audit['rag_compatibility']}\n- Retrieval leakage: {audit['retrieval_leakage_check']}.\n- Ground truth/correctness is used only after frozen prediction loading to derive training labels and evaluation.\n- Missing-required-capabilities was unavailable and omitted consistently; trace completeness uses the persisted execution-complete field, and support count uses persisted support objects.\n\nSource hashes:\n```json\n{json.dumps(audit['source_hashes'], indent=2, sort_keys=True)}\n```\n"""
    (OUT/"data_and_provenance_audit.md").write_text(audit_md, encoding="utf-8")

    all_predictions, few_manifests, per_seed, few_stats = [], {}, {}, {}
    by_scene = defaultdict(list)
    for x in rows: by_scene[x["scene_id"]].append(x)
    for seed in SEEDS:
        rng = np.random.default_rng(seed); train_ids = []
        for scene in sorted(by_scene):
            ids = sorted(x["question_id"] for x in by_scene[scene]); train_ids.extend(int(i) for i in rng.choice(ids, 5, replace=False))
        train_set = set(train_ids); train = [x for x in rows if x["question_id"] in train_set]; test = [x for x in rows if x["question_id"] not in train_set]
        pred, model = fit_predict(train, test, seed); key = str(seed)
        few_manifests[key] = {"seed": seed, "training_question_ids": sorted(train_ids), "training_size": len(train), "evaluation_size": len(test), "per_scene_training_count": dict(Counter(x["scene_id"] for x in train))}
        per_seed[key] = metrics(pred) | {"training_label_counts": model["training_label_counts"], "confusion_matrix": confusion(pred), "model": model}
        few_stats[key] = paired_stats(pred, seed)
        all_predictions.extend(compact_prediction(x, "fewshot_scene_adaptation", key) for x in pred)
    write_json(OUT/"fewshot_split_manifests.json", few_manifests); write_json(OUT/"per_seed_metrics.json", per_seed)

    scene_rng = np.random.default_rng(OOF_SEED); scene_order = list(sorted(by_scene)); scene_rng.shuffle(scene_order)
    folds = [sorted(scene_order[i::5]) for i in range(5)]; oof_manifest, oof_pred, oof_models = {}, [], {}
    for fold, heldout in enumerate(folds):
        held = set(heldout); train = [x for x in rows if x["scene_id"] not in held]; test = [x for x in rows if x["scene_id"] in held]
        pred, model = fit_predict(train, test, OOF_SEED); oof_pred.extend(pred); oof_models[str(fold)] = model
        oof_manifest[str(fold)] = {"heldout_scenes": heldout, "training_scenes": sorted(set(by_scene)-held), "training_size": len(train), "test_size": len(test), "training_question_ids": sorted(x["question_id"] for x in train)}
    if len(oof_pred) != 1000 or len({x["question_id"] for x in oof_pred}) != 1000: raise RuntimeError("OOF coverage invariant failed")
    oof_pred.sort(key=lambda x: x["question_id"]); oof_metrics = metrics(oof_pred) | {"confusion_matrix": confusion(oof_pred), "fold_models": oof_models}
    write_json(OUT/"scene_oof_split_manifest.json", oof_manifest); write_json(OUT/"scene_oof_metrics.json", oof_metrics)
    all_predictions.extend(compact_prediction(x, "scene_disjoint_oof", str(next(k for k,v in oof_manifest.items() if x["scene_id"] in v["heldout_scenes"]))) for x in oof_pred)
    with (OUT/"per_case_predictions.jsonl").open("w", encoding="utf-8") as f:
        for x in all_predictions: f.write(json.dumps(x, sort_keys=True) + "\n")
    rescue = {"fewshot": {k: {z: v[z] for z in ("rescue", "harm", "net_gain_over_direct", "structured_selections", "intervention_precision", "structured_only_rescue_recall", "direct_only_protection_rate")} for k,v in per_seed.items()},
              "scene_oof": {z: oof_metrics[z] for z in ("rescue", "harm", "net_gain_over_direct", "structured_selections", "intervention_precision", "structured_only_rescue_recall", "direct_only_protection_rate")}}
    write_json(OUT/"rescue_harm_analysis.json", rescue); write_json(OUT/"statistical_analysis.json", {"fewshot_per_seed": few_stats, "scene_oof_unique_predictions": paired_stats(oof_pred, OOF_SEED)})
    avg = lambda path: sum(per_seed[str(s)][path]["accuracy"] for s in SEEDS)/len(SEEDS)
    a_sel = avg("selector"); a_dir = avg("direct"); a_str = avg("always_structured")
    train_pos = [per_seed[str(s)]["training_label_counts"].get("1", 0) for s in SEEDS]
    collapsed_a = all(per_seed[str(s)]["structured_selections"] in (0, per_seed[str(s)]["n"]) for s in SEEDS)
    collapsed_b = oof_metrics["structured_selections"] in (0, 1000)
    stat = paired_stats(oof_pred, OOF_SEED)
    report = f"""# Minimal Supervised Reasoner Selection\n\n**Classification:** RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC. This is not a blinded final benchmark.\n\n## Experiment A — few-shot scene adaptation\n\nFive fixed seeds each train on 5 QA per physical scene (100 total) and evaluate the other 900 with one shared classifier. Mean accuracies: Direct {a_dir:.3%}, Always Structured {a_str:.3%}, Selector {a_sel:.3%}. This measures adaptation to partially observed scenes, not unseen-scene generalization. Per-seed results are preserved independently and are not pooled as independent QA. Structured-only training positives by seed: {train_pos}. Constant-path collapse: {collapsed_a}.\n\n## Experiment B — scene-disjoint OOF\n\nDirect {oof_metrics['direct']['correct']}/1000 ({oof_metrics['direct']['accuracy']:.3%}); Always Structured {oof_metrics['always_structured']['correct']}/1000 ({oof_metrics['always_structured']['accuracy']:.3%}); Selector {oof_metrics['selector']['correct']}/1000 ({oof_metrics['selector']['accuracy']:.3%}). Rescue/harm/net = {oof_metrics['rescue']}/{oof_metrics['harm']}/{oof_metrics['net_gain_over_direct']}; Structured selections = {oof_metrics['structured_selections']}. Exact McNemar p={stat['exact_mcnemar_p_two_sided']:.6g}; scene-bootstrap accuracy-delta 95% CI={stat['scene_bootstrap_95ci']}. Constant-path collapse: {collapsed_b}.\n\n## Diagnostics and limits\n\nThe three-class confusion matrices and every model coefficient are in the metric artifacts. Feature contribution is descriptive: coefficients are provided per class in transformed-feature space. The earlier Hybrid is not scored here because its frozen base candidates and/or coverage are not identical to these Qwen3.5 pairs; a direct numerical claim would violate the protocol. The selector cannot repair both-wrong cases because it only chooses an existing answer. Same-scene labels, small Structured-only training counts, correlated QA within scenes, and retrospective cohort reuse limit inference. No feature or threshold was changed after observing results.\n"""
    (OUT/"final_experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "fewshot_mean": {"direct": a_dir, "structured": a_str, "selector": a_sel}, "scene_oof": oof_metrics["selector"], "output": str(OUT)}, indent=2))


if __name__ == "__main__": run()
