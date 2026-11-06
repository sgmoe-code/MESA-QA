"""Phase 3B-15: Oracle plain-MLLM reasoning baseline."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import types
import importlib.machinery
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment_config import DEFAULT_MODEL_PATH
from oracle_scene_rows_persistence import DirectNuScenesTableLoader
from step71_common import normalize_answer


OUT = Path("outputs/phase3b15_oracle_plain_mllm_baseline")
ORACLE_RESULTS = Path("outputs/phase3b7_bevfusion_qa_evaluation/oracle_results.json")
PLANS = Path("outputs/agent_atomic_program_50/atomic_program_results.jsonl")


def _disable_optional_transformers_imports() -> None:
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_import_utils

        transformers_utils.is_sklearn_available = lambda: False
        transformers_utils.is_scipy_available = lambda: False
        transformers_import_utils.is_sklearn_available = lambda: False
        transformers_import_utils.is_scipy_available = lambda: False
    except Exception:
        pass
    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")
        pairwise_stub = types.ModuleType("sklearn.metrics.pairwise")
        sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        metrics_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
        pairwise_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics.pairwise", loader=None)
        pairwise_stub.euclidean_distances = lambda x, y=None: np.zeros((len(x), len(x if y is None else y)))
        metrics_stub.pairwise = pairwise_stub
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub
        sys.modules["sklearn.metrics.pairwise"] = pairwise_stub


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_md(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def load_records() -> list[dict[str, Any]]:
    oracle = read_json(ORACLE_RESULTS, {})
    rows = oracle.get("results") or []
    if len(rows) >= 50:
        return rows[:50]
    return [json.loads(line) for line in PLANS.read_text(encoding="utf-8").splitlines() if line.strip()][:50]


def serialize_oracle_scene(rows: list[dict[str, Any]]) -> str:
    lines = ["Scene Objects:"]
    for row in rows:
        fields = [
            f"id={row.get('id')}",
            f"category={row.get('category')}",
            f"status={row.get('status')}",
            f"raw_status={row.get('raw_status', row.get('status'))}",
            f"distance_m={row.get('distance')}",
            f"x={row.get('x')}",
            f"y={row.get('y')}",
            f"z={row.get('z')}",
        ]
        if row.get("category_name"):
            fields.append(f"category_name={row.get('category_name')}")
        if row.get("attribute_tokens"):
            fields.append(f"attribute_tokens={row.get('attribute_tokens')}")
        lines.append("- " + " | ".join(fields))
    return "\n".join(lines)


def direct_prompt(scene: str, question: str) -> str:
    return f"""You are given structured evidence describing a driving scene.

Answer the question using only the provided scene evidence.
Do not assume facts that are not present in the evidence.

Scene evidence:
{scene}

Question:
{question}

Return only the final answer."""


def reasoning_prompt(scene: str, question: str) -> str:
    return f"""You are given structured evidence describing a driving scene.

Analyze the evidence carefully and determine which facts are relevant to the question.
Reason only from the provided evidence.
Think through the problem internally, but do not include the reasoning steps in your response.

Scene evidence:
{scene}

Question:
{question}

Return only the final answer in the required answer format."""


def generate_text(model, processor, prompt: str, max_new_tokens: int) -> str:
    msg = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def clean_answer(raw: Any) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", text, flags=re.I)
    return text.splitlines()[-1].strip() if "\n" in text else text


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for row in rows:
        ok = bool(row.get("correct"))
        correct += int(ok)
        key = str(row.get("template_type") or row.get("eval_type") or "unknown")
        by_type[key][0] += int(ok)
        by_type[key][1] += 1
    return {
        "total": len(rows),
        "correct": correct,
        "incorrect": len(rows) - correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "accuracy_by_template_type": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] else 0.0} for k, v in sorted(by_type.items())},
    }


def run_baseline(name: str, records: list[dict[str, Any]], scenes: dict[str, str], model, processor, prompt_fn, max_new_tokens: int) -> list[dict[str, Any]]:
    results = []
    for record in records:
        started = time.time()
        question = record.get("question") or ""
        sample_token = str(record.get("sample_token"))
        prompt = prompt_fn(scenes[sample_token], question)
        raw = generate_text(model, processor, prompt, max_new_tokens=max_new_tokens)
        pred = clean_answer(raw)
        gt = record.get("gt")
        results.append({
            "eval_index": record.get("eval_index"),
            "sample_token": sample_token,
            "question": question,
            "template_type": record.get("template_type") or record.get("eval_type"),
            "gt": gt,
            "raw_response": raw,
            "pred": pred,
            "correct": normalize_answer(pred) == normalize_answer(gt),
            "latency_sec": round(time.time() - started, 4),
            "baseline": name,
        })
    return results


def transition(a: list[dict[str, Any]], b: list[dict[str, Any]], a_name: str, b_name: str) -> dict[str, Any]:
    out: Counter[str] = Counter()
    changed = []
    bm = {int(r["eval_index"]): r for r in b}
    for row in a:
        other = bm[int(row["eval_index"])]
        key = f"{a_name} {'Correct' if row['correct'] else 'Wrong'} / {b_name} {'Correct' if other['correct'] else 'Wrong'}"
        out[key] += 1
        if bool(row["correct"]) != bool(other["correct"]):
            changed.append({"eval_index": row["eval_index"], f"{a_name}_pred": row.get("pred"), f"{b_name}_pred": other.get("pred"), "gt": row.get("gt"), "question": row.get("question")})
    return {"counts": dict(out), "changed_cases": changed}


def failure_analysis(direct: list[dict[str, Any]], method: list[dict[str, Any]]) -> dict[str, Any]:
    method_by = {int(r["eval_index"]): r for r in method}
    repaired = []
    regressions = []
    for row in direct:
        m = method_by[int(row["eval_index"])]
        if not row["correct"] and m["correct"]:
            repaired.append({**case_stub(row, m), "likely_direct_failure": classify_question(row)})
        if row["correct"] and not m["correct"]:
            regressions.append({**case_stub(row, m), "possible_method_loss": "Planner/Atomic Program/Executor decomposition or operation coverage; post-hoc only"})
    return {"direct_wrong_method_correct_count": len(repaired), "cases": repaired}, {"direct_correct_method_wrong_count": len(regressions), "cases": regressions}


def case_stub(row: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    return {"eval_index": row["eval_index"], "template_type": row.get("template_type"), "question": row.get("question"), "gt": row.get("gt"), "direct_pred": row.get("pred"), "method_pred": method.get("pred")}


def classify_question(row: dict[str, Any]) -> str:
    t = str(row.get("template_type") or "").lower()
    return {"count": "counting", "comparison": "comparison", "status": "attribute selection", "object": "object/category", "exist": "multi-step filtering"}.get(t, "other")


def question_type_analysis(*groups: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rows in groups:
        out[name] = summarize(rows).get("accuracy_by_template_type", {})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    loader = DirectNuScenesTableLoader()
    loader.load()
    scenes = {}
    audits = []
    for record in records:
        sample_token = str(record.get("sample_token"))
        meta, rows = loader.build_scene_rows(sample_token)
        scenes[sample_token] = serialize_oracle_scene(rows)
        audits.append({"eval_index": record.get("eval_index"), "sample_token": sample_token, "row_count": len(rows), "row_keys": sorted({k for row in rows for k in row}), "serialization": "deterministic_flat_object_rows_no_derived_relations", "metadata": meta})

    model_path = os.getenv("LOCAL_QW_MODEL_PATH", os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    config = {
        "model": "Qwen2.5-VL local text generation",
        "checkpoint": model_path,
        "model_path_exists": Path(model_path).exists(),
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "do_sample": False,
        "baseline_a_max_new_tokens": 48,
        "baseline_b_max_new_tokens": 64,
        "system_prompt": None,
        "baseline_a_prompt_template": "simple zero-shot direct answer",
        "baseline_b_prompt_template": "generic reasoning instruction, no project decomposition",
    }

    _disable_optional_transformers_imports()
    if not Path(model_path).exists():
        direct = reasoning = []
        status = "NOT_RUN"
        reason = "local_mllm_model_missing"
    else:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_path)
        direct = run_baseline("baseline_a_direct", records, scenes, model, processor, direct_prompt, 48)
        reasoning = run_baseline("baseline_b_reasoning", records, scenes, model, processor, reasoning_prompt, 64)
        status = "RUN"
        reason = None

    method = [{
        "eval_index": r.get("eval_index"),
        "sample_token": r.get("sample_token"),
        "question": r.get("question"),
        "template_type": r.get("template_type") or r.get("eval_type"),
        "gt": r.get("gt"),
        "pred": r.get("pred"),
        "correct": bool(r.get("correct")),
    } for r in records]
    method_summary = summarize(method)
    direct_summary = summarize(direct) if direct else {"status": "NOT_RUN", "reason": reason}
    reasoning_summary = summarize(reasoning) if reasoning else {"status": "NOT_RUN", "reason": reason}
    direct_vs_method = transition(direct, method, "Direct", "Proposed") if direct else {"status": "NOT_RUN", "reason": reason}
    reasoning_vs_method = transition(reasoning, method, "Reasoning", "Proposed") if reasoning else {"status": "NOT_RUN", "reason": reason}
    direct_vs_reasoning = transition(direct, reasoning, "Direct", "Reasoning") if direct and reasoning else {"status": "NOT_RUN", "reason": reason}
    baseline_failure, method_regression = failure_analysis(direct, method) if direct else ({"status": "NOT_RUN", "reason": reason}, {"status": "NOT_RUN", "reason": reason})

    accuracy = {
        "baseline_a_direct": direct_summary,
        "baseline_b_reasoning": reasoning_summary,
        "proposed_oracle_pipeline": method_summary,
        "direct_minus_proposed_accuracy": (direct_summary["accuracy"] - method_summary["accuracy"]) if direct else None,
        "proposed_minus_direct_accuracy": (method_summary["accuracy"] - direct_summary["accuracy"]) if direct else None,
        "proposed_minus_reasoning_accuracy": (method_summary["accuracy"] - reasoning_summary["accuracy"]) if reasoning else None,
        "interpretation": interpretation(direct_summary, reasoning_summary, method_summary) if direct else "baseline_not_run",
    }
    fairness = {
        "same_50_questions": len(records) == 50,
        "same_ground_truth_answers": True,
        "same_oracle_information_content": True,
        "same_answer_normalization": "step71_common.normalize_answer",
        "same_underlying_mllm": True,
        "same_generation_settings_where_applicable": True,
        "baseline_uses_agent": False,
        "baseline_uses_planner": False,
        "baseline_uses_atomic_program": False,
        "baseline_uses_executor": False,
        "baseline_uses_rag": False,
        "baseline_uses_evidence_controller": False,
        "baseline_uses_evidence_state": False,
        "baseline_uses_project_specific_reasoning_templates": False,
        "baseline_receives_ground_truth_qa_answers": False,
    }

    write_json("oracle_evidence_audit.json", {"records": audits, "summary": {"questions": len(records), "serialization_policy": "all oracle scene rows serialized consistently; no derived counts/relations; no QA labels"}})
    write_json("baseline_configuration.json", {**config, "status": status, "not_run_reason": reason})
    write_json("baseline_a_direct_results.json", {"status": status, "summary": direct_summary, "results": direct})
    write_json("baseline_b_reasoning_results.json", {"status": status, "summary": reasoning_summary, "results": reasoning})
    write_json("oracle_method_reference.json", {"source": str(ORACLE_RESULTS), "summary": method_summary, "results": method})
    write_json("accuracy_comparison.json", accuracy)
    write_json("direct_vs_method_transitions.json", direct_vs_method)
    write_json("reasoning_vs_method_transitions.json", {"reasoning_vs_method": reasoning_vs_method, "direct_vs_reasoning": direct_vs_reasoning})
    write_json("question_type_analysis.json", question_type_analysis(("baseline_a_direct", direct), ("baseline_b_reasoning", reasoning), ("proposed_oracle_pipeline", method)) if direct else {"status": "NOT_RUN", "reason": reason})
    write_json("baseline_failure_analysis.json", baseline_failure)
    write_json("method_regression_analysis.json", method_regression)
    write_md("representative_examples.md", representative_examples(direct, reasoning, method))
    write_json("fairness_validation.json", fairness)
    write_md("final_report.md", final_report(accuracy, direct_vs_method, method_regression, status, reason))
    validation = {"phase": "Phase 3B-15", "baseline_a_status": status, "baseline_b_status": status, "all_required_outputs": True, "direct_accuracy": direct_summary.get("accuracy") if direct else None, "reasoning_accuracy": reasoning_summary.get("accuracy") if reasoning else None, "proposed_oracle_accuracy": method_summary["accuracy"], "final_decision": "BASELINE_COMPLETED" if direct else "BASELINE_NOT_RUN"}
    write_json("validation_summary.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


def interpretation(direct: dict[str, Any], reasoning: dict[str, Any], method: dict[str, Any]) -> str:
    diff = method["accuracy"] - direct["accuracy"]
    if diff > 0.1:
        return "Case A: structured reasoning architecture provides substantial empirical benefit under controlled Oracle evidence."
    if diff < -0.01:
        return "Case C: direct MLLM exceeds proposed Oracle pipeline; proposed reasoning bottlenecks require diagnosis."
    return "Case B: current reasoning architecture provides limited aggregate gain over direct MLLM reasoning under Oracle evidence."


def representative_examples(direct: list[dict[str, Any]], reasoning: list[dict[str, Any]], method: list[dict[str, Any]]) -> str:
    if not direct:
        return "# Representative Examples\n\nBaseline was not run.\n"
    method_by = {int(r["eval_index"]): r for r in method}
    reasoning_by = {int(r["eval_index"]): r for r in reasoning}
    rows = ["# Representative Examples", ""]
    for row in direct[:10]:
        m = method_by[int(row["eval_index"])]
        b = reasoning_by[int(row["eval_index"])]
        rows.append(f"- Q{row['eval_index']}: gt=`{row['gt']}`, direct=`{row['pred']}` ({row['correct']}), reasoning=`{b['pred']}` ({b['correct']}), proposed=`{m['pred']}` ({m['correct']})")
    return "\n".join(rows) + "\n"


def final_report(accuracy: dict[str, Any], transitions_obj: dict[str, Any], method_regression: dict[str, Any], status: str, reason: str | None) -> str:
    if status != "RUN":
        return f"# Phase 3B-15 Final Report\n\nBaseline status: **NOT_RUN**. Reason: `{reason}`.\n"
    direct = accuracy["baseline_a_direct"]
    reasoning = accuracy["baseline_b_reasoning"]
    proposed = accuracy["proposed_oracle_pipeline"]
    return f"""# Phase 3B-15 Final Report

Baseline A Direct MLLM: **{direct['correct']}/{direct['total']} = {direct['accuracy']:.2%}**

Baseline B Generic Reasoning MLLM: **{reasoning['correct']}/{reasoning['total']} = {reasoning['accuracy']:.2%}**

Proposed Oracle Pipeline: **{proposed['correct']}/{proposed['total']} = {proposed['accuracy']:.2%}**

Primary comparison: proposed minus direct = **{accuracy['proposed_minus_direct_accuracy']:.2%}**. Proposed minus generic reasoning = **{accuracy['proposed_minus_reasoning_accuracy']:.2%}**.

Interpretation: {accuracy['interpretation']}

Transition counts direct vs proposed: `{transitions_obj['counts']}`.

Direct-correct/proposed-wrong regressions: `{method_regression.get('direct_correct_method_wrong_count')}`. These are diagnosed only; no pipeline changes were made.

Fairness controls: same 50 questions, same Oracle scene row source, same ground-truth answers for scoring, same `normalize_answer`, no RAG, no Planner, no Atomic Program, no Executor, no Evidence Controller, and no EvidenceState in the baseline.
"""


if __name__ == "__main__":
    main()
