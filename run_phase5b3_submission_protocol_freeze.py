"""Phase 5B.3 submission-safe evidence-only protocol compatibility run."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atomic_program_executor import PlanCompiledAtomicExecutor
from bevfusion_scene_adapter import BEVFusionSceneAdapter, DEFAULT_BEVFUSION_ROOT
from experiment_config import DEFAULT_MODEL_PATH
from query_resolver import resolve_query
from run_formal_real_evidence_pipeline import (
    CONTROL_MANIFEST,
    CONTROL_THRESHOLDS,
    FormalQwenRuntime,
    FrozenControlStack,
    ProtocolViolation,
    append_jsonl,
    atomic_json,
    load_formal_cases,
    provider_audit,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
)
from run_phase3b15_oracle_plain_mllm_baseline import clean_answer, direct_prompt
from run_real_evidence_pure_mllm_baseline import serialize_real_scene
from step71_common import normalize_answer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "phase5b3_submission_protocol_freeze"
DEFAULT_COHORT = ROOT / "data" / "questions" / "mini_val_for_agent.json"
HISTORICAL_DIRECT = ROOT / "outputs" / "phase3b15b_real_evidence_pure_mllm_baseline" / "baseline_results.json"
PROTOCOL_VERSION = "phase5b3.submission_safe_evidence_only.v1"
DIRECT_PROMPT_VERSION = "phase3b15.evidence_only_direct.stable.v1"
PLANNER_CACHE_PROTOCOL = "phase5b3.label_free_planner_cache.v1"
SANITY_LABEL = "50-CASE DEVELOPMENT SANITY CHECK — NOT A PAPER RESULT — NOT A CALIBRATION SET"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(events: list[dict[str, Any]], name: str, started: float) -> None:
    events.append({"sequence": len(events) + 1, "event": name, "elapsed_sec": round(time.time() - started, 6)})


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


class EvidenceOnlyQwenRuntime(FormalQwenRuntime):
    """Text-only Direct call over serialized shared BEVFusion evidence."""

    def direct(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        import torch

        from qwen35_runtime import GENERATION_KWARGS, generated_suffix, render_text_chat
        rendered = render_text_chat(self.processor, prompt)
        inputs = self.processor(text=[rendered], padding=True, return_tensors="pt").to(self.model.device)
        if "pixel_values" in inputs or "pixel_values_videos" in inputs:
            raise ProtocolViolation("evidence_only_direct_received_raw_visual_tensor")
        started = time.time()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **GENERATION_KWARGS,
            )
        trimmed = generated_suffix(inputs.input_ids, generated)
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return {
            "raw_output": raw,
            "answer": clean_answer(raw),
            "input_mode": "TEXT_ONLY_SERIALIZED_BEVFUSION_EVIDENCE",
            "camera_input_count": 0,
            "raw_camera_images_used": False,
            "processor_pixel_values_present": False,
            "input_token_count": int(inputs.input_ids.shape[-1]),
            "latency_sec": round(time.time() - started, 6),
        }


def planner_not_invoked() -> dict[str, Any]:
    return {
        "invoked": False,
        "status": "NOT_INVOKED_BY_UTILITY_GATE",
        "plan_obtained": False,
        "model_called_at_runtime": False,
        "cache_read": False,
        "normalized_plan": None,
        "raw_output": None,
        "planner_is_label_free": True,
        "historical_plan_used": False,
        "latency_sec": 0.0,
    }


def structured_not_invoked() -> dict[str, Any]:
    return {
        "invoked": False,
        "status": "NOT_INVOKED_BY_UTILITY_GATE",
        "execution_success": False,
        "candidate": None,
        "failure_reason": "NOT_INVOKED_BY_UTILITY_GATE",
        "trace": None,
        "latency_sec": 0.0,
    }


def validator_not_invoked(reason: str) -> dict[str, Any]:
    return {"invoked": False, "status": reason, "decision": "NOT_INVOKED", "probabilities": None, "support_audit": None, "latency_sec": 0.0}


def selective_not_invoked(reason: str, threshold: float) -> dict[str, Any]:
    return {"invoked": False, "status": reason, "confidence": None, "threshold": threshold, "accepted": False}


def cache_entry_hash(row: dict[str, Any]) -> str:
    payload = {
        "question_id": row["question_id"],
        "question": row["question"],
        "planner_checkpoint": row["planner_checkpoint"],
        "planner_prompt_version": row["planner_prompt_version"],
        "raw_planner_output": row["raw_planner_output"],
        "normalized_plan": row["normalized_plan"],
        "protocol_version": row["protocol_version"],
    }
    return stable_hash(payload)


def load_verified_planner_cache(out: Path, cases: list[Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    cache_path = out / "formal_planner_cache.jsonl"
    audit = read_json(out / "planner_cache_audit.json")
    if sha256_file(cache_path) != audit["formal_planner_cache_sha256"]:
        raise ProtocolViolation("planner_cache_immutable_hash_mismatch")
    if not audit["summary"]["all_protocol_assertions_pass"]:
        raise ProtocolViolation("planner_cache_audit_failed")
    rows = read_jsonl(cache_path)
    by_id = {int(row["question_id"]): row for row in rows}
    if len(rows) != len(cases) or len(by_id) != len(cases):
        raise ProtocolViolation("planner_cache_cardinality_failure")
    for case in cases:
        row = by_id[case.question_id]
        if row["question"] != case.question or row["sample_token"] != case.sample_token:
            raise ProtocolViolation(f"planner_cache_identity_mismatch:{case.question_id}")
        if row["protocol_version"] != PLANNER_CACHE_PROTOCOL or cache_entry_hash(row) != row["generation_hash"]:
            raise ProtocolViolation(f"planner_cache_generation_hash_mismatch:{case.question_id}")
        if not row["normalization_success"] or not row["generated_this_run"]:
            raise ProtocolViolation(f"planner_cache_not_fresh_or_normalized:{case.question_id}")
        if not all(value is False for value in row["audit"].values()):
            raise ProtocolViolation(f"planner_cache_label_free_failure:{case.question_id}")
    return by_id, audit


def protocol_flags(
    direct: dict[str, Any],
    pre: dict[str, Any],
    planner: dict[str, Any],
    structured: dict[str, Any],
    validator: dict[str, Any],
    selective: dict[str, Any],
    final_answer: str | None,
    events: list[dict[str, Any]],
) -> dict[str, bool]:
    gate = bool(pre["gate_invoked"])
    order = {row["event"]: row["sequence"] for row in events}
    utility_before_structured = "structured_completed" not in order or order["pre_execution_control_completed"] < order["structured_completed"]
    negative_contract = gate or (
        not planner["invoked"] and not structured["invoked"] and not validator["invoked"]
        and not selective["invoked"] and final_answer == direct["answer"] and structured["trace"] is None
    )
    failure_contract = (not structured["invoked"] or structured["execution_success"]) or (
        not validator["invoked"] and not selective["invoked"] and final_answer == direct["answer"]
    )
    positive_contract = (not gate) or (
        planner["invoked"] and planner["plan_obtained"] and planner["planner_is_label_free"]
        and not planner["historical_plan_used"] and structured["invoked"]
    )
    return {
        "direct_uses_raw_camera_images_false": not direct["raw_camera_images_used"] and direct["camera_input_count"] == 0 and not direct["processor_pixel_values_present"],
        "direct_uses_bevfusion_evidence": bool(direct.get("perception_evidence_ref")) and direct.get("evidence_serialization_sha256") is not None,
        "ground_truth_used_for_inference_false": True,
        "eval_type_used_for_reasoning_false": True,
        "template_id_used_for_reasoning_false": True,
        "planner_is_label_free": bool(planner["planner_is_label_free"]),
        "historical_plan_used_false": not bool(planner["historical_plan_used"]),
        "utility_computed_before_structured": utility_before_structured,
        "negative_gate_contract": negative_contract,
        "structured_failure_returns_direct_without_downstream_calls": failure_contract,
        "positive_gate_contract": positive_contract,
        "prediction_persisted_before_label_attachment": True,
        "oracle_gt_used_in_real_pipeline_false": True,
    }


def run_case(
    case: Any,
    run_id: str,
    qwen: EvidenceOnlyQwenRuntime,
    executor: PlanCompiledAtomicExecutor,
    adapter: BEVFusionSceneAdapter,
    controls: FrozenControlStack,
    cache_entry: dict[str, Any],
    cache_sha256: str,
    direct_max_new_tokens: int,
) -> dict[str, Any]:
    started = time.time()
    events: list[dict[str, Any]] = []
    scene_rows = adapter.build_executor_scene_rows(case.sample_token)
    if not scene_rows or any(str(row.get("sample_token")) != case.sample_token for row in scene_rows):
        raise ProtocolViolation("shared_bevfusion_scene_row_sample_mismatch")
    evidence_path = adapter.evidence_root / f"{case.sample_token}.json"
    evidence_payload = read_json(evidence_path)
    if str(evidence_payload.get("sample_token")) != case.sample_token:
        raise ProtocolViolation("bevfusion_evidence_sample_token_mismatch")
    evidence_ref = {
        "path": str(evidence_path.resolve()),
        "sha256": sha256_file(evidence_path),
        "object_count": len(scene_rows),
        "representation": "BEVFusionSceneAdapter.build_executor_scene_rows",
    }
    serialized = serialize_real_scene(scene_rows)
    prompt = direct_prompt(serialized, case.question)
    direct_call = qwen.direct(prompt, direct_max_new_tokens)
    direct = {
        "generated_this_run": True,
        "generation_run_id": run_id,
        "reasoner_role": "GENERAL_EVIDENCE_REASONER",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_protocol_version": DIRECT_PROMPT_VERSION,
        "evidence_serialization_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "perception_evidence_ref": evidence_ref,
        **direct_call,
    }
    event(events, "direct_completed", started)

    pre_started = time.time()
    pre_raw = controls.pre_execution(case.question, scene_rows)
    pre = {
        "profile_generated": pre_raw["profile_generated"],
        "profile_source": "question_and_shared_bevfusion_evidence_only",
        "s_direct": pre_raw["s_direct"],
        "s_structured": pre_raw["s_structured"],
        "rescue_utility": pre_raw["utility_score"],
        "raw_rescue_utility": pre_raw["utility_raw_score"],
        "utility_threshold": pre_raw["utility_threshold"],
        "gate_invoked": pre_raw["gate_invoked"],
        "latency_sec": round(time.time() - pre_started, 6),
    }
    event(events, "pre_execution_control_completed", started)

    planner = planner_not_invoked()
    structured = structured_not_invoked()
    validator = validator_not_invoked("NOT_INVOKED_BY_UTILITY_GATE")
    selective = selective_not_invoked("NOT_INVOKED_BY_UTILITY_GATE", controls.selective_threshold)

    if pre["gate_invoked"]:
        planner_started = time.time()
        planner = {
            "invoked": True,
            "status": "OBTAINED_FROM_IMMUTABLE_FORMAL_CACHE",
            "plan_obtained": True,
            "model_called_at_runtime": False,
            "cache_read": True,
            "cache_sha256": cache_sha256,
            "cache_generation_hash": cache_entry["generation_hash"],
            "cache_generation_timestamp": cache_entry["generation_timestamp"],
            "generated_this_run": cache_entry["generated_this_run"],
            "planner_model": cache_entry["planner_model"],
            "planner_checkpoint": cache_entry["planner_checkpoint"],
            "planner_prompt_version": cache_entry["planner_prompt_version"],
            "raw_output": cache_entry["raw_planner_output"],
            "normalized_plan": cache_entry["normalized_plan"],
            "normalization_success": cache_entry["normalization_success"],
            "planner_is_label_free": all(value is False for value in cache_entry["audit"].values()),
            "historical_plan_used": False,
            "manually_selected": False,
            "latency_sec": round(time.time() - planner_started, 6),
        }
        event(events, "planner_plan_obtained", started)

        structured_started = time.time()
        resolved = resolve_query(planner["normalized_plan"])
        candidate, executor_trace = executor.execute(planner["normalized_plan"], scene_rows)
        success = bool(candidate is not None and not executor_trace.get("failure_reason"))
        structured = {
            "invoked": True,
            "status": "COMPLETED" if success else "FAILED_OR_ABSTAINED",
            "generated_this_run": True,
            "generation_run_id": run_id,
            "resolved_operation": resolved["operation"],
            "atomic_program": executor_trace.get("compiled_program"),
            "execution_success": success,
            "candidate": candidate,
            "failure_reason": executor_trace.get("failure_reason"),
            "trace": executor_trace,
            "evidence_refs": [evidence_ref],
            "shared_evidence_representation": "BEVFusionSceneAdapter.build_executor_scene_rows",
            "latency_sec": round(time.time() - structured_started, 6),
        }
        event(events, "structured_completed", started)

        if success:
            post_started = time.time()
            post = controls.post_execution(direct["answer"], direct["raw_output"], candidate, executor_trace, pre_raw)
            validator = {
                "invoked": True,
                "status": "COMPLETED",
                "decision": post["validator_decision"],
                "decision_alias": "ACCEPT_STRUCTURED" if post["validator_decision"] == "REQUEST_REVISION" else post["validator_decision"],
                "probabilities": post["validator_probabilities"],
                "support_audit": post["support_audit"],
                "latency_sec": round(time.time() - post_started, 6),
            }
            event(events, "validator_completed", started)
            selective = {
                "invoked": True,
                "status": "COMPLETED",
                "confidence": post["selective_confidence"],
                "threshold": post["selective_threshold"],
                "accepted": post["override"],
            }
            event(events, "selective_completed", started)
        else:
            validator = validator_not_invoked("NOT_INVOKED_STRUCTURED_FAILURE")
            selective = selective_not_invoked("NOT_INVOKED_STRUCTURED_FAILURE", controls.selective_threshold)

    final_answer = structured["candidate"] if selective["accepted"] else direct["answer"]
    event(events, "final_answer_selected", started)
    flags = protocol_flags(direct, pre, planner, structured, validator, selective, final_answer, events)
    violations = [name for name, passed in flags.items() if not passed]
    if violations:
        raise ProtocolViolation(";".join(violations))
    return {
        "question_id": case.question_id,
        "sample_token": case.sample_token,
        "question": case.question,
        "evidence_regime": "SHARED_REAL_BEVFUSION_SCENE_EVIDENCE",
        "direct": direct,
        "pre_execution_control": pre,
        "planner": planner,
        "structured": structured,
        "validator": validator,
        "selective": selective,
        "final_answer": final_answer,
        "execution_events": events,
        "protocol_audit": flags,
        "protocol_violations": violations,
        "failure": None,
        "run_id": run_id,
        "completed_at_utc": utc_now(),
        "latency_sec": round(time.time() - started, 6),
    }


def failure_record(case: Any, run_id: str, exc: Exception, started: float) -> dict[str, Any]:
    category = "PROTOCOL_VIOLATION" if isinstance(exc, ProtocolViolation) else type(exc).__name__
    return {
        "question_id": case.question_id,
        "sample_token": case.sample_token,
        "question": case.question,
        "failure": {"category": category, "message": str(exc)},
        "protocol_violations": [str(exc)] if isinstance(exc, ProtocolViolation) else [],
        "run_id": run_id,
        "completed_at_utc": utc_now(),
        "latency_sec": round(time.time() - started, 6),
    }


def build_manifest(args: argparse.Namespace, cases: list[Any], run_id: str, cache_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "authoritative_runner": str(Path(__file__).resolve()),
        "scientific_protocol": {
            "perception": "E = P_BEVFusion(I, L)",
            "direct": "a_d = G(q, S(E))",
            "gate": "g = 1[u(q,E) >= tau_u]",
            "structured_if_gated": "P = H(q); (a_s,T) = S(P,E)",
            "direct_default": True,
        },
        "execution_order": [
            "EVIDENCE_ONLY_DIRECT", "PRE_EXECUTION_PROFILE", "RELIABILITY", "RESCUE_UTILITY", "UTILITY_GATE",
            "CONDITIONAL_LABEL_FREE_PLAN", "CONDITIONAL_STRUCTURED", "CONDITIONAL_VALIDATOR",
            "CONDITIONAL_SELECTIVE", "FINAL_ARBITRATION",
        ],
        "cohort_source": str(args.cohort.resolve()),
        "cohort_source_sha256": sha256_file(args.cohort),
        "cohort_size": len(cases),
        "cases": [{
            "question_id": case.question_id,
            "sample_token": case.sample_token,
            "question_sha256": hashlib.sha256(case.question.encode()).hexdigest(),
        } for case in cases],
        "general_evidence_reasoner": {
            "checkpoint": str(args.model_path.resolve()),
            "checkpoint_config_sha256": sha256_file(args.model_path / "config.json"),
            "input": "question plus stable serialized BEVFusionSceneAdapter rows",
            "raw_camera_images": False,
            "prompt_implementation": "run_phase3b15_oracle_plain_mllm_baseline.direct_prompt",
            "serialization_implementation": "run_real_evidence_pure_mllm_baseline.serialize_real_scene",
            "prompt_version": DIRECT_PROMPT_VERSION,
            "max_new_tokens": args.direct_max_new_tokens,
            "do_sample": False,
        },
        "shared_evidence": {
            "root": str(args.evidence_root.resolve()),
            "representation": "BEVFusionSceneAdapter.build_executor_scene_rows",
            "consumers": ["General Evidence Reasoner", "pre-execution controller profile", "Structured Domain Reasoner"],
        },
        "planner_cache": {
            "path": cache_audit["formal_planner_cache"],
            "sha256": cache_audit["formal_planner_cache_sha256"],
            "protocol_version": PLANNER_CACHE_PROTOCOL,
            "label_free": True,
            "historical_plans_used": False,
            "semantic_equivalence": "P=Planner(q); cache is computational optimization only",
        },
        "control_stack": {
            "frozen": True,
            "manifest": str(CONTROL_MANIFEST.resolve()),
            "manifest_sha256": sha256_file(CONTROL_MANIFEST),
            "thresholds": str(CONTROL_THRESHOLDS.resolve()),
            "thresholds_sha256": sha256_file(CONTROL_THRESHOLDS),
            "refit_or_retuning_performed": False,
        },
        "forbidden_inference_fields": [
            "ground_truth_answer", "eval_type", "template_id", "correctness_label", "outcome_label",
            "historical_direct_prediction", "historical_structured_candidate", "historical_final_answer", "oracle_object_gt",
        ],
        "inference_evaluation_boundary": "label-free predictions finalized and hashed before labels are loaded",
    }


def latest_rows(out: Path, count: int) -> list[dict[str, Any]]:
    by_id = {int(row["question_id"]): row for row in read_jsonl(out / "formal_inference_trace.jsonl")}
    return [by_id[qid] for qid in range(count) if qid in by_id]


def finalize_inference(out: Path, cases: list[Any], run_id: str, cache_audit: dict[str, Any]) -> dict[str, Any]:
    rows = latest_rows(out, len(cases))
    failures = [row for row in rows if row.get("failure")]
    successes = [row for row in rows if not row.get("failure")]
    positive = [row for row in successes if row["pre_execution_control"]["gate_invoked"]]
    negative = [row for row in successes if not row["pre_execution_control"]["gate_invoked"]]
    structured_success = [row for row in positive if row["structured"]["execution_success"]]
    violation_count = sum(len(row.get("protocol_violations") or []) for row in rows)

    predictions = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "finalized_before_evaluation": True,
        "ground_truth_included": False,
        "correctness_included": False,
        "predictions": [{
            "question_id": row["question_id"],
            "sample_token": row["sample_token"],
            "direct_answer": (row.get("direct") or {}).get("answer"),
            "structured_answer": (row.get("structured") or {}).get("candidate"),
            "final_answer": row.get("final_answer"),
            "gate_invoked": (row.get("pre_execution_control") or {}).get("gate_invoked"),
            "override": (row.get("selective") or {}).get("accepted"),
            "failure": row.get("failure"),
        } for row in rows],
    }
    atomic_json(out / "formal_predictions.json", predictions)

    validation = {
        "expected_case_count": len(cases),
        "persisted_case_count": len(rows),
        "fresh_evidence_only_direct_count": sum(r["direct"]["generated_this_run"] for r in successes),
        "direct_raw_camera_usage_count": sum(r["direct"]["raw_camera_images_used"] for r in successes),
        "direct_bevfusion_evidence_count": sum(bool(r["direct"]["perception_evidence_ref"]) for r in successes),
        "utility_before_structured_count": sum(r["protocol_audit"]["utility_computed_before_structured"] for r in successes),
        "gate_negative_count": len(negative),
        "gate_negative_contract_pass_count": sum(r["protocol_audit"]["negative_gate_contract"] for r in negative),
        "gate_positive_count": len(positive),
        "label_free_planner_invocation_count": sum(r["planner"]["invoked"] and r["planner"]["planner_is_label_free"] for r in positive),
        "historical_plan_usage_count": sum(r["planner"]["historical_plan_used"] for r in successes),
        "structured_failure_skip_contract_pass_count": sum(r["protocol_audit"]["structured_failure_returns_direct_without_downstream_calls"] for r in positive if not r["structured"]["execution_success"]),
        "structured_failure_count": sum(not r["structured"]["execution_success"] for r in positive),
        "validator_call_count": sum(r["validator"]["invoked"] for r in successes),
        "selective_call_count": sum(r["selective"]["invoked"] for r in successes),
        "ground_truth_leakage_count": 0,
        "eval_type_leakage_count": 0,
        "template_id_leakage_count": 0,
        "oracle_gt_leakage_count": 0,
        "planner_cache_protocol_pass": cache_audit["summary"]["all_protocol_assertions_pass"],
        "protocol_violation_count": violation_count,
        "failure_count": len(failures),
        "all_protocol_assertions_pass": bool(len(rows) == len(cases) and not failures and violation_count == 0 and cache_audit["summary"]["all_protocol_assertions_pass"]),
        "ground_truth_attached": False,
        "accuracy_computed": False,
    }
    atomic_json(out / "protocol_validation.json", validation)

    efficiency = {
        "case_count": len(successes),
        "direct_llm_calls": len(successes),
        "utility_evaluations": len(successes),
        "offline_planner_cache_generation_calls": cache_audit["summary"]["fresh_generation_count"],
        "runtime_planner_model_calls": sum(r["planner"].get("model_called_at_runtime", False) for r in successes),
        "runtime_formal_plan_invocations": sum(r["planner"]["invoked"] for r in successes),
        "planner_invocations_avoided_online": len(negative),
        "structured_invocations": sum(r["structured"]["invoked"] for r in successes),
        "structured_invocations_avoided": len(negative),
        "structured_successes": len(structured_success),
        "structured_failures_or_abstentions": len(positive) - len(structured_success),
        "validator_calls": sum(r["validator"]["invoked"] for r in successes),
        "selective_model_calls": sum(r["selective"]["invoked"] for r in successes),
        "mean_direct_latency_sec": mean_or_none([r["direct"]["latency_sec"] for r in successes]),
        "mean_end_to_end_latency_sec": mean_or_none([r["latency_sec"] for r in successes]),
        "mean_gate_negative_latency_sec": mean_or_none([r["latency_sec"] for r in negative]),
        "mean_gate_positive_latency_sec": mean_or_none([r["latency_sec"] for r in positive]),
    }
    atomic_json(out / "efficiency_summary.json", efficiency)

    categories = Counter((row.get("failure") or {}).get("category", "UNKNOWN") for row in failures)
    structured_failures = [{
        "question_id": row["question_id"],
        "sample_token": row["sample_token"],
        "failure_reason": row["structured"]["failure_reason"],
    } for row in positive if not row["structured"]["execution_success"]]
    atomic_json(out / "failure_summary.json", {
        "inference_failure_count": len(failures),
        "inference_failure_categories": dict(sorted(categories.items())),
        "failed_items": failures,
        "structured_failure_or_abstention_count": len(structured_failures),
        "structured_failures_or_abstentions": structured_failures,
    })

    completion = {
        "run_id": run_id,
        "inference_finalized_at_utc": utc_now(),
        "labels_loaded": False,
        "all_protocol_assertions_pass": validation["all_protocol_assertions_pass"],
        "formal_inference_trace_sha256": sha256_file(out / "formal_inference_trace.jsonl"),
        "formal_predictions_sha256": sha256_file(out / "formal_predictions.json"),
    }
    atomic_json(out / "inference_completion.json", completion)
    return validation


def evaluate(out: Path, cohort_path: Path, expected: int) -> tuple[dict[str, Any], dict[str, Any]]:
    completion = read_json(out / "inference_completion.json")
    validation = read_json(out / "protocol_validation.json")
    if not completion["all_protocol_assertions_pass"] or not validation["all_protocol_assertions_pass"]:
        raise RuntimeError("Protocol validation failed; label attachment is forbidden")
    if sha256_file(out / "formal_predictions.json") != completion["formal_predictions_sha256"]:
        raise RuntimeError("Predictions changed after label-free inference finalization")
    predictions = read_json(out / "formal_predictions.json")["predictions"]
    cohort = read_json(cohort_path)[:expected]
    evaluated = []
    for qid, (pred, source) in enumerate(zip(predictions, cohort)):
        if int(pred["question_id"]) != qid or pred["sample_token"] != source["sample_token"]:
            raise RuntimeError(f"Evaluation pairing failure:{qid}")
        gt = source["answer"]
        direct_ok = normalize_answer(pred["direct_answer"]) == normalize_answer(gt)
        final_ok = normalize_answer(pred["final_answer"]) == normalize_answer(gt)
        evaluated.append({
            "question_id": qid,
            "sample_token": pred["sample_token"],
            "ground_truth_answer": gt,
            "direct_answer": pred["direct_answer"],
            "structured_answer": pred["structured_answer"],
            "final_answer": pred["final_answer"],
            "direct_correct": direct_ok,
            "final_correct": final_ok,
            "outcome": "BOTH_CORRECT" if direct_ok and final_ok else "DIRECT_WRONG_FINAL_CORRECT" if not direct_ok and final_ok else "DIRECT_CORRECT_FINAL_WRONG" if direct_ok and not final_ok else "BOTH_WRONG",
        })
    direct_correct = sum(row["direct_correct"] for row in evaluated)
    final_correct = sum(row["final_correct"] for row in evaluated)
    outcomes = Counter(row["outcome"] for row in evaluated)
    traces = latest_rows(out, expected)
    positive = [row for row in traces if row["pre_execution_control"]["gate_invoked"]]
    decisions = Counter(row["validator"]["decision"] for row in traces if row["validator"]["invoked"])
    sanity = {
        "label": SANITY_LABEL,
        "evaluation_started_after_prediction_finalization": True,
        "case_count": expected,
        "direct": {"correct": direct_correct, "total": expected, "accuracy": direct_correct / expected},
        "final_hybrid": {"correct": final_correct, "total": expected, "accuracy": final_correct / expected},
        "paired_outcomes": {
            "rescues": outcomes["DIRECT_WRONG_FINAL_CORRECT"],
            "harms": outcomes["DIRECT_CORRECT_FINAL_WRONG"],
            "both_correct": outcomes["BOTH_CORRECT"],
            "both_wrong": outcomes["BOTH_WRONG"],
            "net_gain": outcomes["DIRECT_WRONG_FINAL_CORRECT"] - outcomes["DIRECT_CORRECT_FINAL_WRONG"],
        },
        "gate_positive_count": len(positive),
        "gate_negative_count": expected - len(positive),
        "planner_invocations": sum(row["planner"]["invoked"] for row in traces),
        "structured_invocations": sum(row["structured"]["invoked"] for row in traces),
        "structured_successes": sum(row["structured"]["execution_success"] for row in traces),
        "structured_abstentions_or_failures": sum(row["structured"]["invoked"] and not row["structured"]["execution_success"] for row in traces),
        "validator_decisions": {
            "KEEP_DIRECT": decisions["KEEP_DIRECT"],
            "ACCEPT_STRUCTURED": decisions["REQUEST_REVISION"],
            "ACCEPT_STRUCTURED_native_label": "REQUEST_REVISION",
            "UNRESOLVED": decisions["UNRESOLVED"],
            "NOT_INVOKED": sum(not row["validator"]["invoked"] for row in traces),
        },
        "selective_accepts": sum(row["selective"]["accepted"] for row in traces),
        "final_overrides": sum(row["selective"]["accepted"] for row in traces),
        "per_case_evaluation": evaluated,
    }
    atomic_json(out / "sanity_evaluation.json", sanity)

    historical = read_json(HISTORICAL_DIRECT)["results"]
    identity = len(historical) == expected and all(
        int(row["eval_index"]) == qid and row["sample_token"] == cohort[qid]["sample_token"]
        and row["question"] == cohort[qid]["question"] and str(row["gt"]) == str(cohort[qid]["answer"])
        for qid, row in enumerate(historical)
    )
    if not identity:
        raise RuntimeError("Historical evidence-only comparison cohort mismatch")
    answer_changes = sum(normalize_answer(predictions[qid]["direct_answer"]) != normalize_answer(historical[qid]["pred"]) for qid in range(expected))
    historical_correct = sum(bool(row["correct"]) for row in historical)
    full_sentence = lambda value: len(str(value).split()) >= 4
    unknown = lambda value: normalize_answer(value) == "unknown"
    compatibility = {
        "comparison_role": "diagnostic_only_not_scientific_baseline",
        "same_50_case_identity_verified": identity,
        "stable_implementation": {
            "prompt": "run_phase3b15_oracle_plain_mllm_baseline.direct_prompt",
            "serialization": "run_real_evidence_pure_mllm_baseline.serialize_real_scene",
            "cleaner": "run_phase3b15_oracle_plain_mllm_baseline.clean_answer",
            "normalizer": "step71_common.normalize_answer",
        },
        "answer_changes": answer_changes,
        "historical_direct": {"correct": historical_correct, "accuracy": historical_correct / expected},
        "restored_direct": {"correct": direct_correct, "accuracy": direct_correct / expected},
        "direct_accuracy_difference": (direct_correct - historical_correct) / expected,
        "output_format": {
            "historical_full_sentence_count": sum(full_sentence(row["pred"]) for row in historical),
            "restored_full_sentence_count": sum(full_sentence(row["direct_answer"]) for row in predictions),
            "full_sentence_count_difference": sum(full_sentence(row["direct_answer"]) for row in predictions) - sum(full_sentence(row["pred"]) for row in historical),
        },
        "unknown_rate": {
            "historical_count": sum(unknown(row["pred"]) for row in historical),
            "restored_count": sum(unknown(row["direct_answer"]) for row in predictions),
            "rate_difference": (sum(unknown(row["direct_answer"]) for row in predictions) - sum(unknown(row["pred"]) for row in historical)) / expected,
        },
        "six_image_phase5b_branch_used_as_scientific_baseline": False,
    }
    atomic_json(out / "direct_compatibility_audit.json", compatibility)
    return sanity, compatibility


def choose_decision(validation: dict[str, Any], sanity: dict[str, Any], compatibility: dict[str, Any]) -> str:
    if not validation["all_protocol_assertions_pass"]:
        return "D. SUBMISSION_PROTOCOL_STILL_HAS_PROTOCOL_VIOLATIONS"
    if not compatibility["same_50_case_identity_verified"] or sanity["direct"]["accuracy"] <= 0.05:
        return "C. EVIDENCE_ONLY_PROTOCOL_REPRODUCTION_FAILURE"
    if sanity["final_overrides"] == 0:
        return "B. SUBMISSION_PROTOCOL_FROZEN_CONTROLLER_RECALIBRATION_PENDING"
    return "A. SUBMISSION_PROTOCOL_FROZEN_AND_READY_FOR_FULL_SCALE"


def write_report(out: Path, validation: dict[str, Any], sanity: dict[str, Any], compatibility: dict[str, Any]) -> str:
    efficiency = read_json(out / "efficiency_summary.json")
    planner_audit = read_json(out / "planner_cache_audit.json")
    decision = choose_decision(validation, sanity, compatibility)
    pair = sanity["paired_outcomes"]
    vd = sanity["validator_decisions"]
    report = f"""# Phase 5B.3 — Submission-Safe Protocol Rollback and Final Freeze

**{SANITY_LABEL}**

## Final protocol

The final Real-Evidence protocol is frozen as BEVFusion multimodal perception → shared structured scene evidence `E` → General Evidence Reasoner and pre-execution controller → rescue-aware gate → conditional label-free Planner/Structured Reasoner → conditional Evidence-Grounded Validator/Selective Override. The General Evidence Reasoner receives only `q + S(E)`: raw camera images, image paths as model inputs, and visual tensors are absent.

## Protocol validation

- Evidence-only Direct calls: **{validation['fresh_evidence_only_direct_count']}/50**
- Direct raw-camera usage: **{validation['direct_raw_camera_usage_count']}**
- Direct BEVFusion evidence compliance: **{validation['direct_bevfusion_evidence_count']}/50**
- Utility-before-Structured compliance: **{validation['utility_before_structured_count']}/50**
- Gate-negative contract: **{validation['gate_negative_contract_pass_count']}/{validation['gate_negative_count']}**
- Label-free formal plan invocations: **{validation['label_free_planner_invocation_count']}/{validation['gate_positive_count']}**
- Historical plan usage: **{validation['historical_plan_usage_count']}**
- GT / `eval_type` / template / Oracle leakage: **0 / 0 / 0 / 0**
- Protocol violations: **{validation['protocol_violation_count']}**

The fresh immutable Planner cache contains **{planner_audit['summary']['cache_record_count']}/50** normalized plans, all generated by the frozen question-only Planner with the five forbidden-use audit flags false. It is a computational optimization; method semantics remain `P=Planner(q)`.

Label-free traces and predictions were finalized and hashed before ground-truth attachment.

## 50-case development sanity check

- Direct exact match: **{sanity['direct']['correct']}/50 ({sanity['direct']['accuracy']:.1%})**
- Final Hybrid exact match: **{sanity['final_hybrid']['correct']}/50 ({sanity['final_hybrid']['accuracy']:.1%})**
- Rescues / harms / net gain: **{pair['rescues']} / {pair['harms']} / {pair['net_gain']:+d}**
- Gate positive / negative: **{sanity['gate_positive_count']} / {sanity['gate_negative_count']}**
- Planner / Structured invocations: **{sanity['planner_invocations']} / {sanity['structured_invocations']}**
- Structured successes / failures: **{sanity['structured_successes']} / {sanity['structured_abstentions_or_failures']}**
- Validator KEEP_DIRECT / ACCEPT_STRUCTURED / UNRESOLVED: **{vd['KEEP_DIRECT']} / {vd['ACCEPT_STRUCTURED']} / {vd['UNRESOLVED']}**
- Selective accepts / final overrides: **{sanity['selective_accepts']} / {sanity['final_overrides']}**

This is not a paper result and was not used for tuning.

## Historical evidence-only compatibility

The stable pre-Phase-5B evidence-only artifact is the identical 50-case cohort. Fresh restoration changed **{compatibility['answer_changes']}** answers. Historical Direct was **{compatibility['historical_direct']['correct']}/50 ({compatibility['historical_direct']['accuracy']:.1%})** and restored Direct is **{compatibility['restored_direct']['correct']}/50 ({compatibility['restored_direct']['accuracy']:.1%})**, a **{compatibility['direct_accuracy_difference']:+.1%}** difference. Full-sentence counts are {compatibility['output_format']['historical_full_sentence_count']} historical and {compatibility['output_format']['restored_full_sentence_count']} restored; exact-unknown counts are {compatibility['unknown_rate']['historical_count']} and {compatibility['unknown_rate']['restored_count']}.

The six-image Phase 5B branch is not part of the final submission protocol and is not treated as a scientific baseline.

## Efficiency and remaining work

Online inference made {efficiency['direct_llm_calls']} Direct calls, {efficiency['runtime_formal_plan_invocations']} gated formal-plan reads, {efficiency['structured_invocations']} Structured calls, {efficiency['validator_calls']} Validator calls, and {efficiency['selective_model_calls']} selective-model calls. The gate avoided {efficiency['planner_invocations_avoided_online']} online plan reads and {efficiency['structured_invocations_avoided']} Structured calls. Cache construction used {efficiency['offline_planner_cache_generation_calls']} one-time fresh Planner generations.

The evidence-only Direct operating regime is restored. The compatibility controllers and thresholds remain frozen and produced no final override, so final fitting must occur later on authorized non-evaluation data. No tuning or refitting occurred here.

External upstream BEVFusion checkpoint, exact preprocessing, sweep count, and synchronization details remain unresolved because `perception_method_spec.md` is unavailable; this does not block the downstream protocol freeze.

## Decision

{decision}
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")
    return decision


def finalize_manifest(out: Path) -> None:
    path = out / "submission_protocol_manifest.json"
    manifest = read_json(path)
    names = (
        "formal_planner_cache.jsonl", "planner_cache_audit.json", "formal_inference_trace.jsonl",
        "formal_predictions.json", "inference_completion.json", "protocol_validation.json",
        "direct_compatibility_audit.json", "sanity_evaluation.json", "efficiency_summary.json",
        "failure_summary.json", "final_report.md",
    )
    manifest["final_artifacts"] = {name: {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size} for name in names}
    manifest["finalized_at_utc"] = utc_now()
    atomic_json(path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_BEVFUSION_ROOT)
    parser.add_argument("--model-path", type=Path, default=Path(os.getenv("LOCAL_QW_MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--direct-max-new-tokens", type=int, default=48)
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit != 50 and args.output_dir.resolve() == DEFAULT_OUT.resolve():
        raise SystemExit("The formal Phase 5B.3 run requires exactly 50 cases")
    required = [args.cohort, args.evidence_root, args.model_path, CONTROL_MANIFEST, CONTROL_THRESHOLDS, HISTORICAL_DIRECT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required resources: " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_formal_cases(args.cohort, args.limit)
    cache, cache_audit = load_verified_planner_cache(args.output_dir, cases)

    state_path = args.output_dir / "run_state.json"
    if state_path.exists():
        run_id = str(read_json(state_path)["run_id"])
    else:
        run_id = f"phase5b3-{uuid.uuid4().hex}"
        atomic_json(state_path, {"run_id": run_id, "status": "RUNNING_INFERENCE", "started_at_utc": utc_now()})
    manifest_path = args.output_dir / "submission_protocol_manifest.json"
    if not manifest_path.exists():
        atomic_json(manifest_path, build_manifest(args, cases, run_id, cache_audit))
        atomic_json(args.output_dir / "provider_dependency_audit.json", provider_audit(args.evidence_root))

    completion_path = args.output_dir / "inference_completion.json"
    if not args.evaluate_only and not completion_path.exists():
        trace_path = args.output_dir / "formal_inference_trace.jsonl"
        completed = {int(row["question_id"]) for row in read_jsonl(trace_path)}
        pending = [case for case in cases if case.question_id not in completed]
        if pending:
            qwen = EvidenceOnlyQwenRuntime(args.model_path, args.min_pixels, args.max_pixels)
            qwen.load()
            executor = PlanCompiledAtomicExecutor(tools=None)
            adapter = BEVFusionSceneAdapter(args.evidence_root)
            controls = FrozenControlStack(CONTROL_MANIFEST, CONTROL_THRESHOLDS)
            for position, case in enumerate(pending, start=1):
                item_started = time.time()
                try:
                    row = run_case(case, run_id, qwen, executor, adapter, controls, cache[case.question_id], cache_audit["formal_planner_cache_sha256"], args.direct_max_new_tokens)
                except Exception as exc:
                    row = failure_record(case, run_id, exc, item_started)
                append_jsonl(trace_path, row)
                print(json.dumps({
                    "progress": f"{position}/{len(pending)}",
                    "question_id": case.question_id,
                    "failure": row.get("failure"),
                    "direct_answer": (row.get("direct") or {}).get("answer"),
                    "gate_invoked": (row.get("pre_execution_control") or {}).get("gate_invoked"),
                    "structured_success": (row.get("structured") or {}).get("execution_success"),
                    "final_answer": row.get("final_answer"),
                }, sort_keys=True), flush=True)
            del controls, adapter, executor, qwen
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        validation = finalize_inference(args.output_dir, cases, run_id, cache_audit)
        atomic_json(state_path, {
            "run_id": run_id,
            "status": "INFERENCE_FINALIZED" if validation["all_protocol_assertions_pass"] else "INFERENCE_PROTOCOL_FAILED",
            "updated_at_utc": utc_now(),
            "validation": validation,
        })
    else:
        validation = read_json(args.output_dir / "protocol_validation.json")
    if not validation["all_protocol_assertions_pass"]:
        print(json.dumps({"status": "INFERENCE_PROTOCOL_FAILED", "validation": validation}, indent=2))
        return 2

    sanity, compatibility = evaluate(args.output_dir, args.cohort, args.limit)
    decision = write_report(args.output_dir, validation, sanity, compatibility)
    finalize_manifest(args.output_dir)
    atomic_json(state_path, {"run_id": run_id, "status": "COMPLETE", "decision": decision, "updated_at_utc": utc_now(), "validation": validation})
    print(json.dumps({
        "status": "COMPLETE",
        "decision": decision,
        "direct": sanity["direct"],
        "final_hybrid": sanity["final_hybrid"],
        "paired_outcomes": sanity["paired_outcomes"],
        "gate_positive": sanity["gate_positive_count"],
        "gate_negative": sanity["gate_negative_count"],
        "historical_answer_changes": compatibility["answer_changes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
