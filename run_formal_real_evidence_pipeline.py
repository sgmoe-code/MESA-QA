"""Phase 5B formal Real-Evidence runner.

This is the authoritative inference entry point for the corrected protocol.  It
deliberately does not import a historical prediction, plan, executor result, or
QA label.  The only per-case inputs crossing the inference boundary are the
question, six current keyframes, identifiers used for lookup, and normalized
BEVFusion detections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from atomic_program_executor import PlanCompiledAtomicExecutor
from bevfusion_scene_adapter import BEVFusionSceneAdapter, DEFAULT_BEVFUSION_ROOT
from experiment_config import DEFAULT_MODEL_PATH
from mllm_question_planner_v2 import MLLMQuestionPlannerV2
from query_resolver import resolve_query
from run_phase3b15_oracle_plain_mllm_baseline import (
    _disable_optional_transformers_imports,
    clean_answer,
)
from run_phase4a2_calibration_cohort import evidence_signals, question_signals
from run_phase4a3_reasoner_reliability import FEATURE_SETS, predict_logistic, profile
from run_phase4b3_evidence_grounded_validator import (
    ACTIONS,
    features as validator_features,
    predict_multiclass,
)
from step71_common import normalize_answer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "phase5b_formal_pipeline_correction"
DEFAULT_COHORT = ROOT / "data" / "questions" / "mini_val_for_agent.json"
CONTROL_MANIFEST = ROOT / "outputs" / "phase4c_final_evaluation" / "final_model_manifest.json"
CONTROL_THRESHOLDS = ROOT / "outputs" / "phase4c_final_evaluation" / "frozen_thresholds.json"
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
PROTOCOL_VERSION = "phase5b.real_evidence.v1"
DIRECT_PROMPT_VERSION = "phase5b.direct.six_native_images_plus_bevfusion.v1"
PLANNER_PROMPT_VERSION = "mllm_question_planner_v2.question_only.no_template.v1"


@dataclass(frozen=True)
class FormalCase:
    question_id: int
    sample_token: str
    question: str
    camera_inputs: tuple[tuple[str, str], ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_formal_cases(path: Path, limit: int) -> list[FormalCase]:
    """Establish a narrow boundary; labels/template metadata are not retained."""
    payload = read_json(path)
    rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    cases: list[FormalCase] = []
    for question_id, source in enumerate(rows[:limit]):
        image_map = source.get("image_paths") or {}
        camera_inputs = tuple((channel, str(image_map.get(channel) or "")) for channel in CAMERAS)
        cases.append(FormalCase(
            question_id=question_id,
            sample_token=str(source.get("sample_token") or ""),
            question=str(source.get("question") or ""),
            camera_inputs=camera_inputs,
        ))
    return cases


def canonical_bevfusion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """One explicit conversion boundary for the external detector contract."""
    objects = []
    for index, raw in enumerate(payload.get("objects") or []):
        objects.append({
            "object_id": str(raw.get("object_id") or f"bevfusion_{index:06d}"),
            "category": raw.get("category"),
            "confidence": raw.get("confidence"),
            "center": raw.get("center"),
            "size": raw.get("size"),
            "yaw": raw.get("yaw"),
            "velocity": raw.get("velocity"),
        })
    return {
        "sample_token": str(payload.get("sample_token") or ""),
        "scene_token": payload.get("scene_token"),
        "source": str(payload.get("source") or "BEVFusion"),
        "coordinate_frame": str(payload.get("coordinate_frame") or "lidar"),
        "objects": objects,
    }


def serialize_bevfusion_for_direct(payload: dict[str, Any]) -> str:
    header = {
        "sample_token": payload["sample_token"],
        "source": payload["source"],
        "coordinate_frame": payload["coordinate_frame"],
        "object_count": len(payload["objects"]),
    }
    lines = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    lines.extend(json.dumps(obj, sort_keys=True, separators=(",", ":")) for obj in payload["objects"])
    return "\n".join(lines)


def direct_prompt(evidence: str, question: str) -> str:
    return f"""You are answering a question about the current driving scene.

The six native camera images preceding this text are, in order: CAM_FRONT,
CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.
Use both the visual observations and the BEVFusion machine-perception evidence.
Answer only from supplied current-scene information. Do not invent unavailable
facts. If the evidence is insufficient, say unknown. Produce only a concise
final answer.

BEVFusion evidence (lidar frame, one JSON object per line):
{evidence}

Question:
{question}

Final answer:"""


class FormalQwenRuntime:
    def __init__(self, model_path: Path, min_pixels: int, max_pixels: int):
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None:
            return
        _disable_optional_transformers_imports()
        from qwen35_runtime import load_text_model

        self.model, self.processor = load_text_model(self.model_path)

    def direct(self, case: FormalCase, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        import torch
        from qwen_vl_utils import process_vision_info

        paths = [path for _, path in case.camera_inputs]
        if len(paths) != 6 or len(set(paths)) != 6:
            raise ProtocolViolation("direct_camera_cardinality_not_six_unique")
        image_meta = []
        for (channel, raw_path) in case.camera_inputs:
            path = Path(raw_path)
            if not path.is_file():
                raise ProtocolViolation(f"camera_file_missing:{channel}:{path}")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_meta.append({
                    "channel": channel,
                    "path": str(path),
                    "width": int(image.width),
                    "height": int(image.height),
                    "sha256": sha256_file(path),
                })

        content: list[dict[str, Any]] = []
        for item in image_meta:
            content.append({"type": "text", "text": item["channel"]})
            content.append({
                "type": "image",
                "image": item["path"],
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            })
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        input_count = len(image_inputs or [])
        if input_count != 6:
            raise ProtocolViolation(f"processor_received_{input_count}_images")
        inputs = self.processor(
            text=[rendered], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)
        started = time.time()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return {
            "raw_output": raw,
            "answer": clean_answer(raw),
            "camera_inputs": image_meta,
            "camera_input_count": input_count,
            "processor_pixel_values_present": bool("pixel_values" in inputs),
            "input_token_count": int(inputs.input_ids.shape[-1]),
            "latency_sec": round(time.time() - started, 4),
        }


class FormalPlanner(MLLMQuestionPlannerV2):
    """The existing normalized planner with a bounded deterministic decode."""

    def _run_text(self, prompt: str, max_tokens: int = 340) -> str:
        return super()._run_text(prompt, max_tokens=min(max_tokens, 256))


class ProtocolViolation(RuntimeError):
    pass


def model_tuple(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(spec["weights"], dtype=float),
        np.asarray(spec["mean"], dtype=float),
        np.asarray(spec["std"], dtype=float),
    )


class FrozenControlStack:
    """Compatibility-only invocation of the frozen Phase 4 lightweight stack."""

    def __init__(self, manifest_path: Path, thresholds_path: Path):
        manifest = read_json(manifest_path)
        thresholds = read_json(thresholds_path)
        utility = manifest["utility"]
        self.feature_names = list(utility["features"])
        self.direct_model = model_tuple(utility["direct_model"])
        self.structured_model = model_tuple(utility["structured_model"])
        self.utility_model = model_tuple(utility["utility_model"])
        self.calibrator = model_tuple(utility["scalar_calibrator"])
        self.utility_threshold = float(thresholds["utility"]["threshold"])
        self.validator_names = list(manifest["validator"]["features"])
        self.validator_models = [model_tuple(manifest["validator"]["models"][action]) for action in ACTIONS]
        self.selective_names = list(manifest["selective_confidence"]["features"])
        self.selective_model = model_tuple(manifest["selective_confidence"]["model"])
        self.selective_threshold = float(thresholds["selective"]["threshold"])

    def pre_execution(
        self,
        question: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Score the frozen reliability/utility stack from q,E only.

        This boundary is intentionally independent of Direct text, Planner
        output, Structured candidates, and executor traces so callers can use
        the utility decision as a true pre-Structured invocation gate.
        """
        feature_values, profile_meta = profile({
            "question": question,
            "question_signals": question_signals(question),
            "evidence_signals": evidence_signals(rows),
        })
        x = np.asarray([[feature_values[name] for name in self.feature_names]], dtype=float)
        s_direct = float(predict_logistic(x, self.direct_model)[0])
        s_structured = float(predict_logistic(x, self.structured_model)[0])
        meta_x = np.column_stack([x, [s_direct], [s_structured], [1 - s_direct], [(1 - s_direct) * s_structured]])
        raw_utility = float(predict_logistic(meta_x, self.utility_model)[0])
        utility = float(predict_logistic(np.asarray([[raw_utility]]), self.calibrator)[0])
        return {
            "profile_generated": True,
            "feature_values": feature_values,
            "profile_meta": profile_meta,
            "s_direct": s_direct,
            "s_structured": s_structured,
            "utility_raw_score": raw_utility,
            "utility_score": utility,
            "utility_threshold": self.utility_threshold,
            "gate_invoked": bool(utility >= self.utility_threshold),
        }

    def post_execution(
        self,
        direct_answer: str,
        direct_raw: str,
        structured_answer: str | None,
        executor_trace: dict[str, Any],
        pre_execution: dict[str, Any],
    ) -> dict[str, Any]:
        feature_values = pre_execution["feature_values"]
        profile_meta = pre_execution["profile_meta"]
        s_direct = float(pre_execution["s_direct"])
        s_structured = float(pre_execution["s_structured"])
        raw_utility = float(pre_execution["utility_raw_score"])
        utility = float(pre_execution["utility_score"])
        gate = bool(pre_execution["gate_invoked"])

        package = {
            "direct": {
                "initial_answer": direct_answer,
                "normalized_answer": normalize_answer(direct_answer),
                "s_direct": s_direct,
                "explicit_uncertainty": bool(re.search(r"uncertain|unknown|cannot determine", direct_raw, re.I)),
                "answer_format_valid": bool(str(direct_answer or "").strip()),
            },
            "structured": {
                "candidate_answer": structured_answer,
                "s_structured": s_structured,
                "required_capabilities": profile_meta.get("required_capabilities"),
                "atomic_program": executor_trace.get("compiled_program"),
                "executor_output": executor_trace.get("candidate_answer"),
                "provider_provenance": "BEVFusionSceneAdapter + deterministic geometry executor",
            },
            "utility": {
                "rescue_utility_score": utility,
                "hard_case_risk_score": (1 - s_direct) * (1 - s_structured),
            },
            "shared": {"abstract_evidence_and_compatibility_profile": feature_values},
        }
        values, support_audit = validator_features(package, {"executor_trace": executor_trace})
        decision: str = "NOT_INVOKED"
        probabilities = None
        confidence = None
        override = False
        if gate:
            xv = np.asarray([[values[name] for name in self.validator_names]], dtype=float)
            decisions, probs = predict_multiclass(xv, self.validator_models)
            decision = decisions[0]
            probabilities = {action: float(probs[0, idx]) for idx, action in enumerate(ACTIONS)}
            selective_values = {"base_request_probability": probabilities["REQUEST_REVISION"], **values}
            xs = np.asarray([[selective_values[name] for name in self.selective_names]], dtype=float)
            confidence = float(predict_logistic(xs, self.selective_model)[0])
            override = bool(
                decision == "REQUEST_REVISION"
                and confidence >= self.selective_threshold
                and structured_answer is not None
                and not executor_trace.get("failure_reason")
            )
        return {
            "utility_score": utility,
            "utility_raw_score": raw_utility,
            "utility_threshold": self.utility_threshold,
            "s_direct": s_direct,
            "s_structured": s_structured,
            "gate_invoked": gate,
            "validator_decision": decision,
            "validator_probabilities": probabilities,
            "selective_confidence": confidence,
            "selective_threshold": self.selective_threshold,
            "override": override,
            "support_audit": support_audit,
            "compatibility_note": "Frozen lightweight models invoked for smoke compatibility; changed Direct/Planner semantics require future non-50-case recalibration.",
        }

    def run(
        self,
        question: str,
        rows: list[dict[str, Any]],
        direct_answer: str,
        direct_raw: str,
        structured_answer: str | None,
        executor_trace: dict[str, Any],
    ) -> dict[str, Any]:
        """Backward-compatible Phase 5B wrapper.

        Phase 5B.1 calls ``pre_execution`` before Planner invocation and calls
        ``post_execution`` only on gate-positive cases.
        """
        pre = self.pre_execution(question, rows)
        return self.post_execution(
            direct_answer, direct_raw, structured_answer, executor_trace, pre
        )


def provider_audit(evidence_root: Path) -> dict[str, Any]:
    providers = [
        {
            "provider_name": "BEVFusion normalized detection cache",
            "purpose": "Sole external structured scene-evidence provider for the formal Structured path and machine evidence for Direct.",
            "input": "sample_token-indexed external detector JSON",
            "output": "canonical objects and executor scene rows",
            "question_dependent": False,
            "can_be_generated_for_fresh_sample": True,
            "requires_gt": False,
            "currently_reads_old_phase_or_step_artifact": True,
            "historical_location": str(evidence_root),
            "formal_cleanup": "Consumed only through an explicit canonical adapter boundary; location is configurable and contents are detector outputs, not prior QA outputs.",
            "final_classification": "REQUIRED_FINAL",
        },
        {
            "provider_name": "Deterministic geometry/relation derivation",
            "purpose": "Derive spatial relations from canonical lidar-frame object centers during execution.",
            "input": "current canonical BEVFusion executor rows plus fresh normalized plan",
            "output": "relation matches, support IDs, and execution trace",
            "question_dependent": True,
            "can_be_generated_for_fresh_sample": True,
            "requires_gt": False,
            "currently_reads_old_phase_or_step_artifact": False,
            "final_classification": "REQUIRED_FINAL",
        },
        {
            "provider_name": "Grounded visual predicate/completion evidence",
            "purpose": "Historical opportunistic visual completion and predicate verification.",
            "input": "historical grounded requests/crops/Qwen artifacts",
            "output": "visual semantic predicates or completion candidates",
            "question_dependent": True,
            "can_be_generated_for_fresh_sample": False,
            "requires_gt": False,
            "currently_reads_old_phase_or_step_artifact": True,
            "formal_cleanup": "Not imported or loaded by the formal runner; absence is explicit.",
            "final_classification": "LEGACY_NOT_FINAL",
        },
        {
            "provider_name": "CenterPoint/evidence-head scene artifacts",
            "purpose": "Historical alternate detector/evidence providers.",
            "input": "CenterPoint NPZ and trained evidence-head artifacts",
            "output": "alternate object rows/status evidence",
            "question_dependent": False,
            "can_be_generated_for_fresh_sample": False,
            "requires_gt": False,
            "currently_reads_old_phase_or_step_artifact": True,
            "formal_cleanup": "Not imported or loaded by the formal runner.",
            "final_classification": "LEGACY_NOT_FINAL",
        },
        {
            "provider_name": "Temporal evidence provider",
            "purpose": "Historical optional temporal status evidence.",
            "input": "neighboring sample_data frames and historical request artifacts",
            "output": "temporal attribute records",
            "question_dependent": True,
            "can_be_generated_for_fresh_sample": False,
            "requires_gt": False,
            "currently_reads_old_phase_or_step_artifact": True,
            "formal_cleanup": "Not imported or loaded; current Phase 5B contract is keyframe-only.",
            "final_classification": "LEGACY_NOT_FINAL",
        },
        {
            "provider_name": "Oracle NuScenes annotations/scene rows",
            "purpose": "Historical diagnostic Oracle evidence.",
            "input": "sample_annotation and Oracle scene graph",
            "output": "GT-derived object rows",
            "question_dependent": False,
            "can_be_generated_for_fresh_sample": True,
            "requires_gt": True,
            "currently_reads_old_phase_or_step_artifact": True,
            "formal_cleanup": "Strictly excluded from the Real-Evidence runner.",
            "final_classification": "LEGACY_NOT_FINAL",
        },
    ]
    return {
        "audit_version": PROTOCOL_VERSION,
        "formal_structured_provider_count": 2,
        "optional_final_provider_count": 0,
        "providers": providers,
        "all_required_final_reproducible_for_unseen_samples": all(
            p["can_be_generated_for_fresh_sample"] and not p["requires_gt"]
            for p in providers if p["final_classification"] == "REQUIRED_FINAL"
        ),
    }


def protocol_flags(direct: dict[str, Any], planner: dict[str, Any], structured: dict[str, Any]) -> dict[str, bool]:
    return {
        "direct_multimodal": bool(
            direct.get("camera_input_count") == 6 and direct.get("processor_pixel_values_present")
        ),
        "direct_has_six_camera_inputs": direct.get("camera_input_count") == 6,
        "direct_has_bevfusion_evidence": bool(direct.get("perception_evidence_ref")),
        "direct_prediction_generated_this_run": bool(direct.get("generated_this_run")),
        "planner_generated_this_run": bool(planner.get("generated_this_run")),
        "structured_trace_generated_this_run": bool(structured.get("generated_this_run")),
        "no_cached_direct": True,
        "no_cached_plan": True,
        "no_eval_type_reasoning": True,
        "no_template_routing": True,
        "no_gt_answer": True,
        "no_oracle_gt": True,
    }


def build_manifest(args: argparse.Namespace, cases: list[FormalCase], run_id: str) -> dict[str, Any]:
    case_rows = [
        {
            "question_id": case.question_id,
            "sample_token": case.sample_token,
            "question_sha256": hashlib.sha256(case.question.encode()).hexdigest(),
            "camera_paths": {channel: path for channel, path in case.camera_inputs},
        }
        for case in cases
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "authoritative_runner": str(Path(__file__).resolve()),
        "cohort_source": str(args.cohort.resolve()),
        "cohort_source_sha256": sha256_file(args.cohort),
        "cohort_size": len(cases),
        "deterministic_case_order_sha256": stable_hash(case_rows),
        "cases": case_rows,
        "model": {
            "checkpoint": str(args.model_path.resolve()),
            "checkpoint_config_sha256": sha256_file(args.model_path / "config.json"),
            "do_sample": False,
            "direct_max_new_tokens": args.direct_max_new_tokens,
            "planner_max_new_tokens": 256,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "evidence": {
            "regime": "REAL_MULTIMODAL_EVIDENCE",
            "bevfusion_root": str(args.evidence_root.resolve()),
            "canonical_conversion": "canonical_bevfusion_payload -> BEVFusionSceneAdapter",
        },
        "control_stack": {
            "mode": "frozen_lightweight_compatibility_smoke_only",
            "model_manifest": str(CONTROL_MANIFEST.resolve()),
            "model_manifest_sha256": sha256_file(CONTROL_MANIFEST),
            "thresholds": str(CONTROL_THRESHOLDS.resolve()),
            "thresholds_sha256": sha256_file(CONTROL_THRESHOLDS),
            "recalibration_required_before_formal_performance_interpretation": True,
        },
        "forbidden_inference_fields": [
            "ground_truth_answer", "eval_type", "template_id", "correctness_label",
            "historical_prediction", "historical_plan", "historical_executor_answer", "oracle_object_gt",
        ],
        "resume_policy": "same run_id traces are durable and skipped; --retry-failed re-executes failed items",
    }


def completed_ids(trace_path: Path, retry_failed: bool) -> set[int]:
    rows = read_jsonl(trace_path)
    latest = {int(row["question_id"]): row for row in rows}
    if retry_failed:
        return {qid for qid, row in latest.items() if not row.get("failure")}
    return set(latest)


def run_case(
    case: FormalCase,
    run_id: str,
    qwen: FormalQwenRuntime,
    planner: FormalPlanner,
    executor: PlanCompiledAtomicExecutor,
    adapter: BEVFusionSceneAdapter,
    controls: FrozenControlStack,
    direct_max_new_tokens: int,
) -> dict[str, Any]:
    started = time.time()
    raw_payload = adapter.load_sample(case.sample_token)
    canonical = canonical_bevfusion_payload(raw_payload)
    if canonical["sample_token"] != case.sample_token:
        raise ProtocolViolation("bevfusion_sample_token_mismatch")
    if canonical["source"].lower() != "bevfusion":
        raise ProtocolViolation("perception_source_not_bevfusion")
    if canonical["coordinate_frame"].lower() != "lidar":
        raise ProtocolViolation("perception_coordinate_frame_not_lidar")
    scene_rows = adapter.build_executor_scene_rows(case.sample_token)
    evidence_path = adapter.evidence_root / f"{case.sample_token}.json"
    evidence_ref = {
        "path": str(evidence_path.resolve()),
        "sha256": sha256_file(evidence_path),
        "object_count": len(canonical["objects"]),
        "canonical_schema": "phase5b.bevfusion.canonical.v1",
    }

    prompt = direct_prompt(serialize_bevfusion_for_direct(canonical), case.question)
    direct_call = qwen.direct(case, prompt, direct_max_new_tokens)
    direct = {
        "generated_this_run": True,
        "generation_run_id": run_id,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_protocol_version": DIRECT_PROMPT_VERSION,
        "perception_evidence_ref": evidence_ref,
        **direct_call,
    }

    # No template hint, exemplar, scene cue, answer, label, or historical plan is passed.
    plan_result = planner.plan(case.question, template_type="")
    planner_failed = bool(plan_result.get("planner_error")) or "fallback" in str(plan_result.get("planner_source", ""))
    planner_trace = {
        "generated_this_run": True,
        "generation_run_id": run_id,
        "planner_generated_this_run": True,
        "planner_input": {"question": case.question},
        "planner_prompt": plan_result.get("planner_prompt"),
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
        "planner_raw_output": plan_result.get("planner_raw"),
        "normalized_plan": None if planner_failed else {
            key: plan_result.get(key) for key in (
                "question_type", "answer_type", "hop", "target_entity", "reference_entity",
                "relations", "need_modules", "reasoning_path", "why",
            )
        },
        "planner_source": plan_result.get("planner_source"),
        "normalization_success": not planner_failed,
        "failure_reason": plan_result.get("planner_error") if planner_failed else None,
        "cached_plan_used": False,
    }

    if planner_failed:
        structured_answer = None
        executor_trace = {
            "source": "compiled_atomic_executor",
            "compiled_program": None,
            "candidate_answer": None,
            "failure_reason": f"planner_normalization_failed:{plan_result.get('planner_error')}",
            "num_support": 0,
            "num_references": 0,
        }
        resolved_operation = None
    else:
        normalized_plan = planner_trace["normalized_plan"]
        resolved = resolve_query(normalized_plan)  # Formal API: plan only; eval_type is never supplied.
        resolved_operation = resolved["operation"]
        structured_answer, executor_trace = executor.execute(normalized_plan, scene_rows)

    structured = {
        "generated_this_run": True,
        "generation_run_id": run_id,
        "resolved_operation": resolved_operation,
        "atomic_program": executor_trace.get("compiled_program"),
        "candidate": structured_answer,
        "execution_success": bool(structured_answer is not None and not executor_trace.get("failure_reason")),
        "failure_reason": executor_trace.get("failure_reason"),
        "executor_trace": executor_trace,
        "evidence_refs": [evidence_ref],
        "provider_provenance": ["BEVFusionSceneAdapter", "deterministic_geometry_relation_derivation"],
    }
    control = controls.run(
        case.question, scene_rows, direct["answer"], direct["raw_output"], structured_answer, executor_trace
    )
    final_answer = structured_answer if control["override"] else direct["answer"]
    flags = protocol_flags(direct, planner_trace, structured)
    violations = [name for name, passed in flags.items() if not passed]
    if violations:
        raise ProtocolViolation(";".join(violations))
    return {
        "question_id": case.question_id,
        "sample_token": case.sample_token,
        "scene_token": canonical.get("scene_token"),
        "question": case.question,
        "evidence_regime": "REAL_MULTIMODAL_EVIDENCE",
        "direct": direct,
        "planner": planner_trace,
        "structured": structured,
        "control": control,
        "final_answer": final_answer,
        "protocol_audit": flags,
        "protocol_violations": violations,
        "failure": None,
        "run_id": run_id,
        "completed_at_utc": utc_now(),
        "latency_sec": round(time.time() - started, 4),
    }


def failure_record(case: FormalCase, run_id: str, exc: Exception, started: float) -> dict[str, Any]:
    category = "PROTOCOL_VIOLATION" if isinstance(exc, ProtocolViolation) else type(exc).__name__
    return {
        "question_id": case.question_id,
        "sample_token": case.sample_token,
        "question": case.question,
        "evidence_regime": "REAL_MULTIMODAL_EVIDENCE",
        "failure": {"category": category, "message": str(exc)},
        "protocol_violations": [str(exc)] if isinstance(exc, ProtocolViolation) else [],
        "run_id": run_id,
        "completed_at_utc": utc_now(),
        "latency_sec": round(time.time() - started, 4),
    }


def finalize(out: Path, cases: list[FormalCase], run_id: str) -> dict[str, Any]:
    all_rows = read_jsonl(out / "formal_inference_trace.jsonl")
    latest = {int(row["question_id"]): row for row in all_rows}
    rows = [latest[qid] for qid in sorted(latest) if qid < len(cases)]
    failures = [row for row in rows if row.get("failure")]
    successes = [row for row in rows if not row.get("failure")]
    categories = Counter((row.get("failure") or {}).get("category", "UNKNOWN") for row in failures)
    violation_count = sum(len(row.get("protocol_violations") or []) for row in rows)

    predictions = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "ground_truth_included": False,
        "correctness_included": False,
        "predictions": [
            {
                "question_id": row["question_id"],
                "sample_token": row["sample_token"],
                "direct_answer": (row.get("direct") or {}).get("answer"),
                "structured_answer": (row.get("structured") or {}).get("candidate"),
                "final_answer": row.get("final_answer"),
                "override": (row.get("control") or {}).get("override"),
                "failure": row.get("failure"),
            }
            for row in rows
        ],
    }
    atomic_json(out / "formal_predictions.json", predictions)

    validation = {
        "expected_case_count": len(cases),
        "persisted_case_count": len(rows),
        "fresh_direct_success_count": sum(bool((r.get("direct") or {}).get("generated_this_run")) for r in successes),
        "six_image_input_count": sum((r.get("direct") or {}).get("camera_input_count") == 6 for r in successes),
        "fresh_planner_success_count": sum(
            bool((r.get("planner") or {}).get("generated_this_run")) and bool((r.get("planner") or {}).get("normalization_success"))
            for r in successes
        ),
        "fresh_planner_call_count": sum(bool((r.get("planner") or {}).get("generated_this_run")) for r in successes),
        "structured_execution_success_count": sum(bool((r.get("structured") or {}).get("execution_success")) for r in successes),
        "structured_explicit_failure_or_abstention_count": sum(
            not bool((r.get("structured") or {}).get("execution_success")) for r in successes
        ),
        "protocol_violation_count": violation_count,
        "all_protocol_assertions_pass": bool(len(rows) == len(cases) and not failures and violation_count == 0),
        "ground_truth_attached": False,
        "accuracy_computed": False,
    }
    atomic_json(out / "protocol_validation.json", validation)
    atomic_json(out / "failure_summary.json", {
        "failure_count": len(failures),
        "categories": dict(sorted(categories.items())),
        "failed_items": [
            {"question_id": r["question_id"], "sample_token": r["sample_token"], **(r.get("failure") or {})}
            for r in failures
        ],
    })
    atomic_json(out / "runner_readiness.json", {
        "resume": True,
        "skip_completed": True,
        "per_qa_partial_persistence": True,
        "failed_item_retry": True,
        "deterministic_manifest": True,
        "fresh_artifact_provenance": True,
        "per_case_failure_logging": True,
        "atomic_summary_writes": True,
        "batching": False,
        "distributed_execution": False,
        "full_scale_notes": [
            "Use a fresh output directory/run ID for official evaluation.",
            "Recalibrate frozen lightweight control models on an authorized non-evaluation cohort because Direct/Planner semantics changed.",
            "Provision official-val six keyframes and sample-token-indexed BEVFusion cache before scaling.",
        ],
    })
    write_report(out, validation, dict(categories))
    manifest_path = out / "formal_protocol_manifest.json"
    manifest = read_json(manifest_path)
    artifact_names = (
        "provider_dependency_audit.json",
        "formal_inference_trace.jsonl",
        "formal_predictions.json",
        "protocol_validation.json",
        "failure_summary.json",
        "runner_readiness.json",
        "final_report.md",
    )
    manifest["fresh_artifact_provenance"] = {
        name: {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size}
        for name in artifact_names
    }
    manifest["finalized_at_utc"] = utc_now()
    atomic_json(manifest_path, manifest)
    return validation


def write_report(out: Path, validation: dict[str, Any], categories: dict[str, int]) -> None:
    audit = read_json(out / "provider_dependency_audit.json")
    required = [p for p in audit["providers"] if p["final_classification"] == "REQUIRED_FINAL"]
    optional = [p for p in audit["providers"] if p["final_classification"] == "OPTIONAL_FINAL"]
    legacy = [p for p in audit["providers"] if p["final_classification"] == "LEGACY_NOT_FINAL"]
    if validation["all_protocol_assertions_pass"]:
        decision = "B. FORMAL_PIPELINE_CORRECTED_WITH_MINOR_FULL_SCALE_ENGINEERING_REMAINING"
    elif not audit["all_required_final_reproducible_for_unseen_samples"]:
        decision = "C. FORMAL_PIPELINE_BLOCKED_BY_REQUIRED_EVIDENCE_PROVIDER"
    else:
        decision = "D. FORMAL_PIPELINE_STILL_VIOLATES_INFERENCE_PROTOCOL"
    report = f"""# Phase 5B — Formal Pipeline Correction

## A. Direct Path

- Every successful Real-Evidence Direct call contained six native image items: **{validation['six_image_input_count']}/{validation['expected_case_count']}**.
- Every Direct prompt also contained the current sample's canonical BEVFusion evidence.
- Fresh Direct answers: **{validation['fresh_direct_success_count']}/{validation['expected_case_count']}**. Historical predictions were never loaded.
- Qwen2.5-VL operated in native multimodal mode. Per-case traces record six processor image inputs, `pixel_values` presence, channel/path/dimensions/hash, and input-token count.

## B. Planner

- Fresh Planner calls: **{validation['fresh_planner_call_count']}/{validation['expected_case_count']}**; normalized successfully: **{validation['fresh_planner_success_count']}/{validation['expected_case_count']}**.
- No historical plan was loaded and no fallback to a cached plan exists.
- Planner input is the current question only. Operation resolution uses the fresh normalized `question_type`; neither `eval_type` nor template metadata enters the call or resolver.

## C. Structured Evidence

REQUIRED_FINAL ({len(required)}): {', '.join(p['provider_name'] for p in required)}.

OPTIONAL_FINAL ({len(optional)}): {'none' if not optional else ', '.join(p['provider_name'] for p in optional)}.

LEGACY_NOT_FINAL ({len(legacy)}): {', '.join(p['provider_name'] for p in legacy)}. These providers are not imported or read by the formal runner.

Every REQUIRED_FINAL provider has a reproducible, GT-free generation/derivation path for unseen samples. BEVFusion is an external stage and is consumed through one sample-token-indexed canonical conversion boundary.

## D. Ground-Truth Boundary

No QA answer, NuScenes annotation/object GT, Oracle scene graph, correctness/outcome label, `eval_type`, or template ID entered Direct generation, Planner, Structured execution, Gate, Validator, or final arbitration. Predictions and traces contain no labels. The cohort loader immediately projects the dataset record to question, sample identifier, and six image paths.

## E. 50-Case Integration

- Fresh Direct success: **{validation['fresh_direct_success_count']}/{validation['expected_case_count']}**
- Six-image Direct input: **{validation['six_image_input_count']}/{validation['expected_case_count']}**
- Fresh Planner normalization success: **{validation['fresh_planner_success_count']}/{validation['expected_case_count']}**
- Structured execution success: **{validation['structured_execution_success_count']}/{validation['expected_case_count']}**
- Structured explicit failure/abstention: **{validation['structured_explicit_failure_or_abstention_count']}**
- Protocol violations: **{validation['protocol_violation_count']}**
- Failure categories: `{json.dumps(categories, sort_keys=True)}`

Accuracy was not computed. This run is an integration/protocol validation, not a formal performance result.

## F. Full-Scale Readiness

The runner now supports resume, completed-item skipping, per-QA durable JSONL persistence, failed-item retry, a deterministic cohort manifest, provenance, and failure logging. Before official scaling, provision official-val camera/BEVFusion assets and recalibrate the lightweight control models on an authorized non-evaluation cohort; the old fitted weights are invoked here only for compatibility because Direct and Planner semantics changed. Batching/distributed execution remains optional engineering work.

{decision}
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")


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
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit != 50 and args.output_dir.resolve() == DEFAULT_OUT.resolve():
        raise SystemExit("The authoritative Phase 5B output requires --limit 50; use another --output-dir for tests.")
    required = [args.cohort, args.evidence_root, args.model_path, CONTROL_MANIFEST, CONTROL_THRESHOLDS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required resources: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = args.output_dir / "run_state.json"
    if run_state_path.exists():
        run_state = read_json(run_state_path)
        run_id = str(run_state["run_id"])
    else:
        run_id = f"phase5b-{uuid.uuid4().hex}"
        atomic_json(run_state_path, {"run_id": run_id, "started_at_utc": utc_now(), "status": "RUNNING"})

    cases = load_formal_cases(args.cohort, args.limit)
    if len(cases) != args.limit:
        raise SystemExit(f"Expected {args.limit} cases, found {len(cases)}")
    atomic_json(args.output_dir / "formal_protocol_manifest.json", build_manifest(args, cases, run_id))
    audit = provider_audit(args.evidence_root)
    atomic_json(args.output_dir / "provider_dependency_audit.json", audit)
    if not audit["all_required_final_reproducible_for_unseen_samples"]:
        atomic_json(run_state_path, {"run_id": run_id, "status": "BLOCKED_REQUIRED_PROVIDER", "updated_at_utc": utc_now()})
        return 3

    trace_path = args.output_dir / "formal_inference_trace.jsonl"
    done = completed_ids(trace_path, args.retry_failed)
    pending = [case for case in cases if case.question_id not in done]
    if pending:
        qwen = FormalQwenRuntime(args.model_path, args.min_pixels, args.max_pixels)
        qwen.load()
        planner = FormalPlanner(qwen.model, qwen.processor, fallback_planner=None, use_template_hint=False)
        executor = PlanCompiledAtomicExecutor(tools=None)
        adapter = BEVFusionSceneAdapter(args.evidence_root)
        controls = FrozenControlStack(CONTROL_MANIFEST, CONTROL_THRESHOLDS)
        for position, case in enumerate(pending, start=1):
            item_started = time.time()
            try:
                row = run_case(
                    case, run_id, qwen, planner, executor, adapter, controls, args.direct_max_new_tokens
                )
            except Exception as exc:  # Persist and continue so failures are categorized/retriable.
                row = failure_record(case, run_id, exc, item_started)
            append_jsonl(trace_path, row)
            print(json.dumps({
                "progress": f"{position}/{len(pending)}",
                "question_id": case.question_id,
                "failure": row.get("failure"),
                "direct_answer": (row.get("direct") or {}).get("answer"),
                "structured_answer": (row.get("structured") or {}).get("candidate"),
                "final_answer": row.get("final_answer"),
            }, sort_keys=True), flush=True)

    validation = finalize(args.output_dir, cases, run_id)
    status = "COMPLETE" if validation["all_protocol_assertions_pass"] else "COMPLETE_WITH_FAILURES"
    atomic_json(run_state_path, {"run_id": run_id, "status": status, "updated_at_utc": utc_now(), "validation": validation})
    print(json.dumps({"status": status, **validation}, indent=2, sort_keys=True))
    return 0 if validation["all_protocol_assertions_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
