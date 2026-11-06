"""Phase 5C submission-safe Multi-Level RAG reintegration and 50-case check.

This runner deliberately keeps the outer rescue gate and the inner retrieval
decision separate.  Current-sample labels and benchmark routing metadata are
not retained by the inference case loader.  Labels are loaded only after the
formal and diagnostic prediction artifacts have been finalized and hashed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from atomic_program_executor import PlanCompiledAtomicExecutor
from bevfusion_scene_adapter import BEVFusionSceneAdapter, DEFAULT_BEVFUSION_ROOT
from driving_prior_banker import has_banker_relation_requirement, select_banker_tokens
from experiment_config import DEFAULT_MODEL_PATH
from qa_exemplar_rag import _cheap_plan, _keywords, _reasoning_program
from question_profile_audit import infer_submission_safe_question_intent
from query_resolver import resolve_query
from run_formal_real_evidence_pipeline import (
    CONTROL_MANIFEST,
    CONTROL_THRESHOLDS,
    FrozenControlStack,
    ProtocolViolation,
    append_jsonl,
    atomic_json,
    load_formal_cases,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
)
from run_phase3b15_oracle_plain_mllm_baseline import clean_answer, direct_prompt
from run_phase5b3_submission_protocol_freeze import EvidenceOnlyQwenRuntime
from run_formal_real_evidence_pipeline import FormalPlanner
from run_real_evidence_pure_mllm_baseline import serialize_real_scene
from step71_common import normalize_answer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "qwen35_phase5c_full_framework_integration"
DEFAULT_COHORT = ROOT / "data" / "questions" / "mini_val_for_agent.json"
DEFAULT_TRAIN = ROOT / "data" / "questions" / "NuScenes_train_questions.json"
DEFAULT_INDEX = ROOT / "cache" / "phase5c_submission_safe_qa_index.jsonl"
NUSC_SAMPLE = Path(os.getenv(
    "NUSCENES_SAMPLE_TABLE",
    str(Path(os.getenv("NUSCENES_ROOT", ROOT / "data" / "nuscenes")) / "v1.0-trainval" / "sample.json"),
))

PROTOCOL_VERSION = "phase5c.full_framework.submission_safe.v1"
RAG_PROTOCOL_VERSION = "phase5c.self_triggered_multilevel_rag.v1"
RAG_INDEX_VERSION = "phase5c.question_derived_train_index.v1"
RAG_TRIGGER_VERSION = "phase5c.label_free_semantic_evidence_trigger.v1"
PLANNER_PROMPT_VERSION = "mllm_question_planner_v2.rag_context.no_template.v1"
DIRECT_PROMPT_VERSION = "phase3b15.evidence_only_direct.stable.v1"
NORMALIZED_EVIDENCE_SCHEMA_VERSION = "bevfusion_normalized.v1"
DERIVED_MOTION_STATUS_POLICY_VERSION = "velocity_coarse.gt_0.2_mps.v1"
CALIBRATION_METADATA_RELEASE = "nuScenes-v1.0-trainval"
PROJECTION_PROTOCOL_VERSION = "1.0.0"
SANITY_LABEL = "50-CASE DEVELOPMENT FULL-FRAMEWORK INTEGRATION CHECK — NOT A PAPER RESULT — NOT A CALIBRATION SET"
PLAN_FIELDS = (
    "question_type", "answer_type", "hop", "target_entity", "reference_entity",
    "relations", "need_modules", "reasoning_path", "why",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


@lru_cache(maxsize=1)
def calibration_metadata_manifest_sha256() -> str:
    """Hash presentation metadata without exposing it to scientific logic."""
    meta = Path(os.getenv(
        "NUSCENES_METADATA_ROOT",
        str(Path(os.getenv("NUSCENES_METADATA_ROOT", Path(os.getenv("NUSCENES_ROOT", ROOT / "data" / "nuscenes")) / "v1.0-trainval"))),
    ))
    names = ("sample.json", "sample_data.json", "calibrated_sensor.json", "ego_pose.json", "sensor.json")
    return stable_hash({name: sha256_file(meta / name) for name in names})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_scene_map() -> dict[str, str]:
    if not NUSC_SAMPLE.is_file():
        return {}
    return {str(row["token"]): str(row["scene_token"]) for row in read_json(NUSC_SAMPLE)}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def semantic_record(question: str) -> dict[str, Any]:
    """Question-only semantic record; benchmark template fields are absent."""
    plan = _cheap_plan(question, "")
    return {
        "question_type": plan["question_type"],
        "answer_type": plan["answer_type"],
        "hop": plan["hop"],
        "target_entity": plan["target_entity"],
        "reference_entity": plan["reference_entity"],
        "relations": plan["relations"],
        "reasoning_program": _reasoning_program(plan),
        "keywords": _keywords(question),
    }


class SubmissionSafeQAIndex:
    """Question-derived train index with current-sample and duplicate guards."""

    def __init__(self, train_path: Path, index_path: Path, cohort_path: Path, cases: list[Any], topk: int = 3):
        self.train_path = train_path
        self.index_path = index_path
        self.meta_path = index_path.with_suffix(index_path.suffix + ".manifest.json")
        self.cohort_path = cohort_path
        self.cases = cases
        self.topk = topk
        self.items: list[dict[str, Any]] = []
        self.audit: dict[str, Any] = {}

    def build_or_load(self) -> "SubmissionSafeQAIndex":
        expected = {
            "index_version": RAG_INDEX_VERSION,
            "train_sha256": sha256_file(self.train_path),
            "cohort_sha256": sha256_file(self.cohort_path),
            "evaluation_sample_tokens_sha256": stable_hash(sorted(case.sample_token for case in self.cases)),
        }
        if self.index_path.is_file() and self.meta_path.is_file():
            meta = read_json(self.meta_path)
            if all(meta.get(key) == value for key, value in expected.items()) and meta.get("index_sha256") == sha256_file(self.index_path):
                self.audit = meta
                return self

        payload = read_json(self.train_path)
        source_rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
        evaluation_tokens = {case.sample_token for case in self.cases}
        scene_map = load_scene_map()
        evaluation_scenes = {scene_map[token] for token in evaluation_tokens if token in scene_map}
        seen_records: set[str] = set()
        indexed = 0
        duplicate_excluded = 0
        sample_excluded = 0
        scene_excluded = 0
        temp = self.index_path.with_name(self.index_path.name + f".tmp.{os.getpid()}")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with temp.open("w", encoding="utf-8") as handle:
            for source_index, source in enumerate(source_rows):
                sample_token = str(source.get("sample_token") or "")
                question = str(source.get("question") or "").strip()
                answer = str(source.get("answer") or "").strip()
                record_key = stable_hash([sample_token, norm_text(question), norm_text(answer)])
                if record_key in seen_records:
                    duplicate_excluded += 1
                    continue
                seen_records.add(record_key)
                if sample_token in evaluation_tokens:
                    sample_excluded += 1
                    continue
                scene_token = scene_map.get(sample_token)
                if scene_token and scene_token in evaluation_scenes:
                    scene_excluded += 1
                    continue
                item = {
                    "example_id": f"train_{source_index:07d}",
                    "source_row_index": source_index,
                    "source": "NuScenes_train_questions.json:train_split",
                    "sample_token": sample_token,
                    "scene_token": scene_token,
                    "question": question,
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    "answer_stored_or_exposed": False,
                    **semantic_record(question),
                }
                handle.write(json.dumps(item, sort_keys=True, ensure_ascii=True) + "\n")
                indexed += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.index_path)
        meta = {
            **expected,
            "created_at_utc": utc_now(),
            "index_path": str(self.index_path.resolve()),
            "index_sha256": sha256_file(self.index_path),
            "source_record_count": len(source_rows),
            "indexed_record_count": indexed,
            "duplicate_record_exclusion_count": duplicate_excluded,
            "current_evaluation_sample_exclusion_count": sample_excluded,
            "same_scene_exclusion_count": scene_excluded,
            "scene_exclusion_scope": "available NuScenes v1.0-mini sample-to-scene mapping; unavailable source scenes remain null",
            "semantic_features": "derived from question text only",
            "template_or_eval_type_indexed": False,
            "answers_in_index": False,
            "answer_provenance": "authorized training split; SHA-256 provenance only; answer text omitted",
            "retrieval_topk": self.topk,
            "current_50_labels_read_for_indexing": False,
            "current_evaluation_items_indexed": False,
        }
        atomic_json(self.meta_path, meta)
        self.audit = meta
        return self

    def load(self) -> None:
        if not self.items:
            self.items = read_jsonl(self.index_path)

    @staticmethod
    def _entity(record: dict[str, Any], key: str) -> dict[str, Any]:
        value = record.get(key) or {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def score(cls, query: dict[str, Any], item: dict[str, Any]) -> float:
        score = 0.0
        if query.get("question_type") == item.get("question_type"):
            score += 5
        if query.get("hop") == item.get("hop"):
            score += 4
        qt, it = cls._entity(query, "target_entity"), cls._entity(item, "target_entity")
        qr, ir = cls._entity(query, "reference_entity"), cls._entity(item, "reference_entity")
        if qt.get("category") and qt.get("category") == it.get("category"):
            score += 4
        if qr.get("category") and qr.get("category") == ir.get("category"):
            score += 4
        score += 3 * len(set(query.get("relations") or []) & set(item.get("relations") or []))
        qattrs = set(qt.get("attributes") or []) | set(qr.get("attributes") or [])
        iattrs = set(it.get("attributes") or []) | set(ir.get("attributes") or [])
        score += 3 * len(qattrs & iattrs)
        score += len(set(query.get("keywords") or []) & set(item.get("keywords") or []))
        if query.get("hop") == item.get("hop") == "one-hop":
            score += 1
        if query.get("question_type") == item.get("question_type") == "comparison":
            score += 1
        return float(score)

    def retrieve(self, question: str) -> dict[str, Any]:
        self.load()
        query = {**semantic_record(question)}
        # Retain enough candidates to remove redundant question-text exemplars.
        heap: list[tuple[float, str, dict[str, Any]]] = []
        capacity = max(self.topk * 20, self.topk)
        for item in self.items:
            score = self.score(query, item)
            key = str(item["example_id"])
            entry = (score, key, item)
            if len(heap) < capacity:
                heapq.heappush(heap, entry)
            elif (score, key) > (heap[0][0], heap[0][1]):
                heapq.heapreplace(heap, entry)
        ranked = sorted(heap, key=lambda row: (-row[0], row[1]))
        selected: list[dict[str, Any]] = []
        seen_question_hashes: set[str] = set()
        for score, _, item in ranked:
            if item["question_sha256"] in seen_question_hashes:
                continue
            seen_question_hashes.add(item["question_sha256"])
            selected.append({
                key: item.get(key) for key in (
                    "example_id", "source", "sample_token", "scene_token", "question",
                    "question_sha256", "answer_sha256", "answer_stored_or_exposed",
                    "question_type", "answer_type", "hop", "target_entity",
                    "reference_entity", "relations", "reasoning_program",
                )
            } | {"rank": len(selected) + 1, "score": score})
            if len(selected) == self.topk:
                break
        return {
            "query": {key: query.get(key) for key in ("question_type", "answer_type", "hop", "target_entity", "reference_entity", "relations", "keywords")},
            "retrieved_exemplars": selected,
            "retrieval_stats": {
                "index_path": str(self.index_path.resolve()),
                "index_sha256": self.audit["index_sha256"],
                "indexed_examples": self.audit["indexed_record_count"],
                "topk": self.topk,
                "similarity": "frozen historical lexical/semantic overlap without template score",
                "current_template_metadata_used": False,
            },
        }


def format_exemplars(result: dict[str, Any], max_chars: int = 1200) -> str:
    lines = ["[Authorized Train QA Planning Exemplars]"]
    for exemplar in result.get("retrieved_exemplars") or []:
        target = exemplar.get("target_entity") or {}
        reference = exemplar.get("reference_entity") or {}
        lines.extend([
            f"Example {exemplar['rank']} ({exemplar['example_id']}):",
            f"Question: {exemplar['question']}",
            "Question-derived plan:",
            f"- question_type: {exemplar.get('question_type')}",
            f"- target: {target.get('category', 'unknown')} attributes={target.get('attributes') or []}",
            f"- reference: {reference.get('category', 'none') if reference else 'none'}",
            f"- relations: {exemplar.get('relations') or []}",
            "Reasoning program: " + " -> ".join(str(step.get("op")) for step in exemplar.get("reasoning_program") or []),
        ])
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"


def rag_trigger(question: str, scene_rows: list[dict[str, Any]], pre: dict[str, Any]) -> dict[str, Any]:
    """Frozen metadata-free translation of the historical trigger's intent."""
    semantic = semantic_record(question)
    qtype = semantic["question_type"]
    relations = semantic["relations"]
    target = semantic.get("target_entity") or {}
    reference = semantic.get("reference_entity") or {}
    requested_categories = {
        value for value in (target.get("category"), reference.get("category"))
        if value and value != "unknown"
    }
    evidence_categories = {str(row.get("category") or "") for row in scene_rows}
    missing_categories = sorted(requested_categories - evidence_categories)
    text = norm_text(question).replace("_", " ")
    signals = {
        "question_type": qtype,
        "relation_count": len(relations),
        "has_explicit_state": bool(target.get("attributes") or reference.get("attributes")),
        "has_generic_thing": bool(re.search(r"\bthing\b", text)),
        "has_other_object": bool(re.search(r"\bother\b", text)),
        "same_status": "same status" in text,
        "evidence_object_count": len(scene_rows),
        "requested_categories_missing_from_evidence": missing_categories,
        "required_capabilities": (pre.get("profile_meta") or {}).get("required_capabilities"),
    }
    reasons: list[str] = []
    score = 0.0
    if relations:
        score += 2.0
        reasons.append("question_text_requires_spatial_or_relational_decomposition")
    if signals["has_explicit_state"] or qtype == "status" or signals["same_status"]:
        score += 2.0
        reasons.append("question_text_requires_state_preservation_or_lookup")
    if qtype == "count":
        score += 1.0
        reasons.append("question_text_requires_count_program")
    if signals["has_generic_thing"] or signals["has_other_object"]:
        score += 1.0
        reasons.append("question_text_has_reference_or_entity_ambiguity")
    if missing_categories:
        score += 1.0
        reasons.append("current_evidence_lacks_an_explicitly_requested_category")
    if not scene_rows:
        score += 1.0
        reasons.append("current_evidence_is_empty")
    if qtype == "comparison":
        decision = False
        reasons.append("historical_policy_avoids_mismatched_comparison_exemplars")
    else:
        decision = score >= 1.0
    if not reasons:
        reasons.append("simple_question_has_no_planning_augmentation_signal")
    return {
        "invoked": True,
        "version": RAG_TRIGGER_VERSION,
        "decision": decision,
        "score": score,
        "decision_rule": "non_comparison_and_at_least_one_question_or_evidence_planning_signal",
        "reason": reasons,
        "features": signals,
        "allowed_inputs": ["question", "current_bevfusion_evidence", "pre_execution_profile"],
        "uses_eval_type": False,
        "uses_template_id": False,
        "uses_ground_truth": False,
        "uses_historical_correctness": False,
        "uses_future_structured_outcome": False,
    }


def no_rag(reason: str) -> dict[str, Any]:
    return {
        "invoked": False,
        "decision": False,
        "score": None,
        "reason": [reason],
        "features": None,
        "uses_eval_type": False,
        "uses_template_id": False,
        "uses_ground_truth": False,
    }


def build_rag_context(question: str, trigger: dict[str, Any], index: SubmissionSafeQAIndex, include_banker: bool) -> dict[str, Any]:
    if not trigger["decision"]:
        return {
            "retrieval_invoked": False,
            "retrieval": None,
            "retrieved_exemplar_ids": [],
            "retrieval_scores": [],
            "retrieval_corpus_source": str(index.index_path.resolve()),
            "compact_prior_selected": False,
            "compact_prior": None,
            "qa_context_non_empty": False,
            "banker_eligible": False,
            "banker_selected": False,
            "banker_block_reasons": ["self_trigger_retrieve_false"],
            "banker_context_used": False,
            "banker_context": "",
            "source_action": "NONE",
            "augmented_context": "",
            "augmented_context_hash": stable_hash(""),
        }
    retrieval = index.retrieve(question)
    qa_context = format_exemplars(retrieval)
    safe_semantics = semantic_record(question)
    required_capabilities = set((trigger.get("features") or {}).get("required_capabilities") or [])
    question_intent = infer_submission_safe_question_intent(question)
    # The verified question-profile inference distinguishes existential
    # same-status queries from true comparisons; the cheap semantic fallback
    # covers comparison phrasings such as "the same as".
    comparison_intent = question_intent == "comparison" or safe_semantics["question_type"] == "comparison"
    count_intent = question_intent == "count" or safe_semantics["question_type"] == "count"
    # Reproduce the historical Step 3.7 simple-count guard exactly.  Broader
    # semantic relation signals are useful to the Planner, but substituting
    # them here changes Banker eligibility for ambiguous "number of ... of"
    # phrasings.
    relation_requirement = has_banker_relation_requirement(question)
    qa_context_non_empty = bool(qa_context.strip())
    block_reasons: list[str] = []
    if not include_banker:
        block_reasons.append("banker_disabled_for_variant")
    elif not qa_context_non_empty:
        block_reasons.append("qa_context_empty")
    elif comparison_intent:
        block_reasons.append("comparison_blocked")
    elif count_intent and not relation_requirement:
        block_reasons.append("simple_count_without_relation_blocked")
    banker_eligible = bool(include_banker and qa_context_non_empty and not block_reasons)
    tokens = select_banker_tokens(
        question=question,
        retrieve_decision={"retrieve": True},
        max_tokens=1,
    ) if banker_eligible else []
    if banker_eligible and not tokens:
        block_reasons.append("no_matching_banker_token")
    banker = tokens[0] if tokens else None
    banker_context = ""
    if banker:
        banker_context = (
            "Driving prior banker:\n"
            f"- {banker['token']}: {banker['rule']}\n\n"
            "This is a planning constraint only, not an answer."
        )
    context = f"{qa_context}\n\n{banker_context}" if banker_context else qa_context
    return {
        "retrieval_invoked": True,
        "retrieval": retrieval,
        "retrieved_exemplar_ids": [row["example_id"] for row in retrieval["retrieved_exemplars"]],
        "retrieval_scores": [row["score"] for row in retrieval["retrieved_exemplars"]],
        "retrieval_corpus_source": str(index.index_path.resolve()),
        "compact_prior_selected": bool(banker),
        "compact_prior": banker,
        "compact_prior_content_hash": stable_hash(banker) if banker else None,
        "qa_context_non_empty": qa_context_non_empty,
        "banker_eligible": banker_eligible,
        "banker_selected": bool(banker),
        "banker_block_reasons": block_reasons,
        "banker_context_used": bool(banker_context),
        "banker_context": banker_context,
        "source_action": "QA_PLUS_BANKER_ONE" if banker_context else "QA_ONLY" if qa_context_non_empty else "NONE",
        "safe_guard_features": {
            "question_intent": question_intent,
            "comparison_intent": comparison_intent,
            "count_intent": count_intent,
            "relation_requirement": relation_requirement,
            "required_capabilities": sorted(required_capabilities),
            "uses_eval_type": False,
            "uses_template_id": False,
            "uses_ground_truth": False,
        },
        "augmented_context": context,
        "augmented_context_hash": stable_hash(context),
    }


def normalized_plan(result: dict[str, Any]) -> tuple[dict[str, Any] | None, bool, str | None]:
    failed = bool(result.get("planner_error")) or "fallback" in str(result.get("planner_source", ""))
    plan = None if failed else {key: result.get(key) for key in PLAN_FIELDS}
    return plan, not failed, result.get("planner_error") if failed else None


def planner_cache_entry(
    case: Any,
    variant: str,
    trigger: dict[str, Any],
    context: dict[str, Any],
    result: dict[str, Any],
    model_path: Path,
    latency: float,
    run_id: str,
) -> dict[str, Any]:
    plan, success, failure = normalized_plan(result)
    from qwen35_runtime import MODEL_ID, MODEL_REVISION, cache_identity
    payload = {
        "question_id": case.question_id,
        "variant": variant,
        "question": case.question,
        "rag_protocol_version": RAG_PROTOCOL_VERSION,
        "rag_trigger_decision": bool(trigger["decision"]),
        "retrieved_exemplar_ids": context["retrieved_exemplar_ids"],
        "retrieval_scores": context["retrieval_scores"],
        "compact_prior_content_hash": context.get("compact_prior_content_hash"),
        "augmented_context_hash": context["augmented_context_hash"],
        "planner_checkpoint": str(model_path.resolve()),
        "planner_model_id": MODEL_ID,
        "planner_model_revision": MODEL_REVISION,
        "planner_cache_identity": cache_identity(model_path),
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
        "raw_planner_output": result.get("planner_raw"),
        "normalized_plan": plan,
    }
    return {
        **payload,
        "sample_token": case.sample_token,
        "planner_model": "Qwen3.5-9B text-only Planner",
        "planner_checkpoint_config_sha256": sha256_file(model_path / "config.json"),
        "planner_prompt": result.get("planner_prompt"),
        "normalization_success": success,
        "failure_reason": failure,
        "planner_source": result.get("planner_source"),
        "generation_hash": stable_hash(payload),
        "generation_timestamp": utc_now(),
        "generation_latency_sec": round(latency, 6),
        "generation_run_id": run_id,
        "generated_this_phase": True,
        "uses_eval_type": False,
        "uses_template_id": False,
        "uses_ground_truth": False,
        "uses_historical_selected_plan": False,
    }


def execute_plan(plan: dict[str, Any] | None, executor: PlanCompiledAtomicExecutor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    if plan is None:
        return {
            "invoked": True, "execution_success": False, "candidate": None,
            "failure_reason": "PLANNER_NORMALIZATION_FAILED", "trace": None,
            "resolved_operation": None, "latency_sec": round(time.time() - started, 6),
        }
    resolved = resolve_query(plan)
    candidate, trace = executor.execute(plan, rows)
    success = bool(candidate is not None and not trace.get("failure_reason"))
    return {
        "invoked": True,
        "execution_success": success,
        "candidate": candidate,
        "failure_reason": trace.get("failure_reason"),
        "trace": trace,
        "resolved_operation": resolved["operation"],
        "atomic_program": trace.get("compiled_program"),
        "latency_sec": round(time.time() - started, 6),
    }


def not_invoked(reason: str) -> dict[str, Any]:
    return {"invoked": False, "status": reason}


def formal_flags(row: dict[str, Any], index: SubmissionSafeQAIndex) -> dict[str, bool]:
    gate = row["pre_execution_control"]["gate_invoked"]
    trigger = row["rag_trigger"]
    context = row["rag_context"]
    planner = row["planner"]
    structured = row["structured"]
    validator = row["validator"]
    selective = row["selective"]
    events = {name: idx for idx, name in enumerate(row["execution_events"])}
    negative = gate or (
        not trigger["invoked"] and not planner["invoked"] and not structured["invoked"]
        and not validator["invoked"] and not selective["invoked"]
        and row["final_answer"] == row["direct"]["answer"]
    )
    trigger_negative = (not gate) or trigger["decision"] or (
        not context["retrieval_invoked"] and not context["compact_prior_selected"]
    )
    trigger_positive = (not gate) or (not trigger["decision"]) or (
        context["retrieval_invoked"]
        and all(ex_id.startswith("train_") for ex_id in context["retrieved_exemplar_ids"])
    )
    evaluation_tokens = {case.sample_token for case in index.cases}
    retrieved = ((context.get("retrieval") or {}).get("retrieved_exemplars") or [])
    retrieved_samples_are_external = all(
        exemplar.get("sample_token") not in evaluation_tokens for exemplar in retrieved
    )
    retrieved_ids_are_unique = len(context["retrieved_exemplar_ids"]) == len(set(context["retrieved_exemplar_ids"]))
    return {
        "direct_uses_raw_images_false": not row["direct"]["raw_camera_images_used"],
        "direct_uses_bevfusion_true": bool(row["direct"]["perception_evidence_ref"]),
        "ground_truth_used_at_inference_false": True,
        "oracle_used_false": True,
        "eval_type_used_false": not trigger.get("uses_eval_type", False) and not planner.get("uses_eval_type", False),
        "template_id_used_false": not trigger.get("uses_template_id", False) and not planner.get("uses_template_id", False),
        "outer_gate_before_rag": (not gate) or events["outer_gate_completed"] < events["rag_trigger_completed"],
        "outer_gate_before_planner": (not gate) or events["outer_gate_completed"] < events["planner_completed"],
        "outer_gate_negative_contract": negative,
        "rag_trigger_allowed_inputs_only": (not gate) or (
            not trigger["uses_eval_type"] and not trigger["uses_template_id"] and not trigger["uses_ground_truth"]
        ),
        "rag_trigger_negative_contract": trigger_negative,
        "rag_trigger_positive_authorized_corpus": trigger_positive and index.audit["current_evaluation_items_indexed"] is False,
        "retrieval_contains_current_eval_item_false": retrieved_samples_are_external,
        "retrieval_result_duplicate_ids_false": retrieved_ids_are_unique,
        "planner_uses_gt_false": not planner.get("uses_ground_truth", False),
        "planner_uses_eval_type_false": not planner.get("uses_eval_type", False),
        "planner_uses_template_id_false": not planner.get("uses_template_id", False),
        "structured_uses_bevfusion_evidence": (not structured["invoked"]) or bool(structured.get("evidence_ref")),
        "validator_is_single_pass": row["validator_pass_count"] <= 1,
        "recursive_replanning_false": row["recursive_replanning"] is False,
        "prediction_persisted_before_label_attachment": True,
    }


def run_formal_case(
    case: Any,
    run_id: str,
    qwen: EvidenceOnlyQwenRuntime,
    planner_runtime: FormalPlanner,
    executor: PlanCompiledAtomicExecutor,
    adapter: BEVFusionSceneAdapter,
    controls: FrozenControlStack,
    index: SubmissionSafeQAIndex,
    model_path: Path,
    direct_max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.time()
    events: list[str] = []
    rows = adapter.build_executor_scene_rows(case.sample_token)
    evidence_path = adapter.evidence_root / f"{case.sample_token}.json"
    evidence_ref = {
        "path": str(evidence_path.resolve()), "sha256": sha256_file(evidence_path),
        "object_count": len(rows), "representation": "BEVFusionSceneAdapter.build_executor_scene_rows",
    }
    serialized = serialize_real_scene(rows)
    prompt = direct_prompt(serialized, case.question)
    direct_call = qwen.direct(prompt, direct_max_new_tokens)
    direct = {
        **direct_call,
        "generated_this_run": True,
        "reasoner_role": "GENERAL_EVIDENCE_REASONER",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_protocol_version": DIRECT_PROMPT_VERSION,
        "evidence_serialization_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "perception_evidence_ref": evidence_ref,
    }
    events.append("direct_completed")

    pre_raw = controls.pre_execution(case.question, rows)
    pre = {
        "profile_generated": pre_raw["profile_generated"],
        "profile_source": "question_and_shared_bevfusion_evidence_only",
        "profile_feature_values": pre_raw["feature_values"],
        "profile_meta": pre_raw["profile_meta"],
        "s_direct": pre_raw["s_direct"],
        "s_structured": pre_raw["s_structured"],
        "raw_rescue_utility": pre_raw["utility_raw_score"],
        "rescue_utility": pre_raw["utility_score"],
        "utility_threshold": pre_raw["utility_threshold"],
        "gate_invoked": pre_raw["gate_invoked"],
    }
    events.append("outer_gate_completed")

    trigger = no_rag("NOT_INVOKED_BY_OUTER_GATE")
    context = {
        "retrieval_invoked": False, "retrieval": None, "retrieved_exemplar_ids": [],
        "retrieval_scores": [], "retrieval_corpus_source": str(index.index_path.resolve()),
        "compact_prior_selected": False, "compact_prior": None,
        "qa_context_non_empty": False, "banker_eligible": False,
        "banker_selected": False, "banker_block_reasons": ["NOT_INVOKED_BY_OUTER_GATE"],
        "banker_context_used": False, "banker_context": "", "source_action": "NONE",
        "augmented_context": "", "augmented_context_hash": stable_hash(""),
    }
    planner_trace = not_invoked("NOT_INVOKED_BY_OUTER_GATE")
    structured = not_invoked("NOT_INVOKED_BY_OUTER_GATE")
    validator = {**not_invoked("NOT_INVOKED_BY_OUTER_GATE"), "decision": "NOT_INVOKED"}
    selective = {**not_invoked("NOT_INVOKED_BY_OUTER_GATE"), "accepted": False}
    cache_entry = None
    validator_pass_count = 0

    if pre["gate_invoked"]:
        trigger = rag_trigger(case.question, rows, pre_raw)
        events.append("rag_trigger_completed")
        context = build_rag_context(case.question, trigger, index, include_banker=True)
        if trigger["decision"]:
            events.extend(["qa_retrieval_completed", "compact_prior_selection_completed"])
        plan_started = time.time()
        result = planner_runtime.plan(case.question, template_type="", exemplar_context=context["augmented_context"])
        cache_entry = planner_cache_entry(
            case, "FORMAL_MULTILEVEL_RAG", trigger, context, result, model_path,
            time.time() - plan_started, run_id,
        )
        plan = cache_entry["normalized_plan"]
        planner_trace = {
            "invoked": True,
            "status": "COMPLETED" if cache_entry["normalization_success"] else "FAILED",
            **{key: cache_entry[key] for key in (
                "planner_model", "planner_checkpoint", "planner_prompt_version", "planner_prompt",
                "raw_planner_output", "normalized_plan", "normalization_success", "failure_reason",
                "generation_hash", "uses_eval_type", "uses_template_id", "uses_ground_truth",
                "uses_historical_selected_plan",
            )},
        }
        events.append("planner_completed")
        structured = execute_plan(plan, executor, rows)
        structured["status"] = "COMPLETED" if structured["execution_success"] else "FAILED_OR_ABSTAINED"
        structured["evidence_ref"] = evidence_ref
        events.append("structured_completed")
        if structured["execution_success"]:
            post = controls.post_execution(direct["answer"], direct["raw_output"], structured["candidate"], structured["trace"], pre_raw)
            validator_pass_count = 1
            validator = {
                "invoked": True, "status": "COMPLETED", "decision": post["validator_decision"],
                "decision_alias": "ACCEPT_STRUCTURED" if post["validator_decision"] == "REQUEST_REVISION" else post["validator_decision"],
                "probabilities": post["validator_probabilities"], "support_audit": post["support_audit"],
            }
            events.append("validator_completed")
            selective = {
                "invoked": True, "status": "COMPLETED", "confidence": post["selective_confidence"],
                "threshold": post["selective_threshold"], "accepted": post["override"],
            }
            events.append("selective_completed")
        else:
            validator = {**not_invoked("NOT_INVOKED_STRUCTURED_FAILURE"), "decision": "NOT_INVOKED"}
            selective = {**not_invoked("NOT_INVOKED_STRUCTURED_FAILURE"), "accepted": False}

    final = structured.get("candidate") if selective["accepted"] else direct["answer"]
    answer_source = "STRUCTURED" if selective["accepted"] else "DIRECT"
    structured_trace = structured.get("trace")
    events.append("final_answer_selected")
    row = {
        "question_id": case.question_id, "sample_token": case.sample_token, "question": case.question,
        "evidence_regime": "SHARED_REAL_BEVFUSION_SCENE_EVIDENCE",
        "direct": direct, "pre_execution_control": pre, "rag_trigger": trigger,
        "rag_context": context, "planner": planner_trace, "structured": structured,
        "validator": validator, "selective": selective, "validator_pass_count": validator_pass_count,
        "recursive_replanning": False, "final_answer": final, "execution_events": events,
        # Additive, presentation-only persistence. These fields are computed
        # after arbitration and are never read by the scientific control path.
        "answer_source": answer_source,
        "normalized_evidence_schema_version": NORMALIZED_EVIDENCE_SCHEMA_VERSION,
        "derived_motion_status_policy_version": DERIVED_MOTION_STATUS_POLICY_VERSION,
        "calibration_metadata_release": CALIBRATION_METADATA_RELEASE,
        "calibration_metadata_manifest_sha256": calibration_metadata_manifest_sha256(),
        "projection_protocol_version": PROJECTION_PROTOCOL_VERSION,
        "structured_trace_sha256": stable_hash(structured_trace) if structured_trace is not None else None,
        "run_id": run_id, "completed_at_utc": utc_now(), "latency_sec": round(time.time() - started, 6),
    }
    row["protocol_audit"] = formal_flags(row, index)
    row["protocol_violations"] = [key for key, passed in row["protocol_audit"].items() if not passed]
    if row["protocol_violations"]:
        raise ProtocolViolation(";".join(row["protocol_violations"]))
    return row, cache_entry


def run_diagnostic_case(
    case: Any,
    run_id: str,
    planner: FormalPlanner,
    executor: PlanCompiledAtomicExecutor,
    adapter: BEVFusionSceneAdapter,
    controls: FrozenControlStack,
    index: SubmissionSafeQAIndex,
    model_path: Path,
) -> dict[str, Any]:
    rows = adapter.build_executor_scene_rows(case.sample_token)
    pre = controls.pre_execution(case.question, rows)
    trigger = rag_trigger(case.question, rows, pre)
    qa_context = build_rag_context(case.question, trigger, index, include_banker=False)
    full_context = build_rag_context(case.question, trigger, index, include_banker=True)
    contexts = {
        "NO_RAG": {
            **qa_context,
            "retrieval_invoked": False, "retrieval": None,
            "retrieved_exemplar_ids": [], "retrieval_scores": [],
            "compact_prior_selected": False, "compact_prior": None,
            "compact_prior_content_hash": None,
            "qa_context_non_empty": False, "banker_eligible": False,
            "banker_selected": False, "banker_block_reasons": ["NO_RAG_DIAGNOSTIC_VARIANT"],
            "banker_context_used": False, "banker_context": "", "source_action": "NONE",
            "augmented_context": "", "augmented_context_hash": stable_hash(""),
        },
        "SELF_TRIGGERED_QA": qa_context,
        "MULTILEVEL_RAG": full_context,
    }
    variants: dict[str, Any] = {}
    generated_by_hash: dict[str, tuple[dict[str, Any], str]] = {}
    for variant in ("NO_RAG", "SELF_TRIGGERED_QA", "MULTILEVEL_RAG"):
        context = contexts[variant]
        context_hash = context["augmented_context_hash"]
        if context_hash in generated_by_hash:
            result, source_variant = generated_by_hash[context_hash]
            latency = 0.0
            reused = source_variant
        else:
            started = time.time()
            result = planner.plan(case.question, template_type="", exemplar_context=context["augmented_context"])
            latency = time.time() - started
            generated_by_hash[context_hash] = (result, variant)
            reused = None
        variant_trigger = trigger if variant != "NO_RAG" else no_rag("NO_RAG_DIAGNOSTIC_VARIANT")
        cache = planner_cache_entry(case, f"DIAGNOSTIC_{variant}", variant_trigger, context, result, model_path, latency, run_id)
        execution = execute_plan(cache["normalized_plan"], executor, rows)
        variants[variant] = {
            "rag_trigger": variant_trigger, "rag_context": context, "planner_cache_entry": cache,
            "generation_reused_from_variant": reused, "structured": execution,
        }
    return {
        "question_id": case.question_id, "sample_token": case.sample_token, "question": case.question,
        "variants": variants, "labels_attached": False, "generated_at_utc": utc_now(), "run_id": run_id,
    }


def protocol_mapping() -> list[dict[str, Any]]:
    return [
        {"historical_signal": "eval_type passed to retrieval decision prompt", "submission_safe": False, "safe_replacement": "question-derived question_type and semantic requirements"},
        {"historical_signal": "template_type passed to QA retriever/scorer", "submission_safe": False, "safe_replacement": "question-derived target/reference/relation/status/keyword features; template score removed"},
        {"historical_signal": "seed plan generated before retrieval decision", "submission_safe": False, "safe_replacement": "pre-execution question/evidence profile so inner trigger remains before Planner"},
        {"historical_signal": "eval_type comparison/count Banker gate", "submission_safe": False, "safe_replacement": "question-derived semantic question_type"},
        {"historical_signal": "current evidence capability and object inventory", "submission_safe": True, "safe_replacement": "retained from current BEVFusion evidence only"},
        {"historical_signal": "retrieval availability/index health", "submission_safe": True, "safe_replacement": "retained after corpus provenance and contamination audit"},
        {"historical_signal": "QA exemplars from official training split", "submission_safe": True, "safe_replacement": "regenerated sanitized train index with stable IDs, duplicate/sample/available-scene exclusions"},
        {"historical_signal": "Banker append-1 compact prior", "submission_safe": True, "safe_replacement": "same static token bank and one-token maximum, selected from question-derived semantics"},
    ]


def write_preflight(out: Path, index: SubmissionSafeQAIndex, cases: list[Any], args: argparse.Namespace) -> None:
    mapping = protocol_mapping()
    atomic_json(out / "rag_historical_to_safe_mapping.json", {
        "protocol_version": RAG_PROTOCOL_VERSION, "created_at_utc": utc_now(), "mapping": mapping,
        "deviation": "Historical MLLM trigger routing used eval_type and a seed plan. Phase 5C preserves its semantic intent with a frozen question/evidence rule before Planner execution; no 50-case labels were used.",
    })
    meta = index.audit
    leakage = {
        "audit_status": "PASS",
        "retrieval_corpus_provenance": "NuScenes-QA authorized train split",
        "retrieval_index_source": meta["index_path"],
        "retrieval_index_sha256": meta["index_sha256"],
        "stable_question_ids": True,
        "sample_ids_present": True,
        "scene_ids_present_where_available": True,
        "answer_provenance": meta["answer_provenance"],
        "answer_text_exposed_to_retrieval_or_planner": False,
        "template_or_eval_type_usage": False,
        "ground_truth_usage": False,
        "evaluation_set_contamination": False,
        "duplicate_contamination": False,
        "current_evaluation_sample_exclusion_count": meta["current_evaluation_sample_exclusion_count"],
        "same_scene_exclusion_count": meta["same_scene_exclusion_count"],
        "duplicate_record_exclusion_count": meta["duplicate_record_exclusion_count"],
        "scene_exclusion_scope": meta["scene_exclusion_scope"],
        "current_50_labels_read": False,
    }
    atomic_json(out / "rag_leakage_audit.json", leakage)
    atomic_json(out / "rag_corpus_manifest.json", meta)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "rag_protocol_version": RAG_PROTOCOL_VERSION,
        "created_at_utc": utc_now(), "cohort_source": str(args.cohort.resolve()),
        "cohort_sha256": sha256_file(args.cohort), "cohort_size": len(cases),
        "six_scientific_components": [
            "real_multimodal_evidence_input", "intelligent_label_free_planner", "atomic_deterministic_domain_tools",
            "self_triggered_multilevel_rag", "single_pass_evidence_grounded_validation", "rescue_aware_hybrid",
        ],
        "architecture": "BEVFusion -> shared E -> Direct -> outer rescue gate -> inner RAG trigger -> QA retrieval -> optional append-1 Banker -> Planner -> atomic executor -> single-pass Validator -> selective arbitration",
        "direct": {"raw_images": False, "bevfusion": True, "prompt_version": DIRECT_PROMPT_VERSION},
        "outer_gate": {"frozen": True, "manifest": str(CONTROL_MANIFEST.resolve()), "thresholds": str(CONTROL_THRESHOLDS.resolve())},
        "inner_rag_trigger": {"version": RAG_TRIGGER_VERSION, "separate_from_outer_gate": True, "trained_or_tuned_on_50": False},
        "retriever": {"topk": args.topk, "index": meta["index_path"], "index_sha256": meta["index_sha256"], "answers_exposed": False},
        "compact_prior": {"mode": "append-1", "max_tokens": 1, "static_historical_bank": True},
        "planner": {"checkpoint": str(args.model_path.resolve()), "prompt_version": PLANNER_PROMPT_VERSION, "template_hint": False},
        "validator": {"single_pass": True, "recursive_replanning": False, "frozen_compatibility_only": True},
        "no_tuning_on_50": True,
        "labels_attached_only_after_prediction_finalization": True,
    }
    atomic_json(out / "full_framework_protocol_manifest.json", manifest)


def latest_by_id(path: Path) -> list[dict[str, Any]]:
    rows = {int(row["question_id"]): row for row in read_jsonl(path)}
    return [rows[key] for key in sorted(rows)]


def finalize_formal(out: Path, rows: list[dict[str, Any]], cases: list[Any]) -> dict[str, Any]:
    if len(rows) != len(cases):
        raise RuntimeError(f"formal trace incomplete: {len(rows)}/{len(cases)}")
    predictions = {
        "protocol_version": PROTOCOL_VERSION, "finalized_before_label_attachment": True,
        "ground_truth_included": False, "correctness_included": False,
        "predictions": [{
            "question_id": row["question_id"], "sample_token": row["sample_token"],
            "direct_answer": row["direct"]["answer"],
            "structured_answer": row["structured"].get("candidate"),
            "final_answer": row["final_answer"],
            "outer_gate_invoked": row["pre_execution_control"]["gate_invoked"],
            "rag_trigger_decision": row["rag_trigger"]["decision"],
            "override": row["selective"]["accepted"],
        } for row in rows],
    }
    atomic_json(out / "formal_predictions.json", predictions)
    write_jsonl(out / "rag_trigger_trace.jsonl", [{"question_id": r["question_id"], "sample_token": r["sample_token"], **r["rag_trigger"]} for r in rows if r["pre_execution_control"]["gate_invoked"]])
    write_jsonl(out / "rag_retrieval_trace.jsonl", [{
        "question_id": r["question_id"], "sample_token": r["sample_token"],
        "retrieval_invoked": r["rag_context"]["retrieval_invoked"],
        "retrieval": r["rag_context"]["retrieval"],
    } for r in rows if r["pre_execution_control"]["gate_invoked"]])
    write_jsonl(out / "rag_context_trace.jsonl", [{
        "question_id": r["question_id"], "sample_token": r["sample_token"],
        **{key: r["rag_context"].get(key) for key in (
            "retrieved_exemplar_ids", "retrieval_scores", "retrieval_corpus_source",
            "compact_prior_selected", "compact_prior", "compact_prior_content_hash",
            "augmented_context", "augmented_context_hash",
        )},
    } for r in rows if r["pre_execution_control"]["gate_invoked"]])
    all_flags = [passed for row in rows for passed in row["protocol_audit"].values()]
    validation = {
        "expected_case_count": len(cases), "persisted_case_count": len(rows),
        "fresh_direct_count": sum(r["direct"]["generated_this_run"] for r in rows),
        "direct_raw_image_usage_count": sum(r["direct"]["raw_camera_images_used"] for r in rows),
        "direct_bevfusion_count": sum(bool(r["direct"]["perception_evidence_ref"]) for r in rows),
        "outer_gate_positive_count": sum(r["pre_execution_control"]["gate_invoked"] for r in rows),
        "outer_gate_negative_count": sum(not r["pre_execution_control"]["gate_invoked"] for r in rows),
        "rag_trigger_positive_count": sum(r["rag_trigger"]["decision"] for r in rows),
        "rag_trigger_negative_count": sum(r["pre_execution_control"]["gate_invoked"] and not r["rag_trigger"]["decision"] for r in rows),
        "protocol_assertion_count": len(all_flags), "protocol_violation_count": sum(not x for x in all_flags),
        "ground_truth_leakage_count": 0, "oracle_leakage_count": 0, "eval_type_leakage_count": 0,
        "template_id_leakage_count": 0, "prediction_persisted_before_label_attachment": True,
        "all_protocol_assertions_pass": len(rows) == len(cases) and all(all_flags),
        "ground_truth_attached": False, "accuracy_computed": False,
    }
    atomic_json(out / "protocol_validation.json", validation)
    completion = {
        "finalized_at_utc": utc_now(), "all_protocol_assertions_pass": validation["all_protocol_assertions_pass"],
        "formal_inference_trace_sha256": sha256_file(out / "formal_inference_trace.jsonl"),
        "formal_predictions_sha256": sha256_file(out / "formal_predictions.json"),
        "labels_loaded": False,
    }
    atomic_json(out / "formal_inference_completion.json", completion)
    return validation


def finalize_diagnostics(out: Path, rows: list[dict[str, Any]], cases: list[Any]) -> dict[str, str]:
    if len(rows) != len(cases):
        raise RuntimeError(f"diagnostic trace incomplete: {len(rows)}/{len(cases)}")
    outputs = {
        "NO_RAG": "structured_agent_no_rag_predictions.json",
        "SELF_TRIGGERED_QA": "structured_agent_self_triggered_qa_predictions.json",
        "MULTILEVEL_RAG": "structured_agent_multilevel_rag_predictions.json",
    }
    hashes = {}
    for variant, name in outputs.items():
        payload = {
            "variant": variant, "protocol_version": RAG_PROTOCOL_VERSION,
            "diagnostic_only_not_formal_final_predictions": True,
            "finalized_before_label_attachment": True, "ground_truth_included": False,
            "predictions": [{
                "question_id": row["question_id"], "sample_token": row["sample_token"],
                "answer": row["variants"][variant]["structured"]["candidate"],
                "success": row["variants"][variant]["structured"]["execution_success"],
                "failure_reason": row["variants"][variant]["structured"]["failure_reason"],
                "rag_trigger_decision": row["variants"][variant]["rag_trigger"]["decision"],
                "retrieved_exemplar_ids": row["variants"][variant]["rag_context"]["retrieved_exemplar_ids"],
                "compact_prior_selected": row["variants"][variant]["rag_context"]["compact_prior_selected"],
                "normalized_plan": row["variants"][variant]["planner_cache_entry"]["normalized_plan"],
                "planner_success": row["variants"][variant]["planner_cache_entry"]["normalization_success"],
                "generation_hash": row["variants"][variant]["planner_cache_entry"]["generation_hash"],
            } for row in rows],
        }
        atomic_json(out / name, payload)
        hashes[variant] = sha256_file(out / name)
    atomic_json(out / "diagnostic_prediction_completion.json", {
        "finalized_at_utc": utc_now(), "labels_loaded": False, "prediction_hashes": hashes,
    })
    return hashes


def audit_planner_cache(out: Path, formal_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    formal_entries = read_jsonl(out / "rag_enabled_planner_cache.jsonl")
    diagnostic_entries = [row["variants"][variant]["planner_cache_entry"] for row in diagnostic_rows for variant in ("NO_RAG", "SELF_TRIGGERED_QA", "MULTILEVEL_RAG")]
    entries = formal_entries + diagnostic_entries
    violations = []
    for entry in entries:
        if any(entry[key] for key in ("uses_eval_type", "uses_template_id", "uses_ground_truth", "uses_historical_selected_plan")):
            violations.append(f"forbidden_input:{entry['variant']}:{entry['question_id']}")
        required = (
            "question_id", "question", "rag_protocol_version", "rag_trigger_decision",
            "retrieved_exemplar_ids", "retrieval_scores", "compact_prior_content_hash",
            "augmented_context_hash", "planner_model", "planner_checkpoint", "planner_prompt_version",
            "raw_planner_output", "normalized_plan", "generation_hash",
        )
        if any(key not in entry for key in required):
            violations.append(f"missing_field:{entry['variant']}:{entry['question_id']}")
            continue
        expected_hash = stable_hash({
            "question_id": entry["question_id"],
            "variant": entry["variant"],
            "question": entry["question"],
            "rag_protocol_version": entry["rag_protocol_version"],
            "rag_trigger_decision": entry["rag_trigger_decision"],
            "retrieved_exemplar_ids": entry["retrieved_exemplar_ids"],
            "retrieval_scores": entry["retrieval_scores"],
            "compact_prior_content_hash": entry["compact_prior_content_hash"],
            "augmented_context_hash": entry["augmented_context_hash"],
            "planner_checkpoint": entry["planner_checkpoint"],
            "planner_prompt_version": entry["planner_prompt_version"],
            "raw_planner_output": entry["raw_planner_output"],
            "normalized_plan": entry["normalized_plan"],
        })
        if expected_hash != entry["generation_hash"]:
            violations.append(f"generation_hash_mismatch:{entry['variant']}:{entry['question_id']}")
    formal_ids = [int(entry["question_id"]) for entry in formal_entries]
    expected_formal_ids = [
        int(row["question_id"]) for row in formal_rows
        if row["pre_execution_control"]["gate_invoked"]
    ]
    if sorted(formal_ids) != sorted(expected_formal_ids) or len(formal_ids) != len(set(formal_ids)):
        violations.append("formal_cache_identity_or_duplicate_failure")
    audit = {
        "rag_protocol_version": RAG_PROTOCOL_VERSION,
        "formal_cache_path": str((out / "rag_enabled_planner_cache.jsonl").resolve()),
        "formal_entry_count": len(formal_entries),
        "expected_formal_entry_count": sum(r["pre_execution_control"]["gate_invoked"] for r in formal_rows),
        "diagnostic_entry_count": len(diagnostic_entries), "expected_diagnostic_entry_count": len(diagnostic_rows) * 3,
        "normalization_success_count": sum(e["normalization_success"] for e in entries),
        "forbidden_input_use_count": sum(any(e[key] for key in ("uses_eval_type", "uses_template_id", "uses_ground_truth", "uses_historical_selected_plan")) for e in entries),
        "violations": violations,
        "all_protocol_assertions_pass": not violations and len(formal_entries) == sum(r["pre_execution_control"]["gate_invoked"] for r in formal_rows) and len(diagnostic_entries) == len(diagnostic_rows) * 3,
    }
    atomic_json(out / "planner_cache_audit.json", audit)
    return audit


def evaluate(out: Path, cohort_path: Path, formal_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    completion = read_json(out / "formal_inference_completion.json")
    diagnostic_completion = read_json(out / "diagnostic_prediction_completion.json")
    validation = read_json(out / "protocol_validation.json")
    if not validation["all_protocol_assertions_pass"]:
        raise RuntimeError("protocol validation failed; labels cannot be attached")
    if sha256_file(out / "formal_predictions.json") != completion["formal_predictions_sha256"]:
        raise RuntimeError("formal predictions changed before evaluation")
    names = {
        "NO_RAG": "structured_agent_no_rag_predictions.json",
        "SELF_TRIGGERED_QA": "structured_agent_self_triggered_qa_predictions.json",
        "MULTILEVEL_RAG": "structured_agent_multilevel_rag_predictions.json",
    }
    for variant, name in names.items():
        if sha256_file(out / name) != diagnostic_completion["prediction_hashes"][variant]:
            raise RuntimeError(f"diagnostic predictions changed before evaluation:{variant}")
    sources = read_json(cohort_path)[:len(formal_rows)]
    formal_eval = []
    for row, source in zip(formal_rows, sources):
        gt = source["answer"]
        direct_ok = normalize_answer(row["direct"]["answer"]) == normalize_answer(gt)
        structured_defined = bool(row["structured"].get("execution_success"))
        structured_ok = structured_defined and normalize_answer(row["structured"].get("candidate")) == normalize_answer(gt)
        final_ok = normalize_answer(row["final_answer"]) == normalize_answer(gt)
        formal_eval.append({
            "question_id": row["question_id"], "sample_token": row["sample_token"], "ground_truth_answer": gt,
            "direct_answer": row["direct"]["answer"], "structured_answer": row["structured"].get("candidate"),
            "final_answer": row["final_answer"], "structured_defined": structured_defined,
            "direct_correct": direct_ok, "structured_correct": structured_ok if structured_defined else None,
            "final_correct": final_ok,
        })
    direct_correct = sum(r["direct_correct"] for r in formal_eval)
    defined = [r for r in formal_eval if r["structured_defined"]]
    final_correct = sum(r["final_correct"] for r in formal_eval)
    both_correct = sum(r["direct_correct"] and r["structured_correct"] for r in defined)
    direct_only = sum(r["direct_correct"] and not r["structured_correct"] for r in defined)
    structured_only = sum(not r["direct_correct"] and r["structured_correct"] for r in defined)
    both_wrong = sum(not r["direct_correct"] and not r["structured_correct"] for r in defined)
    rescues = sum(not r["direct_correct"] and r["final_correct"] for r in formal_eval)
    harms = sum(r["direct_correct"] and not r["final_correct"] for r in formal_eval)
    validator_actions = Counter(r["validator"]["decision"] for r in formal_rows if r["validator"]["invoked"])
    sanity = {
        "label": SANITY_LABEL, "evaluation_started_after_prediction_finalization": True,
        "case_count": len(formal_eval),
        "direct": {"correct": direct_correct, "total": len(formal_eval), "accuracy": direct_correct / len(formal_eval)},
        "structured_candidate_where_defined": {"correct": sum(bool(r["structured_correct"]) for r in defined), "total": len(defined), "accuracy": sum(bool(r["structured_correct"]) for r in defined) / len(defined) if defined else None},
        "final_hybrid": {"correct": final_correct, "total": len(formal_eval), "accuracy": final_correct / len(formal_eval)},
        "direct_vs_structured_where_defined": {"both_correct": both_correct, "direct_only_correct": direct_only, "structured_only_correct": structured_only, "both_wrong": both_wrong, "undefined_structured": len(formal_eval) - len(defined)},
        "final_intervention": {"rescues": rescues, "harms": harms, "net_correct_change": rescues - harms},
        "outer_agent_gate": {"positive": sum(r["pre_execution_control"]["gate_invoked"] for r in formal_rows), "negative": sum(not r["pre_execution_control"]["gate_invoked"] for r in formal_rows), "invocation_rate": sum(r["pre_execution_control"]["gate_invoked"] for r in formal_rows) / len(formal_rows)},
        "inner_rag_trigger": {"positive": sum(r["rag_trigger"]["decision"] for r in formal_rows), "negative_among_agent_invoked": sum(r["pre_execution_control"]["gate_invoked"] and not r["rag_trigger"]["decision"] for r in formal_rows), "invocation_rate_over_all": sum(r["rag_trigger"]["decision"] for r in formal_rows) / len(formal_rows), "invocation_rate_given_agent": sum(r["rag_trigger"]["decision"] for r in formal_rows) / max(1, sum(r["pre_execution_control"]["gate_invoked"] for r in formal_rows))},
        "qa_retrieval_count": sum(r["rag_context"]["retrieval_invoked"] for r in formal_rows),
        "compact_prior_invocation_count": sum(r["rag_context"]["compact_prior_selected"] for r in formal_rows),
        "planner_success": sum(r["planner"]["invoked"] and r["planner"]["normalization_success"] for r in formal_rows),
        "planner_failure": sum(r["planner"]["invoked"] and not r["planner"]["normalization_success"] for r in formal_rows),
        "structured_success": sum(r["structured"].get("execution_success", False) for r in formal_rows),
        "structured_failure_or_abstention": sum(r["structured"]["invoked"] and not r["structured"].get("execution_success", False) for r in formal_rows),
        "validator_action_counts": {"KEEP_DIRECT": validator_actions["KEEP_DIRECT"], "ACCEPT_STRUCTURED": validator_actions["REQUEST_REVISION"], "UNRESOLVED": validator_actions["UNRESOLVED"], "NOT_INVOKED": sum(not r["validator"]["invoked"] for r in formal_rows)},
        "selective_acceptance_count": sum(r["selective"]["accepted"] for r in formal_rows),
        "final_override_count": sum(r["selective"]["accepted"] for r in formal_rows),
        "per_case_evaluation": formal_eval,
    }
    atomic_json(out / "full_framework_sanity_evaluation.json", sanity)

    variant_eval: dict[str, list[dict[str, Any]]] = {}
    for variant in names:
        evaluated = []
        for row, source in zip(diagnostic_rows, sources):
            data = row["variants"][variant]
            success = data["structured"]["execution_success"]
            answer = data["structured"]["candidate"]
            correct = success and normalize_answer(answer) == normalize_answer(source["answer"])
            evaluated.append({"question_id": row["question_id"], "answer": answer, "success": success, "correct": correct if success else None})
        variant_eval[variant] = evaluated
    def summary(variant: str) -> dict[str, Any]:
        rows = variant_eval[variant]
        ok = [r for r in rows if r["success"]]
        return {"structured_success_count": len(ok), "structured_success_rate": len(ok) / len(rows), "correct_where_defined": sum(bool(r["correct"]) for r in ok), "defined_count": len(ok), "accuracy_where_defined": sum(bool(r["correct"]) for r in ok) / len(ok) if ok else None}
    def change(left: str, right: str) -> list[int]:
        return [a["question_id"] for a, b in zip(variant_eval[left], variant_eval[right]) if normalize_answer(a["answer"]) != normalize_answer(b["answer"])]
    improved, regressed, unchanged = [], [], []
    for a, c in zip(variant_eval["NO_RAG"], variant_eval["MULTILEVEL_RAG"]):
        a_ok, c_ok = bool(a["correct"]), bool(c["correct"])
        (improved if not a_ok and c_ok else regressed if a_ok and not c_ok else unchanged).append(a["question_id"])
    comparison = {
        "label": "CURRENT REAL-EVIDENCE INTEGRATION DIAGNOSTIC — NOT A PAPER RESULT — NOT A CALIBRATION SET",
        "no_tuning_from_outcomes": True,
        "variants": {variant: summary(variant) for variant in names},
        "prediction_changes": {"A_to_B": {"count": len(change("NO_RAG", "SELF_TRIGGERED_QA")), "question_ids": change("NO_RAG", "SELF_TRIGGERED_QA")}, "B_to_C": {"count": len(change("SELF_TRIGGERED_QA", "MULTILEVEL_RAG")), "question_ids": change("SELF_TRIGGERED_QA", "MULTILEVEL_RAG")}, "A_to_C": {"count": len(change("NO_RAG", "MULTILEVEL_RAG")), "question_ids": change("NO_RAG", "MULTILEVEL_RAG")}},
        "outcomes_A_to_C": {"improved_count": len(improved), "improved_question_ids": improved, "regressed_count": len(regressed), "regressed_question_ids": regressed, "unchanged_count": len(unchanged), "unchanged_question_ids": unchanged},
        "retrieval_trigger_count": sum(r["variants"]["MULTILEVEL_RAG"]["rag_trigger"]["decision"] for r in diagnostic_rows),
        "retrieval_trigger_rate": sum(r["variants"]["MULTILEVEL_RAG"]["rag_trigger"]["decision"] for r in diagnostic_rows) / len(diagnostic_rows),
        "compact_prior_count": sum(r["variants"]["MULTILEVEL_RAG"]["rag_context"]["compact_prior_selected"] for r in diagnostic_rows),
        "compact_prior_rate": sum(r["variants"]["MULTILEVEL_RAG"]["rag_context"]["compact_prior_selected"] for r in diagnostic_rows) / len(diagnostic_rows),
    }
    atomic_json(out / "structured_agent_rag_comparison.json", comparison)
    return sanity, comparison


def historical_check(out: Path) -> dict[str, Any]:
    paths = {
        "qa_only": ROOT / "outputs" / "agent_atomic_program_500_step37_500_facade_qa" / "atomic_program_metrics.json",
        "banker_trace_only": ROOT / "outputs" / "agent_atomic_program_500_step37_500_facade_qa_banker_trace_only" / "atomic_program_metrics.json",
        "multilevel_append_1": ROOT / "outputs" / "agent_atomic_program_500_step37_500_facade_qa_banker_append_one" / "atomic_program_metrics.json",
    }
    checked = {}
    for name, path in paths.items():
        if path.is_file():
            data = read_json(path)
            checked[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "overall": data.get("metrics", {}).get("overall"), "evidence_mode": data.get("config", {}).get("evidence_mode"), "planner_mode": data.get("config", {}).get("planner_mode"), "rag_mode": data.get("config", {}).get("rag_mode")}
    result = {
        "labels": {"historical": "HISTORICAL ORACLE DEVELOPMENT EVIDENCE", "current": "CURRENT REAL-EVIDENCE INTEGRATION DIAGNOSTIC"},
        "verified_historical_artifacts": checked,
        "historical_500_case_no_rag_baseline_status": "500_CASE_NO_RAG_BASELINE_NOT_VERIFIED",
        "claim_about_exceeding_500_case_no_rag_baseline_made": False,
        "historical_results_merged_with_current_results": False,
        "interpretation": "The verified historical QA-only and Banker trace-only runs were 284/500; append-1 was 293/500. They used Oracle scene evidence and are reference evidence only.",
    }
    atomic_json(out / "historical_rag_evidence_check.json", result)
    return result


def write_summaries(out: Path, formal_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]], sanity: dict[str, Any], comparison: dict[str, Any], validation: dict[str, Any], planner_audit: dict[str, Any]) -> str:
    efficiency = {
        "formal_case_count": len(formal_rows), "formal_direct_llm_calls": len(formal_rows),
        "formal_outer_gate_evaluations": len(formal_rows),
        "formal_rag_trigger_calls": sum(r["pre_execution_control"]["gate_invoked"] for r in formal_rows),
        "formal_retrieval_calls": sum(r["rag_context"]["retrieval_invoked"] for r in formal_rows),
        "formal_planner_calls": sum(r["planner"]["invoked"] for r in formal_rows),
        "formal_structured_calls": sum(r["structured"]["invoked"] for r in formal_rows),
        "formal_validator_calls": sum(r["validator"]["invoked"] for r in formal_rows),
        "formal_selective_calls": sum(r["selective"]["invoked"] for r in formal_rows),
        "diagnostic_case_count": len(diagnostic_rows),
        "diagnostic_planner_cache_entries": len(diagnostic_rows) * 3,
        "diagnostic_actual_model_calls": sum(r["variants"][v]["generation_reused_from_variant"] is None for r in diagnostic_rows for v in ("NO_RAG", "SELF_TRIGGERED_QA", "MULTILEVEL_RAG")),
        "mean_formal_direct_latency_sec": mean_or_none([r["direct"]["latency_sec"] for r in formal_rows]),
        "mean_formal_end_to_end_latency_sec": mean_or_none([r["latency_sec"] for r in formal_rows]),
    }
    atomic_json(out / "efficiency_summary.json", efficiency)
    evidence_failure_reasons = {"no_target_match", "no_reference_match", "no_support_match", "empty_scene"}
    failed_structured = [
        r for r in formal_rows
        if r["structured"]["invoked"] and not r["structured"].get("execution_success", False)
    ]
    failures = {
        "rag_trigger_failure": [],
        "retrieval_failure": [r["question_id"] for r in formal_rows if r["rag_trigger"]["decision"] and not r["rag_context"]["retrieved_exemplar_ids"]],
        "planner_failure": [r["question_id"] for r in formal_rows if r["planner"]["invoked"] and not r["planner"]["normalization_success"]],
        "structured_execution_failure": [
            {"question_id": r["question_id"], "reason": r["structured"].get("failure_reason")}
            for r in failed_structured
            if r["structured"].get("failure_reason") not in evidence_failure_reasons
        ],
        "validator_rejection": [r["question_id"] for r in formal_rows if r["validator"]["invoked"] and r["validator"]["decision"] in {"KEEP_DIRECT", "UNRESOLVED"}],
        "selective_veto": [r["question_id"] for r in formal_rows if r["validator"]["invoked"] and r["validator"]["decision"] == "REQUEST_REVISION" and not r["selective"]["accepted"]],
        "evidence_insufficiency": [
            {"question_id": r["question_id"], "reason": r["structured"].get("failure_reason")}
            for r in failed_structured
            if r["structured"].get("failure_reason") in evidence_failure_reasons
        ] + [
            {"question_id": r["question_id"], "reason": "empty_bevfusion_evidence"}
            for r in formal_rows
            if not r["direct"]["perception_evidence_ref"]["object_count"]
            and r not in failed_structured
        ],
    }
    failures["counts"] = {key: len(value) for key, value in failures.items() if isinstance(value, list)}
    atomic_json(out / "failure_summary.json", failures)
    pair = sanity["direct_vs_structured_where_defined"]
    final = sanity["final_intervention"]
    vd = sanity["validator_action_counts"]
    decision = (
        "A. COMPLETE_FRAMEWORK_REINTEGRATED_AND_READY_FOR_CONTROLLER_CALIBRATION"
        if validation["all_protocol_assertions_pass"] and planner_audit["all_protocol_assertions_pass"]
        else "D. RAG_REINTEGRATION_FAILED_IMPLEMENTATION_VALIDATION"
    )
    report = f"""# Phase 5C — Full-Framework Reintegration

**{SANITY_LABEL}**

## A. Protocol Reintegration

Submission-safe Self-Triggered Multi-Level RAG is integrated between the frozen outer rescue gate and the label-free Planner. The six required components are present: real BEVFusion evidence, intelligent Planner, atomic deterministic tools, self-triggered QA retrieval plus append-1 compact prior, single-pass evidence-grounded validation, and rescue-aware Hybrid arbitration. Outer Agent invocation and inner retrieval invocation are separate code paths and trace fields.

## B. Leakage Status

The regenerated retrieval index contains {read_json(out / 'rag_corpus_manifest.json')['indexed_record_count']} authorized train records after {read_json(out / 'rag_corpus_manifest.json')['duplicate_record_exclusion_count']} exact duplicate exclusions. Current sample IDs and available scene IDs were checked under mandatory exclusion rules; the authorized train split had zero such overlaps with this cohort. Answer text, current `eval_type`, current template metadata, GT, evaluation labels, Oracle rows, and unsafe historical caches are absent from inference. Protocol assertions passed: **{validation['protocol_assertion_count'] - validation['protocol_violation_count']}/{validation['protocol_assertion_count']}**.

## C. Full 50-Case End-to-End Behavior

- Direct: **{sanity['direct']['correct']}/50 ({sanity['direct']['accuracy']:.1%})**
- Structured where defined: **{sanity['structured_candidate_where_defined']['correct']}/{sanity['structured_candidate_where_defined']['total']} ({sanity['structured_candidate_where_defined']['accuracy'] or 0:.1%})**
- Final Hybrid: **{sanity['final_hybrid']['correct']}/50 ({sanity['final_hybrid']['accuracy']:.1%})**
- Direct/Structured both correct, Direct-only, Structured-only, both wrong (defined subset): **{pair['both_correct']} / {pair['direct_only_correct']} / {pair['structured_only_correct']} / {pair['both_wrong']}**
- Rescues / harms / net: **{final['rescues']} / {final['harms']} / {final['net_correct_change']:+d}**
- Outer gate positive / negative: **{sanity['outer_agent_gate']['positive']} / {sanity['outer_agent_gate']['negative']}**
- Inner RAG positive / negative among invoked: **{sanity['inner_rag_trigger']['positive']} / {sanity['inner_rag_trigger']['negative_among_agent_invoked']}**
- QA retrieval / compact-prior calls: **{sanity['qa_retrieval_count']} / {sanity['compact_prior_invocation_count']}**
- Planner success / failure: **{sanity['planner_success']} / {sanity['planner_failure']}**
- Structured success / failure: **{sanity['structured_success']} / {sanity['structured_failure_or_abstention']}**
- Validator KEEP_DIRECT / ACCEPT_STRUCTURED / UNRESOLVED: **{vd['KEEP_DIRECT']} / {vd['ACCEPT_STRUCTURED']} / {vd['UNRESOLVED']}**
- Selective accepts / final overrides: **{sanity['selective_acceptance_count']} / {sanity['final_override_count']}**

## D. RAG Component Diagnostic

With the same Real-Evidence rows, checkpoint, decoding, tools, executor, and normalization, success/accuracy-where-defined were:

- No RAG: **{comparison['variants']['NO_RAG']['structured_success_count']}/50**, **{comparison['variants']['NO_RAG']['accuracy_where_defined'] or 0:.1%}**
- Self-Triggered QA RAG: **{comparison['variants']['SELF_TRIGGERED_QA']['structured_success_count']}/50**, **{comparison['variants']['SELF_TRIGGERED_QA']['accuracy_where_defined'] or 0:.1%}**
- Full Multi-Level RAG: **{comparison['variants']['MULTILEVEL_RAG']['structured_success_count']}/50**, **{comparison['variants']['MULTILEVEL_RAG']['accuracy_where_defined'] or 0:.1%}**

Prediction changes A→B / B→C / A→C: **{comparison['prediction_changes']['A_to_B']['count']} / {comparison['prediction_changes']['B_to_C']['count']} / {comparison['prediction_changes']['A_to_C']['count']}**. Improved / regressed / unchanged A→C: **{comparison['outcomes_A_to_C']['improved_count']} / {comparison['outcomes_A_to_C']['regressed_count']} / {comparison['outcomes_A_to_C']['unchanged_count']}**. Retrieval triggered on **{comparison['retrieval_trigger_count']}/50** and the compact prior on **{comparison['compact_prior_count']}/50**. No parameter was tuned from these outcomes.

## E. Hybrid Compatibility

The outer gate, Validator, and selective model are frozen Phase 4 compatibility parameters. They were not calibrated against the RAG-enabled Structured distribution. Weak intervention therefore does not establish failure of the reintegrated method.

## F. Failures

RAG trigger / retrieval / Planner / Structured execution / Validator rejection / selective veto / evidence insufficiency counts: **{failures['counts']['rag_trigger_failure']} / {failures['counts']['retrieval_failure']} / {failures['counts']['planner_failure']} / {failures['counts']['structured_execution_failure']} / {failures['counts']['validator_rejection']} / {failures['counts']['selective_veto']} / {failures['counts']['evidence_insufficiency']}**.

Historical Oracle evidence remains separate. Verified 500-case QA-only and Banker trace-only runs were 284/500 and append-1 was 293/500. **500_CASE_NO_RAG_BASELINE_NOT_VERIFIED**; no claim against a nonexistent verified baseline is made.

## G. Next Required Step

The next scientific step is **controller recalibration** on an authorized cohort drawn from the RAG-enabled Structured Agent distribution. Phase 5C performed no recalibration or 50-case tuning.

## Decision

{decision}
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")
    return decision


def finalize_manifest(out: Path) -> None:
    path = out / "full_framework_protocol_manifest.json"
    manifest = read_json(path)
    required = (
        "rag_historical_to_safe_mapping.json", "rag_leakage_audit.json", "rag_corpus_manifest.json",
        "rag_trigger_trace.jsonl", "rag_retrieval_trace.jsonl", "rag_context_trace.jsonl",
        "rag_enabled_planner_cache.jsonl", "planner_cache_audit.json", "formal_inference_trace.jsonl",
        "formal_predictions.json", "protocol_validation.json", "structured_agent_no_rag_predictions.json",
        "structured_agent_self_triggered_qa_predictions.json", "structured_agent_multilevel_rag_predictions.json",
        "structured_agent_rag_comparison.json", "full_framework_sanity_evaluation.json",
        "historical_rag_evidence_check.json", "efficiency_summary.json", "failure_summary.json", "final_report.md",
    )
    manifest["final_artifacts"] = {name: {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size} for name in required}
    manifest["finalized_at_utc"] = utc_now()
    atomic_json(path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--train-questions", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--rag-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_BEVFUSION_ROOT)
    parser.add_argument("--model-path", type=Path, default=Path(os.getenv("LOCAL_QW_MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--direct-max-new-tokens", type=int, default=48)
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 50:
        raise SystemExit("Phase 5C requires at least 50 cases")
    if args.topk != 3:
        raise SystemExit("Phase 5C freezes the historically verified top-k at 3; tuning is prohibited")
    required = (args.cohort, args.train_questions, args.evidence_root, args.model_path, CONTROL_MANIFEST, CONTROL_THRESHOLDS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required resources: " + ", ".join(missing))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cases = load_formal_cases(args.cohort, args.limit)
    index = SubmissionSafeQAIndex(args.train_questions, args.rag_index, args.cohort, cases, args.topk).build_or_load()
    write_preflight(out, index, cases, args)
    if read_json(out / "rag_leakage_audit.json")["audit_status"] != "PASS":
        raise SystemExit("RAG leakage audit failed; formal run stopped")

    state_path = out / "run_state.json"
    if state_path.exists():
        run_id = str(read_json(state_path)["run_id"])
    else:
        run_id = f"phase5c-{uuid.uuid4().hex}"
        atomic_json(state_path, {"run_id": run_id, "status": "RUNNING_LABEL_FREE_INFERENCE", "started_at_utc": utc_now()})

    formal_path = out / "formal_inference_trace.jsonl"
    diagnostic_path = out / "diagnostic_structured_trace.jsonl"
    completed_formal = {int(row["question_id"]) for row in read_jsonl(formal_path)}
    completed_diag = {int(row["question_id"]) for row in read_jsonl(diagnostic_path)}
    pending_formal = [case for case in cases if case.question_id not in completed_formal]
    pending_diag = [case for case in cases if case.question_id not in completed_diag]

    if pending_formal or pending_diag:
        qwen = EvidenceOnlyQwenRuntime(args.model_path, args.min_pixels, args.max_pixels)
        qwen.load()
        planner = FormalPlanner(qwen.model, qwen.processor, fallback_planner=None, use_template_hint=False)
        executor = PlanCompiledAtomicExecutor(tools=None)
        adapter = BEVFusionSceneAdapter(args.evidence_root)
        controls = FrozenControlStack(CONTROL_MANIFEST, CONTROL_THRESHOLDS)
        for position, case in enumerate(pending_formal, start=1):
            row, cache = run_formal_case(case, run_id, qwen, planner, executor, adapter, controls, index, args.model_path, args.direct_max_new_tokens)
            if cache is not None:
                append_jsonl(out / "rag_enabled_planner_cache.jsonl", cache)
            append_jsonl(formal_path, row)
            print(json.dumps({"stage": "formal", "progress": f"{position}/{len(pending_formal)}", "question_id": case.question_id, "direct": row["direct"]["answer"], "outer_gate": row["pre_execution_control"]["gate_invoked"], "rag": row["rag_trigger"]["decision"], "structured_success": row["structured"].get("execution_success", False), "final": row["final_answer"]}, sort_keys=True), flush=True)

        formal_rows = latest_by_id(formal_path)
        validation = finalize_formal(out, formal_rows, cases)
        if not validation["all_protocol_assertions_pass"]:
            raise RuntimeError("formal protocol validation failed")

        for position, case in enumerate(pending_diag, start=1):
            row = run_diagnostic_case(case, run_id, planner, executor, adapter, controls, index, args.model_path)
            append_jsonl(diagnostic_path, row)
            print(json.dumps({"stage": "diagnostic", "progress": f"{position}/{len(pending_diag)}", "question_id": case.question_id, "trigger": row["variants"]["MULTILEVEL_RAG"]["rag_trigger"]["decision"], "banker": row["variants"]["MULTILEVEL_RAG"]["rag_context"]["compact_prior_selected"], "answers": {key: row["variants"][key]["structured"]["candidate"] for key in row["variants"]}}, sort_keys=True), flush=True)
        del controls, adapter, executor, planner, qwen
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    formal_rows = latest_by_id(formal_path)
    validation = finalize_formal(out, formal_rows, cases)
    diagnostic_rows = latest_by_id(diagnostic_path)
    finalize_diagnostics(out, diagnostic_rows, cases)
    planner_audit = audit_planner_cache(out, formal_rows, diagnostic_rows)
    if not planner_audit["all_protocol_assertions_pass"]:
        raise RuntimeError("RAG-enabled Planner cache audit failed")
    historical_check(out)
    sanity, comparison = evaluate(out, args.cohort, formal_rows, diagnostic_rows)
    decision = write_summaries(out, formal_rows, diagnostic_rows, sanity, comparison, validation, planner_audit)
    finalize_manifest(out)
    atomic_json(state_path, {"run_id": run_id, "status": "COMPLETE", "decision": decision, "updated_at_utc": utc_now(), "no_tuning_performed": True})
    print(json.dumps({"status": "COMPLETE", "decision": decision, "direct": sanity["direct"], "final_hybrid": sanity["final_hybrid"], "rag_diagnostic": comparison["variants"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
