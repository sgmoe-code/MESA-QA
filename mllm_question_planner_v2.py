# mllm_question_planner_v2.py
# Step1b-v2: MLLM-guided Planner with stricter prompt and robust normalization.
#
# Improvements over v1:
# 1. Prompt explicitly says: do NOT join module names with "|".
# 2. If there is no relation, return [] instead of ["unknown"].
# 3. Parser defensively splits module strings containing "|".
# 4. Parser removes unknown/null relations.
# 5. Count/comparison/one-hop plans are forced to include scene_graph_retriever + symbolic_executor + evidence_verifier.
#
# This file keeps fallback to GuidedQuestionPlanner for safety, but metrics should record planner_source.

import json
from dataclasses import asdict, dataclass
import re
from typing import Dict, Any, Optional, Tuple


ALLOWED_QUESTION_TYPES = {"exist", "count", "object", "status", "comparison", "risk", "rule", "unknown"}
ALLOWED_ANSWER_TYPES = {"yes_no", "number", "category", "status", "text"}
ALLOWED_HOPS = {"zero-hop", "one-hop", "multi-hop", "unknown"}

ALLOWED_MODULES = {
    "visual_tool",
    "scene_graph_retriever",
    "symbolic_executor",
    "open_rag",
    "result_parser",
    "evidence_verifier",
}

ALLOWED_CATEGORIES = {
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
    "thing", "unknown"
}

ALLOWED_STATUSES = {
    "moving", "parked", "stopped", "standing", "not_standing",
    "stationary", "with_rider", "without_rider", "unknown"
}

ALLOWED_RELATIONS = {
    "front", "back", "left", "right",
    "front-left", "front-right", "back-left", "back-right",
    "same-status-as", "same-category-as", "near"
}


ALLOWED_RAG_SOURCES = {"none", "qa_exemplar"}


@dataclass
class RetrievalDecision:
    retrieve: bool
    rag_source: str
    refined_query: str
    reason: str
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return default


def _parse_bool_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _parse_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    text = re.sub(r"^```json", "", text, flags=re.I).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw or "", flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def fallback_retrieval_decision(original_question: str, config: Any = None) -> Dict[str, Any]:
    retrieve = bool(_cfg_value(config, "self_triggered_default_retrieve", False))
    source = "qa_exemplar" if retrieve else "none"
    refined_query = original_question if retrieve else ""
    return RetrievalDecision(
        retrieve=retrieve,
        rag_source=source,
        refined_query=refined_query,
        reason="fallback_after_invalid_retrieval_decision",
        confidence=None,
    ).to_dict()


def parse_retrieval_decision(raw: str, original_question: str, config: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trace = {
        "decision_valid_json": False,
        "decision_fallback_used": False,
        "decision_raw_preview": (raw or "")[:1000],
    }
    obj = _extract_json_object(raw or "")
    if obj is None:
        trace["decision_fallback_used"] = True
        return fallback_retrieval_decision(original_question, config), trace

    trace["decision_valid_json"] = True
    required = {"retrieve", "rag_source", "refined_query", "reason"}
    if not required.issubset(obj):
        trace["decision_fallback_used"] = True
        trace["decision_missing_fields"] = sorted(required - set(obj))
        return fallback_retrieval_decision(original_question, config), trace

    retrieve = _parse_bool_value(obj.get("retrieve"))
    rag_source = str(obj.get("rag_source", "none")).strip().lower()
    if retrieve is None or rag_source not in ALLOWED_RAG_SOURCES:
        trace["decision_fallback_used"] = True
        trace["decision_error"] = "invalid_retrieve_or_rag_source"
        return fallback_retrieval_decision(original_question, config), trace

    if retrieve:
        rag_source = "qa_exemplar"
        refined_query = str(obj.get("refined_query") or original_question).strip()
    else:
        rag_source = "none"
        refined_query = ""

    decision = RetrievalDecision(
        retrieve=bool(retrieve),
        rag_source=rag_source,
        refined_query=refined_query,
        reason=str(obj.get("reason", "")).strip()[:500],
        confidence=_parse_confidence(obj.get("confidence")),
    ).to_dict()
    return decision, trace


def summarize_seed_plan_for_retrieval(seed_plan: Optional[Dict[str, Any]]) -> str:
    if not isinstance(seed_plan, dict):
        return ""
    target = seed_plan.get("target_entity") or {}
    reference = seed_plan.get("reference_entity") or {}
    relations = seed_plan.get("relations") or []
    if isinstance(relations, str):
        relations = [relations]
    target_category = target.get("category", "") if isinstance(target, dict) else ""
    reference_category = reference.get("category", "") if isinstance(reference, dict) else ""
    return "\n".join([
        "question_type: " + str(seed_plan.get("question_type", "")),
        "target: " + str(target_category),
        "reference: " + str(reference_category),
        "relation: " + ", ".join(str(rel) for rel in relations),
        "answer_type: " + str(seed_plan.get("answer_type", "")),
    ])


class MLLMQuestionPlannerV2:
    def __init__(self, model, processor, fallback_planner=None, use_template_hint: bool = False, verbose: bool = False):
        self.model = model
        self.processor = processor
        self.fallback_planner = fallback_planner
        self.use_template_hint = use_template_hint
        self.verbose = verbose

    def plan(
        self,
        question: str,
        template_type: str = "",
        exemplar_context: str = "",
        scene_query_extension: Optional[str] = None,
        scene_strategy_token: Optional[str] = None,
        scene_strategy_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        exemplar_context = (exemplar_context or "").strip()
        exemplar_context_used = bool(exemplar_context)
        prompt = self._build_prompt(
            question,
            template_type,
            exemplar_context=exemplar_context,
            scene_query_extension=scene_query_extension,
            scene_strategy_token=scene_strategy_token,
            scene_strategy_instruction=scene_strategy_instruction,
        )
        raw = self._run_text(prompt, max_tokens=340)

        plan, error = self._parse_and_validate(raw)

        if plan is None:
            if self.verbose:
                print("[MLLMPlannerV2] invalid plan, fallback. Error:", error)
                print("[MLLMPlannerV2] raw:", raw[:500])
            if self.fallback_planner is not None:
                fb = self.fallback_planner.plan(question, template_type=template_type).to_dict()
                fb["planner_source"] = "rule_fallback_after_mllm_exemplar" if exemplar_context_used else "rule_fallback"
                fb["planner_raw"] = raw
                fb["planner_error"] = error
                fb["exemplar_context_used"] = exemplar_context_used
                fb["planner_prompt"] = prompt
                return fb
            fb = self._default_plan(question, raw=raw, error=error, exemplar_context_used=exemplar_context_used)
            fb["planner_prompt"] = prompt
            return fb

        plan["planner_source"] = "mllm_exemplar" if exemplar_context_used else "mllm"
        plan["planner_raw"] = raw
        plan["planner_error"] = None
        plan["exemplar_context_used"] = exemplar_context_used
        plan["planner_prompt"] = prompt
        return plan

    def plan_with_drivelm_hints(
        self,
        question: str,
        template_type: str = "",
        seed_plan: Optional[Dict[str, Any]] = None,
        drivelm_context: str = "",
    ) -> Dict[str, Any]:
        drivelm_context = (drivelm_context or "").strip()
        prompt = self._build_drivelm_prompt(
            question=question,
            template_type=template_type,
            seed_plan=seed_plan,
            drivelm_context=drivelm_context,
        )
        raw = self._run_text(prompt, max_tokens=380)
        plan, error = self._parse_and_validate(raw)
        if plan is None:
            if self.verbose:
                print("[MLLMPlannerV2] invalid DriveLM plan, fallback. Error:", error)
                print("[MLLMPlannerV2] raw:", raw[:500])
            if isinstance(seed_plan, dict):
                fb = dict(seed_plan)
                fb["planner_source"] = "seed_fallback_after_mllm_drivelm"
                fb["planner_raw"] = raw
                fb["planner_error"] = error
                fb["drivelm_context_used"] = bool(drivelm_context)
                return fb
            if self.fallback_planner is not None:
                fb = self.fallback_planner.plan(question, template_type=template_type).to_dict()
                fb["planner_source"] = "rule_fallback_after_mllm_drivelm"
                fb["planner_raw"] = raw
                fb["planner_error"] = error
                fb["drivelm_context_used"] = bool(drivelm_context)
                return fb
            return self._default_plan(question, raw=raw, error=error, exemplar_context_used=False)
        plan["planner_source"] = "mllm_drivelm"
        plan["planner_raw"] = raw
        plan["planner_error"] = None
        plan["drivelm_context_used"] = bool(drivelm_context)
        return plan

    def plan_with_controller_context(
        self,
        question: str,
        template_type: str = "",
        seed_plan: Optional[Dict[str, Any]] = None,
        controller_context: str = "",
    ) -> Dict[str, Any]:
        controller_context = (controller_context or "").strip()
        prompt = self._build_controller_prompt(
            question=question,
            template_type=template_type,
            seed_plan=seed_plan,
            controller_context=controller_context,
        )
        raw = self._run_text(prompt, max_tokens=380)
        plan, error = self._parse_and_validate(raw)
        if plan is None:
            if self.verbose:
                print("[MLLMPlannerV2] invalid controller plan, fallback. Error:", error)
                print("[MLLMPlannerV2] raw:", raw[:500])
            if isinstance(seed_plan, dict):
                fb = dict(seed_plan)
                fb["planner_source"] = "seed_fallback_after_mllm_minimal_controller"
                fb["planner_raw"] = raw
                fb["planner_error"] = error
                fb["controller_context_used"] = bool(controller_context)
                return fb
            if self.fallback_planner is not None:
                fb = self.fallback_planner.plan(question, template_type=template_type).to_dict()
                fb["planner_source"] = "rule_fallback_after_mllm_minimal_controller"
                fb["planner_raw"] = raw
                fb["planner_error"] = error
                fb["controller_context_used"] = bool(controller_context)
                return fb
            return self._default_plan(question, raw=raw, error=error, exemplar_context_used=False)
        plan["planner_source"] = "mllm_minimal_controller"
        plan["planner_raw"] = raw
        plan["planner_error"] = None
        plan["controller_context_used"] = bool(controller_context)
        return plan

    def planner_analyze_with_optional_controller_request(
        self,
        question: str,
        eval_type: Optional[str] = None,
        template_type: str = "",
    ) -> Dict[str, Any]:
        prompt = self._build_tool_controller_analysis_prompt(question, eval_type=eval_type, template_type=template_type)
        raw = self._run_text(prompt, max_tokens=520)
        obj = self._extract_json(raw)
        if not isinstance(obj, dict):
            fallback = self.plan(question, template_type=template_type)
            return {
                "task_analysis": self._task_analysis_from_plan(fallback, eval_type=eval_type),
                "controller_request": {"call_controller": False, "allowed_sources": [], "reason": "analysis_json_parse_failed"},
                "atomic_program": fallback,
                "planner_raw": raw,
                "planner_error": "json_parse_failed",
            }
        task_analysis = self._normalize_task_analysis(obj.get("task_analysis"), eval_type=eval_type)
        controller_request = self._normalize_controller_request(obj.get("controller_request"), task_analysis)
        atomic_program = obj.get("atomic_program")
        normalized_program = None
        atomic_error = None
        if isinstance(atomic_program, dict):
            normalized_program, atomic_error = self._parse_and_validate(json.dumps(atomic_program, ensure_ascii=False))
        return {
            "task_analysis": task_analysis,
            "controller_request": controller_request,
            "atomic_program": normalized_program,
            "atomic_program_error": atomic_error,
            "planner_raw": raw,
            "planner_error": None,
        }

    def planner_finalize_with_controller_evidence(
        self,
        question: str,
        task_analysis: Dict[str, Any],
        controller_context: str,
        draft_atomic_program: Optional[Dict[str, Any]] = None,
        template_type: str = "",
    ) -> Dict[str, Any]:
        prompt = self._build_tool_controller_finalization_prompt(
            question=question,
            task_analysis=task_analysis,
            controller_context=controller_context,
            draft_atomic_program=draft_atomic_program,
            template_type=template_type,
        )
        raw = self._run_text(prompt, max_tokens=420)
        plan, error = self._parse_and_validate(raw)
        if plan is None:
            if isinstance(draft_atomic_program, dict):
                fb = dict(draft_atomic_program)
                fb["planner_source"] = "tool_controller_draft_fallback_after_finalization"
                fb["planner_raw"] = raw
                fb["planner_error"] = error
                return fb
            fallback = self.plan(question, template_type=template_type)
            fallback["planner_source"] = "tool_controller_normal_fallback_after_finalization"
            fallback["planner_raw"] = raw
            fallback["planner_error"] = error
            return fallback
        plan["planner_source"] = "mllm_tool_controller_finalized"
        plan["planner_raw"] = raw
        plan["planner_error"] = None
        return plan

    def _normalize_task_analysis(self, value: Any, eval_type: Optional[str] = None) -> Dict[str, Any]:
        value = value if isinstance(value, dict) else {}
        qtype = str(value.get("question_type") or eval_type or "unknown").strip().lower()
        if qtype not in ALLOWED_QUESTION_TYPES:
            qtype = str(eval_type or "unknown").strip().lower()
        if qtype not in ALLOWED_QUESTION_TYPES:
            qtype = "unknown"
        atype = str(value.get("answer_type") or {"count": "number", "exist": "yes_no", "comparison": "yes_no", "object": "category", "status": "status"}.get(qtype, "text")).strip().lower()
        if atype not in ALLOWED_ANSWER_TYPES:
            atype = "text"
        relation = value.get("relation")
        if isinstance(relation, list):
            relation = ", ".join(str(item) for item in relation if item)
        relation = str(relation or "").strip().lower().replace("_", " ") or None
        state = str(value.get("state") or "").strip().lower().replace("_", " ") or None
        target = str(value.get("target") or "").strip().lower().replace("_", " ") or None
        reference = str(value.get("reference") or "").strip().lower().replace("_", " ") or None
        optional_scene_hints = value.get("optional_scene_hints") if isinstance(value.get("optional_scene_hints"), dict) else {}
        return {
            "question_type": qtype,
            "answer_type": atype,
            "target": target,
            "reference": reference,
            "relation": relation,
            "state": state,
            "requires_relation_reasoning": bool(value.get("requires_relation_reasoning") or relation),
            "requires_state_reasoning": bool(value.get("requires_state_reasoning") or state),
            "requires_count_reasoning": bool(value.get("requires_count_reasoning") or qtype == "count"),
            "requires_comparison_reasoning": bool(value.get("requires_comparison_reasoning") or qtype == "comparison"),
            "optional_scene_hints": {
                "scene_type": optional_scene_hints.get("scene_type"),
                "candidate_ambiguity": optional_scene_hints.get("candidate_ambiguity"),
                "relation_complexity": optional_scene_hints.get("relation_complexity"),
                "state_ambiguity": optional_scene_hints.get("state_ambiguity"),
            },
        }

    def _normalize_controller_request(self, value: Any, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        value = value if isinstance(value, dict) else {}
        call = _parse_bool_value(value.get("call_controller"))
        if call is None:
            call = False
        sources = value.get("allowed_sources") or []
        if isinstance(sources, str):
            sources = [sources]
        allowed = []
        for source in sources:
            source = str(source).strip().lower()
            if source in {"qa_exemplar", "driving_prior_banker"} and source not in allowed:
                allowed.append(source)
        if call and not allowed:
            allowed = ["qa_exemplar"]
        return {"call_controller": bool(call), "allowed_sources": allowed, "reason": str(value.get("reason") or "").strip()[:500]}

    def _task_analysis_from_plan(self, plan: Dict[str, Any], eval_type: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return self._normalize_task_analysis({}, eval_type=eval_type)
        target = plan.get("target_entity") or {}
        reference = plan.get("reference_entity") or {}
        relations = plan.get("relations") or []
        if isinstance(relations, str):
            relations = [relations]
        attrs = []
        if isinstance(target, dict):
            attrs.extend(target.get("attributes") or [])
        if isinstance(reference, dict):
            attrs.extend(reference.get("attributes") or [])
        return self._normalize_task_analysis({
            "question_type": plan.get("question_type") or eval_type,
            "answer_type": plan.get("answer_type"),
            "target": target.get("category") if isinstance(target, dict) else None,
            "reference": reference.get("category") if isinstance(reference, dict) else None,
            "relation": ", ".join(str(rel) for rel in relations),
            "state": attrs[0] if attrs else None,
        }, eval_type=eval_type)

    def decide_retrieval_for_question(
        self,
        question: str,
        eval_type: Optional[str] = None,
        seed_plan: Optional[Dict[str, Any]] = None,
        available_rag_sources: Optional[list] = None,
        scene_decision_context: Optional[str] = None,
        config: Any = None,
    ) -> Dict[str, Any]:
        prompt = self._build_retrieval_decision_prompt(
            question=question,
            eval_type=eval_type,
            seed_plan=seed_plan,
            available_rag_sources=available_rag_sources,
            scene_decision_context=scene_decision_context,
        )
        try:
            retries = int(_cfg_value(config, "self_triggered_max_decision_retries", 1))
        except (TypeError, ValueError):
            retries = 1
        attempts = max(1, retries + 1)
        decision = fallback_retrieval_decision(question, config)
        trace = {"decision_valid_json": False, "decision_fallback_used": True}
        for attempt in range(attempts):
            raw = self._run_text(prompt, max_tokens=220)
            decision, trace = parse_retrieval_decision(raw, question, config)
            trace["decision_attempts"] = attempt + 1
            if not trace.get("decision_fallback_used"):
                break
        return {"decision": decision, "trace": trace}

    def _build_tool_controller_analysis_prompt(
        self,
        question: str,
        eval_type: Optional[str] = None,
        template_type: str = "",
    ) -> str:
        template_hint = ""
        if self.use_template_hint and template_type:
            template_hint = f"\nOptional benchmark hint: template_type = {template_type}\n"
        return f"""
You are the Planner of a public-safety cross-modal QA Agent.
First analyze the question type and explicit constraints.
Then decide whether the evidence controller is useful.
Call the evidence controller only when QA exemplars or driving priors are likely to help construct a better atomic program.
Do not call the controller for simple comparison questions.
Do not call the controller only to get an answer.
The controller can provide QA exemplars and abstract driving priors, but it does not provide ground-truth answers.
If no evidence is needed, generate the atomic program directly.

Hard guidance:
- Comparison questions usually do not call controller.
- Simple count questions without relation usually do not call controller, unless examples are clearly needed.
- Relation-heavy object/exist/status questions may call controller.
- Same-status, other-object, or generic "thing" questions may call controller.
- Status questions involving relation and state words may call controller.

Available modules for atomic programs:
["visual_tool", "scene_graph_retriever", "symbolic_executor", "open_rag", "result_parser", "evidence_verifier"]
{template_hint}
eval_type: {eval_type or 'unknown'}
Question:
{question}

Return JSON only. No markdown. Use exactly this wrapper schema:
{{
  "task_analysis": {{
    "question_type": "count|exist|object|status|comparison",
    "answer_type": "number|yes_no|category|status|text",
    "target": "car|thing|unknown",
    "reference": "car|pedestrian|thing|null",
    "relation": "front|left|front-left|null",
    "state": "moving|parked|stopped|standing|null",
    "requires_relation_reasoning": false,
    "requires_state_reasoning": false,
    "requires_count_reasoning": false,
    "requires_comparison_reasoning": false,
    "optional_scene_hints": {{
      "scene_type": null,
      "candidate_ambiguity": null,
      "relation_complexity": null,
      "state_ambiguity": null
    }}
  }},
  "controller_request": {{
    "call_controller": false,
    "allowed_sources": [],
    "reason": "short planner reason"
  }},
  "atomic_program": null
}}

If controller_request.call_controller is false, atomic_program must be a complete normal planner JSON object with keys question_type, answer_type, hop, target_entity, reference_entity, relations, need_modules, reasoning_path, why.
If controller_request.call_controller is true, atomic_program may be null or a draft normal planner JSON object.
""".strip()

    def _build_tool_controller_finalization_prompt(
        self,
        question: str,
        task_analysis: Dict[str, Any],
        controller_context: str,
        draft_atomic_program: Optional[Dict[str, Any]] = None,
        template_type: str = "",
    ) -> str:
        template_hint = ""
        if self.use_template_hint and template_type:
            template_hint = f"\nOptional benchmark hint: template_type = {template_type}\n"
        task_json = json.dumps(task_analysis or {}, ensure_ascii=False, indent=2)
        draft_json = json.dumps(draft_atomic_program or {}, ensure_ascii=False, indent=2)
        return f"""
You are the same Planner that requested evidence from the evidence controller.
Use the returned evidence only to improve atomic program construction.
QA exemplars are planning templates, not answers.
Driving prior banker tokens are constraints, not answers.
Do not add relation/state constraints absent from the question.
Preserve explicit relation and state constraints from the question.
Preserve question_type and answer_type from task_analysis.
Output the same atomic program JSON schema used by the normal Planner.
Do not output free-form reasoning.

Available modules:
["visual_tool", "scene_graph_retriever", "symbolic_executor", "open_rag", "result_parser", "evidence_verifier"]

Task analysis:
{task_json}

Draft atomic program, if any:
{draft_json}

Evidence controller context:
{controller_context}
{template_hint}
Question:
{question}

Return JSON only using this normal planner schema:
{{
  "question_type": "count|exist|object|status|comparison",
  "answer_type": "number|yes_no|category|status|text",
  "hop": "zero-hop|one-hop|multi-hop|unknown",
  "target_entity": {{"category": "car", "attributes": []}},
  "reference_entity": null,
  "relations": [],
  "need_modules": ["scene_graph_retriever", "symbolic_executor", "result_parser"],
  "reasoning_path": ["scene_graph_retriever", "symbolic_executor", "result_parser"],
  "why": "Short explanation of the reasoning plan, not an answer."
}}
""".strip()

    def _build_retrieval_decision_prompt(
        self,
        question: str,
        eval_type: Optional[str] = None,
        seed_plan: Optional[Dict[str, Any]] = None,
        available_rag_sources: Optional[list] = None,
        scene_decision_context: Optional[str] = None,
    ) -> str:
        sources = available_rag_sources or ["qa_exemplar"]
        source_lines = []
        if "qa_exemplar" in sources:
            source_lines.append("- qa_exemplar: similar NuScenes-QA training questions with templates and reasoning paths. It may help with exist/count/object/status questions, but can hurt comparison questions if examples are mismatched.")
        seed_summary = summarize_seed_plan_for_retrieval(seed_plan)
        seed_block = ""
        if seed_summary:
            seed_block = "\nSeed plan summary:\n" + seed_summary
        scene_block = ""
        if scene_decision_context and str(scene_decision_context).strip():
            scene_block = "\n".join([
                "",
                "Optional macro visual scene profile:",
                str(scene_decision_context).strip(),
                "",
                "Use the macro visual scene profile only to judge whether external QA exemplar evidence may help construct a reliable atomic program.",
                "Do not use it to answer the question.",
                "Do not infer exact object counts, object identities, or statuses from it.",
            ])
        return "\n".join([
            "You are a retrieval controller for an autonomous-driving visual question answering system.",
            "Your job is not to answer the question.",
            "Your job is only to decide whether retrieved QA exemplars from the NuScenes-QA training split would help the planner generate a better reasoning path.",
            "",
            "Available RAG source:",
            *source_lines,
            "",
            "Return only valid JSON with fields:",
            "- retrieve: boolean",
            "- rag_source: qa_exemplar or none",
            "- refined_query: string",
            "- reason: string",
            "- confidence: number between 0 and 1",
            "",
            "Decision guidelines:",
            "- Prefer retrieve=true when the question requires object filtering, spatial relation reasoning, counting, or status lookup.",
            "- Prefer retrieve=false when the question is a simple comparison or when exemplars may introduce conflicting answer patterns.",
            "- Do not retrieve just because retrieval is available.",
            "- If retrieve=true, write a concise refined_query that captures the question type, target object, reference object, relation, and attribute if available.",
            "- Do not include any answer label in the refined_query.",
            "- Do not use test answers.",
            "",
            "eval_type: " + str(eval_type or "unknown"),
            seed_block,
            scene_block,
            "Question:",
            question,
        ]).strip()

    def _build_controller_prompt(
        self,
        question: str,
        template_type: str = "",
        seed_plan: Optional[Dict[str, Any]] = None,
        controller_context: str = "",
    ) -> str:
        template_hint = ""
        if self.use_template_hint and template_type:
            template_hint = f"\nOptional benchmark hint: template_type = {template_type}\n"
        seed_json = json.dumps(seed_plan or {}, ensure_ascii=False, indent=2)
        context = (controller_context or "").strip()
        return f"""
You are the Planner of a public-safety cross-modal QA Agent.
Your task is NOT to answer the question.
Your task is to refine a structured atomic tool-use plan.

You are given a source action selected by an evidence controller.
Do not change the selected evidence sources.
Use the selected context only to refine the atomic plan.
Do not treat priors or scene cues as answers.
Do not add constraints absent from the question.
Preserve explicit relation and state constraints from the question.
Preserve question_type and answer_type unless the seed plan is invalid.
Do not invent unsupported atomic operations or module names.

Available modules. You may only choose from this exact list:
["visual_tool", "scene_graph_retriever", "symbolic_executor", "open_rag", "result_parser", "evidence_verifier"]

Seed plan JSON:
{seed_json}

Controller-selected context:
{context}
{template_hint}
Question:
{question}

Return JSON only. No markdown. No explanation outside JSON.
Use the same schema as the seed plan:
{{
  "question_type": "count|exist|object|status|comparison",
  "answer_type": "number|yes_no|category|status|text",
  "hop": "zero-hop|one-hop|multi-hop|unknown",
  "target_entity": {{"category": "car", "attributes": []}},
  "reference_entity": null,
  "relations": [],
  "need_modules": ["scene_graph_retriever", "symbolic_executor", "result_parser"],
  "reasoning_path": ["scene_graph_retriever", "symbolic_executor", "result_parser"],
  "why": "Short explanation of the reasoning plan, not an answer."
}}
""".strip()

    def _build_drivelm_prompt(
        self,
        question: str,
        template_type: str = "",
        seed_plan: Optional[Dict[str, Any]] = None,
        drivelm_context: str = "",
    ) -> str:
        template_hint = ""
        if self.use_template_hint and template_type:
            template_hint = f"\nOptional benchmark hint: template_type = {template_type}\n"
        seed_json = json.dumps(seed_plan or {}, ensure_ascii=False, indent=2)
        context = (drivelm_context or "").strip()
        return f"""
You are the Planner of a public-safety cross-modal QA Agent.
Your task is NOT to answer the question.
Your task is to refine a structured tool-use plan.

The seed plan is the anchor. Preserve its question_type and answer_type unless it is clearly invalid.
The DriveLM hints are retrieved from other driving scenes and should only guide reasoning order.
They are not evidence about the current scene.
Do not copy any object, answer, count, status, or conclusion from the hints.
The final answer will be computed only by the deterministic atomic executor from current-scene evidence.
Do not invent unsupported atomic operations or module names.

Available modules. You may only choose from this exact list:
["visual_tool", "scene_graph_retriever", "symbolic_executor", "open_rag", "result_parser", "evidence_verifier"]

Seed plan JSON:
{seed_json}

DriveLM planner-safe context:
{context}
{template_hint}
Question:
{question}

Return JSON only. No markdown. No explanation outside JSON.
Use the same schema as the seed plan:
{{
  "question_type": "count|exist|object|status|comparison",
  "answer_type": "number|yes_no|category|status|text",
  "hop": "zero-hop|one-hop|multi-hop|unknown",
  "target_entity": {{"category": "car", "attributes": []}},
  "reference_entity": null,
  "relations": [],
  "need_modules": ["scene_graph_retriever", "symbolic_executor", "result_parser"],
  "reasoning_path": ["scene_graph_retriever", "symbolic_executor", "result_parser"],
  "why": "Short explanation of the reasoning plan, not an answer."
}}
""".strip()

    def _build_prompt(
        self,
        question: str,
        template_type: str,
        exemplar_context: str = "",
        scene_query_extension: Optional[str] = None,
        scene_strategy_token: Optional[str] = None,
        scene_strategy_instruction: Optional[str] = None,
    ) -> str:
        template_hint = ""
        if self.use_template_hint and template_type:
            template_hint = f"\nOptional benchmark hint: template_type = {template_type}\n"

        exemplar_section = ""
        exemplar_context = (exemplar_context or "").strip()
        if exemplar_context:
            answer_warning = ""
            if "Answer:" in exemplar_context:
                answer_warning = (
                    "\nThe exemplar answers are from different training scenes. "
                    "Do not copy them as the current answer."
                )
            exemplar_section = f"""
Retrieved reasoning exemplars from the official NuScenes-QA training split are provided below.
They are from different scenes.
Use them only to infer the current question's structure and reasoning path.
Do NOT copy any object IDs, object counts, or answers from exemplars.
Focus on the pattern: question_type, target entity, reference entity, relations, hop, and reasoning program.{answer_warning}

{exemplar_context}
"""
        scene_section = ""
        scene_query_extension = (scene_query_extension or "").strip()
        if scene_query_extension:
            scene_section = f"""
Optional macro visual scene background:
{scene_query_extension}

This background is only a high-level visual summary. It must not be used as answer evidence.
Do not infer exact object counts, object identities, object statuses, or final answers from it.
Use it only to make the atomic program more careful about relation, status, count, and reference-object constraints.
"""
        scene_strategy_section = ""
        scene_strategy_instruction = (scene_strategy_instruction or "").strip()
        scene_strategy_token = (scene_strategy_token or "").strip()
        if scene_strategy_instruction and scene_strategy_token:
            scene_strategy_section = f"""
Optional Scene Strategy:
{scene_strategy_instruction}
"""

        support_before_question = f"{template_hint}\n{exemplar_section}".strip()
        if scene_section:
            support_before_question = f"{support_before_question}\n{scene_section}".strip()
        if scene_strategy_section:
            support_before_question = f"{support_before_question}\n{scene_strategy_section}".strip()

        return f"""
You are the Planner of a public-safety cross-modal QA Agent.
Your task is NOT to answer the question.
Your task is to analyze the question and output a structured tool-use plan.

Available modules. You may only choose from this exact list:
["visual_tool", "scene_graph_retriever", "symbolic_executor", "open_rag", "result_parser", "evidence_verifier"]

Module selection rules:
- Each module name must be a separate string in the JSON array.
- Do NOT join multiple module names with "|".
- Do NOT invent new module names.
- Always include "result_parser".
- Use "scene_graph_retriever" for count, comparison, status query, object query with spatial relation, and any one-hop question.
- Use "symbolic_executor" for count, comparison, exact filtering, exact status lookup, and object-object spatial relation.
- Use "evidence_verifier" for count, comparison, one-hop relation, and safety/risk questions.
- Use "open_rag" only for traffic rule, public safety rule, evacuation, risk, or action recommendation questions.

Question decomposition guidance:
- "how many" or "what number" usually means question_type=count and answer_type=number.
- "is/are/do/does" usually means question_type=exist or comparison and answer_type=yes_no.
- "what is it", "what object", or "the ... thing is what" usually means question_type=object and answer_type=category.
- "what status" or "in what status" means question_type=status and answer_type=status.
- Nouns identify target or reference entities: car, truck, bus, pedestrian, traffic cone, etc.
- Adjectives/status words identify filters: moving, parked, stopped, standing, not standing.
- Spatial words identify relations: front, back, left, right, front-left, front-right, back-left, back-right.
- If the question asks about an object relative to another object, it is one-hop.
- If it asks about objects relative to ego/me only, it is zero-hop.
- If there is no spatial or comparison relation, return "relations": [].
- Do NOT return ["unknown"] for relations.
- Use "same-status-as" only when the question asks whether two objects have the same status.

{support_before_question}

Question:
{question}

Return JSON only. No markdown. No explanation outside JSON.

Required JSON schema example:
{{
  "question_type": "count",
  "answer_type": "number",
  "hop": "zero-hop",
  "target_entity": {{
    "category": "car",
    "attributes": []
  }},
  "reference_entity": null,
  "relations": [],
  "need_modules": ["scene_graph_retriever", "symbolic_executor", "result_parser", "evidence_verifier"],
  "reasoning_path": ["scene_graph_retriever", "symbolic_executor", "result_parser", "evidence_verifier"],
  "why": "Count cars using scene graph and executor."
}}
""".strip()

    def _run_text(self, prompt: str, max_tokens: int = 340) -> str:
        from qwen35_runtime import GENERATION_KWARGS, generated_suffix, render_text_chat
        text = render_text_chat(self.processor, prompt)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)

        import torch
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **GENERATION_KWARGS,
            )
        gen_ids_trim = generated_suffix(inputs.input_ids, gen_ids)
        return self.processor.batch_decode(gen_ids_trim, skip_special_tokens=True)[0].strip()

    def _parse_and_validate(self, raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        obj = self._extract_json(raw)
        if obj is None:
            return None, "json_parse_failed"

        try:
            plan = self._normalize_plan(obj)
            ok, err = self._validate_plan(plan)
            if not ok:
                return None, err
            return plan, None
        except Exception as e:
            return None, f"exception:{repr(e)}"

    def _extract_json(self, raw: str) -> Optional[Dict[str, Any]]:
        text = raw.strip()
        text = re.sub(r"^```json", "", text, flags=re.I).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return None

        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

        return None

    def _normalize_plan(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        qtype = str(obj.get("question_type", "unknown")).lower().strip()
        atype = str(obj.get("answer_type", "text")).lower().strip()
        hop = str(obj.get("hop", "unknown")).lower().strip()

        target = obj.get("target_entity") or {}
        if not isinstance(target, dict):
            target = {}

        reference = obj.get("reference_entity")
        if reference is not None and not isinstance(reference, dict):
            reference = None

        relations = obj.get("relations") or []
        if isinstance(relations, str):
            relations = [relations]

        need_modules = obj.get("need_modules") or []
        if isinstance(need_modules, str):
            need_modules = [need_modules]

        path = obj.get("reasoning_path") or need_modules
        if isinstance(path, str):
            path = [path]

        plan = {
            "question_type": qtype if qtype in ALLOWED_QUESTION_TYPES else "unknown",
            "answer_type": atype if atype in ALLOWED_ANSWER_TYPES else "text",
            "hop": hop if hop in ALLOWED_HOPS else "unknown",
            "target_entity": self._normalize_entity(target),
            "reference_entity": self._normalize_entity(reference) if reference is not None else None,
            "relations": self._normalize_relations(relations),
            "need_modules": self._normalize_modules(need_modules),
            "reasoning_path": self._normalize_modules(path),
            "why": str(obj.get("why", "")).strip()[:300],
        }

        self._apply_required_modules(plan)
        return plan

    def _apply_required_modules(self, plan: Dict[str, Any]) -> None:
        self._ensure_module(plan, "result_parser")

        if plan["question_type"] in {"count", "comparison"}:
            for m in ["scene_graph_retriever", "symbolic_executor", "evidence_verifier"]:
                self._ensure_module(plan, m)

        if plan["question_type"] in {"status"}:
            # In oracle prototype, status lookup is often more reliable from scene evidence.
            self._ensure_module(plan, "scene_graph_retriever")
            self._ensure_module(plan, "symbolic_executor")

        if plan["relations"] or plan["hop"] in {"one-hop", "multi-hop"}:
            self._ensure_module(plan, "scene_graph_retriever")
            self._ensure_module(plan, "symbolic_executor")
            self._ensure_module(plan, "evidence_verifier")

        if not plan["need_modules"]:
            plan["need_modules"] = ["visual_tool", "result_parser"]
        if not plan["reasoning_path"]:
            plan["reasoning_path"] = plan["need_modules"][:]

    def _ensure_module(self, plan: Dict[str, Any], module: str) -> None:
        if module not in plan["need_modules"]:
            plan["need_modules"].append(module)
        if module not in plan["reasoning_path"]:
            plan["reasoning_path"].append(module)

    def _normalize_entity(self, ent: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if ent is None:
            return {"category": "unknown", "attributes": []}

        category = str(ent.get("category", "unknown")).lower().strip().replace(" ", "_")
        category = self._map_category(category)

        attrs = ent.get("attributes") or []
        if isinstance(attrs, str):
            attrs = [attrs]

        norm_attrs = []
        for a in attrs:
            # Occasionally model outputs "moving|parked|..."
            parts = str(a).lower().strip().replace(" ", "_").split("|")
            for p in parts:
                p = p.strip()
                if p in {"", "none", "null", "n/a"}:
                    continue
                if p in ALLOWED_STATUSES and p not in norm_attrs and p != "unknown":
                    norm_attrs.append(p)

        return {
            "category": category,
            "attributes": norm_attrs,
        }

    def _map_category(self, category: str) -> str:
        mapping = {
            "trafficcone": "traffic_cone",
            "traffic_cone": "traffic_cone",
            "cone": "traffic_cone",
            "person": "pedestrian",
            "people": "pedestrian",
            "human": "pedestrian",
            "adult": "pedestrian",
            "child": "pedestrian",
            "vehicle": "car",
            "construction": "construction_vehicle",
            "construction_vehicle": "construction_vehicle",
        }
        if category in ALLOWED_CATEGORIES:
            return category
        return mapping.get(category, "unknown")

    def _normalize_relations(self, relations):
        out = []
        for r in relations:
            # split accidental "front|same-status-as"
            for part in str(r).lower().strip().replace(" ", "-").split("|"):
                part = part.strip()
                if part in {"", "unknown", "none", "null", "n/a"}:
                    continue
                if part in ALLOWED_RELATIONS and part not in out:
                    out.append(part)
        return out

    def _normalize_modules(self, modules):
        out = []
        for m in modules:
            # split accidental "visual_tool|scene_graph_retriever"
            for part in str(m).lower().strip().split("|"):
                part = part.strip()
                if part in {"", "unknown", "none", "null", "n/a"}:
                    continue
                if part in ALLOWED_MODULES and part not in out:
                    out.append(part)
        return out

    def _validate_plan(self, plan: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if plan["question_type"] not in ALLOWED_QUESTION_TYPES:
            return False, "invalid_question_type"
        if plan["answer_type"] not in ALLOWED_ANSWER_TYPES:
            return False, "invalid_answer_type"
        if plan["hop"] not in ALLOWED_HOPS:
            return False, "invalid_hop"
        if not isinstance(plan["target_entity"], dict):
            return False, "invalid_target_entity"
        if not isinstance(plan["relations"], list):
            return False, "invalid_relations"
        if not isinstance(plan["need_modules"], list):
            return False, "invalid_need_modules"
        if len(plan["need_modules"]) == 0:
            return False, "empty_need_modules"
        return True, None

    def _default_plan(self, question: str, raw: str = "", error: str = "", exemplar_context_used: bool = False) -> Dict[str, Any]:
        return {
            "question": question,
            "question_type": "unknown",
            "answer_type": "text",
            "hop": "unknown",
            "target_entity": {"category": "unknown", "attributes": []},
            "reference_entity": None,
            "relations": [],
            "need_modules": ["visual_tool", "result_parser"],
            "reasoning_path": ["visual_tool", "result_parser"],
            "why": "Fallback default plan.",
            "planner_source": "default_fallback_after_mllm_exemplar" if exemplar_context_used else "default_fallback",
            "planner_raw": raw,
            "planner_error": error,
            "exemplar_context_used": exemplar_context_used,
        }
