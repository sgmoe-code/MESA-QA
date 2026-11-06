"""Central experiment switches for NuScenes-QA agent runs."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, Set


AGENT_MODES = {"schema", "atomic_program", "react_atomic", "hybrid"}
PLANNER_MODES = {"rule", "mllm", "mllm_exemplar", "mllm_self_triggered", "mllm_self_triggered_controller", "mllm_drivelm", "mllm_minimal_controller", "mllm_minimal_controller_passthrough", "mllm_minimal_controller_decision_only", "mllm_tool_controller"}
RAG_MODES = {"none", "qa_exemplar", "controller_qa", "minimal_controller", "tool_controller", "drivelm", "drivelm_reasoning", "qa_exemplar_drivelm", "pred_scene", "hybrid", "oracle_scene_debug"}
SCENE_PROFILE_MODES = {"none", "structured", "visual", "hybrid"}
EVIDENCE_MODES = {"oracle_scene_graph", "retrieved_scene", "full_scene_text"}
VERIFIER_MODES = {"none", "result_parser", "mllm_verifier"}
QA_EXEMPLAR_CONTEXT_MODES = {"none", "plain", "sparc_light"}
QA_EXEMPLAR_QUERY_MODES = {"single", "multi"}


PROJECT_ROOT = Path(os.environ.get("MESA_QA_ROOT", Path(__file__).resolve().parent))
DEFAULT_MODEL_PATH = os.environ.get("MESA_QA_MODEL_PATH", str(PROJECT_ROOT / "models" / "Qwen3.5-9B"))
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "questions" / "mini_val_for_agent.json")
DEFAULT_DATAROOT = os.environ.get("NUSCENES_ROOT", str(PROJECT_ROOT / "data" / "nuscenes"))
DEFAULT_VERSION = "v1.0-mini"
DEFAULT_OUTPUT_ROOT = os.environ.get("MESA_QA_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))
DEFAULT_QA_EXEMPLAR_TRAIN_PATH = str(PROJECT_ROOT / "data" / "questions" / "NuScenes_train_questions.json")
DEFAULT_QA_EXEMPLAR_INDEX_PATH = str(PROJECT_ROOT / "cache" / "qa_exemplar_index.jsonl")
DEFAULT_DRIVELM_JSON_PATH = str(PROJECT_ROOT / "data" / "DriveLM" / "v1_1_train_nus.json")
DEFAULT_DRIVELM_INDEX_PATH = str(PROJECT_ROOT / "cache" / "drivelm_reasoning_index.jsonl")
DEFAULT_DRIVELM_COMPACT_INDEX_PATH = str(PROJECT_ROOT / "cache" / "drivelm_reasoning_compact_index.jsonl")


@dataclass
class ExperimentConfig:
    model_path: str
    data_path: str
    dataroot: str
    version: str

    max_samples: int
    agent_mode: str

    planner_mode: str
    rag_mode: str
    scene_profile_mode: str
    evidence_mode: str
    verifier_mode: str

    use_template_hint: bool
    scene_rag_topk_objects: int
    scene_rag_topk_relations: int
    scene_rag_topk_summaries: int
    scene_rag_include_relations: bool
    scene_rag_include_objects: bool
    scene_rag_debug: bool
    scene_rag_max_relation_chunks: int
    allow_oracle_rag: bool

    qa_exemplar_train_path: str
    qa_exemplar_index_path: str
    qa_exemplar_topk: int
    qa_exemplar_use_answer: bool
    qa_exemplar_use_template: bool
    qa_exemplar_use_plan: bool
    qa_exemplar_use_reasoning_program: bool
    qa_exemplar_rebuild_index: bool
    qa_exemplar_debug: bool
    qa_exemplar_context_mode: str
    qa_exemplar_final_topk: int
    qa_exemplar_raw_topk_per_query: int
    qa_exemplar_multi_query: bool
    qa_exemplar_query_mode: str
    qa_exemplar_filter_same_type: bool
    qa_exemplar_filter_same_template: bool
    qa_exemplar_min_score: float
    qa_exemplar_context_strategy: str
    qa_exemplar_max_per_template: int
    qa_exemplar_include_filtered_trace: bool
    exemplar_context_max_chars: int
    self_triggered_rag_source: str
    self_triggered_default_retrieve: bool
    self_triggered_max_decision_retries: int
    self_triggered_allow_comparison_rag: bool
    scene_profile_enabled: bool
    scene_profile_trace_only: bool
    scene_aware_self_trigger_enabled: bool
    scene_query_extension_enabled: bool
    scene_query_extension_trace_only: bool
    scene_query_extension_max_chars: int
    scene_query_extension_source: str
    scene_guard_shadow_enabled: bool
    scene_guard_source: str
    scene_guard_max_tokens: int
    scene_guard_active_routing_enabled: bool
    scene_strategy_enabled: bool
    scene_strategy_trace_only: bool
    scene_strategy_source: str
    scene_strategy_max_tokens: int
    scene_profile_source: str
    scene_decision_context_max_chars: int
    front_camera_image_root: str
    front_camera_scene_profile_cache_dir: str
    local_qw_model_path: str
    local_qw_device: str
    local_qw_max_new_tokens: int
    local_qw_temperature: float
    controller_qa_enabled: bool
    controller_banker_enabled: bool
    controller_banker_trace_only: bool
    controller_banker_append_one: bool
    controller_banker_max_tokens: int
    controller_max_context_chars: int
    controller_force_exact_qa_context: bool
    minimal_controller_enabled: bool
    minimal_controller_use_prior: bool
    minimal_controller_use_scene_cues: bool
    minimal_controller_max_priors: int
    minimal_controller_max_context_chars: int
    minimal_controller_enable_comparison_context: bool
    minimal_controller_prior_trace_only: bool
    minimal_controller_scene_trace_only: bool
    tool_controller_use_banker: bool
    tool_controller_qa_only: bool
    tool_controller_banker_trace_only: bool
    tool_controller_disable_comparison: bool
    tool_controller_max_priors: int
    tool_controller_max_context_chars: int
    tool_controller_default_qa_topk: int

    drivelm_json_path: str
    drivelm_index_path: str
    drivelm_compact_index_path: str
    drivelm_keep_raw_answer: bool
    drivelm_max_units_per_qa: int
    drivelm_debug_topk: int
    drivelm_topk: int
    drivelm_raw_topk: int
    drivelm_max_hints: int
    drivelm_max_context_chars: int
    drivelm_allowed_unit_types: list[str]
    drivelm_allow_comparison_rag: bool
    drivelm_enable_plan_guard: bool
    drivelm_enable_answer_masking: bool
    drivelm_enable_hint_abstraction: bool
    drivelm_enable_overlap_guard: bool
    drivelm_min_score: float
    drivelm_load_mode: str

    output_dir: str
    result_jsonl: str
    metric_json: str
    change_summary_md: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def switch_dict(self) -> Dict[str, Any]:
        return {
            "agent_mode": self.agent_mode,
            "planner_mode": self.planner_mode,
            "rag_mode": self.rag_mode,
            "scene_profile_mode": self.scene_profile_mode,
            "evidence_mode": self.evidence_mode,
            "verifier_mode": self.verifier_mode,
            "max_samples": self.max_samples,
            "scene_rag_topk_objects": self.scene_rag_topk_objects,
            "scene_rag_topk_relations": self.scene_rag_topk_relations,
            "scene_rag_topk_summaries": self.scene_rag_topk_summaries,
            "scene_rag_include_relations": self.scene_rag_include_relations,
            "scene_rag_include_objects": self.scene_rag_include_objects,
            "scene_rag_debug": self.scene_rag_debug,
            "scene_rag_max_relation_chunks": self.scene_rag_max_relation_chunks,
            "allow_oracle_rag": self.allow_oracle_rag,
            "qa_exemplar_train_path": self.qa_exemplar_train_path,
            "qa_exemplar_index_path": self.qa_exemplar_index_path,
            "qa_exemplar_topk": self.qa_exemplar_topk,
            "qa_exemplar_use_answer": self.qa_exemplar_use_answer,
            "qa_exemplar_use_template": self.qa_exemplar_use_template,
            "qa_exemplar_use_plan": self.qa_exemplar_use_plan,
            "qa_exemplar_use_reasoning_program": self.qa_exemplar_use_reasoning_program,
            "qa_exemplar_rebuild_index": self.qa_exemplar_rebuild_index,
            "qa_exemplar_debug": self.qa_exemplar_debug,
            "qa_exemplar_context_mode": self.qa_exemplar_context_mode,
            "qa_exemplar_final_topk": self.qa_exemplar_final_topk,
            "qa_exemplar_raw_topk_per_query": self.qa_exemplar_raw_topk_per_query,
            "qa_exemplar_multi_query": self.qa_exemplar_multi_query,
            "qa_exemplar_query_mode": self.qa_exemplar_query_mode,
            "qa_exemplar_filter_same_type": self.qa_exemplar_filter_same_type,
            "qa_exemplar_filter_same_template": self.qa_exemplar_filter_same_template,
            "qa_exemplar_min_score": self.qa_exemplar_min_score,
            "qa_exemplar_context_strategy": self.qa_exemplar_context_strategy,
            "qa_exemplar_max_per_template": self.qa_exemplar_max_per_template,
            "qa_exemplar_include_filtered_trace": self.qa_exemplar_include_filtered_trace,
            "exemplar_context_max_chars": self.exemplar_context_max_chars,
            "self_triggered_rag_source": self.self_triggered_rag_source,
            "self_triggered_default_retrieve": self.self_triggered_default_retrieve,
            "self_triggered_max_decision_retries": self.self_triggered_max_decision_retries,
            "self_triggered_allow_comparison_rag": self.self_triggered_allow_comparison_rag,
            "scene_profile_enabled": self.scene_profile_enabled,
            "scene_profile_trace_only": self.scene_profile_trace_only,
            "scene_aware_self_trigger_enabled": self.scene_aware_self_trigger_enabled,
            "scene_query_extension_enabled": self.scene_query_extension_enabled,
            "scene_query_extension_trace_only": self.scene_query_extension_trace_only,
            "scene_query_extension_max_chars": self.scene_query_extension_max_chars,
            "scene_query_extension_source": self.scene_query_extension_source,
            "scene_guard_shadow_enabled": self.scene_guard_shadow_enabled,
            "scene_guard_source": self.scene_guard_source,
            "scene_guard_max_tokens": self.scene_guard_max_tokens,
            "scene_guard_active_routing_enabled": self.scene_guard_active_routing_enabled,
            "scene_strategy_enabled": self.scene_strategy_enabled,
            "scene_strategy_trace_only": self.scene_strategy_trace_only,
            "scene_strategy_source": self.scene_strategy_source,
            "scene_strategy_max_tokens": self.scene_strategy_max_tokens,
            "scene_profile_source": self.scene_profile_source,
            "scene_decision_context_max_chars": self.scene_decision_context_max_chars,
            "front_camera_image_root": self.front_camera_image_root,
            "front_camera_scene_profile_cache_dir": self.front_camera_scene_profile_cache_dir,
            "local_qw_model_path": self.local_qw_model_path,
            "local_qw_device": self.local_qw_device,
            "local_qw_max_new_tokens": self.local_qw_max_new_tokens,
            "local_qw_temperature": self.local_qw_temperature,
            "controller_qa_enabled": self.controller_qa_enabled,
            "controller_banker_enabled": self.controller_banker_enabled,
            "controller_banker_trace_only": self.controller_banker_trace_only,
            "controller_banker_append_one": self.controller_banker_append_one,
            "controller_banker_max_tokens": self.controller_banker_max_tokens,
            "controller_max_context_chars": self.controller_max_context_chars,
            "controller_force_exact_qa_context": self.controller_force_exact_qa_context,
            "minimal_controller_enabled": self.minimal_controller_enabled,
            "minimal_controller_use_prior": self.minimal_controller_use_prior,
            "minimal_controller_use_scene_cues": self.minimal_controller_use_scene_cues,
            "minimal_controller_max_priors": self.minimal_controller_max_priors,
            "minimal_controller_max_context_chars": self.minimal_controller_max_context_chars,
            "minimal_controller_enable_comparison_context": self.minimal_controller_enable_comparison_context,
            "minimal_controller_prior_trace_only": self.minimal_controller_prior_trace_only,
            "minimal_controller_scene_trace_only": self.minimal_controller_scene_trace_only,
            "tool_controller_use_banker": self.tool_controller_use_banker,
            "tool_controller_qa_only": self.tool_controller_qa_only,
            "tool_controller_banker_trace_only": self.tool_controller_banker_trace_only,
            "tool_controller_disable_comparison": self.tool_controller_disable_comparison,
            "tool_controller_max_priors": self.tool_controller_max_priors,
            "tool_controller_max_context_chars": self.tool_controller_max_context_chars,
            "tool_controller_default_qa_topk": self.tool_controller_default_qa_topk,
            "drivelm_json_path": self.drivelm_json_path,
            "drivelm_index_path": self.drivelm_index_path,
            "drivelm_compact_index_path": self.drivelm_compact_index_path,
            "drivelm_keep_raw_answer": self.drivelm_keep_raw_answer,
            "drivelm_max_units_per_qa": self.drivelm_max_units_per_qa,
            "drivelm_debug_topk": self.drivelm_debug_topk,
            "drivelm_topk": self.drivelm_topk,
            "drivelm_raw_topk": self.drivelm_raw_topk,
            "drivelm_max_hints": self.drivelm_max_hints,
            "drivelm_max_context_chars": self.drivelm_max_context_chars,
            "drivelm_allowed_unit_types": self.drivelm_allowed_unit_types,
            "drivelm_allow_comparison_rag": self.drivelm_allow_comparison_rag,
            "drivelm_enable_plan_guard": self.drivelm_enable_plan_guard,
            "drivelm_enable_answer_masking": self.drivelm_enable_answer_masking,
            "drivelm_enable_hint_abstraction": self.drivelm_enable_hint_abstraction,
            "drivelm_enable_overlap_guard": self.drivelm_enable_overlap_guard,
            "drivelm_min_score": self.drivelm_min_score,
            "drivelm_load_mode": self.drivelm_load_mode,
        }


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_first(names: list[str], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
    return default


def _parse_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _parse_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _parse_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _parse_int_first(names: list[str], default: int) -> int:
    label = "/".join(names)
    raw = _env_first(names, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


def _parse_float_first(names: list[str], default: float) -> float:
    label = "/".join(names)
    raw = _env_first(names, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


def _validate_choice(name: str, value: str, allowed: Set[str]) -> str:
    normalized = value.strip().lower()
    if name == "RAG_MODE" and normalized == "scene":
        raise ValueError(
            "RAG_MODE=\'scene\' has been retired because it uses GT-derived scene_rows. "
            "Use RAG_MODE=oracle_scene_debug with ALLOW_ORACLE_RAG=1 for debug-only oracle retrieval."
        )
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {name}={value!r}. Allowed values: {choices}")
    return normalized


def load_config() -> ExperimentConfig:
    model_path = _env("MODEL_PATH", DEFAULT_MODEL_PATH)
    data_path = _env("DATA_PATH", DEFAULT_DATA_PATH)
    dataroot = _env("DATAROOT", DEFAULT_DATAROOT)
    version = _env("NUSCENES_VERSION", DEFAULT_VERSION)

    max_samples = _parse_int("MAX_SAMPLES", 50)
    agent_mode = _validate_choice("AGENT_MODE", _env("AGENT_MODE", "atomic_program"), AGENT_MODES)
    planner_mode = _validate_choice("PLANNER_MODE", _env("PLANNER_MODE", "mllm"), PLANNER_MODES)
    rag_mode = _validate_choice("RAG_MODE", _env("RAG_MODE", "none"), RAG_MODES)
    if planner_mode == "mllm_exemplar" and rag_mode not in {"qa_exemplar", "hybrid"}:
        raise ValueError("PLANNER_MODE=mllm_exemplar requires RAG_MODE=qa_exemplar or RAG_MODE=hybrid.")
    if planner_mode == "mllm_self_triggered" and rag_mode not in {"qa_exemplar", "hybrid"}:
        raise ValueError("PLANNER_MODE=mllm_self_triggered requires RAG_MODE=qa_exemplar or RAG_MODE=hybrid.")
    if planner_mode == "mllm_self_triggered_controller" and rag_mode != "controller_qa":
        raise ValueError("PLANNER_MODE=mllm_self_triggered_controller requires RAG_MODE=controller_qa.")
    if planner_mode == "mllm_drivelm" and rag_mode != "drivelm_reasoning":
        raise ValueError("PLANNER_MODE=mllm_drivelm requires RAG_MODE=drivelm_reasoning.")
    if planner_mode.startswith("mllm_minimal_controller") and rag_mode != "minimal_controller":
        raise ValueError("PLANNER_MODE=mllm_minimal_controller* requires RAG_MODE=minimal_controller.")
    if planner_mode == "mllm_tool_controller" and rag_mode != "tool_controller":
        raise ValueError("PLANNER_MODE=mllm_tool_controller requires RAG_MODE=tool_controller.")
    scene_profile_mode = _validate_choice(
        "SCENE_PROFILE_MODE",
        _env("SCENE_PROFILE_MODE", "none"),
        SCENE_PROFILE_MODES,
    )
    evidence_mode = _validate_choice(
        "EVIDENCE_MODE",
        _env("EVIDENCE_MODE", "oracle_scene_graph"),
        EVIDENCE_MODES,
    )
    verifier_mode = _validate_choice(
        "VERIFIER_MODE",
        _env("VERIFIER_MODE", "result_parser"),
        VERIFIER_MODES,
    )
    use_template_hint = _parse_bool("USE_TEMPLATE_HINT", False)
    scene_rag_topk_objects = _parse_int("SCENE_RAG_TOPK_OBJECTS", 20)
    scene_rag_topk_relations = _parse_int("SCENE_RAG_TOPK_RELATIONS", 20)
    scene_rag_topk_summaries = _parse_int("SCENE_RAG_TOPK_SUMMARIES", 3)
    scene_rag_include_relations = _parse_bool("SCENE_RAG_INCLUDE_RELATIONS", True)
    scene_rag_include_objects = _parse_bool("SCENE_RAG_INCLUDE_OBJECTS", True)
    scene_rag_debug = _parse_bool("SCENE_RAG_DEBUG", False)
    scene_rag_max_relation_chunks = _parse_int("SCENE_RAG_MAX_RELATION_CHUNKS", 3000)
    allow_oracle_rag = _parse_bool("ALLOW_ORACLE_RAG", False)
    if rag_mode == "oracle_scene_debug" and not allow_oracle_rag:
        raise ValueError(
            "RAG_MODE=oracle_scene_debug uses oracle scene_rows and is only allowed when ALLOW_ORACLE_RAG=1."
        )

    qa_exemplar_train_path = _env("QA_EXEMPLAR_TRAIN_PATH", DEFAULT_QA_EXEMPLAR_TRAIN_PATH)
    qa_exemplar_index_path = _env("QA_EXEMPLAR_INDEX_PATH", DEFAULT_QA_EXEMPLAR_INDEX_PATH)
    qa_exemplar_topk = _parse_int("QA_EXEMPLAR_TOPK", 3)
    qa_exemplar_use_answer = _parse_bool("QA_EXEMPLAR_USE_ANSWER", False)
    qa_exemplar_use_template = _parse_bool("QA_EXEMPLAR_USE_TEMPLATE", True)
    qa_exemplar_use_plan = _parse_bool("QA_EXEMPLAR_USE_PLAN", True)
    qa_exemplar_use_reasoning_program = _parse_bool("QA_EXEMPLAR_USE_REASONING_PROGRAM", True)
    qa_exemplar_rebuild_index = _parse_bool("QA_EXEMPLAR_REBUILD_INDEX", False)
    qa_exemplar_debug = _parse_bool("QA_EXEMPLAR_DEBUG", False)
    qa_exemplar_context_mode = _validate_choice(
        "RAG_CONTEXT_MODE/QA_EXEMPLAR_CONTEXT_MODE",
        _env_first(["RAG_CONTEXT_MODE", "QA_EXEMPLAR_CONTEXT_MODE"], "none"),
        QA_EXEMPLAR_CONTEXT_MODES,
    )
    qa_exemplar_final_topk = _parse_int("QA_EXEMPLAR_FINAL_TOPK", qa_exemplar_topk)
    qa_exemplar_raw_topk_per_query = _parse_int("QA_EXEMPLAR_RAW_TOPK_PER_QUERY", qa_exemplar_topk)
    if qa_exemplar_raw_topk_per_query < 1:
        raise ValueError(f"QA_EXEMPLAR_RAW_TOPK_PER_QUERY must be >= 1, got {qa_exemplar_raw_topk_per_query}")
    qa_exemplar_query_mode = _validate_choice(
        "QA_EXEMPLAR_QUERY_MODE",
        _env("QA_EXEMPLAR_QUERY_MODE", "multi" if _parse_bool("QA_EXEMPLAR_MULTI_QUERY", False) else "single"),
        QA_EXEMPLAR_QUERY_MODES,
    )
    qa_exemplar_multi_query = qa_exemplar_query_mode == "multi"
    qa_exemplar_filter_same_type = _parse_bool("QA_EXEMPLAR_FILTER_SAME_TYPE", False)
    qa_exemplar_filter_same_template = _parse_bool("QA_EXEMPLAR_FILTER_SAME_TEMPLATE", False)
    qa_exemplar_min_score = _parse_float_first(["QA_EXEMPLAR_SCORE_THRESHOLD", "QA_EXEMPLAR_MIN_SCORE"], 0.0)
    qa_exemplar_context_strategy = _env(
        "QA_EXEMPLAR_CONTEXT_STRATEGY",
        "sparc_light" if qa_exemplar_context_mode == "sparc_light" else "topk",
    ).strip().lower()
    qa_exemplar_max_per_template = _parse_int("QA_EXEMPLAR_MAX_PER_TEMPLATE", max(1, qa_exemplar_final_topk))
    if qa_exemplar_max_per_template < 1:
        raise ValueError(f"QA_EXEMPLAR_MAX_PER_TEMPLATE must be >= 1, got {qa_exemplar_max_per_template}")
    qa_exemplar_include_filtered_trace = _parse_bool("QA_EXEMPLAR_INCLUDE_FILTERED_TRACE", True)
    if qa_exemplar_context_mode == "sparc_light" and rag_mode not in {"qa_exemplar", "controller_qa", "hybrid"}:
        raise ValueError("QA_EXEMPLAR_CONTEXT_MODE=sparc_light requires RAG_MODE=qa_exemplar, RAG_MODE=controller_qa, or RAG_MODE=hybrid.")
    exemplar_context_max_chars = _parse_int("EXEMPLAR_CONTEXT_MAX_CHARS", 4000)
    self_triggered_rag_source = _validate_choice(
        "SELF_TRIGGERED_RAG_SOURCE",
        _env("SELF_TRIGGERED_RAG_SOURCE", "qa_exemplar"),
        {"qa_exemplar"},
    )
    self_triggered_default_retrieve = _parse_bool("SELF_TRIGGERED_DEFAULT_RETRIEVE", False)
    self_triggered_max_decision_retries = _parse_int("SELF_TRIGGERED_MAX_DECISION_RETRIES", 1)
    self_triggered_allow_comparison_rag = _parse_bool("SELF_TRIGGERED_ALLOW_COMPARISON_RAG", False)
    scene_profile_enabled = _parse_bool("SCENE_PROFILE_ENABLED", False)
    scene_profile_trace_only = _parse_bool("SCENE_PROFILE_TRACE_ONLY", True)
    scene_aware_self_trigger_enabled = _parse_bool("SCENE_AWARE_SELF_TRIGGER_ENABLED", False)
    scene_query_extension_enabled = _parse_bool("SCENE_QUERY_EXTENSION_ENABLED", False)
    scene_query_extension_trace_only = _parse_bool("SCENE_QUERY_EXTENSION_TRACE_ONLY", True)
    scene_query_extension_max_chars = _parse_int("SCENE_QUERY_EXTENSION_MAX_CHARS", 700)
    scene_query_extension_source = _env("SCENE_QUERY_EXTENSION_SOURCE", "local_qw")
    scene_guard_shadow_enabled = _parse_bool("SCENE_GUARD_SHADOW_ENABLED", False)
    scene_guard_source = _env("SCENE_GUARD_SOURCE", "local_qw")
    scene_guard_max_tokens = _parse_int("SCENE_GUARD_MAX_TOKENS", 1)
    scene_guard_active_routing_enabled = _parse_bool("SCENE_GUARD_ACTIVE_ROUTING_ENABLED", False)
    if scene_guard_active_routing_enabled:
        raise ValueError("SCENE_GUARD_ACTIVE_ROUTING_ENABLED must remain False in Step 3.8.5.")
    scene_strategy_enabled = _parse_bool("SCENE_STRATEGY_ENABLED", False)
    scene_strategy_trace_only = _parse_bool("SCENE_STRATEGY_TRACE_ONLY", True)
    scene_strategy_source = _env("SCENE_STRATEGY_SOURCE", "local_qw")
    scene_strategy_max_tokens = _parse_int("SCENE_STRATEGY_MAX_TOKENS", 1)
    if scene_strategy_max_tokens != 1:
        raise ValueError("SCENE_STRATEGY_MAX_TOKENS must be 1 in Step 3.10.")
    scene_profile_source = _env("SCENE_PROFILE_SOURCE", "visual_mllm")
    scene_decision_context_max_chars = _parse_int("SCENE_DECISION_CONTEXT_MAX_CHARS", 500)
    front_camera_image_root = _env("FRONT_CAMERA_IMAGE_ROOT", "")
    front_camera_scene_profile_cache_dir = _env("FRONT_CAMERA_SCENE_PROFILE_CACHE_DIR", "cache/front_camera_scene_profiles")
    local_qw_model_path = _env("LOCAL_QW_MODEL_PATH", model_path)
    local_qw_device = _env("LOCAL_QW_DEVICE", "cuda")
    local_qw_max_new_tokens = _parse_int("LOCAL_QW_MAX_NEW_TOKENS", 180)
    local_qw_temperature = _parse_float("LOCAL_QW_TEMPERATURE", 0.0)
    controller_qa_enabled = _parse_bool("CONTROLLER_QA_ENABLED", True)
    controller_banker_enabled = _parse_bool("CONTROLLER_BANKER_ENABLED", False)
    controller_banker_trace_only = _parse_bool("CONTROLLER_BANKER_TRACE_ONLY", True)
    controller_banker_append_one = _parse_bool("CONTROLLER_BANKER_APPEND_ONE", False)
    controller_banker_max_tokens = _parse_int("CONTROLLER_BANKER_MAX_TOKENS", 1)
    controller_max_context_chars = _parse_int("CONTROLLER_MAX_CONTEXT_CHARS", 1200)
    controller_force_exact_qa_context = _parse_bool("CONTROLLER_FORCE_EXACT_QA_CONTEXT", False)
    minimal_controller_enabled = _parse_bool("MINIMAL_CONTROLLER_ENABLED", False)
    minimal_controller_use_prior = _parse_bool("MINIMAL_CONTROLLER_USE_PRIOR", True)
    minimal_controller_use_scene_cues = _parse_bool("MINIMAL_CONTROLLER_USE_SCENE_CUES", True)
    minimal_controller_max_priors = _parse_int("MINIMAL_CONTROLLER_MAX_PRIORS", 3)
    minimal_controller_max_context_chars = _parse_int("MINIMAL_CONTROLLER_MAX_CONTEXT_CHARS", 1200)
    minimal_controller_enable_comparison_context = _parse_bool("MINIMAL_CONTROLLER_ENABLE_COMPARISON_CONTEXT", False)
    minimal_controller_prior_trace_only = _parse_bool("MINIMAL_CONTROLLER_PRIOR_TRACE_ONLY", False)
    minimal_controller_scene_trace_only = _parse_bool("MINIMAL_CONTROLLER_SCENE_TRACE_ONLY", False)
    tool_controller_use_banker = _parse_bool("TOOL_CONTROLLER_USE_BANKER", True)
    tool_controller_qa_only = _parse_bool("TOOL_CONTROLLER_QA_ONLY", False)
    tool_controller_banker_trace_only = _parse_bool("TOOL_CONTROLLER_BANKER_TRACE_ONLY", False)
    tool_controller_disable_comparison = _parse_bool("TOOL_CONTROLLER_DISABLE_COMPARISON", True)
    tool_controller_max_priors = _parse_int("TOOL_CONTROLLER_MAX_PRIORS", 2)
    tool_controller_max_context_chars = _parse_int("TOOL_CONTROLLER_MAX_CONTEXT_CHARS", 1200)
    tool_controller_default_qa_topk = _parse_int("TOOL_CONTROLLER_DEFAULT_QA_TOPK", 3)

    drivelm_json_path = _env("DRIVELM_JSON_PATH", DEFAULT_DRIVELM_JSON_PATH)
    drivelm_index_path = _env("DRIVELM_INDEX_PATH", DEFAULT_DRIVELM_INDEX_PATH)
    drivelm_compact_index_path = _env("DRIVELM_COMPACT_INDEX_PATH", DEFAULT_DRIVELM_COMPACT_INDEX_PATH)
    drivelm_keep_raw_answer = _parse_bool("DRIVELM_KEEP_RAW_ANSWER", False)
    drivelm_max_units_per_qa = _parse_int("DRIVELM_MAX_UNITS_PER_QA", 5)
    drivelm_debug_topk = _parse_int("DRIVELM_DEBUG_TOPK", 5)
    drivelm_topk = _parse_int("DRIVELM_TOPK", 3)
    drivelm_raw_topk = _parse_int("DRIVELM_RAW_TOPK", 20)
    drivelm_max_hints = _parse_int("DRIVELM_MAX_HINTS", 2)
    drivelm_max_context_chars = _parse_int("DRIVELM_MAX_CONTEXT_CHARS", 1800)
    drivelm_allowed_unit_types = [
        item.strip()
        for item in _env("DRIVELM_ALLOWED_UNIT_TYPES", "relation_centric,state_behavior,interaction_risk,object_centric").split(",")
        if item.strip()
    ]
    drivelm_allow_comparison_rag = _parse_bool("DRIVELM_ALLOW_COMPARISON_RAG", False)
    drivelm_enable_plan_guard = _parse_bool("DRIVELM_ENABLE_PLAN_GUARD", True)
    drivelm_enable_answer_masking = _parse_bool("DRIVELM_ENABLE_ANSWER_MASKING", True)
    drivelm_enable_hint_abstraction = _parse_bool("DRIVELM_ENABLE_HINT_ABSTRACTION", True)
    drivelm_enable_overlap_guard = _parse_bool("DRIVELM_ENABLE_OVERLAP_GUARD", True)
    drivelm_min_score = _parse_float("DRIVELM_MIN_SCORE", 0.0)
    drivelm_load_mode = _env("DRIVELM_LOAD_MODE", "compact").strip().lower()
    if drivelm_load_mode not in {"compact", "full"}:
        raise ValueError(f"DRIVELM_LOAD_MODE must be compact or full, got {drivelm_load_mode!r}")

    output_root = _env("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    run_tag = os.environ.get("RUN_TAG", "").strip()
    run_name = f"agent_{agent_mode}_{max_samples}"
    if run_tag:
        run_name = f"{run_name}_{run_tag}"

    output_dir = os.path.join(output_root, run_name)
    result_jsonl = os.path.join(output_dir, f"{agent_mode}_results.jsonl")
    metric_json = os.path.join(output_dir, f"{agent_mode}_metrics.json")
    change_summary_md = os.path.join(output_dir, "change_summary.md")

    return ExperimentConfig(
        model_path=model_path,
        data_path=data_path,
        dataroot=dataroot,
        version=version,
        max_samples=max_samples,
        agent_mode=agent_mode,
        planner_mode=planner_mode,
        rag_mode=rag_mode,
        scene_profile_mode=scene_profile_mode,
        evidence_mode=evidence_mode,
        verifier_mode=verifier_mode,
        use_template_hint=use_template_hint,
        scene_rag_topk_objects=scene_rag_topk_objects,
        scene_rag_topk_relations=scene_rag_topk_relations,
        scene_rag_topk_summaries=scene_rag_topk_summaries,
        scene_rag_include_relations=scene_rag_include_relations,
        scene_rag_include_objects=scene_rag_include_objects,
        scene_rag_debug=scene_rag_debug,
        scene_rag_max_relation_chunks=scene_rag_max_relation_chunks,
        allow_oracle_rag=allow_oracle_rag,
        qa_exemplar_train_path=qa_exemplar_train_path,
        qa_exemplar_index_path=qa_exemplar_index_path,
        qa_exemplar_topk=qa_exemplar_topk,
        qa_exemplar_use_answer=qa_exemplar_use_answer,
        qa_exemplar_use_template=qa_exemplar_use_template,
        qa_exemplar_use_plan=qa_exemplar_use_plan,
        qa_exemplar_use_reasoning_program=qa_exemplar_use_reasoning_program,
        qa_exemplar_rebuild_index=qa_exemplar_rebuild_index,
        qa_exemplar_debug=qa_exemplar_debug,
        qa_exemplar_context_mode=qa_exemplar_context_mode,
        qa_exemplar_final_topk=qa_exemplar_final_topk,
        qa_exemplar_raw_topk_per_query=qa_exemplar_raw_topk_per_query,
        qa_exemplar_multi_query=qa_exemplar_multi_query,
        qa_exemplar_query_mode=qa_exemplar_query_mode,
        qa_exemplar_filter_same_type=qa_exemplar_filter_same_type,
        qa_exemplar_filter_same_template=qa_exemplar_filter_same_template,
        qa_exemplar_min_score=qa_exemplar_min_score,
        qa_exemplar_context_strategy=qa_exemplar_context_strategy,
        qa_exemplar_max_per_template=qa_exemplar_max_per_template,
        qa_exemplar_include_filtered_trace=qa_exemplar_include_filtered_trace,
        exemplar_context_max_chars=exemplar_context_max_chars,
        self_triggered_rag_source=self_triggered_rag_source,
        self_triggered_default_retrieve=self_triggered_default_retrieve,
        self_triggered_max_decision_retries=self_triggered_max_decision_retries,
        self_triggered_allow_comparison_rag=self_triggered_allow_comparison_rag,
        scene_profile_enabled=scene_profile_enabled,
        scene_profile_trace_only=scene_profile_trace_only,
        scene_aware_self_trigger_enabled=scene_aware_self_trigger_enabled,
        scene_query_extension_enabled=scene_query_extension_enabled,
        scene_query_extension_trace_only=scene_query_extension_trace_only,
        scene_query_extension_max_chars=scene_query_extension_max_chars,
        scene_query_extension_source=scene_query_extension_source,
        scene_guard_shadow_enabled=scene_guard_shadow_enabled,
        scene_guard_source=scene_guard_source,
        scene_guard_max_tokens=scene_guard_max_tokens,
        scene_guard_active_routing_enabled=scene_guard_active_routing_enabled,
        scene_strategy_enabled=scene_strategy_enabled,
        scene_strategy_trace_only=scene_strategy_trace_only,
        scene_strategy_source=scene_strategy_source,
        scene_strategy_max_tokens=scene_strategy_max_tokens,
        scene_profile_source=scene_profile_source,
        scene_decision_context_max_chars=scene_decision_context_max_chars,
        front_camera_image_root=front_camera_image_root,
        front_camera_scene_profile_cache_dir=front_camera_scene_profile_cache_dir,
        local_qw_model_path=local_qw_model_path,
        local_qw_device=local_qw_device,
        local_qw_max_new_tokens=local_qw_max_new_tokens,
        local_qw_temperature=local_qw_temperature,
        controller_qa_enabled=controller_qa_enabled,
        controller_banker_enabled=controller_banker_enabled,
        controller_banker_trace_only=controller_banker_trace_only,
        controller_banker_append_one=controller_banker_append_one,
        controller_banker_max_tokens=controller_banker_max_tokens,
        controller_max_context_chars=controller_max_context_chars,
        controller_force_exact_qa_context=controller_force_exact_qa_context,
        minimal_controller_enabled=minimal_controller_enabled,
        minimal_controller_use_prior=minimal_controller_use_prior,
        minimal_controller_use_scene_cues=minimal_controller_use_scene_cues,
        minimal_controller_max_priors=minimal_controller_max_priors,
        minimal_controller_max_context_chars=minimal_controller_max_context_chars,
        minimal_controller_enable_comparison_context=minimal_controller_enable_comparison_context,
        minimal_controller_prior_trace_only=minimal_controller_prior_trace_only,
        minimal_controller_scene_trace_only=minimal_controller_scene_trace_only,
        tool_controller_use_banker=tool_controller_use_banker,
        tool_controller_qa_only=tool_controller_qa_only,
        tool_controller_banker_trace_only=tool_controller_banker_trace_only,
        tool_controller_disable_comparison=tool_controller_disable_comparison,
        tool_controller_max_priors=tool_controller_max_priors,
        tool_controller_max_context_chars=tool_controller_max_context_chars,
        tool_controller_default_qa_topk=tool_controller_default_qa_topk,
        drivelm_json_path=drivelm_json_path,
        drivelm_index_path=drivelm_index_path,
        drivelm_compact_index_path=drivelm_compact_index_path,
        drivelm_keep_raw_answer=drivelm_keep_raw_answer,
        drivelm_max_units_per_qa=drivelm_max_units_per_qa,
        drivelm_debug_topk=drivelm_debug_topk,
        drivelm_topk=drivelm_topk,
        drivelm_raw_topk=drivelm_raw_topk,
        drivelm_max_hints=drivelm_max_hints,
        drivelm_max_context_chars=drivelm_max_context_chars,
        drivelm_allowed_unit_types=drivelm_allowed_unit_types,
        drivelm_allow_comparison_rag=drivelm_allow_comparison_rag,
        drivelm_enable_plan_guard=drivelm_enable_plan_guard,
        drivelm_enable_answer_masking=drivelm_enable_answer_masking,
        drivelm_enable_hint_abstraction=drivelm_enable_hint_abstraction,
        drivelm_enable_overlap_guard=drivelm_enable_overlap_guard,
        drivelm_min_score=drivelm_min_score,
        drivelm_load_mode=drivelm_load_mode,
        output_dir=output_dir,
        result_jsonl=result_jsonl,
        metric_json=metric_json,
        change_summary_md=change_summary_md,
    )
