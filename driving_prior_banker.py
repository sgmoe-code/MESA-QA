"""Tiny static DriveLM-inspired driving prior banker.

This module does not load DriveLM data and does not retrieve answers. It only
selects abstract planning constraints from a fixed local token bank.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


STATE_WORDS = {"moving", "parked", "stopped", "standing", "walking", "sitting"}
RELATION_KEYWORDS = ["front", "behind", "left", "right", "near", "next to", "beside", "of"]
COMPOUND_RELATIONS = {"front left", "front right", "back left", "back right", "front-left", "front-right", "back-left", "back-right"}
STEP37_ENABLED_TOKENS = {
    "<preserve_state_constraint>",
    "<reference_first>",
    "<preserve_compound_relation>",
    "<same_status_two_stage>",
}
STEP37_PRIORITY = [
    "<same_status_two_stage>",
    "<preserve_compound_relation>",
    "<preserve_state_constraint>",
    "<reference_first>",
]


BANKER_TOKENS = [
    {
        "token": "<preserve_state_constraint>",
        "rule": "Preserve explicit state words in the question when constructing the atomic program.",
        "prior_type": "state_disambiguation",
    },
    {
        "token": "<reference_first>",
        "rule": "Resolve the reference object first, then match target candidates by the requested relation.",
        "prior_type": "reference_resolution",
    },
    {
        "token": "<preserve_compound_relation>",
        "rule": "Preserve compound spatial relations such as front-left or front-right as one constraint.",
        "prior_type": "relation_preservation",
    },
    {
        "token": "<separate_target_reference>",
        "rule": "Keep target and reference objects separate; the object being asked about is not necessarily the object mentioned after \"of\".",
        "prior_type": "entity_role_control",
    },
    {
        "token": "<same_status_two_stage>",
        "rule": "For same-status questions, first identify the reference object's status, then match candidates with that status.",
        "prior_type": "state_disambiguation",
    },
    {
        "token": "<exclude_other_reference>",
        "rule": "When the question says \"other\", exclude the reference object itself from candidate targets.",
        "prior_type": "entity_role_control",
    },
    {
        "token": "<no_extra_constraint>",
        "rule": "Do not add relation or state constraints that are not explicitly required by the question.",
        "prior_type": "constraint_control",
    },
    {
        "token": "<thing_grounding>",
        "rule": "Resolve \"thing\" as an object candidate constrained by the current relation or state; do not default it to pedestrian or vehicle without scene evidence.",
        "prior_type": "entity_grounding",
    },
]
BANKER_BY_TOKEN = {item["token"]: item for item in BANKER_TOKENS}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _relation(task_analysis: Dict[str, Any]) -> str:
    relation = task_analysis.get("relation")
    if isinstance(relation, list):
        relation = " ".join(str(item) for item in relation)
    return _norm(relation)


def _question_type(task_analysis: Dict[str, Any]) -> str:
    return _norm(task_analysis.get("question_type"))


def _add(out: List[Dict[str, Any]], token: str, reason: str) -> None:
    item = dict(BANKER_BY_TOKEN[token])
    item["reasons"] = [reason]
    for existing in out:
        if existing.get("token") == token:
            if reason not in existing["reasons"]:
                existing["reasons"].append(reason)
            return
    out.append(item)


def select_banker_tokens_legacy(
    question: str,
    task_analysis: dict,
    max_tokens: int = 2,
    enable_for_comparison: bool = False,
) -> list[dict]:
    task_analysis = task_analysis if isinstance(task_analysis, dict) else {}
    q = _norm(question)
    qtype = _question_type(task_analysis)
    relation = _relation(task_analysis)
    if qtype == "comparison" and not enable_for_comparison:
        return []

    out: List[Dict[str, Any]] = []
    if "same status" in q:
        _add(out, "<same_status_two_stage>", "same_status")
    if "other" in q:
        _add(out, "<exclude_other_reference>", "other_object")
    if any(rel in q for rel in COMPOUND_RELATIONS) or any(rel in relation for rel in COMPOUND_RELATIONS):
        _add(out, "<preserve_compound_relation>", "compound_relation")
    if relation:
        _add(out, "<reference_first>", "relation_present")
    if "thing" in q and relation:
        _add(out, "<separate_target_reference>", "generic_thing_with_relation")
        _add(out, "<thing_grounding>", "generic_thing")
    if any(word in q for word in STATE_WORDS) or _norm(task_analysis.get("state")) in STATE_WORDS:
        _add(out, "<preserve_state_constraint>", "state_word")
    if qtype == "count" and not relation:
        out = [item for item in out if item.get("token") != "<reference_first>"]
        _add(out, "<no_extra_constraint>", "simple_count")

    return out[:max(0, int(max_tokens))]


def _has_relation(text: str) -> bool:
    non_of_keywords = [keyword for keyword in RELATION_KEYWORDS if keyword != "of"]
    if any(keyword in text for keyword in non_of_keywords):
        return True
    if " of " not in f" {text} ":
        return False
    simple_count_phrases = ("number of", "how many", "amount of")
    return not any(phrase in text for phrase in simple_count_phrases)


def has_banker_relation_requirement(question: str) -> bool:
    """Public submission-safe wrapper used by the frozen Phase 5C runner."""
    return _has_relation(_norm(question))


def select_banker_tokens(
    question: str,
    eval_type: Optional[str] = None,
    retrieve_decision: Optional[Dict[str, Any]] = None,
    max_tokens: int = 1,
    enabled_token_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    decision = dict(retrieve_decision or {})
    if decision.get("retrieve") is not True:
        return []

    q = _norm(question)
    qtype = _norm(eval_type)
    if qtype == "comparison":
        return []
    if qtype == "count" and not _has_relation(q):
        return []

    enabled = set(enabled_token_names or STEP37_ENABLED_TOKENS)
    candidates: Dict[str, Dict[str, Any]] = {}

    def add(token: str, reason: str, score: float) -> None:
        if token not in enabled:
            return
        item = dict(BANKER_BY_TOKEN[token])
        item["reasons"] = [reason]
        item["score"] = score
        candidates[token] = item

    if "same status" in q:
        add("<same_status_two_stage>", "same_status_detected", 4.0)
    if any(rel in q for rel in COMPOUND_RELATIONS):
        add("<preserve_compound_relation>", "compound_relation_detected", 3.5)
    if any(word in q for word in STATE_WORDS):
        add("<preserve_state_constraint>", "state_word_detected", 3.0)
    if _has_relation(q):
        add("<reference_first>", "relation_word_detected", 3.0)

    ordered = [candidates[token] for token in STEP37_PRIORITY if token in candidates]
    return ordered[: max(0, int(max_tokens))]
