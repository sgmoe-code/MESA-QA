"""Analysis-only question profiling for Step 3.9 RAG retrieval audit."""

from __future__ import annotations

import re
from typing import Any, Dict, List


RELATION_ALIASES = {
    "front left": ["front left", "front-left"],
    "front right": ["front right", "front-right"],
    "back left": ["back left", "back-left", "rear left", "rear-left"],
    "back right": ["back right", "back-right", "rear right", "rear-right"],
    "front": ["front", "in front", "ahead"],
    "behind": ["behind", "back", "rear"],
    "left": ["left"],
    "right": ["right"],
    "near": ["near", "nearby", "close", "next to", "beside"],
}
COMPOUND_RELATIONS = {"front left", "front right", "back left", "back right"}
STATUS_ALIASES = {
    "moving": ["moving", "driving", "proceeding"],
    "parked": ["parked", "parking"],
    "stopped": ["stopped", "stop", "not standing", "not_standing"],
    "standing": ["standing"],
    "walking": ["walking"],
    "sitting": ["sitting", "lying"],
    "turning": ["turning"],
    "waiting": ["waiting"],
}
OBJECT_ALIASES = {
    "car": ["car", "cars", "vehicle", "vehicles", "sedan", "suv", "van"],
    "truck": ["truck", "trucks"],
    "bus": ["bus", "buses"],
    "pedestrian": ["pedestrian", "pedestrians", "person", "people", "human"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike"],
    "traffic cone": ["traffic cone", "traffic cones", "cone", "cones"],
    "barrier": ["barrier", "barriers"],
    "construction vehicle": ["construction vehicle", "construction vehicles"],
    "trailer": ["trailer", "trailers"],
}
EVAL_TYPES = {"exist", "count", "object", "status", "comparison"}


def _norm(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _norm(phrase)
    if " " in phrase or "-" in phrase:
        return phrase in text
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _find_terms(text: str, aliases: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for canonical, forms in aliases.items():
        if any(_contains_phrase(text, form) for form in forms):
            out.append(canonical)
    return out


def _infer_eval_type(question: str, eval_type: str | None) -> str:
    value = _norm(eval_type).split("_")[0]
    if value in EVAL_TYPES:
        return value
    q = _norm(question)
    if q.startswith("how many") or "number of" in q or "amount of" in q:
        return "count"
    if q.startswith("is there") or q.startswith("are there") or "visible" in q:
        return "exist"
    if "same status" in q or "more than" in q or "less than" in q or "fewer than" in q:
        return "comparison"
    if "status" in q or "doing" in q:
        return "status"
    if q.startswith("what") or q.startswith("which"):
        return "object"
    return "unknown"


def infer_submission_safe_question_intent(question: str) -> str:
    """Infer intent from question text only, without benchmark metadata."""
    return _infer_eval_type(question, None)


def _reference_phrases(text: str, statuses: List[str], objects: List[str]) -> List[str]:
    phrases = []
    for status in statuses:
        for obj in objects:
            phrase = f"{status} {obj}"
            if _contains_phrase(text, phrase):
                phrases.append(phrase)
    for obj in objects:
        if _contains_phrase(text, f"the {obj}") or _contains_phrase(text, f"it"):
            phrases.append(obj)
    return sorted(set(phrases))


def _candidate_banker_token(
    q: str,
    eval_type: str,
    compound: List[str],
    statuses: List[str],
    has_reference: bool,
    has_relation: bool,
) -> str | None:
    if "same status" in q:
        return "<same_status_two_stage>"
    if compound:
        return "<preserve_compound_relation>"
    if statuses:
        return "<preserve_state_constraint>"
    if has_reference or (has_relation and eval_type != "count"):
        return "<reference_first>"
    return None


def _reasoning_signature(
    eval_type: str,
    has_status: bool,
    has_reference: bool,
    has_relation: bool,
    has_compound: bool,
    has_same_status: bool,
) -> str:
    if eval_type == "comparison":
        return "comparison"
    parts = [eval_type if eval_type in EVAL_TYPES else "unknown"]
    if has_same_status:
        parts.append("same_status")
    if has_status:
        parts.append("status")
    if has_reference:
        parts.append("reference")
    if has_compound:
        parts.append("compound_relation")
    elif has_relation:
        parts.append("spatial_relation")
    return "+".join(parts)


def extract_question_profile(question: str, eval_type: str | None = None) -> dict:
    """Extract deterministic symbolic features without using answers."""
    q = _norm(question)
    inferred_eval = _infer_eval_type(q, eval_type)
    relation_terms = _find_terms(q, RELATION_ALIASES)
    compound_relation_terms = [term for term in relation_terms if term in COMPOUND_RELATIONS]
    status_terms = _find_terms(q, STATUS_ALIASES)
    object_terms = _find_terms(q, OBJECT_ALIASES)
    refs = _reference_phrases(q, status_terms, object_terms)
    has_same_status = "same status" in q
    has_reference = bool(refs) or " of the " in f" {q} " or " it" in f" {q} "
    has_relation = bool(relation_terms)
    has_compound = bool(compound_relation_terms)
    has_status = bool(status_terms)
    token = _candidate_banker_token(q, inferred_eval, compound_relation_terms, status_terms, has_reference, has_relation)
    signature = _reasoning_signature(
        inferred_eval,
        has_status,
        has_reference,
        has_relation,
        has_compound,
        has_same_status,
    )
    return {
        "eval_type": inferred_eval,
        "normalized_question": q,
        "relation_terms": relation_terms,
        "compound_relation_terms": compound_relation_terms,
        "status_terms": status_terms,
        "object_terms": object_terms,
        "reference_phrases": refs,
        "has_same_status": has_same_status,
        "has_reference_object": has_reference,
        "has_spatial_relation": has_relation,
        "has_compound_relation": has_compound,
        "has_status_constraint": has_status,
        "candidate_banker_token": token,
        "reasoning_signature": signature,
    }
