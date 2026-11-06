# query_resolver.py
# Converts MLLM Planner output into a normalized executable query.

from typing import Dict, Any, Optional
from copy import deepcopy

from domain_schema import (
    OBJECT_TYPES, ANY_OBJECT_TYPES, ATTRIBUTES, RELATIONS,
    SPATIAL_RELATIONS, COMPARISON_RELATIONS, QUESTION_TO_OPERATION,
    CATEGORY_ALIASES, ATTRIBUTE_ALIASES, RELATION_ALIASES,
)


def normalize_category(x: Any) -> str:
    s = str(x or "unknown").lower().strip().replace("-", "_")
    s_space = s.replace("_", " ")
    if s in OBJECT_TYPES:
        return s
    if s_space in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[s_space]
    if s in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[s]
    return "unknown"


def normalize_attribute(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).lower().strip().replace("-", " ").replace("_", " ")
    if s in {"", "none", "null", "n/a", "unknown"}:
        return None
    if s in ATTRIBUTE_ALIASES:
        return ATTRIBUTE_ALIASES[s]
    s_u = s.replace(" ", "_")
    if s_u in ATTRIBUTES:
        return s_u
    return None


def normalize_relation(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).lower().strip()
    if s in {"", "none", "null", "n/a", "unknown"}:
        return None
    s_dash = s.replace("_", "-").replace(" ", "-")
    s_space = s.replace("_", " ").replace("-", " ")
    if s_dash in RELATIONS:
        return s_dash
    if s in RELATION_ALIASES:
        return RELATION_ALIASES[s]
    if s_space in RELATION_ALIASES:
        return RELATION_ALIASES[s_space]
    if s_dash in RELATION_ALIASES:
        return RELATION_ALIASES[s_dash]
    return None


def normalize_entity(entity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    entity = entity or {}
    category = normalize_category(entity.get("category", "unknown"))
    raw_attrs = entity.get("attributes", []) or []
    if isinstance(raw_attrs, str):
        raw_attrs = [raw_attrs]
    attrs = []
    moved_relations = []
    for attr in raw_attrs:
        rel = normalize_relation(attr)
        if rel in SPATIAL_RELATIONS or rel in COMPARISON_RELATIONS:
            moved_relations.append(rel)
            continue
        norm = normalize_attribute(attr)
        if norm and norm not in attrs:
            attrs.append(norm)
    return {"category": category, "attributes": attrs, "_moved_relations": moved_relations}


def infer_operation(plan: Dict[str, Any], eval_type: Optional[str] = None) -> str:
    qtype = (eval_type or plan.get("question_type") or "unknown").lower()
    return QUESTION_TO_OPERATION.get(qtype, "answer")


def resolve_query(plan: Dict[str, Any], eval_type: Optional[str] = None) -> Dict[str, Any]:
    plan = deepcopy(plan)
    operation = infer_operation(plan, eval_type=eval_type)
    target = normalize_entity(plan.get("target_entity"))
    reference = normalize_entity(plan.get("reference_entity"))
    raw_relations = plan.get("relations", []) or []
    if isinstance(raw_relations, str):
        raw_relations = [raw_relations]
    relations = []
    for r in raw_relations:
        for part in str(r).split("|"):
            rel = normalize_relation(part)
            if rel and rel not in relations:
                relations.append(rel)
    for rel in target.pop("_moved_relations", []):
        if rel not in relations:
            relations.append(rel)
    for rel in reference.pop("_moved_relations", []):
        if rel not in relations:
            relations.append(rel)
    spatial_relations = [r for r in relations if r in SPATIAL_RELATIONS]
    comparison_relations = [r for r in relations if r in COMPARISON_RELATIONS]
    hop = plan.get("hop", "unknown")
    if spatial_relations and reference.get("category") not in ANY_OBJECT_TYPES:
        hop = "one-hop"
    if reference.get("category") in ANY_OBJECT_TYPES and not reference.get("attributes") and not spatial_relations:
        reference = None
    return {
        "operation": operation,
        "target_filter": {"category": target.get("category", "unknown"), "attributes": target.get("attributes", [])},
        "reference_filter": None if reference is None else {"category": reference.get("category", "unknown"), "attributes": reference.get("attributes", [])},
        "relations": relations,
        "spatial_relations": spatial_relations,
        "comparison_relations": comparison_relations,
        "hop": hop,
        "raw_question_type": plan.get("question_type"),
        "raw_plan": plan,
    }
