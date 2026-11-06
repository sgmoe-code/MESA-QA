# scene_graph_query.py
# Generic scene graph query operators.

from typing import Dict, Any, List, Optional, Tuple
from domain_schema import ANY_OBJECT_TYPES


def object_matches(obj: Dict[str, Any], category: str = "unknown", attributes: List[str] = None) -> bool:
    attributes = attributes or []
    if category not in ANY_OBJECT_TYPES:
        if obj.get("category") != category:
            return False
    for attr in attributes:
        if attr in {"", "unknown", None}:
            continue
        status = obj.get("status", "unknown")
        raw_status = obj.get("raw_status", "unknown")
        if attr == "not_standing":
            if status == "standing" or raw_status == "standing":
                return False
        elif attr == "standing":
            if status != "standing" and raw_status != "standing":
                return False
        else:
            if status != attr and raw_status != attr:
                return False
    return True


def sort_objects(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(objects, key=lambda o: (float(o.get("distance", 1e9)), str(o.get("id", ""))))


def select_objects(scene_rows: List[Dict[str, Any]], object_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
    category = object_filter.get("category", "unknown")
    attrs = object_filter.get("attributes", []) or []
    return sort_objects([obj for obj in scene_rows if object_matches(obj, category=category, attributes=attrs)])


def relation_between(target: Dict[str, Any], reference: Dict[str, Any], x_th: float = 2.0, y_th: float = 2.0) -> str:
    dx = float(target.get("x", 0.0)) - float(reference.get("x", 0.0))
    dy = float(target.get("y", 0.0)) - float(reference.get("y", 0.0))
    rel = []
    if dx > x_th:
        rel.append("front")
    elif dx < -x_th:
        rel.append("back")
    if dy > y_th:
        rel.append("left")
    elif dy < -y_th:
        rel.append("right")
    return "-".join(rel) if rel else "near"


def relation_matches(actual: str, required: str) -> bool:
    if not required or actual == required:
        return True
    if required == "front" and actual in {"front", "front-left", "front-right"}:
        return True
    if required == "back" and actual in {"back", "back-left", "back-right"}:
        return True
    if required == "left" and actual in {"left", "front-left", "back-left"}:
        return True
    if required == "right" and actual in {"right", "front-right", "back-right"}:
        return True
    return False


def select_related(targets: List[Dict[str, Any]], references: List[Dict[str, Any]], spatial_relations: List[str], allow_self: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not spatial_relations:
        return targets, []
    matched, edges, seen_ids = [], [], set()
    for t in targets:
        for r in references:
            if not allow_self and t.get("id") == r.get("id"):
                continue
            actual = relation_between(t, r)
            ok = any(relation_matches(actual, req) for req in spatial_relations)
            if ok:
                edges.append({
                    "subject": t.get("id"),
                    "relation": actual,
                    "object": r.get("id"),
                    "target_category": t.get("category"),
                    "reference_category": r.get("category"),
                })
                if t.get("id") not in seen_ids:
                    matched.append(t)
                    seen_ids.add(t.get("id"))
                break
    return sort_objects(matched), edges


def choose_primary(objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not objects:
        return None
    return sort_objects(objects)[0]
