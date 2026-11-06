"""General driving-scene category ontology for SceneKG v1.1."""

from __future__ import annotations

from typing import Any, Dict


VEHICLES = {"car", "truck", "bus", "trailer", "construction_vehicle", "construction vehicle", "motorcycle", "bicycle"}
HUMANS = {"pedestrian", "person", "human"}
STATIC_OBJECTS = {"traffic_cone", "traffic cone", "cone", "barrier", "bicycle rack"}
ANY_OBJECT = {"object", "thing", "unknown", "any", "anything"}


def normalize_category_label(raw_category: str | None) -> dict:
    raw = None if raw_category is None else str(raw_category)
    value = str(raw_category or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if value.startswith("vehicle.car"):
        canonical = "car"
    elif value.startswith("vehicle.truck"):
        canonical = "truck"
    elif value.startswith("vehicle.bus"):
        canonical = "bus"
    elif value.startswith("vehicle.trailer"):
        canonical = "trailer"
    elif value.startswith("vehicle.construction") or value in {"construction_vehicle", "construction"}:
        canonical = "construction_vehicle"
    elif value.startswith("vehicle.motorcycle") or value == "motorbike":
        canonical = "motorcycle"
    elif value.startswith("vehicle.bicycle") or value == "bike":
        canonical = "bicycle"
    elif value.startswith("human.pedestrian") or value in {"pedestrian", "person", "people", "human"}:
        canonical = "pedestrian"
    elif value in {"movable_object.trafficcone", "trafficcone", "traffic_cone", "cone"}:
        canonical = "traffic_cone"
    elif value in {"movable_object.barrier", "barrier"}:
        canonical = "barrier"
    elif value in {"ego", "ego_vehicle", "self"}:
        canonical = "ego"
    elif value == "bicycle_rack":
        canonical = "bicycle rack"
    else:
        canonical = value

    if canonical in {"car", "truck", "bus", "trailer", "construction_vehicle", "motorcycle", "bicycle"}:
        group = "vehicle"
    elif canonical == "pedestrian":
        group = "human"
    elif canonical in {"traffic_cone", "barrier", "bicycle rack"}:
        group = "static_object"
    elif canonical == "ego":
        group = "ego"
    else:
        group = "unknown"
    aliases = sorted(_aliases_for(canonical, group))
    return {
        "raw_category": raw,
        "canonical_category": canonical,
        "category_group": group,
        "aliases": aliases,
        "is_vehicle": group == "vehicle",
        "is_human": group == "human",
        "is_two_wheeler": canonical in {"motorcycle", "bicycle"},
        "is_static_object": group == "static_object",
    }


def category_matches(
    node_category_info: dict,
    query_category: str | None,
    allow_group_match: bool = True,
) -> bool:
    query = normalize_category_label(query_category)
    q = query["canonical_category"]
    if q in ANY_OBJECT or str(query_category or "").strip().lower().replace("_", " ") in ANY_OBJECT:
        return True
    node_canon = node_category_info.get("canonical_category")
    if node_canon == q:
        return True
    if q in node_category_info.get("aliases", []):
        return True
    if allow_group_match and query["category_group"] != "unknown":
        return node_category_info.get("category_group") == query["category_group"]
    return False


def _aliases_for(canonical: str, group: str) -> set[str]:
    aliases = {canonical, canonical.replace("_", " ")}
    if group == "vehicle":
        aliases.add("vehicle")
    if canonical == "pedestrian":
        aliases.update({"person", "human", "people"})
    if canonical == "traffic_cone":
        aliases.update({"traffic cone", "cone"})
    if canonical == "ego":
        aliases.update({"ego vehicle", "self", "our vehicle"})
    if canonical in ANY_OBJECT:
        aliases.update(ANY_OBJECT)
    return aliases
