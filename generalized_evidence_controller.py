"""Capability-driven evidence acquisition for Phase 3B-13.

The controller consumes semantic requirements and provider contracts.  It has no
knowledge of benchmark questions, question ids, or diagnostic phase labels.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from run_phase3b11a_visual_evidence_infrastructure import (
    TargetedVisualEvidenceController,
    merge_visual_evidence_into_scene_rows,
)
from scene_graph_query import object_matches


SUPPORTED_VISUAL_ATTRIBUTES = {"parked", "stopped", "standing", "moving", "stationary"}
UNKNOWN_VALUES = {None, "", "UNKNOWN", "unknown", "insufficient"}


def evidence_sufficiency(requirement: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess whether the predicate can be evaluated, independently of its truth."""
    target = requirement.get("target") or {}
    anchors = _candidate(rows, {"category": target.get("category", "unknown"), "predicates": []})
    if not anchors:
        return {"evidence_sufficient": False, "reason": "target_entity_missing", "predicate_evaluable": False}
    reference = requirement.get("reference")
    if reference and not _candidate(rows, {"category": reference.get("category", "unknown"), "predicates": []}):
        return {"evidence_sufficient": False, "reason": "reference_entity_missing", "predicate_evaluable": False}
    predicates = target.get("predicates") or []
    if predicates:
        available = []
        for row in anchors:
            status = row.get("status") or row.get("raw_status")
            semantic = row.get("semantic_attributes") or {}
            available.extend([status, row.get("raw_status")])
            available.extend(value.get("value") for value in semantic.values() if isinstance(value, dict))
        requested = set(str(predicate) for predicate in predicates)
        informative_values = requested | {f"not_{predicate}" for predicate in requested}
        informative = any(value in informative_values for value in available)
        if not informative:
            return {"evidence_sufficient": False, "reason": "requested_property_evidence_missing", "predicate_evaluable": False}
    if requirement.get("spatial_scope") and not any(row.get("geometry") and row.get("x") is not None and row.get("y") is not None for row in anchors):
        return {"evidence_sufficient": False, "reason": "required_geometry_missing", "predicate_evaluable": False}
    return {"evidence_sufficient": True, "reason": "evidence_available_for_evaluation", "predicate_evaluable": bool(predicates or anchors)}


def evidence_polarity(requirement: dict[str, Any], evidence: dict[str, Any]) -> str:
    value = evidence.get("attribute_value")
    if value in UNKNOWN_VALUES:
        return "UNKNOWN_OR_INSUFFICIENT"
    predicate = ((requirement.get("target") or {}).get("predicates") or [None])[0]
    if predicate and value == predicate:
        return "POSITIVE_INFORMATIVE"
    if predicate and value == f"not_{predicate}":
        return "NEGATIVE_INFORMATIVE"
    return "POSITIVE_INFORMATIVE"


def requirement_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Translate the existing atomic program into a provider-neutral contract."""
    target = plan.get("target_entity") or {}
    reference = plan.get("reference_entity") or {}
    relations = list(plan.get("relations") or [])
    return {
        "requirement_id": "atomic_program_requirement",
        "evidence_kinds": [
            "entity" if target else "context",
            *(["attribute"] if target.get("attributes") else []),
            *(["relation", "geometry"] if relations else []),
            *(["motion"] if any(a in {"moving", "stationary", "stopped"} for a in target.get("attributes", [])) else []),
        ],
        "target": {
            "category": target.get("category") or "unknown",
            "predicates": list(target.get("attributes") or []),
        },
        "reference": {
            "category": reference.get("category") or "unknown",
            "predicates": list(reference.get("attributes") or []),
        } if reference else None,
        "spatial_scope": relations,
        "temporal_scope": None,
        "minimum_confidence": "available",
        "source_agnostic": True,
    }


def _candidate(rows: list[dict[str, Any]], entity: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in rows if object_matches(row, entity.get("category", "unknown"), entity.get("predicates", []))]


def requirement_satisfied(requirement: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[bool, str]:
    targets = _candidate(rows, requirement["target"])
    if not targets:
        return False, "target_entity_or_predicate_missing"
    reference = requirement.get("reference")
    if reference and not _candidate(rows, reference):
        return False, "reference_entity_or_predicate_missing"
    if requirement.get("spatial_scope") and not any(
        row.get("geometry") and row.get("x") is not None and row.get("y") is not None for row in targets
    ):
        return False, "required_geometry_missing"
    return True, "all_required_evidence_present"


class EvidenceProvider:
    name = "base"
    kind = "source"
    priority = 100

    def capabilities(self) -> dict[str, Any]:
        raise NotImplementedError

    def supports_capability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"capability_supported": False, "reason": "not_implemented"}

    def assess_serviceability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"request_serviceable": False, "reason": "not_implemented"}

    def can_handle(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def acquire(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def normalize(self, result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        return result


class DirectStructuredProvider(EvidenceProvider):
    name = "bevfusion_structured"
    kind = "direct_perception"
    priority = 10

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "evidence_types": ["entity", "category", "geometry", "velocity", "confidence"],
            "prerequisites": [], "input": "BEVFusion normalized sample rows",
            "output": "executor scene rows", "confidence": True, "abstention": True,
            "provenance": True, "availability": "available",
        }

    def can_handle(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        check = self.assess_serviceability(requirement, context)
        return {"compatible": check["request_serviceable"], "satisfies": check["request_serviceable"], "reason": check["reason"]}

    def supports_capability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"capability_supported": True, "reason": "direct_structured_capability_registered"}

    def assess_serviceability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        check = evidence_sufficiency(requirement, context["scene_rows"])
        return {"request_serviceable": False, "reason": "direct_source_already_checked_or_insufficient", "evidence_check": check}

    def acquire(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "already_available", "source": self.name, "derived": False,
                "provenance": {"sample_token": context.get("sample_token"), "source_fields": "normalized BEVFusion object rows"}}


class DeterministicGeometryProvider(EvidenceProvider):
    name = "deterministic_geometry"
    kind = "derived_deterministic"
    priority = 20

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.name, "evidence_types": ["geometry", "relation", "distance"],
            "prerequisites": ["object centers x/y"], "input": "executor scene rows",
            "output": "derived relation evidence", "confidence": True, "abstention": True,
            "provenance": True, "availability": "available",
        }

    def can_handle(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        needed = bool(requirement.get("spatial_scope"))
        usable = all(row.get("x") is not None and row.get("y") is not None for row in context["scene_rows"])
        return {"compatible": needed and usable, "satisfies": needed and usable,
                "reason": "center_geometry_available" if needed and usable else "no_generic_geometry_requirement"}

    def acquire(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "derived", "source": self.name, "derived": True,
                "provenance": {"sample_token": context.get("sample_token"), "source_fields": ["x", "y", "geometry.center_xyz"], "method": "existing scene_graph_query relation operators"}}


class TargetedVisualProvider(EvidenceProvider):
    """Adapter for the existing 11B provider contract and frozen outputs."""
    name = "targeted_visual_semantic"
    kind = "targeted_learned"
    priority = 30

    def __init__(self, visual_results_path: Path, visual_controller: TargetedVisualEvidenceController):
        self.visual_controller = visual_controller
        self.results = json.loads(visual_results_path.read_text(encoding="utf-8")) if visual_results_path.exists() else []
        self.index = {}
        for row in self.results:
            result = row.get("visual_result") or {}
            key = (str(row.get("sample_token")), str(result.get("object_id")), str(result.get("attribute_name")))
            self.index[key] = result
        self.invocations: list[dict[str, Any]] = []

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.name, "evidence_types": ["object_attribute", "semantic_status"],
            "prerequisites": ["localized object anchor", "camera projection", "ROI"],
            "input": "11A targeted request", "output": "normalized object semantic attribute",
            "confidence": True, "abstention": True, "provenance": True,
            "availability": "available_for_existing_targeted_artifacts",
        }

    def can_handle(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        support = self.supports_capability(requirement, context)
        service = self.assess_serviceability(requirement, context)
        return {"compatible": support["capability_supported"] and service["request_serviceable"], "satisfies": False, "reason": service["reason"], "capability": support, "serviceability": service}

    def supports_capability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        attrs = requirement["target"].get("predicates") or []
        if len(attrs) != 1 or attrs[0] not in SUPPORTED_VISUAL_ATTRIBUTES:
            return {"capability_supported": False, "reason": "unsupported_visual_attribute_shape"}
        return {"capability_supported": True, "reason": "object_localized_visual_semantic_capability_registered"}

    def assess_serviceability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        support = self.supports_capability(requirement, context)
        if not support["capability_supported"]:
            return {"request_serviceable": False, "reason": support["reason"]}
        attrs = requirement["target"].get("predicates") or []
        anchors = _candidate(context["scene_rows"], {"category": requirement["target"].get("category", "unknown"), "predicates": []})
        if not anchors:
            return {"request_serviceable": False, "reason": "localized_object_anchor_missing", "prerequisites": {"localized_object_anchor": False}}
        key_prefix = (str(context["sample_token"]), str(anchors[0].get("id")), attrs[0])
        if key_prefix not in self.index:
            return {"request_serviceable": False, "reason": "frozen_visual_artifact_unavailable", "prerequisites": {"localized_object_anchor": True, "projection_or_roi": "UNKNOWN", "frozen_artifact": False}}
        return {"request_serviceable": True, "reason": "all_visual_request_prerequisites_available", "prerequisites": {"localized_object_anchor": True, "projection_or_roi": True, "frozen_artifact": True}}

    def acquire(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        anchor = _candidate(context["scene_rows"], {"category": request["requirement"]["target"].get("category", "unknown"), "predicates": []})[0]
        req = {"evidence_type": "semantic_status", "attribute": request["requirement"]["target"]["predicates"][0], "target_category": request["requirement"]["target"].get("category")}
        visual_request = self.visual_controller.build_request(context["sample_token"], {
            "object_id": anchor["id"], "category": anchor.get("category"), "confidence": anchor.get("confidence"),
            "geometry": anchor.get("geometry"),
        }, req)
        result = copy.deepcopy(self.index[(str(context["sample_token"]), str(anchor["id"]), req["attribute"])])
        self.invocations.append({"request": visual_request, "result": result})
        return {"status": "acquired", "source": self.name, "provider_result": result,
                "provenance": result.get("provenance") or {}, "request": visual_request}

    def normalize(self, result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        visual = result["provider_result"]
        return {"object_id": visual.get("object_id"), "attribute_name": visual.get("attribute_name"),
                "attribute_value": visual.get("attribute_value"), "confidence": visual.get("confidence"),
                "source": self.name, "provenance": visual.get("provenance") or {},
                "sufficient": visual.get("attribute_value") not in UNKNOWN_VALUES}


class EntityCompletionUnavailable(EvidenceProvider):
    name = "entity_completion"
    kind = "targeted_learned"
    priority = 90

    def capabilities(self) -> dict[str, Any]:
        return {"provider": self.name, "evidence_types": ["entity_completion"], "prerequisites": ["reliable detector output"],
                "input": "sample perception inputs", "output": "executor scene rows", "confidence": True,
                "abstention": True, "provenance": True, "availability": "CAPABILITY_UNAVAILABLE"}

    def can_handle(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"compatible": False, "satisfies": False, "reason": "no_reliable_generic_completion_provider"}

    def acquire(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "unsupported", "source": self.name}


class EvidenceController:
    def __init__(self, providers: list[EvidenceProvider]):
        self.providers = sorted(providers, key=lambda provider: provider.priority)

    def acquire(self, requirement: dict[str, Any], scene_rows: list[dict[str, Any]], sample_token: str) -> dict[str, Any]:
        context = {"scene_rows": scene_rows, "sample_token": sample_token}
        trace = {"requirement": copy.deepcopy(requirement), "steps": [], "depth": 0}
        initial = evidence_sufficiency(requirement, scene_rows)
        trace["steps"].append({"step": 0, "action": "check_current_evidence", "evidence_sufficient": initial["evidence_sufficient"], "satisfied": initial["evidence_sufficient"], "predicate_evaluable": initial["predicate_evaluable"], "reason": initial["reason"]})
        if initial["evidence_sufficient"]:
            return {"scene_rows": scene_rows, "satisfied": True, "evidence_sufficient": True, "reason": initial["reason"], "trace": trace}

        for provider in self.providers:
            capability = provider.supports_capability(requirement, context)
            serviceability = provider.assess_serviceability(requirement, context)
            check = {"capability_supported": capability.get("capability_supported", False), "request_serviceable": serviceability.get("request_serviceable", False), "capability_reason": capability.get("reason"), "serviceability_reason": serviceability.get("reason"), "prerequisites": serviceability.get("prerequisites", {})}
            trace["steps"].append({"step": len(trace["steps"]), "action": "discover_provider", "provider": provider.name, "capability_kind": provider.kind, "capability_supported": check["capability_supported"], "request_serviceable": check["request_serviceable"], "check": check})
            if not check["capability_supported"] or not check["request_serviceable"]:
                continue
            request = {"requirement": requirement, "sample_token": sample_token}
            acquired = provider.acquire(request, context)
            trace["steps"].append({"step": len(trace["steps"]), "action": "acquire", "provider": provider.name, "status": acquired.get("status"), "provider_invoked": True, "prerequisites": provider.capabilities().get("prerequisites", [])})
            if acquired.get("status") not in {"acquired", "derived"}:
                continue
            normalized = provider.normalize(acquired, request)
            if normalized.get("object_id"):
                merged = merge_visual_evidence_into_scene_rows(scene_rows, {
                    "object_id": normalized["object_id"], "attribute_name": normalized.get("attribute_name"),
                    "attribute_value": normalized.get("attribute_value"), "confidence": normalized.get("confidence"),
                    "source": normalized.get("source"), "provenance": normalized.get("provenance") or {},
                })
            else:
                merged = scene_rows
            context["scene_rows"] = merged
            post = evidence_sufficiency(requirement, merged)
            satisfied, reason = post["evidence_sufficient"], post["reason"]
            trace["depth"] = 2 if provider.name == "targeted_visual_semantic" else 1
            polarity = evidence_polarity(requirement, normalized)
            item_sufficient = bool(normalized.get("object_id")) and polarity != "UNKNOWN_OR_INSUFFICIENT"
            trace["steps"].append({"step": len(trace["steps"]), "action": "sufficiency_recheck", "provider": provider.name, "evidence_sufficient": item_sufficient, "requirement_complete": satisfied, "satisfied": item_sufficient, "predicate_evaluable": item_sufficient, "evidence_informative": polarity != "UNKNOWN_OR_INSUFFICIENT", "evidence_polarity": polarity, "reason": reason if satisfied else "acquired_evidence_item_is_informative_but_composite_requirement_is_incomplete", "normalized": normalized})
            if item_sufficient:
                return {"scene_rows": merged, "satisfied": True, "evidence_sufficient": True, "requirement_complete": satisfied, "reason": reason, "trace": trace}
            return {"scene_rows": merged, "satisfied": False, "evidence_sufficient": False, "requirement_complete": satisfied, "reason": reason, "trace": trace}
        trace["depth"] = "unsupported"
        return {"scene_rows": scene_rows, "satisfied": False, "evidence_sufficient": False, "reason": "no_serviceable_provider", "trace": trace}


def capability_inventory(providers: list[EvidenceProvider]) -> dict[str, Any]:
    return {"providers": [provider.capabilities() for provider in providers], "unmet_capabilities": [
        "ENTITY_COMPLETION", "TEMPORAL_CORRESPONDENCE", "MULTI_OBJECT_VISUAL_COMPARISON"
    ]}
