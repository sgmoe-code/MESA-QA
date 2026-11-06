"""Generic object-evidence completion provider backed by existing frozen artifacts.

This provider does not run a model. It exposes already-produced CenterPoint camera
candidate evidence through the Phase 3B-13B capability/serviceability contract.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from generalized_evidence_controller import EvidenceProvider
from scene_kg_category_ontology import normalize_category_label


class ObjectCompletionProvider(EvidenceProvider):
    name = "centerpoint_grounded_object_completion"
    kind = "direct_perception_artifact"
    priority = 5

    def __init__(self, evidence_path: Path, manifest_path: Path):
        self.evidence_path = evidence_path
        self.manifest = {row.get("evidence_id"): row for row in _read(manifest_path, [])}
        self.rows = _read(evidence_path, [])
        self.index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            if row.get("predicate_type") != "candidate_validation" or row.get("support") != "yes":
                continue
            category = normalize_category_label(row.get("target_category") or row.get("category"))["canonical_category"]
            self.index.setdefault((str(row.get("sample_token")), category), []).append(row)
        self.invocations: list[dict[str, Any]] = []

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "evidence_types": ["entity", "category", "object_localization"],
            "prerequisites": ["supported entity category", "camera candidate artifact", "reliable localization record"],
            "input": "structured EvidenceRequirement plus current evidence state",
            "output": "bounded candidate object evidence with 3D localization where available",
            "localization_levels": ["LOCALIZATION_3D", "LOCALIZATION_2D"],
            "confidence": True, "abstention": True, "provenance": True,
            "availability": "available_from_frozen_centerpoint_grounded_artifacts",
        }

    def supports_capability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        category = _category(requirement)
        return {"capability_supported": category not in {"unknown", "thing"} and any(key[1] == category for key in self.index), "reason": "category_supported_by_frozen_candidate_artifacts" if any(key[1] == category for key in self.index) else "category_not_supported_by_frozen_candidate_artifacts"}

    def assess_serviceability(self, requirement: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        category = _category(requirement)
        candidates = self.index.get((str(context.get("sample_token")), category), [])
        supported = self.supports_capability(requirement, context)["capability_supported"]
        return {"request_serviceable": bool(supported and candidates), "reason": "candidate_artifact_and_localization_available" if supported and candidates else "request_candidate_artifact_unavailable", "candidate_count": len(candidates), "localization": "LOCALIZATION_3D" if candidates and all((self.manifest.get(row.get("evidence_id")) or {}).get("geometry") for row in candidates) else "LOCALIZATION_2D"}

    def acquire(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        category = _category(request["requirement"])
        raw = self.index.get((str(context.get("sample_token")), category), [])
        candidates = self._deduplicate(raw)
        normalized = [self._normalize_candidate(row) for row in candidates]
        self.invocations.append({"request": copy.deepcopy(request), "sample_token": context.get("sample_token"), "category": category, "candidate_count": len(candidates), "candidates": copy.deepcopy(candidates), "result": {"candidates": normalized}})
        return {"status": "acquired" if candidates else "abstained", "source": self.name, "candidates": candidates, "provenance": [{"evidence_id": row.get("evidence_id"), "sample_token": row.get("sample_token"), "object_id": row.get("object_id")} for row in candidates]}

    def normalize(self, result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        candidates = [self._normalize_candidate(row) for row in result.get("candidates", [])]
        return {"source": self.name, "candidates": candidates, "candidate_count": len(candidates), "evidence_type": "entity", "localization_level": "LOCALIZATION_3D" if candidates and all(row.get("geometry") for row in candidates) else "LOCALIZATION_2D", "provenance": result.get("provenance", []), "sufficient": bool(candidates)}

    def _deduplicate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chosen: dict[str, dict[str, Any]] = {}
        for row in rows:
            object_id = str(row.get("object_id"))
            if object_id not in chosen or _confidence_rank(row.get("confidence")) > _confidence_rank(chosen[object_id].get("confidence")):
                chosen[object_id] = row
        return list(chosen.values())

    def _normalize_candidate(self, row: dict[str, Any]) -> dict[str, Any]:
        package = self.manifest.get(row.get("evidence_package_id")) or self.manifest.get(row.get("evidence_id"), {})
        projection = (package.get("projection") or {}).get("primary") or {}
        category = normalize_category_label(row.get("target_category") or row.get("category"))["canonical_category"]
        geometry = package.get("geometry") or projection.get("geometry")
        return {"id": f"object_completion:{row.get('sample_token')}:{row.get('object_id')}", "category": category, "confidence": row.get("confidence") or "medium", "localization_level": "LOCALIZATION_3D" if geometry else "LOCALIZATION_2D", "geometry": geometry, "source": self.name, "provenance": {"provider": self.name, "source_modality": "camera_plus_centerpoint_projection", "sample_token": row.get("sample_token"), "camera": projection.get("camera"), "bbox": projection.get("bbox_clipped"), "roi": (package.get("artifacts") or {}).get("primary_object_crop"), "raw_evidence_id": row.get("evidence_id"), "raw_request_id": row.get("request_id"), "confidence": row.get("confidence"), "normalization": "existing_category_ontology"}}


def _read(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _category(requirement: dict[str, Any]) -> str:
    return normalize_category_label(((requirement.get("target") or {}).get("category") or "unknown"))["canonical_category"]


def _confidence_rank(value: Any) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value).lower(), 0)
