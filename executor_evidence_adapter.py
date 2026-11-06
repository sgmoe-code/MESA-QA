"""Executor adapter for entity-centric EvidenceState."""

from __future__ import annotations

import math
from typing import Any

from evidence_state import EvidenceState, LOCALIZATION_3D
from query_resolver import resolve_query


GEOMETRY_OPERATIONS = {"distance", "direction", "nearest", "farthest"}


def operation_requirements(plan: dict[str, Any], eval_type: str | None = None) -> dict[str, Any]:
    query = resolve_query(plan, eval_type=eval_type)
    requires_3d = bool(query.get("spatial_relations")) or query.get("operation") in GEOMETRY_OPERATIONS
    return {
        "operation": query.get("operation"),
        "requires_category": True,
        "requires_status": bool((query.get("target_filter") or {}).get("attributes") or (query.get("reference_filter") or {}).get("attributes")),
        "requires_3d": requires_3d,
        "query": query,
    }


class ExecutorEvidenceAdapter:
    def __init__(self, state: EvidenceState):
        self.state = state

    def materialize_scene_rows(self, plan: dict[str, Any] | None = None, eval_type: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        requirements = operation_requirements(plan or {}, eval_type=eval_type) if plan is not None else {"requires_3d": False, "operation": None}
        rows: list[dict[str, Any]] = []
        filtered = []
        for entity in self.state.entities.values():
            obs = self._select_localization_observation(entity.observations, require_3d=bool(requirements.get("requires_3d")))
            if obs is None:
                filtered.append({"entity_id": entity.entity_id, "reason": "required_localization_unavailable"})
                continue
            row = {
                "id": entity.entity_id,
                "category": entity.category,
                "status": self._materialized_status(entity.attributes),
                "raw_status": self._materialized_status(entity.attributes),
                "source": "evidence_state_adapter",
                "sample_token": self.state.sample_token,
                "evidence_state_entity_id": entity.entity_id,
                "supporting_observation_ids": list(entity.observations),
                "materialized_observation_id": obs.observation_id,
                "provider_sources": sorted({self.state.observations[o].provider for o in entity.observations if o in self.state.observations}),
                "localization_level": obs.localization_level,
                "provenance_ref": {
                    "entity_id": entity.entity_id,
                    "observation_id": obs.observation_id,
                    "provider": obs.provider,
                    "attribute_ids": [a.attribute_id for values in entity.attributes.values() for a in values],
                    "localization_source": obs.observation_id,
                },
            }
            if obs.geometry:
                row["geometry"] = obs.geometry
                center = (obs.geometry or {}).get("center_xyz") or []
                if len(center) >= 2:
                    x, y = _float(center[0]), _float(center[1])
                    if x is not None:
                        row["x"] = round(x, 2)
                    if y is not None:
                        row["y"] = round(y, 2)
                    if len(center) > 2 and _float(center[2]) is not None:
                        row["z"] = round(float(center[2]), 2)
                    if x is not None and y is not None:
                        row["distance"] = round(math.sqrt(x * x + y * y), 2)
            rows.append(row)
        rows = sorted(rows, key=lambda row: (float(row.get("distance", 1e9)), str(row.get("id", ""))))
        trace = {
            "operation_requirements": requirements,
            "materialized_count": len(rows),
            "filtered_count": len(filtered),
            "filtered_entities": filtered,
            "entity_count": len(self.state.entities),
            "observation_count": len(self.state.observations),
        }
        return rows, trace

    def _select_localization_observation(self, observation_ids: list[str], require_3d: bool):
        observations = [self.state.observations[o] for o in observation_ids if o in self.state.observations]
        if require_3d:
            observations = [obs for obs in observations if obs.localization_level == LOCALIZATION_3D and obs.geometry]
        if not observations:
            return None
        observations = sorted(observations, key=lambda obs: (obs.localization_level != LOCALIZATION_3D, obs.provider, obs.observation_id))
        return observations[0]

    @staticmethod
    def _materialized_status(attributes: dict[str, list[Any]]) -> str:
        candidates = []
        for key in ("status", "raw_status", "parked", "stopped", "standing", "moving", "stationary"):
            candidates.extend(attributes.get(key, []))
        informative = [a for a in candidates if a.value not in {None, "", "UNKNOWN", "unknown", "insufficient"}]
        if not informative:
            return "unknown"
        provider_rank = {"targeted_visual_semantic": 0, "stub_semantic_status": 0, "bevfusion": 1}
        informative = sorted(informative, key=lambda a: (provider_rank.get(a.provider, 2), a.attribute_id))
        return str(informative[0].value)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
