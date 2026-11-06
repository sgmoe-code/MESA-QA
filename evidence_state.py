"""Entity-centric evidence state for Phase 3B-14B.

The state separates physical entity hypotheses from provider observations.  It
is intentionally small and does not perform QA reasoning or geometric relation
computation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any


LOCALIZATION_3D = "LOCALIZATION_3D"
LOCALIZATION_2D = "LOCALIZATION_2D"
LOCALIZATION_CATEGORY_ONLY = "LOCALIZATION_CATEGORY_ONLY"
LOCALIZATION_UNKNOWN = "UNKNOWN"
MATCHED = "MATCHED"
POSSIBLE_MATCH = "POSSIBLE_MATCH"
UNASSOCIATED = "UNASSOCIATED"


@dataclass
class Observation:
    observation_id: str
    provider: str
    source: str
    category: str
    confidence: Any = None
    localization_level: str = LOCALIZATION_UNKNOWN
    geometry: dict[str, Any] | None = None
    bbox_2d: Any = None
    camera: str | None = None
    temporal_scope: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributeEvidence:
    attribute_id: str
    key: str
    value: Any
    confidence: Any
    provider: str
    observation_id: str | None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Association:
    entity_id: str
    observation_id: str
    state: str
    confidence: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    entity_id: str
    category: str
    observations: list[str] = field(default_factory=list)
    attributes: dict[str, list[AttributeEvidence]] = field(default_factory=dict)
    source_refs: dict[str, list[str]] = field(default_factory=dict)
    association_state: str = MATCHED
    provenance: dict[str, Any] = field(default_factory=dict)


class EvidenceState:
    def __init__(self, sample_token: str):
        self.sample_token = sample_token
        self.entities: dict[str, Entity] = {}
        self.observations: dict[str, Observation] = {}
        self.associations: list[Association] = []
        self.source_index: dict[str, list[str]] = {}
        self._next_entity = 0
        self._next_attribute = 0

    @classmethod
    def from_scene_rows(cls, sample_token: str, rows: list[dict[str, Any]]) -> "EvidenceState":
        state = cls(sample_token)
        for row in rows:
            state.add_entity_from_bevfusion_row(row)
        return state

    def add_entity_from_bevfusion_row(self, row: dict[str, Any]) -> str:
        entity_id = self._make_entity_id("entity")
        provider_id = str(row.get("bevfusion_object_id") or row.get("id"))
        obs_id = f"bevfusion:{self.sample_token}:{provider_id}"
        geometry = copy.deepcopy(row.get("geometry"))
        obs = Observation(
            observation_id=obs_id,
            provider="bevfusion",
            source="structured_3d",
            category=str(row.get("category") or "unknown"),
            confidence=row.get("confidence"),
            localization_level=LOCALIZATION_3D if geometry and (row.get("x") is not None and row.get("y") is not None) else LOCALIZATION_UNKNOWN,
            geometry=geometry,
            temporal_scope={"sample_token": self.sample_token},
            provenance={
                "source": row.get("source"),
                "sample_token": self.sample_token,
                "provider_object_id": provider_id,
                "feature_ref": row.get("feature_ref"),
                "metadata": copy.deepcopy(row.get("metadata") or {}),
            },
            raw=copy.deepcopy(row),
        )
        ent = Entity(
            entity_id=entity_id,
            category=obs.category,
            observations=[obs_id],
            source_refs={"bevfusion": [provider_id]},
            association_state=MATCHED,
            provenance={"initial_provider": "bevfusion", "sample_token": self.sample_token},
        )
        self.entities[entity_id] = ent
        self._add_observation(obs)
        self.associations.append(Association(entity_id, obs_id, MATCHED, 1.0, {"rule": "initial_bevfusion_ingestion"}))
        for key in ("status", "raw_status"):
            value = row.get(key)
            if value not in {None, "", "unknown", "UNKNOWN"}:
                self.attach_attribute(entity_id, key, value, row.get("confidence"), "bevfusion", obs_id, obs.provenance)
        return entity_id

    def integrate_object_candidate(self, candidate: dict[str, Any], max_3d_match_distance: float = 1.5) -> dict[str, Any]:
        obs = self._observation_from_candidate(candidate)
        match = self._find_confident_match(obs, max_3d_match_distance)
        if match:
            entity_id, distance = match
            self.entities[entity_id].observations.append(obs.observation_id)
            self.entities[entity_id].source_refs.setdefault(obs.provider, []).append(obs.observation_id)
            association_state = MATCHED
            action = "associated_existing_entity"
            provenance = {"rule": "category_and_3d_distance", "distance": distance, "threshold": max_3d_match_distance}
        else:
            entity_id = self._make_entity_id("entity")
            self.entities[entity_id] = Entity(
                entity_id=entity_id,
                category=obs.category,
                observations=[obs.observation_id],
                source_refs={obs.provider: [obs.observation_id]},
                association_state=UNASSOCIATED,
                provenance={"initial_provider": obs.provider, "sample_token": self.sample_token},
            )
            association_state = UNASSOCIATED
            action = "inserted_new_entity"
            provenance = {"rule": "no_confident_generic_association"}
        self._add_observation(obs)
        self.associations.append(Association(entity_id, obs.observation_id, association_state, None, provenance))
        return {"action": action, "entity_id": entity_id, "observation_id": obs.observation_id, "association_state": association_state, "localization_level": obs.localization_level}

    def attach_visual_attribute(self, visual_result: dict[str, Any]) -> dict[str, Any]:
        object_id = str(visual_result.get("object_id"))
        entity_id = self.entity_for_provider_ref("bevfusion", object_id)
        if entity_id is None:
            return {"attached": False, "reason": "anchor_entity_missing", "object_id": object_id}
        obs_id = f"visual_semantic:{self.sample_token}:{object_id}:{visual_result.get('attribute_name')}"
        obs = Observation(
            observation_id=obs_id,
            provider=str(visual_result.get("source") or "targeted_visual_semantic"),
            source="semantic_attribute",
            category=str(visual_result.get("category") or self.entities[entity_id].category),
            confidence=visual_result.get("confidence"),
            localization_level=LOCALIZATION_2D,
            bbox_2d=(visual_result.get("provenance") or {}).get("bbox"),
            camera=(visual_result.get("provenance") or {}).get("camera"),
            temporal_scope={"sample_token": self.sample_token},
            provenance=copy.deepcopy(visual_result.get("provenance") or {}),
            raw=copy.deepcopy(visual_result),
        )
        self._add_observation(obs)
        self.entities[entity_id].observations.append(obs_id)
        self.entities[entity_id].source_refs.setdefault(obs.provider, []).append(obs_id)
        attr = self.attach_attribute(
            entity_id,
            str(visual_result.get("attribute_name") or "status"),
            visual_result.get("attribute_value"),
            visual_result.get("confidence"),
            obs.provider,
            obs_id,
            obs.provenance,
        )
        self.associations.append(Association(entity_id, obs_id, MATCHED, visual_result.get("confidence"), {"rule": "anchored_visual_object_id"}))
        return {"attached": True, "entity_id": entity_id, "observation_id": obs_id, "attribute_id": attr.attribute_id}

    def attach_attribute(self, entity_id: str, key: str, value: Any, confidence: Any, provider: str, observation_id: str | None, provenance: dict[str, Any]) -> AttributeEvidence:
        self._next_attribute += 1
        attr = AttributeEvidence(
            attribute_id=f"attr_{self._next_attribute:06d}",
            key=str(key),
            value=value,
            confidence=confidence,
            provider=provider,
            observation_id=observation_id,
            provenance=copy.deepcopy(provenance),
        )
        self.entities[entity_id].attributes.setdefault(str(key), []).append(attr)
        return attr

    def entity_for_provider_ref(self, provider: str, provider_id: str) -> str | None:
        for entity in self.entities.values():
            if provider_id in entity.source_refs.get(provider, []):
                return entity.entity_id
            for obs_id in entity.observations:
                raw = self.observations.get(obs_id)
                if raw and str((raw.raw or {}).get("id")) == provider_id:
                    return entity.entity_id
        return None

    def statistics(self) -> dict[str, Any]:
        loc_counts: dict[str, int] = {}
        provider_counts: dict[str, int] = {}
        attr_count = 0
        for obs in self.observations.values():
            loc_counts[obs.localization_level] = loc_counts.get(obs.localization_level, 0) + 1
            provider_counts[obs.provider] = provider_counts.get(obs.provider, 0) + 1
        for entity in self.entities.values():
            attr_count += sum(len(v) for v in entity.attributes.values())
        return {
            "sample_token": self.sample_token,
            "entity_count": len(self.entities),
            "observation_count": len(self.observations),
            "association_count": len(self.associations),
            "attribute_evidence_count": attr_count,
            "localization_distribution": loc_counts,
            "provider_distribution": provider_counts,
        }

    def _add_observation(self, obs: Observation) -> None:
        self.observations[obs.observation_id] = obs
        self.source_index.setdefault(obs.provider, []).append(obs.observation_id)

    def _make_entity_id(self, prefix: str) -> str:
        self._next_entity += 1
        return f"{prefix}_{self._next_entity:06d}"

    def _observation_from_candidate(self, candidate: dict[str, Any]) -> Observation:
        prov = copy.deepcopy(candidate.get("provenance") or {})
        geometry = copy.deepcopy(candidate.get("geometry"))
        level = str(candidate.get("localization_level") or (LOCALIZATION_3D if geometry else LOCALIZATION_2D if prov.get("bbox") else LOCALIZATION_CATEGORY_ONLY))
        obs_id = str(candidate.get("id") or f"object_completion:{self.sample_token}:{len(self.observations):06d}")
        return Observation(
            observation_id=obs_id,
            provider=str(candidate.get("source") or prov.get("provider") or "object_completion"),
            source=str(prov.get("source_modality") or "object_completion"),
            category=str(candidate.get("category") or "unknown"),
            confidence=candidate.get("confidence"),
            localization_level=level,
            geometry=geometry,
            bbox_2d=prov.get("bbox"),
            camera=prov.get("camera"),
            temporal_scope={"sample_token": self.sample_token},
            provenance=prov,
            raw=copy.deepcopy(candidate),
        )

    def _find_confident_match(self, obs: Observation, threshold: float) -> tuple[str, float] | None:
        if obs.localization_level != LOCALIZATION_3D or not obs.geometry:
            return None
        center = (obs.geometry or {}).get("center_xyz")
        if not center or len(center) < 2:
            return None
        best: tuple[str, float] | None = None
        for entity in self.entities.values():
            if entity.category != obs.category:
                continue
            for existing_id in entity.observations:
                existing = self.observations.get(existing_id)
                if not existing or existing.localization_level != LOCALIZATION_3D or not existing.geometry:
                    continue
                ecenter = (existing.geometry or {}).get("center_xyz")
                if not ecenter or len(ecenter) < 2:
                    continue
                distance = math.sqrt((float(center[0]) - float(ecenter[0])) ** 2 + (float(center[1]) - float(ecenter[1])) ** 2)
                if distance <= threshold and (best is None or distance < best[1]):
                    best = (entity.entity_id, round(distance, 4))
        return best
