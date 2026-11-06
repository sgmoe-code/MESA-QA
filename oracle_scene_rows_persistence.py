"""Persistence and direct reload utilities for full oracle scene_rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pyquaternion import Quaternion

from experiment_config import DEFAULT_DATAROOT, DEFAULT_VERSION


PERSISTENCE_VERSION = "oracle_scene_rows_v1"


def serialize_oracle_scene_rows(scene_rows) -> dict | list:
    notes: list[str] = []

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, tuple):
            notes.append("tuple_converted_to_list")
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            notes.append("ndarray_converted_to_list")
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        notes.append(f"{type(value).__name__}_converted_to_string")
        return str(value)

    converted = convert(scene_rows)
    if isinstance(converted, list):
        return converted
    return converted


def persist_oracle_scene_rows_for_sample(
    output_dir: str,
    idx: int,
    sample_id: str | None,
    scene_rows,
    metadata: dict | None = None,
) -> dict:
    root = Path(output_dir) / "oracle_scene_rows"
    root.mkdir(parents=True, exist_ok=True)
    rows = serialize_oracle_scene_rows(scene_rows)
    if not isinstance(rows, list):
        rows = []
    filename = f"idx_{idx:04d}.json"
    rel_path = f"oracle_scene_rows/{filename}"
    row_keys = sorted({key for row in rows if isinstance(row, dict) for key in row.keys()})
    payload = {
        "idx": idx,
        "sample_id": sample_id,
        "scene_id": (metadata or {}).get("scene_id"),
        "question": (metadata or {}).get("question"),
        "scene_rows": rows,
        "num_rows": len(rows),
        "row_keys": row_keys,
        "metadata": metadata or {},
        "source": "oracle",
        "persistence_version": PERSISTENCE_VERSION,
    }
    path = root / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    index_path = root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"entries": []}
    index["entries"] = [entry for entry in index.get("entries", []) if entry.get("idx") != idx]
    entry = {
        "idx": idx,
        "sample_id": sample_id,
        "scene_id": payload["scene_id"],
        "question": payload["question"],
        "scene_rows_file": rel_path,
        "num_rows": len(rows),
        "row_keys": row_keys,
        "source": "oracle",
        "persistence_version": PERSISTENCE_VERSION,
    }
    index["entries"].append(entry)
    index["entries"] = sorted(index["entries"], key=lambda item: item["idx"])
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return {**entry, "json_serializable": True, "serialization_notes": []}


def load_persisted_oracle_scene_rows(
    output_dir: str,
    idx: int | None = None,
    sample_id: str | None = None,
) -> dict | list:
    root = Path(output_dir) / "oracle_scene_rows"
    index = json.loads((root / "index.json").read_text())
    for entry in index.get("entries", []):
        if idx is not None and entry.get("idx") != idx:
            continue
        if sample_id is not None and entry.get("sample_id") != sample_id:
            continue
        return json.loads((Path(output_dir) / entry["scene_rows_file"]).read_text())
    raise FileNotFoundError(f"No persisted oracle scene_rows for idx={idx}, sample_id={sample_id}")


class DirectNuScenesTableLoader:
    """Minimal JSON-table loader for v1.0-mini scene_rows without importing nuscenes."""

    def __init__(self, dataroot: str = DEFAULT_DATAROOT, version: str = DEFAULT_VERSION):
        self.dataroot = Path(dataroot)
        self.version = version
        self.table_dir = self.dataroot / version
        self.tables: dict[str, dict[str, dict]] = {}

    def load(self) -> None:
        for name in ["sample", "sample_data", "ego_pose", "sample_annotation", "attribute", "instance", "category", "calibrated_sensor", "sensor"]:
            rows = json.loads((self.table_dir / f"{name}.json").read_text())
            self.tables[name] = {row["token"]: row for row in rows}
        self.sample_data_by_sample: dict[str, list[dict]] = {}
        for row in self.tables["sample_data"].values():
            self.sample_data_by_sample.setdefault(row["sample_token"], []).append(row)
        self.annotations_by_sample: dict[str, list[dict]] = {}
        for row in self.tables["sample_annotation"].values():
            self.annotations_by_sample.setdefault(row["sample_token"], []).append(row)

    def get(self, table: str, token: str) -> dict:
        if not self.tables:
            self.load()
        return self.tables[table][token]

    def build_scene_rows(self, sample_token: str, max_lines: int = 250) -> tuple[dict, list[dict]]:
        sample = self.get("sample", sample_token)
        lidar_data = self._lidar_sample_data(sample_token)
        ego_pose = self.get("ego_pose", lidar_data["ego_pose_token"])
        rows = []
        anns = self.annotations_by_sample.get(sample_token, [])
        for idx, ann in enumerate(anns):
            category_name = self._category_name(ann)
            raw_status = "unknown"
            if ann.get("attribute_tokens"):
                raw_status = self.get("attribute", ann["attribute_tokens"][0])["name"].split(".")[-1]
            status = canonical_status(raw_status)
            x, y, z = global_to_ego(ego_pose, ann["translation"])
            dist = math.sqrt(x * x + y * y)
            rows.append(
                {
                    "id": f"obj_{idx}",
                    "category": canonical_category(category_name),
                    "status": status,
                    "raw_status": raw_status,
                    "distance": round(dist, 2),
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "z": round(float(z), 2),
                    "hidden_token": ann["token"],
                    "category_name": category_name,
                    "attribute_tokens": ann.get("attribute_tokens") or [],
                }
            )
        rows = sorted(rows, key=lambda row: row["distance"])[:max_lines]
        metadata = {
            "sample_id": sample_token,
            "sample_token": sample_token,
            "scene_id": sample.get("scene_token"),
            "dataroot": str(self.dataroot),
            "version": self.version,
            "loader": "DirectNuScenesTableLoader",
            "matches_current_loader": "reimplements run_agent_step1b_v21.build_true_scene_graph from NuScenes JSON tables",
        }
        return metadata, rows

    def _lidar_sample_data(self, sample_token: str) -> dict:
        candidates = self.sample_data_by_sample.get(sample_token, [])
        for row in candidates:
            calibrated = self.get("calibrated_sensor", row["calibrated_sensor_token"])
            sensor = self.get("sensor", calibrated["sensor_token"])
            if sensor.get("channel") == "LIDAR_TOP":
                return row
        if candidates:
            return candidates[0]
        raise KeyError(f"No sample_data rows for sample_token={sample_token}")

    def _category_name(self, ann: dict) -> str:
        instance = self.get("instance", ann["instance_token"])
        category = self.get("category", instance["category_token"])
        return category["name"]


def canonical_category(category_name: str) -> str:
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name == "vehicle.car":
        return "car"
    if category_name == "vehicle.truck":
        return "truck"
    if category_name in {"vehicle.bus.rigid", "vehicle.bus.bendy"}:
        return "bus"
    if category_name == "vehicle.trailer":
        return "trailer"
    if category_name == "vehicle.construction":
        return "construction_vehicle"
    if category_name == "vehicle.motorcycle":
        return "motorcycle"
    if category_name == "vehicle.bicycle":
        return "bicycle"
    if category_name == "movable_object.trafficcone":
        return "traffic_cone"
    if category_name == "movable_object.barrier":
        return "barrier"
    return category_name.split(".")[-1]


def canonical_status(attr_name: str | None) -> str:
    attr_name = (attr_name or "unknown").split(".")[-1]
    mapping = {
        "moving": "moving",
        "parked": "parked",
        "stopped": "stopped",
        "standing": "standing",
        "sitting_lying_down": "not_standing",
        "with_rider": "with_rider",
        "without_rider": "without_rider",
    }
    return mapping.get(attr_name, attr_name)


def global_to_ego(ego_pose: dict, global_xyz) -> tuple[float, float, float]:
    global_xyz = np.array(global_xyz)
    ego_translation = np.array(ego_pose["translation"])
    ego_rotation = Quaternion(ego_pose["rotation"])
    p_ego = ego_rotation.inverse.rotate(global_xyz - ego_translation)
    return float(p_ego[0]), float(p_ego[1]), float(p_ego[2])
