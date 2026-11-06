from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scene_kg_category_ontology import normalize_category_label
from step71_common import OUTPUT_DIR, counts_dict, write_json, write_md


DEFAULT_BEVFUSION_ROOT = OUTPUT_DIR / "phase3b5_bevfusion_mini_evidence" / "normalized"


class BEVFusionSceneAdapter:
    """Convert transferred BEVFusion detections into existing executor scene_rows."""

    def __init__(self, evidence_root: str | Path = DEFAULT_BEVFUSION_ROOT, velocity_threshold: float = 0.2):
        self.evidence_root = Path(evidence_root)
        self.velocity_threshold = velocity_threshold

    def available_sample_tokens(self) -> list[str]:
        tokens: list[str] = []
        for path in sorted(self.evidence_root.glob("*.json")):
            payload = self._read_json(path)
            if isinstance(payload.get("objects"), list):
                tokens.append(str(payload.get("sample_token") or path.stem))
        return tokens

    def load_sample(self, sample_token: str) -> dict[str, Any]:
        path = self.evidence_root / f"{sample_token}.json"
        payload = self._read_json(path)
        if not isinstance(payload.get("objects"), list):
            raise ValueError(f"BEVFusion sample JSON has no object list: {path}")
        return payload

    def normalize_objects(self, sample_token: str) -> list[dict[str, Any]]:
        payload = self.load_sample(sample_token)
        return [
            self._normalize_one(sample_token, payload, obj, index)
            for index, obj in enumerate(payload.get("objects") or [])
        ]

    def build_executor_scene_rows(self, sample_token: str, max_lines: int | None = 250) -> list[dict[str, Any]]:
        rows = [self._canonical_object_to_executor_row(obj) for obj in self.normalize_objects(sample_token)]
        rows = sorted(rows, key=lambda row: (float(row.get("distance", 1e9)), str(row.get("id", ""))))
        return rows[:max_lines] if max_lines else rows

    def _normalize_one(
        self,
        sample_token: str,
        payload: dict[str, Any],
        obj: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        center = _float_list(obj.get("center"), length=3)
        size = _float_list(obj.get("size"), length=3)
        velocity = _float_list(obj.get("velocity"), length=2)
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2) if velocity else None
        raw_category = obj.get("category")
        category = normalize_category_label(raw_category)["canonical_category"]
        confidence = _optional_float(obj.get("confidence"))
        status = "moving" if speed is not None and speed > self.velocity_threshold else "stationary" if speed is not None else "unknown"
        object_id = str(obj.get("object_id") or f"bevfusion_{index:06d}")
        return {
            "object_id": object_id,
            "sample_token": sample_token,
            "category": category,
            "geometry": {
                "center_xyz": center,
                "size_wlh": size,
                "yaw": _optional_float(obj.get("yaw")),
                "coordinate_frame": payload.get("coordinate_frame") or "lidar",
                "raw_box": [v for v in [*center, *size, _optional_float(obj.get("yaw"))] if v is not None],
            },
            "velocity_xy": velocity,
            "speed": speed,
            "status": status,
            "attribute": None,
            "confidence": confidence,
            "source": "bevfusion",
            "feature_ref": f"{sample_token}.json:objects[{index}]",
            "metadata": {
                "raw_label": raw_category,
                "source": payload.get("source") or "bevfusion",
                "scene_token": payload.get("scene_token"),
                "coordinate_frame": payload.get("coordinate_frame") or "lidar",
                "status_source": "velocity_coarse" if speed is not None else "missing_attribute_and_velocity",
                "unavailable_fields": ["fine_grained_status", "oracle_attribute_tokens", "oracle_annotation_token"],
            },
        }

    def _canonical_object_to_executor_row(self, obj: dict[str, Any]) -> dict[str, Any]:
        center = ((obj.get("geometry") or {}).get("center_xyz") or [None, None, None])
        x = _optional_float(center[0]) if len(center) > 0 else None
        y = _optional_float(center[1]) if len(center) > 1 else None
        z = _optional_float(center[2]) if len(center) > 2 else None
        distance = math.sqrt(x * x + y * y) if x is not None and y is not None else None
        row: dict[str, Any] = {
            "id": obj["object_id"],
            "category": obj.get("category") or "unknown",
            "status": obj.get("status") or "unknown",
            "raw_status": obj.get("status") or "unknown",
            "source": "bevfusion",
            "confidence": obj.get("confidence"),
            "sample_token": obj.get("sample_token"),
            "coordinate_frame": (obj.get("geometry") or {}).get("coordinate_frame") or "lidar",
            "geometry": obj.get("geometry"),
            "velocity_xy": obj.get("velocity_xy"),
            "speed": obj.get("speed"),
            "bevfusion_object_id": obj.get("object_id"),
        }
        if distance is not None:
            row["distance"] = round(distance, 2)
        if x is not None:
            row["x"] = round(x, 2)
        if y is not None:
            row["y"] = round(y, 2)
        if z is not None:
            row["z"] = round(z, 2)
        return row

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))


def summarize_adapter(adapter: BEVFusionSceneAdapter, max_samples: int) -> dict[str, Any]:
    tokens = adapter.available_sample_tokens()[:max_samples]
    category_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    object_total = 0
    confidence_count = 0
    velocity_count = 0
    empty_samples: list[str] = []
    for token in tokens:
        objects = adapter.normalize_objects(token)
        object_total += len(objects)
        if not objects:
            empty_samples.append(token)
        for obj in objects:
            category_counts[obj.get("category") or "unknown"] += 1
            status_counts[obj.get("status") or "unknown"] += 1
            confidence_count += int(obj.get("confidence") is not None)
            velocity_count += int(obj.get("velocity_xy") is not None)
    return {
        "evidence_root": str(adapter.evidence_root),
        "available_sample_count": len(adapter.available_sample_tokens()),
        "sample_count_validated": len(tokens),
        "object_count": object_total,
        "category_distribution": counts_dict(category_counts),
        "status_distribution": counts_dict(status_counts),
        "confidence_available_count": confidence_count,
        "velocity_available_count": velocity_count,
        "empty_samples": empty_samples,
        "executor_row_contract": ["id", "category", "status", "distance", "x", "y", "z"],
        "source_independence": "Downstream executor receives scene_rows and does not branch on source.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_root", default=str(DEFAULT_BEVFUSION_ROOT))
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--velocity_threshold", type=float, default=0.2)
    args = parser.parse_args()

    adapter = BEVFusionSceneAdapter(args.evidence_root, velocity_threshold=args.velocity_threshold)
    summary = summarize_adapter(adapter, max_samples=args.max_samples)
    write_json(OUTPUT_DIR / "phase3b6_bevfusion_adapter_summary.json", summary)
    write_md(
        OUTPUT_DIR / "phase3b6_bevfusion_adapter_summary.md",
        "# Phase 3B.6 BEVFusion Adapter Summary\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in summary.items()),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _float_list(value: Any, length: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(v) for v in list(value)]
    except (TypeError, ValueError):
        return None
    if len(values) < length:
        return None
    return values[:length]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
