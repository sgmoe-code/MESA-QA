from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from canonical_scene_object_step71 import CanonicalSceneObject
from scene_kg_category_ontology import normalize_category_label
from step71_common import OUTPUT_DIR, counts_dict, distance_xy, jsonable, load_npz_objects, load_questions, write_json, write_md


class CenterPointSceneAdapter:
    def __init__(self, centerpoint_dir: str = "data/features/CenterPoint", filter_policy: str = "combined_30m", velocity_threshold: float = 0.2):
        self.centerpoint_dir = centerpoint_dir
        self.filter_policy = filter_policy
        self.velocity_threshold = velocity_threshold

    def load_predictions(self, sample_token: str) -> list:
        return load_npz_objects(sample_token)

    def normalize_objects(self, sample_token: str) -> list[dict]:
        return [self._normalize_one(sample_token, obj, i).to_dict() for i, obj in enumerate(self.load_predictions(sample_token))]

    def derive_available_status(self, obj: dict) -> dict:
        attr = obj.get("attribute_name") or obj.get("attribute")
        if attr:
            return {"status": str(attr).lower().replace(" ", "_"), "source": "explicit_attribute"}
        velocity = obj.get("velocity")
        if velocity is None:
            box = np.asarray(obj.get("box") if obj.get("box") is not None else [], dtype=np.float32).reshape(-1)
            velocity = box[7:9].tolist() if len(box) >= 9 else None
        if velocity is not None and len(velocity) >= 2:
            speed = math.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2)
            return {"status": "moving" if speed > self.velocity_threshold else "stationary", "source": "velocity_coarse", "speed": speed}
        return {"status": "unknown", "source": "missing_attribute_and_velocity"}

    def emit_canonical_objects(self, sample_token: str) -> list[dict]:
        objects = self.normalize_objects(sample_token)
        if self.filter_policy == "combined_30m":
            return [o for o in objects if (distance_xy(o) is None or distance_xy(o) <= 30.0)]
        return objects

    def _normalize_one(self, sample_token: str, obj: dict, index: int) -> CanonicalSceneObject:
        box = np.asarray(obj.get("box") if obj.get("box") is not None else [], dtype=np.float32).reshape(-1).tolist()
        center = box[:3] if len(box) >= 3 else [None, None, None]
        size = box[3:6] if len(box) >= 6 else [None, None, None]
        yaw = box[6] if len(box) >= 7 else None
        velocity = obj.get("velocity")
        if velocity is None and len(box) >= 9:
            velocity = box[7:9]
        velocity = [float(velocity[0]), float(velocity[1])] if velocity is not None and len(velocity) >= 2 else None
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2) if velocity else None
        status_info = self.derive_available_status(obj)
        raw_label = obj.get("label") or obj.get("category")
        category = normalize_category_label(raw_label)["canonical_category"]
        score = obj.get("score") if obj.get("score") is not None else obj.get("detection_score")
        confidence = float(score) if score is not None else None
        return CanonicalSceneObject(
            object_id=f"cp_{index:04d}",
            sample_token=sample_token,
            category=category,
            geometry={"center_xyz": center, "size_wlh": size, "yaw": yaw, "raw_box": box},
            velocity_xy=velocity,
            speed=speed,
            status=status_info["status"],
            attribute=obj.get("attribute_name") or obj.get("attribute"),
            confidence=confidence,
            source="centerpoint",
            feature_ref=f"{sample_token}.npz:results[{index}].feats" if obj.get("feats") is not None else None,
            metadata={"raw_label": jsonable(raw_label), "status_source": status_info["source"], "box_length": len(box), "filter_policy": self.filter_policy},
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--filter_policy", default="combined_30m")
    ap.add_argument("--velocity_threshold", type=float, default=0.2)
    args = ap.parse_args()
    adapter = CenterPointSceneAdapter(filter_policy=args.filter_policy, velocity_threshold=args.velocity_threshold)
    rows = load_questions(args.max_samples)
    by_sample = {}
    raw_total = filtered_total = 0
    cats = Counter()
    statuses = Counter()
    velocity = score = attr = unknown = 0
    empty = []
    for token in sorted({str(r.get("sample_token")) for r in rows}):
        raw = adapter.normalize_objects(token)
        filtered = adapter.emit_canonical_objects(token)
        raw_total += len(raw)
        filtered_total += len(filtered)
        if not filtered:
            empty.append(token)
        for obj in filtered:
            cats[obj["category"]] += 1
            statuses[obj["status"]] += 1
            velocity += int(obj.get("velocity_xy") is not None)
            score += int(obj.get("confidence") is not None)
            attr += int(obj.get("attribute") is not None)
            unknown += int(obj.get("status") == "unknown")
        by_sample[token] = {"raw_count": len(raw), "filtered_count": len(filtered), "objects": filtered}
    report = {
        "filter_policy": args.filter_policy,
        "velocity_threshold": args.velocity_threshold,
        "sample_count": len(by_sample),
        "objects_before_filtering": raw_total,
        "objects_after_filtering": filtered_total,
        "filtered_out_objects": raw_total - filtered_total,
        "category_distribution": counts_dict(cats),
        "status_distribution": counts_dict(statuses),
        "unknown_status_count": unknown,
        "velocity_available_count": velocity,
        "score_available_count": score,
        "attribute_available_count": attr,
        "samples_with_no_objects": empty,
    }
    write_json(OUTPUT_DIR / "step71_centerpoint_canonical_objects.json", {"metadata": report, "samples": by_sample})
    write_md(OUTPUT_DIR / "step71_centerpoint_adapter_report.md", "# Step 7.1 CenterPoint Adapter Report\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in report.items()))


if __name__ == "__main__":
    main()
