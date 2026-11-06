from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from centerpoint_scene_adapter_step71 import CenterPointSceneAdapter
from step71_common import OUTPUT_DIR, counts_dict, distance_xy, load_questions, normalize_answer, write_json, write_md


POLICIES = [
    "raw_all",
    "distance_20m",
    "distance_30m",
    "distance_50m",
    "combined_30m",
    "combined_50m",
    "category_preserving_30m",
    "category_preserving_50m",
]


def raw_objects_by_sample(max_samples: int = 50) -> dict[str, list[dict]]:
    adapter = CenterPointSceneAdapter(filter_policy="raw_all")
    tokens = sorted({str(r.get("sample_token")) for r in load_questions(max_samples)})
    return {token: adapter.normalize_objects(token) for token in tokens}


def apply_policy(objects: list[dict], policy: str) -> list[dict]:
    if policy == "raw_all":
        return list(objects)
    if policy.startswith("distance_"):
        limit = float(policy.split("_")[1].replace("m", ""))
        return [o for o in objects if distance_xy(o) is None or distance_xy(o) <= limit]
    if policy.startswith("combined_"):
        limit = float(policy.split("_")[1].replace("m", ""))
        return [o for o in objects if distance_xy(o) is None or distance_xy(o) <= limit]
    if policy.startswith("category_preserving_"):
        limit = float(policy.split("_")[2].replace("m", ""))
        kept = [o for o in objects if distance_xy(o) is None or distance_xy(o) <= limit]
        kept_categories = {o.get("category") for o in kept}
        by_cat: dict[str, list[dict]] = {}
        for obj in objects:
            by_cat.setdefault(str(obj.get("category")), []).append(obj)
        for category, cat_objects in by_cat.items():
            if category not in kept_categories:
                nearest = sorted(cat_objects, key=lambda o: distance_xy(o) if distance_xy(o) is not None else 1e9)[0]
                kept.append(nearest)
        return sorted(kept, key=lambda o: str(o.get("object_id")))
    raise ValueError(f"unknown policy: {policy}")


def separated_status(obj: dict) -> dict:
    status = str(obj.get("status") or "unknown")
    if status in {"moving", "stationary"}:
        motion_status = status
        semantic_status = "unknown"
    elif status == "unknown":
        motion_status = "unknown"
        semantic_status = "unknown"
    else:
        motion_status = "unknown"
        semantic_status = status
    return {"motion_status": motion_status, "semantic_status": semantic_status}


def summarize_objects(objects: list[dict]) -> dict:
    return {
        "count": len(objects),
        "category_distribution": counts_dict(Counter(str(o.get("category")) for o in objects)),
        "motion_status_distribution": counts_dict(Counter(separated_status(o)["motion_status"] for o in objects)),
        "semantic_status_distribution": counts_dict(Counter(separated_status(o)["semantic_status"] for o in objects)),
    }


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def eval_rows(preds: list[dict]) -> dict:
    rows = load_questions(50)
    correct = [normalize_answer(r.get("answer")) == normalize_answer(p.get("prediction")) for r, p in zip(rows, preds)]
    by_type: dict[str, list[int]] = {}
    for row, ok in zip(rows, correct):
        t = str(row.get("template_type") or "unknown")
        by_type.setdefault(t, [0, 0])
        by_type[t][0] += int(ok)
        by_type[t][1] += 1
    return {
        "total": len(rows),
        "correct": int(sum(correct)),
        "accuracy": sum(correct) / len(rows) if rows else 0.0,
        "accuracy_by_question_type": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] else 0.0} for k, v in sorted(by_type.items())},
    }
