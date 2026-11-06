from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "CenterPoint"
MINI_VAL_QA = PROJECT_ROOT / "data" / "questions" / "mini_val_for_agent.json"
BASELINE_50_DIR = OUTPUT_DIR / "agent_atomic_program_50_step385_50_no_scene_banker_append_one"


def ensure_outputs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def write_json(path: str | Path, data: Any) -> None:
    ensure_outputs()
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_md(path: str | Path, text: str) -> None:
    ensure_outputs()
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def load_questions(limit: int | None = 50) -> list[dict]:
    data = json.loads(MINI_VAL_QA.read_text(encoding="utf-8"))
    rows = data.get("questions", data) if isinstance(data, dict) else data
    rows = list(rows)
    return rows[:limit] if limit else rows


def load_npz_objects(sample_token: str) -> list[dict]:
    path = FEATURE_DIR / f"{sample_token}.npz"
    if not path.exists():
        return []
    with np.load(path, allow_pickle=True) as z:
        results = z.get("results")
        if results is None:
            return []
        return [dict(x) for x in results if isinstance(x, dict)]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def normalize_answer(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ")


def evaluate_predictions(rows: list[dict], predictions: list[str]) -> dict:
    correct = [normalize_answer(r.get("answer")) == normalize_answer(p) for r, p in zip(rows, predictions)]
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, ok in zip(rows, correct):
        t = str(row.get("template_type") or "unknown")
        by_type[t][0] += int(ok)
        by_type[t][1] += 1
    return {
        "total": len(rows),
        "correct": int(sum(correct)),
        "accuracy": float(sum(correct) / len(rows)) if rows else 0.0,
        "accuracy_by_question_type": {
            k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] else 0.0}
            for k, v in sorted(by_type.items())
        },
    }


def object_xy(obj: dict) -> tuple[float | None, float | None]:
    geom = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else {}
    center = geom.get("center_xyz") or obj.get("translation") or []
    if len(center) < 2:
        return None, None
    try:
        return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None, None


def distance_xy(obj: dict) -> float | None:
    x, y = object_xy(obj)
    if x is None or y is None:
        return None
    return math.sqrt(x * x + y * y)


def infer_category(question: str) -> str | None:
    q = question.lower()
    aliases = {
        "construction_vehicle": ["construction vehicle", "construction vehicles"],
        "traffic_cone": ["traffic cone", "traffic cones", "cone", "cones"],
        "pedestrian": ["pedestrian", "pedestrians", "person", "people"],
        "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
        "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
        "trailer": ["trailer", "trailers"],
        "truck": ["truck", "trucks"],
        "bus": ["bus", "buses"],
        "barrier": ["barrier", "barriers"],
        "car": ["car", "cars", "vehicle", "vehicles"],
    }
    for category, words in aliases.items():
        if any(w in q for w in words):
            return category
    if "thing" in q or "object" in q:
        return "thing"
    return None


def infer_status_terms(question: str) -> list[str]:
    q = question.lower().replace("-", " ")
    terms = []
    if "not standing" in q:
        terms.append("not_standing")
    for term in ["moving", "stationary", "standing", "parked", "stopped", "sitting", "with rider", "without rider"]:
        if term in q:
            terms.append(term.replace(" ", "_"))
    return terms


def infer_relation_terms(question: str) -> list[str]:
    q = question.lower().replace("-", " ")
    out = []
    for phrase in ["front left", "front right", "back left", "back right", "front", "back", "left", "right"]:
        if phrase in q:
            out.append(phrase.replace(" ", "_"))
    return out


def load_baseline_records(limit: int = 50) -> list[dict]:
    path = BASELINE_50_DIR / "atomic_program_results.jsonl"
    if not path.exists():
        path = OUTPUT_DIR / "agent_atomic_program_50" / "atomic_program_results.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def counts_dict(counter: Counter) -> dict:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}
