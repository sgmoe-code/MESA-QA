from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from step71_common import OUTPUT_DIR, PROJECT_ROOT, load_questions, normalize_answer, write_json, write_md


CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

STEP74A_PREFIX = "step74a_"
DEFAULT_MODEL_PATHS = [
    os.getenv("LOCAL_QW_MODEL_PATH"),
    os.getenv("MODEL_PATH"),
    os.getenv("MESA_QA_MODEL_PATH"),
    str(Path(__file__).resolve().parent / "models" / "Qwen3.5-9B"),
]


def out_path(name: str) -> Path:
    if not name.startswith(STEP74A_PREFIX):
        raise ValueError(f"Step 7.4A output must start with {STEP74A_PREFIX}: {name}")
    return OUTPUT_DIR / name


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def count_by(rows: list[dict], key: str) -> dict:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def first_existing_model_path() -> str | None:
    for raw in DEFAULT_MODEL_PATHS:
        if raw and Path(raw).exists():
            return str(Path(raw))
    return None


def qwen_backend_config() -> dict:
    model_path = first_existing_model_path()
    return {
        "backend_name": "local_qwen_vl_transformers",
        "model_path": model_path,
        "enable_local_mllm": bool(model_path),
        "device": os.getenv("LOCAL_QW_DEVICE", "cuda"),
        "max_new_tokens": int(os.getenv("LOCAL_QW_MAX_NEW_TOKENS", "220")),
    }


def image_readability(path: str | None) -> dict:
    if not path:
        return {"exists": False, "readable": False, "dimensions": None, "error": "missing_path"}
    p = Path(path)
    if not p.exists():
        return {"exists": False, "readable": False, "dimensions": None, "error": "path_not_found"}
    try:
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            return {"exists": True, "readable": True, "dimensions": [int(img.width), int(img.height)], "error": None}
    except Exception as exc:
        return {"exists": True, "readable": False, "dimensions": None, "error": f"{type(exc).__name__}: {exc}"}


def load_image_manifest() -> dict:
    data = read_json(out_path("step74a_image_asset_manifest.json"), {})
    return data.get("samples", {}) if isinstance(data, dict) else {}


def selected_image_paths(request: dict, manifest: dict) -> list[str]:
    sample = manifest.get(str(request.get("sample_token"))) or {}
    views = sample.get("views") if isinstance(sample.get("views"), dict) else {}
    paths = []
    for view in request.get("view_hints") or []:
        row = views.get(view) or {}
        if row.get("readable") and row.get("path"):
            paths.append(row["path"])
    if paths:
        return paths
    stitched = sample.get("stitched_image_path")
    if stitched and Path(stitched).exists():
        return [stitched]
    for view in CAMERA_CHANNELS:
        row = views.get(view) or {}
        if row.get("readable") and row.get("path"):
            return [row["path"]]
    return []


def request_sort_key(request: dict) -> tuple[int, str]:
    order = {"semantic_status": 0, "validated_count": 1, "reference_grounding": 2, "visibility": 3}
    return (order.get(str(request.get("predicate_type")), 9), str(request.get("request_id")))


def evidence_from_validated(request: dict, adapter_result: dict, parsed: dict, validation: dict) -> dict:
    predicate = request.get("predicate_type")
    target = {"category": request.get("target_category")}
    if predicate == "semantic_status":
        target["status"] = request.get("target_status")
    candidate_ids = [c.get("object_id") for c in request.get("candidate_objects", []) if c.get("object_id")]
    return {
        "request_id": request.get("request_id"),
        "idx": request.get("idx"),
        "sample_token": request.get("sample_token"),
        "predicate_type": predicate,
        "operation_group": request.get("operation_group"),
        "target": target,
        "source": "vision_predicate_completion_step74a",
        "support": normalize_support(parsed.get("support", "yes" if predicate == "validated_count" else "unknown")),
        "confidence": normalize_confidence(parsed.get("confidence")),
        "evidence_value": parsed,
        "candidate_object_ids": candidate_ids,
        "used_views": [path_to_view(p) for p in adapter_result.get("used_image_paths", [])],
        "validated": bool(validation.get("valid")),
        "rejected_reason": None,
        "not_answering_full_question": parsed.get("not_answering_full_question") is True,
        "mode_used": adapter_result.get("mode_used"),
        "latency_sec": adapter_result.get("latency_sec"),
    }


def rejected_evidence(request: dict, reason: str, adapter_result: dict | None = None) -> dict:
    adapter_result = adapter_result or {}
    return {
        "request_id": request.get("request_id"),
        "idx": request.get("idx"),
        "sample_token": request.get("sample_token"),
        "predicate_type": request.get("predicate_type"),
        "operation_group": request.get("operation_group"),
        "target": {"category": request.get("target_category"), "status": request.get("target_status")},
        "source": "vision_predicate_completion_step74a",
        "support": "unsupported",
        "confidence": "none",
        "evidence_value": {},
        "candidate_object_ids": [c.get("object_id") for c in request.get("candidate_objects", []) if c.get("object_id")],
        "used_views": [path_to_view(p) for p in adapter_result.get("used_image_paths", [])],
        "validated": False,
        "rejected_reason": reason,
        "not_answering_full_question": True,
        "mode_used": adapter_result.get("mode_used", "unavailable"),
        "latency_sec": adapter_result.get("latency_sec"),
    }


def path_to_view(path: str) -> str:
    for cam in CAMERA_CHANNELS:
        if cam in path:
            return cam
    return "six_view_stitched" if "six_view" in path or "stitched" in path else "unknown"


def normalize_support(value: Any) -> str:
    v = str(value).strip().lower()
    return v if v in {"yes", "no", "unknown", "unsupported"} else "unknown"


def normalize_confidence(value: Any) -> str:
    v = str(value).strip().lower()
    return v if v in {"low", "medium", "high"} else "none"


def completion_report(requests: list[dict], raw_rows: list[dict], evidence: list[dict], rejected: list[dict]) -> dict:
    attempted = len(requests)
    call_success = sum(1 for r in raw_rows if r.get("call_success"))
    parsed = sum(1 for r in raw_rows if r.get("parse_success"))
    valid = sum(1 for r in raw_rows if r.get("schema_valid"))
    latencies = [r.get("latency_sec") for r in raw_rows if isinstance(r.get("latency_sec"), (int, float))]
    return {
        "requests_attempted": attempted,
        "call_success_rate": call_success / attempted if attempted else 0.0,
        "parse_success_rate": parsed / attempted if attempted else 0.0,
        "schema_validation_rate": valid / attempted if attempted else 0.0,
        "support_distribution": count_by(evidence + rejected, "support"),
        "confidence_distribution": count_by(evidence + rejected, "confidence"),
        "unknown_rate": sum(1 for e in evidence if e.get("support") == "unknown") / len(evidence) if evidence else 0.0,
        "rejected_outputs": len(rejected),
        "direct_answer_violations": sum(1 for r in raw_rows if r.get("direct_answer_violation")),
        "comparison_answer_violations": sum(1 for r in raw_rows if r.get("comparison_answer_violation")),
        "backend_errors": [r.get("error") for r in raw_rows if r.get("error")][:10],
        "latency_sec": {
            "count": len(latencies),
            "avg": statistics.mean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def evaluation_from_predictions(predictions_name: str, traces_name: str) -> tuple[dict, list[dict]]:
    rows = load_questions(50)
    preds = read_json(out_path(predictions_name), [])
    traces = read_json(out_path(traces_name), [])
    correct = [normalize_answer(r.get("answer")) == normalize_answer(p.get("prediction")) for r, p in zip(rows, preds)]
    by_type = defaultdict(lambda: [0, 0])
    change_by_group = defaultdict(lambda: {"improved": 0, "regressed": 0, "unchanged": 0})
    prev = read_json(OUTPUT_DIR / "step74_source_aware_vision_predictions.json", [])
    case_table = []
    for row, pred, ok, old in zip(rows, preds, correct, prev or [{}] * len(preds)):
        t = str(row.get("template_type") or "unknown")
        by_type[t][0] += int(ok)
        by_type[t][1] += 1
        old_ok = normalize_answer(row.get("answer")) == normalize_answer(old.get("prediction"))
        state = "improved" if ok and not old_ok else "regressed" if old_ok and not ok else "unchanged"
        change_by_group[t][state] += 1
        case_table.append({
            "idx": pred.get("idx"),
            "template_type": t,
            "answer": row.get("answer"),
            "step74_prediction": old.get("prediction"),
            "step74a_prediction": pred.get("prediction"),
            "improvement_state": state,
            "execution_status": pred.get("execution_status"),
            "unsupported_reason": pred.get("unsupported_reason"),
        })
    ev = read_json(out_path("step74a_completed_predicate_evidence.json"), [])
    evaluation = {
        "total": len(rows),
        "correct": int(sum(correct)),
        "accuracy": sum(correct) / len(rows) if rows else 0.0,
        "accuracy_by_question_type": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] else 0.0} for k, v in sorted(by_type.items())},
        "execution_status_distribution": count_by(preds, "execution_status"),
        "valid_answer_rate": sum(1 for p in preds if p.get("prediction") != "unknown") / len(preds) if preds else 0.0,
        "unsupported_reason_distribution": dict(Counter(p.get("unsupported_reason") for p in preds if p.get("unsupported_reason"))),
        "predicate_gap_reduction": {
            "requested": len(ev),
            "usable": sum(1 for e in ev if e.get("validated") and not e.get("rejected_reason") and e.get("support") != "unsupported"),
            "unsupported": sum(1 for e in ev if e.get("support") == "unsupported"),
        },
        "change_by_question_type": dict(change_by_group),
        "trace_count": len(traces),
    }
    return evaluation, case_table

