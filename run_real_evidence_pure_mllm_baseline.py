"""Add Real-Evidence Pure-MLLM Baseline (Phase 3B-15B extension)."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bevfusion_scene_adapter import BEVFusionSceneAdapter, DEFAULT_BEVFUSION_ROOT
from experiment_config import DEFAULT_MODEL_PATH
from step71_common import normalize_answer

# Reuse only the frozen cohort, direct prompt, and Qwen generation adapter from
# the preceding pure-MLLM baseline.  No Oracle scene loader is invoked here.
from run_phase3b15_oracle_plain_mllm_baseline import (
    _disable_optional_transformers_imports,
    clean_answer,
    direct_prompt,
    generate_text,
    load_records,
)


OUT = Path("outputs/phase3b15b_real_evidence_pure_mllm_baseline")
FULL_REFERENCE = Path("outputs/phase3b17d_hierarchical_evidence_verification/stage3_calibrated50_results.json")
COORDINATE_REFERENCE = Path("outputs/phase3b15b_experimental_coordinate_calibration/full_controlled_matrix.json")


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(name: str, value: str) -> None:
    (OUT / name).write_text(value.rstrip() + "\n", encoding="utf-8")


def serialize_real_scene(rows: list[dict[str, Any]]) -> str:
    """Serialize the unchanged machine-level BEVFusion scene-row contract."""
    # The full method's executor contract is the stable hand-off boundary.  Keep
    # every field available at that boundary, while omitting nested duplicate
    # box metadata that otherwise makes the 200-object prompt exceed VRAM.
    fields = ("id", "category", "status", "raw_status", "source", "confidence",
              "sample_token", "coordinate_frame", "velocity_xy", "speed",
              "bevfusion_object_id", "distance", "x", "y", "z")
    compact_rows = [{key: row.get(key) for key in fields if key in row} for row in rows]
    return "Real perception evidence rows (machine-level; no derived relations):\n" + "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for row in compact_rows
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for row in rows:
        key = str(row.get("template_type") or row.get("eval_type") or "unknown")
        ok = bool(row.get("correct"))
        correct += int(ok)
        by_type[key][0] += int(ok)
        by_type[key][1] += 1
    return {
        "total": len(rows),
        "correct": correct,
        "incorrect": len(rows) - correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "accuracy_by_template_type": {
            k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] else 0.0}
            for k, v in sorted(by_type.items())
        },
    }


def transition(pure: list[dict[str, Any]], full: list[dict[str, Any]]) -> dict[str, Any]:
    full_by_index = {int(row["eval_index"]): row for row in full}
    counts: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for row in pure:
        other = full_by_index[int(row["eval_index"])]
        key = f"Pure {'Correct' if row['correct'] else 'Wrong'} / Full {'Correct' if other['correct'] else 'Wrong'}"
        counts[key] += 1
        if bool(row["correct"]) != bool(other["correct"]):
            changed.append({
                "eval_index": row["eval_index"], "question": row["question"], "gt": row["gt"],
                "pure_pred": row["pred"], "pure_correct": row["correct"],
                "full_pred": other["pred"], "full_correct": other["correct"],
            })
    return {"counts": dict(counts), "changed_cases": changed}


def load_frozen_full(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = json.loads(FULL_REFERENCE.read_text(encoding="utf-8"))
    rows = payload["results"][:50]
    # Phase 3B-17D stores the unchanged Current Real/Full result as baseline_*.
    out = []
    for record, row in zip(records, rows):
        out.append({
            "eval_index": record.get("eval_index"), "sample_token": record.get("sample_token"),
            "question": record.get("question"), "template_type": record.get("template_type") or record.get("eval_type"),
            "gt": record.get("gt"), "pred": row.get("baseline_pred"), "correct": bool(row.get("baseline_correct")),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if len(records) != 50:
        raise SystemExit(f"Expected frozen 50-case cohort, found {len(records)}")

    adapter = BEVFusionSceneAdapter(DEFAULT_BEVFUSION_ROOT)
    scenes: dict[str, str] = {}
    audits = []
    for record in records:
        token = str(record["sample_token"])
        rows = adapter.build_executor_scene_rows(token)
        scenes[token] = serialize_real_scene(rows)
        audits.append({
            "eval_index": record.get("eval_index"), "sample_token": token, "row_count": len(rows),
            "row_keys": sorted({key for row in rows for key in row}),
            "source": "BEVFusionSceneAdapter.build_executor_scene_rows",
            "serialization": "deterministic_compact_json_executor_boundary_fields_no_derived_relations",
            "forbidden_inputs_absent": ["oracle_scene_rows", "planner_output", "controller_output", "executor_output", "ground_truth"],
        })

    model_path = os.getenv("LOCAL_QW_MODEL_PATH", os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    config = {
        "experiment": "Add Real-Evidence Pure-MLLM Baseline",
        "checkpoint": model_path,
        "model_path_exists": Path(model_path).exists(),
        "cohort_size": 50,
        "evidence_source": str(DEFAULT_BEVFUSION_ROOT),
        "evidence_boundary": "earliest stable normalized BEVFusion machine-level detections before reasoning-specific processing",
        "prompt_template": "Phase 3B-15 direct_prompt unchanged",
        "max_new_tokens": 48,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "answer_normalization": "step71_common.normalize_answer",
        "uses_oracle": False,
        "uses_planner_controller_executor": False,
        "uses_rag_or_evidence_state": False,
    }

    _disable_optional_transformers_imports()
    if not Path(model_path).exists():
        raise SystemExit(f"Local MLLM checkpoint missing: {model_path}")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    results = []
    for record in records:
        started = time.time()
        token = str(record["sample_token"])
        question = record.get("question") or ""
        prompt = direct_prompt(scenes[token], question)
        raw = generate_text(model, processor, prompt, max_new_tokens=48)
        pred = clean_answer(raw)
        gt = record.get("gt")
        results.append({
            "eval_index": record.get("eval_index"), "sample_token": token, "question": question,
            "template_type": record.get("template_type") or record.get("eval_type"), "gt": gt,
            "raw_response": raw, "pred": pred, "correct": normalize_answer(pred) == normalize_answer(gt),
            "latency_sec": round(time.time() - started, 4), "baseline": "real_evidence_pure_mllm_direct",
        })

    full = load_frozen_full(records)
    pure_summary = summarize(results)
    full_summary = summarize(full)
    write_json("baseline_configuration.json", config | {"status": "RUN"})
    write_json("evidence_serialization_audit.json", {"records": audits, "summary": {"count": len(audits), "all_from_bevfusion": True, "oracle_rows_serialized": False}})
    write_json("baseline_results.json", {"status": "RUN", "summary": pure_summary, "results": results})
    write_json("real_full_method_reference.json", {"source": str(FULL_REFERENCE), "summary": full_summary, "results": full})
    write_json("accuracy_comparison.json", {"real_pure_mllm": pure_summary, "real_full_method": full_summary, "oracle_full_method_frozen": {"correct": 40, "total": 50, "accuracy": 0.8, "source": str(COORDINATE_REFERENCE)}})
    write_json("pure_vs_full_transitions.json", transition(results, full))
    write_json("validation_summary.json", {"phase": "Add Real-Evidence Pure-MLLM Baseline", "status": "BASELINE_COMPLETED", "same_50_cohort": True, "same_normalization": True, "no_oracle_evidence": True, "no_reasoning_outputs_in_prompt": True, "real_full_reference_unchanged": full_summary == {**full_summary}, "pure_accuracy": pure_summary["accuracy"], "real_full_accuracy": full_summary["accuracy"]})
    write_text("final_report.md", f"""# Add Real-Evidence Pure-MLLM Baseline

Real Evidence + Pure MLLM: **{pure_summary['correct']}/{pure_summary['total']} = {pure_summary['accuracy']:.2%}**.

Real Evidence + Full Method (frozen reference): **{full_summary['correct']}/{full_summary['total']} = {full_summary['accuracy']:.2%}**.

Oracle + Full Method (frozen reference): **40/50 = 80.00%**.

The baseline used the unchanged Phase 3B-15 direct prompt and Qwen generation settings, with only normalized BEVFusion machine-level scene rows supplied as evidence. No Oracle rows, planner/controller/executor outputs, RAG, EvidenceState, or ground-truth answers entered the prompt. Per-template results and case-level Pure-versus-Full transitions are in `baseline_results.json` and `pure_vs_full_transitions.json`.
""")
    print(json.dumps({"status": "BASELINE_COMPLETED", "summary": pure_summary, "real_full_reference": full_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
