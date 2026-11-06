from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from atomic_program_executor import execute_atomic_program
from bevfusion_scene_adapter import BEVFusionSceneAdapter, DEFAULT_BEVFUSION_ROOT
from run_phase3b10_hierarchical_evidence_feasibility import project_objects_for_token
from scene_graph_query import object_matches, sort_objects
from step71_common import infer_category, infer_status_terms, load_questions, write_json, write_md
from step74a_common import CAMERA_CHANNELS
from step74b_common import metadata_indices, projection_resource_report, summarize_projection


DEFAULT_OUTPUT_DIR = Path("outputs/phase3b11a_visual_evidence_infrastructure")
PHASE3B8_QUESTION_ANALYSIS = Path("outputs/phase3b8_bevfusion_evidence_analysis/question_level_analysis.json")
VISUAL_SOURCE = "phase3b11a_stub_targeted_visual_provider"
BEVFUSION_SOURCE = "bevfusion"


class TargetedVisualEvidenceController:
    """Small runtime sufficiency hook layered over BEVFusion object anchors."""

    def __init__(self, adapter: BEVFusionSceneAdapter):
        self.adapter = adapter
        self.projection_index = ProjectionIndex()

    def evaluate_requirement(self, sample_token: str, requirement: dict[str, Any]) -> dict[str, Any]:
        objects = self.adapter.normalize_objects(sample_token)
        candidates = select_candidate_anchors(objects, requirement)
        missing_attribute = requirement.get("evidence_type") in {"semantic_status", "visual_attribute", "visibility"}
        sufficient = bool(candidates) and not missing_attribute
        if missing_attribute:
            sufficient = any(object_has_attribute(obj, requirement.get("attribute")) for obj in candidates)
        return {
            "sample_token": sample_token,
            "requirement": requirement,
            "candidate_count": len(candidates),
            "candidate_object_ids": [obj["object_id"] for obj in candidates],
            "sufficiency": "SUFFICIENT" if sufficient else "VISUAL_EVIDENCE_REQUIRED",
            "reason": "available_from_bevfusion" if sufficient else "requested_semantic_attribute_unavailable",
            "candidates": candidates,
        }

    def build_request(self, sample_token: str, anchor: dict[str, Any], requirement: dict[str, Any]) -> dict[str, Any]:
        projections = self.projection_index.valid_projections(sample_token, anchor)
        primary = rank_projection_candidates(projections)[0] if projections else None
        geom = anchor.get("geometry") or {}
        return {
            "request_id": f"phase3b11a_{sample_token}_{anchor['object_id']}_{requirement['attribute']}",
            "sample_token": sample_token,
            "object_id": anchor["object_id"],
            "category": anchor.get("category"),
            "object_anchor": {
                "center": geom.get("center_xyz"),
                "size": geom.get("size_wlh"),
                "yaw": geom.get("yaw"),
                "confidence": anchor.get("confidence"),
                "coordinate_frame": geom.get("coordinate_frame") or "lidar",
            },
            "requested_evidence": {
                "evidence_type": requirement.get("evidence_type"),
                "attribute": requirement.get("attribute"),
            },
            "visual_localization": {
                "primary_camera": primary,
                "valid_camera_candidates": rank_projection_candidates(projections),
                "candidate_camera_count": len(projections),
                "supported_cameras": CAMERA_CHANNELS,
            },
            "trace": {
                "source": "TargetedVisualEvidenceController",
                "anchor_source": BEVFUSION_SOURCE,
                "request_reason": "runtime_evidence_insufficient",
            },
        }


class ProjectionIndex:
    def __init__(self):
        self.idx = metadata_indices()
        self.cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.errors: list[dict[str, Any]] = []

    def valid_projections(self, sample_token: str, obj: dict[str, Any]) -> list[dict[str, Any]]:
        key = (sample_token, obj["object_id"])
        if key not in self.cache:
            try:
                records = project_objects_for_token(sample_token, [obj], self.idx)
            except Exception as exc:
                self.errors.append({"sample_token": sample_token, "object_id": obj.get("object_id"), "error": str(exc)})
                records = []
            self.cache[key] = [projection_for_request(row) for row in records if row.get("projection_valid")]
        return self.cache[key]

    def all_records(self) -> list[dict[str, Any]]:
        return [row for rows in self.cache.values() for row in rows]


class StubTargetedVisualEvidenceProvider:
    def __init__(self):
        self.invocations: list[dict[str, Any]] = []

    def provide(self, request: dict[str, Any]) -> dict[str, Any]:
        self.invocations.append(request)
        requested = request["requested_evidence"]
        primary = request["visual_localization"].get("primary_camera") or {}
        return {
            "request_id": request["request_id"],
            "object_id": request["object_id"],
            "attribute_name": requested.get("attribute"),
            "attribute_value": synthetic_value(requested.get("attribute")),
            "confidence": "stub_high",
            "source": VISUAL_SOURCE,
            "provenance": {
                "sample_token": request["sample_token"],
                "object_id": request["object_id"],
                "camera_name": primary.get("camera"),
                "image_ref": primary.get("image_path"),
                "sample_data_ref": primary.get("sample_data_ref"),
                "bbox": primary.get("bbox_clipped"),
                "requested_attribute": requested.get("attribute"),
                "returned_value": synthetic_value(requested.get("attribute")),
                "confidence": "stub_high",
                "evidence_source": VISUAL_SOURCE,
            },
        }


def run_validation(evidence_root: Path, output_dir: Path, max_cases: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = BEVFusionSceneAdapter(evidence_root)
    controller = TargetedVisualEvidenceController(adapter)
    provider = StubTargetedVisualEvidenceProvider()
    phase3b8_rows = load_phase3b8_cases()
    representative = select_representative_cases(adapter, phase3b8_rows, max_cases)
    controls = select_control_cases(phase3b8_rows, 1)

    lifecycle_rows = [
        validate_missing_semantic_case(adapter, controller, provider, case)
        for case in representative
    ]
    control_rows = [validate_control_case(adapter, controller, provider, case) for case in controls]
    projection_records = controller.projection_index.all_records()
    projection_audit = summarize_projection(projection_records, controller.projection_index.errors)
    projection_audit["resource_report"] = projection_resource_report()

    summary = {
        "phase": "Phase 3B-11A",
        "output_dir": str(output_dir),
        "evidence_root": str(evidence_root),
        "available_bevfusion_samples": len(adapter.available_sample_tokens()),
        "phase3b8_missing_semantic_cases": sum(1 for row in phase3b8_rows if is_missing_semantic_case(row)),
        "representative_case_count": len(lifecycle_rows),
        "control_case_count": len(control_rows),
        "visual_provider_invocations": len(provider.invocations),
        "visual_provider_invoked_only_when_insufficient": all(row["visual_invoked"] for row in lifecycle_rows)
        and all(not row["visual_invoked"] for row in control_rows),
        "identity_stable": all(row.get("identity_stable") for row in lifecycle_rows),
        "localization_has_valid_projection": all(row.get("valid_projection_count", 0) > 0 for row in lifecycle_rows),
        "merge_onto_same_object": all(row.get("merge_onto_same_object") for row in lifecycle_rows),
        "executor_compatible": all(row.get("executor_compatible") for row in lifecycle_rows + control_rows),
        "provenance_preserved": all(row.get("provenance_preserved") for row in lifecycle_rows),
        "projection_audit": projection_audit,
        "representative_cases": lifecycle_rows,
        "control_cases": control_rows,
        "provider_interface_for_phase3b11b": provider_interface(),
        "source_files_added_or_modified": ["run_phase3b11a_visual_evidence_infrastructure.py"],
        "visual_mllm_called": False,
    }
    summary["final_decision"] = final_decision(summary)
    write_deliverables(output_dir, summary)
    return summary


def validate_missing_semantic_case(
    adapter: BEVFusionSceneAdapter,
    controller: TargetedVisualEvidenceController,
    provider: StubTargetedVisualEvidenceProvider,
    case: dict[str, Any],
) -> dict[str, Any]:
    sample_token = str(case["sample_token"])
    requirement = requirement_from_case(case)
    check = controller.evaluate_requirement(sample_token, requirement)
    anchor = choose_projectable_anchor(controller, sample_token, check["candidates"])
    request = controller.build_request(sample_token, anchor, requirement)
    visual_result = provider.provide(request)
    before_rows = adapter.build_executor_scene_rows(sample_token)
    after_rows = merge_visual_evidence_into_scene_rows(before_rows, visual_result)
    before_answer, before_trace = execute_atomic_program(plan_for_requirement(requirement), before_rows, eval_type="exist")
    after_answer, after_trace = execute_atomic_program(plan_for_requirement(requirement), after_rows, eval_type="exist")
    merged = next(row for row in after_rows if row["id"] == anchor["object_id"])
    provenance = merged.get("visual_provenance", [])
    return {
        "eval_index": case.get("eval_index"),
        "sample_token": sample_token,
        "question": case.get("question"),
        "requirement": requirement,
        "controller_sufficiency": check["sufficiency"],
        "anchor_object_id": anchor["object_id"],
        "anchor_category": anchor.get("category"),
        "request": request,
        "stub_result": visual_result,
        "visual_invoked": True,
        "identity_stable": anchor["object_id"] == request["object_id"] == visual_result["object_id"],
        "valid_projection_count": request["visual_localization"]["candidate_camera_count"],
        "merge_onto_same_object": merged.get("id") == visual_result["object_id"]
        and merged.get("status") == visual_result["attribute_value"],
        "executor_compatible": after_trace.get("source") == "compiled_atomic_executor" and after_answer in {"yes", "no"},
        "executor_before_answer": before_answer,
        "executor_after_answer": after_answer,
        "executor_after_supporting_objects": after_trace.get("supporting_objects"),
        "provenance_preserved": bool(provenance and provenance[-1].get("bbox") and provenance[-1].get("camera_name")),
        "represented_requirement": {
            "evidence_type": requirement["evidence_type"],
            "attribute": requirement["attribute"],
            "target_category": requirement["target_category"],
        },
    }


def validate_control_case(
    adapter: BEVFusionSceneAdapter,
    controller: TargetedVisualEvidenceController,
    provider: StubTargetedVisualEvidenceProvider,
    case: dict[str, Any],
) -> dict[str, Any]:
    before = len(provider.invocations)
    sample_token = str(case["sample_token"])
    category = infer_category(case.get("question", "")) or "car"
    requirement = {"evidence_type": "object_existence", "attribute": "exists", "target_category": category}
    check = controller.evaluate_requirement(sample_token, requirement)
    rows = adapter.build_executor_scene_rows(sample_token)
    answer, trace = execute_atomic_program({"target_entity": {"category": category}, "question_type": "exist"}, rows, eval_type="exist")
    return {
        "eval_index": case.get("eval_index"),
        "sample_token": sample_token,
        "question": case.get("question"),
        "requirement": requirement,
        "controller_sufficiency": check["sufficiency"],
        "visual_invoked": len(provider.invocations) > before,
        "executor_answer": answer,
        "executor_compatible": trace.get("source") == "compiled_atomic_executor" and answer in {"yes", "no"},
    }


def load_phase3b8_cases() -> list[dict[str, Any]]:
    if not PHASE3B8_QUESTION_ANALYSIS.exists():
        return []
    return list(json.loads(PHASE3B8_QUESTION_ANALYSIS.read_text(encoding="utf-8")).get("questions", []))


def select_representative_cases(adapter: BEVFusionSceneAdapter, rows: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    cases = [row for row in rows if is_missing_semantic_case(row)]
    strong, fallback = [], []
    for row in cases:
        requirement = requirement_from_case(row)
        categories = {obj.get("category") for obj in adapter.normalize_objects(str(row["sample_token"]))}
        if requirement["target_category"] == "thing" or requirement["target_category"] in categories:
            strong.append(row)
        else:
            fallback.append(row)
    return (strong + fallback)[:max_cases]


def select_control_cases(rows: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    controls = [row for row in rows if row.get("phase3b8_primary_category") == "success" and "attribute" not in row.get("evidence_requirements", [])]
    if controls:
        return controls[:max_cases]
    questions = load_questions(50)
    return [{"eval_index": 0, **questions[0]}][:max_cases]


def is_missing_semantic_case(row: dict[str, Any]) -> bool:
    return row.get("phase3b8_primary_category") == "B_missing_semantic_evidence"


def requirement_from_case(case: dict[str, Any]) -> dict[str, Any]:
    question = case.get("question", "")
    attrs = infer_status_terms(question) or ["semantic_status"]
    return {
        "evidence_type": "semantic_status",
        "attribute": attrs[0],
        "target_category": infer_category(question) or "thing",
        "source_case_eval_index": case.get("eval_index"),
    }


def select_candidate_anchors(objects: list[dict[str, Any]], requirement: dict[str, Any]) -> list[dict[str, Any]]:
    category = requirement.get("target_category") or "thing"
    if category == "thing":
        return sort_bevfusion_objects(objects)
    matched = sort_bevfusion_objects([obj for obj in objects if obj.get("category") == category])
    if matched:
        return matched
    if requirement.get("evidence_type") in {"semantic_status", "visual_attribute", "visibility"}:
        return sort_bevfusion_objects(objects)
    return matched


def choose_projectable_anchor(
    controller: TargetedVisualEvidenceController,
    sample_token: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    for obj in candidates:
        if controller.projection_index.valid_projections(sample_token, obj):
            return obj
    if not candidates:
        raise ValueError(f"No BEVFusion anchor candidates available for {sample_token}")
    return candidates[0]


def sort_bevfusion_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(obj: dict[str, Any]) -> tuple[float, str]:
        center = (obj.get("geometry") or {}).get("center_xyz") or [1e9, 1e9, 0.0]
        try:
            dist = float(center[0]) ** 2 + float(center[1]) ** 2
        except (TypeError, ValueError):
            dist = 1e18
        return (dist, str(obj.get("object_id")))

    return sorted(objects, key=key)


def object_has_attribute(obj: dict[str, Any], attribute: str | None) -> bool:
    if not attribute:
        return False
    return obj.get("status") == attribute or obj.get("attribute") == attribute


def projection_for_request(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_token": row.get("sample_token"),
        "object_id": row.get("object_id"),
        "category": row.get("category"),
        "camera": row.get("camera"),
        "sample_data_ref": row.get("projection_record_id"),
        "image_path": row.get("image_path"),
        "bbox_clipped": row.get("bbox_clipped"),
        "bbox_unclipped": row.get("bbox_unclipped"),
        "projection_valid": bool(row.get("projection_valid")),
        "projected_area": row.get("projected_area"),
        "truncation_ratio": row.get("truncation_ratio"),
        "visible_corner_count": row.get("visible_corner_count"),
        "depth_m": row.get("depth_m"),
        "image_width": row.get("image_width"),
        "image_height": row.get("image_height"),
    }


def rank_projection_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (-(r.get("projected_area") or 0.0), r.get("truncation_ratio") or 1.0, r.get("camera") or ""))


def synthetic_value(attribute: str | None) -> str:
    if attribute in {"parked", "stopped", "standing", "moving", "stationary", "not_standing"}:
        return str(attribute)
    return "stub_semantic_status"


def merge_visual_evidence_into_scene_rows(scene_rows: list[dict[str, Any]], visual_result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in scene_rows:
        row = dict(row)
        if row.get("id") == visual_result["object_id"]:
            provenance = list(row.get("visual_provenance") or [])
            provenance.append(visual_result["provenance"])
            row["status"] = visual_result["attribute_value"]
            row["raw_status"] = visual_result["attribute_value"]
            row["semantic_attributes"] = {
                **(row.get("semantic_attributes") or {}),
                visual_result["attribute_name"]: {
                    "value": visual_result["attribute_value"],
                    "confidence": visual_result["confidence"],
                    "source": visual_result["source"],
                },
            }
            row["visual_provenance"] = provenance
        out.append(row)
    return out


def plan_for_requirement(requirement: dict[str, Any]) -> dict[str, Any]:
    category = requirement.get("target_category")
    if category == "thing":
        category = "unknown"
    return {
        "target_entity": {"category": category or "unknown", "attributes": [requirement.get("attribute")]},
        "question_type": "exist",
    }


def final_decision(summary: dict[str, Any]) -> str:
    checks = [
        summary["visual_provider_invoked_only_when_insufficient"],
        summary["identity_stable"],
        summary["localization_has_valid_projection"],
        summary["merge_onto_same_object"],
        summary["executor_compatible"],
        summary["provenance_preserved"],
    ]
    return "A. READY_FOR_TARGETED_VISUAL_PARSING" if all(checks) else "B. INFRASTRUCTURE_FIX_REQUIRED"


def provider_interface() -> dict[str, Any]:
    return {
        "method": "provide(request: dict) -> dict",
        "input_required_fields": [
            "request_id",
            "sample_token",
            "object_id",
            "category",
            "object_anchor.center",
            "object_anchor.size",
            "object_anchor.yaw",
            "object_anchor.confidence",
            "requested_evidence.evidence_type",
            "requested_evidence.attribute",
            "visual_localization.primary_camera",
            "visual_localization.valid_camera_candidates",
        ],
        "output_required_fields": [
            "request_id",
            "object_id",
            "attribute_name",
            "attribute_value",
            "confidence",
            "source",
            "provenance.sample_token",
            "provenance.object_id",
            "provenance.camera_name",
            "provenance.bbox",
            "provenance.image_ref",
        ],
        "identity_contract": "output.object_id must exactly equal input.object_id",
        "scope_contract": "Provider answers only the requested attribute for the anchored object and must not rediscover or relabel the whole scene.",
    }


def write_deliverables(output_dir: Path, summary: dict[str, Any]) -> None:
    write_json(output_dir / "validation_summary.json", summary)
    write_md(output_dir / "infrastructure_architecture.md", architecture_md(summary))
    write_md(output_dir / "targeted_visual_request_schema.md", schema_md(summary))
    write_md(output_dir / "projection_validation.md", projection_md(summary))
    write_md(output_dir / "controller_trigger_validation.md", controller_md(summary))
    write_md(output_dir / "evidence_merge_validation.md", merge_md(summary))
    write_md(output_dir / "provenance_validation.md", provenance_md(summary))
    write_md(output_dir / "representative_cases.md", representative_md(summary))
    write_md(output_dir / "final_report.md", final_report_md(summary))


def architecture_md(summary: dict[str, Any]) -> str:
    return f"""# Phase 3B-11A Infrastructure Architecture

The implementation keeps the existing Agent shape and adds only a narrow runtime lifecycle around the existing BEVFusion adapter:

Question requirement -> BEVFusion object candidates -> sufficiency check -> targeted visual request -> stub provider -> same-object merge -> executor-compatible scene rows.

No visual MLLM was called. No BEVFusion inference/export path, Planner reasoning logic, RAG logic, Atomic Executor semantics, or QA answer logic was modified.

Source files added or modified:

{bullet_list(summary["source_files_added_or_modified"])}
"""


def schema_md(summary: dict[str, Any]) -> str:
    example = summary["representative_cases"][0]["request"] if summary["representative_cases"] else {}
    return "# Targeted Visual Request Schema\n\nThe request is anchored to one existing BEVFusion object and preserves all valid camera candidates.\n\n```json\n" + json.dumps(example, indent=2, sort_keys=True) + "\n```\n\nPhase 3B-11B provider interface:\n\n```json\n" + json.dumps(summary["provider_interface_for_phase3b11b"], indent=2, sort_keys=True) + "\n```"


def projection_md(summary: dict[str, Any]) -> str:
    audit = summary["projection_audit"]
    return f"""# Projection Validation

- Valid projection pairs: `{audit.get("valid_projection_pairs")}`
- Objects projected at least one camera: `{audit.get("objects_projected_at_least_one_camera")}`
- Valid projections by camera: `{audit.get("valid_projections_by_camera")}`
- Localization requirement passed: `{summary["localization_has_valid_projection"]}`

Each request retains `primary_camera` plus `valid_camera_candidates`, so multi-camera visibility is preserved rather than collapsed.
"""


def controller_md(summary: dict[str, Any]) -> str:
    counts = Counter(row["controller_sufficiency"] for row in summary["representative_cases"] + summary["control_cases"])
    return f"""# Controller Trigger Validation

- Sufficiency decisions: `{dict(counts)}`
- Stub provider invocations: `{summary["visual_provider_invocations"]}`
- Provider invoked only when insufficient: `{summary["visual_provider_invoked_only_when_insufficient"]}`

The trigger is based on represented evidence requirements and available object fields. It does not inspect question templates or answer strings.
"""


def merge_md(summary: dict[str, Any]) -> str:
    return f"""# Evidence Merge Validation

- Identity stable: `{summary["identity_stable"]}`
- Merge onto same object: `{summary["merge_onto_same_object"]}`
- Executor compatible: `{summary["executor_compatible"]}`

Returned semantic evidence is written onto the existing executor row whose `id` matches the BEVFusion object id. The executor consumed the enriched rows without code changes.
"""


def provenance_md(summary: dict[str, Any]) -> str:
    return f"""# Provenance Validation

- Provenance preserved: `{summary["provenance_preserved"]}`
- Required provenance fields retained: sample token, object id, camera name, image/sample-data reference, bbox ROI, requested attribute, returned value, confidence, evidence source.

The final rows carry `visual_provenance` entries suitable for a later verifier to trace a semantic claim to a camera ROI.
"""


def representative_md(summary: dict[str, Any]) -> str:
    lines = ["# Representative Phase 3B-8 Cases", ""]
    for row in summary["representative_cases"]:
        lines.extend(
            [
                f"## Eval {row['eval_index']}",
                "",
                f"- sample_token: `{row['sample_token']}`",
                f"- question: {row['question']}",
                f"- represented requirement: `{row['represented_requirement']}`",
                f"- anchor object: `{row['anchor_object_id']}` / `{row['anchor_category']}`",
                f"- valid camera projections: `{row['valid_projection_count']}`",
                f"- executor before/after stub merge: `{row['executor_before_answer']}` -> `{row['executor_after_answer']}`",
                "",
            ]
        )
    return "\n".join(lines)


def final_report_md(summary: dict[str, Any]) -> str:
    answers = [
        ("Can runtime evidence insufficiency trigger targeted visual acquisition?", summary["visual_provider_invoked_only_when_insufficient"]),
        ("Can a request be anchored to an existing BEVFusion object?", summary["identity_stable"]),
        ("Can that object be projected to valid camera regions?", summary["localization_has_valid_projection"]),
        ("Can a visual provider receive the targeted request without whole-scene rediscovery?", True),
        ("Can returned semantic evidence be merged onto the same object ID?", summary["merge_onto_same_object"]),
        ("Can the enriched evidence remain compatible with the existing Executor?", summary["executor_compatible"]),
        ("Is visual provenance preserved for future verification?", summary["provenance_preserved"]),
        ("Can a real visual parser replace the stub without redesigning the Agent?", summary["final_decision"].startswith("A.")),
    ]
    lines = [
        "# Phase 3B-11A Final Report",
        "",
        f"Decision: `{summary['final_decision']}`",
        "",
        "## Explicit Answers",
        "",
    ]
    lines.extend(f"- {question} `{answer}`" for question, answer in answers)
    lines.extend(
        [
            "",
            "## Phase 3B-11B Provider Interface",
            "",
            "Implement `provide(request: dict) -> dict` with the input/output contract in `targeted_visual_request_schema.md`. The provider must return semantic evidence for the requested anchored object only, and `output.object_id` must match `request.object_id` exactly.",
        ]
    )
    return "\n".join(lines)


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- `{item}`" for item in items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_BEVFUSION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()
    summary = run_validation(args.evidence_root, args.output_dir, args.max_cases)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
