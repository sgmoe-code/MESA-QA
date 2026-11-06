from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from bevfusion_scene_adapter import BEVFusionSceneAdapter, DEFAULT_BEVFUSION_ROOT
from operation_predicate_mapping_step73 import build_mapping
from source_aware_evidence_router_step73 import SourceAwareEvidenceRouter
from step71_common import write_json, write_md
from step74a_common import CAMERA_CHANNELS
from step74b_common import box_corners, metadata_indices, projection_resource_report, summarize_projection, transform_matrix


DEFAULT_OUTPUT_DIR = Path("outputs/phase3b10_hierarchical_evidence_feasibility")
PHASE3B8_QUESTION_ANALYSIS = Path("outputs/phase3b8_bevfusion_evidence_analysis/question_level_analysis.json")
PHASE3B9_SUMMARY = Path("outputs/phase3b9_evidence_controller_analysis/validation_summary.json")
BEVFUSION_SOURCE = "bevfusion_predicted_scene_graph"
VISION_SOURCE = "targeted_visual_evidence_module"
RAG_SOURCE = "rag_context_non_scene_fact"


def build_registry(evidence_root: Path) -> dict[str, dict[str, Any]]:
    return {
        BEVFUSION_SOURCE: {
            "active": True,
            "input_artifacts": str(evidence_root),
            "supports": [
                "object_existence",
                "object_category",
                "object_candidates",
                "object_geometry",
                "ego_spatial_relation",
                "object_object_spatial_relation",
                "distance",
                "motion_status",
                "proposal_count",
            ],
            "does_not_support": [
                "semantic_status",
                "visual_attribute",
                "visibility",
                "validated_count",
                "comparison",
                "fine_grained_reference_grounding",
            ],
        },
        VISION_SOURCE: {
            "active": True,
            "dependent_on": BEVFUSION_SOURCE,
            "future_module_only": True,
            "supports": [
                "semantic_status",
                "visual_attribute",
                "visibility",
                "validated_count",
                "reference_grounding",
            ],
            "requires_anchor_fields": [
                "sample_token",
                "object_id",
                "category",
                "center_xyz",
                "size_wlh",
                "yaw",
                "confidence",
            ],
        },
        RAG_SOURCE: {
            "active": False,
            "supports": ["reasoning_context", "exemplar_context"],
            "does_not_support": ["direct_scene_fact"],
        },
    }


def run_feasibility(evidence_root: Path, output_dir: Path, max_cases: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = BEVFusionSceneAdapter(evidence_root)
    registry = build_registry(evidence_root)
    router = SourceAwareEvidenceRouter(registry=registry, mapping=visual_aware_mapping())
    phase3b8 = load_phase3b8_cases()
    phase3b9 = read_json(PHASE3B9_SUMMARY, {})
    semantic_cases = select_representative_cases(phase3b8, max_cases)

    control_matrix = evidence_control_matrix(router)
    anchor_checks = [anchor_feasibility(adapter, token) for token in unique_tokens(row["sample_token"] for row in semantic_cases)]
    projection_checks, projection_audit = project_case_anchors(adapter, semantic_cases)
    case_analyses = build_case_analyses(semantic_cases, anchor_checks, projection_checks)

    summary = {
        "phase": "Phase 3B-10",
        "output_dir": str(output_dir),
        "phase3b8_question_analysis": str(PHASE3B8_QUESTION_ANALYSIS),
        "phase3b9_summary": str(PHASE3B9_SUMMARY),
        "available_bevfusion_samples": len(adapter.available_sample_tokens()),
        "semantic_failure_cases_total": sum(
            1 for row in phase3b8 if row.get("phase3b8_primary_category") == "B_missing_semantic_evidence"
        ),
        "representative_case_count": len(case_analyses),
        "registry": registry,
        "control_matrix": control_matrix,
        "anchor_checks": anchor_checks,
        "projection_resource_report": projection_resource_report(),
        "projection_audit": projection_audit,
        "projection_checks": projection_checks,
        "representative_cases": case_analyses,
        "phase3b9_validation_passed": bool(phase3b9.get("validation_passed")),
        "final_decision": "B. MINOR_CONTROLLER_EXTENSION_REQUIRED",
        "minimum_next_interface": minimum_visual_interface(),
        "visual_module_implemented": False,
        "bevfusion_inference_rerun": False,
        "planner_redesigned": False,
        "atomic_executor_redesigned": False,
    }

    write_json(output_dir / "feasibility_summary.json", summary)
    write_md(output_dir / "architecture_compatibility.md", architecture_compatibility_md(summary))
    write_md(output_dir / "evidence_sufficiency_analysis.md", evidence_sufficiency_analysis_md(summary))
    write_md(output_dir / "bevfusion_visual_handoff.md", bevfusion_visual_handoff_md(summary))
    write_md(output_dir / "representative_cases.md", representative_cases_md(summary))
    write_md(output_dir / "final_decision.md", final_decision_md(summary))
    return summary


def visual_aware_mapping() -> dict[str, dict[str, Any]]:
    mapping = build_mapping()
    mapping["CHECK_STATUS"] = {
        "required_predicates": ["semantic_status"],
        "auxiliary_predicates": ["motion_status", "object_candidates", "object_geometry"],
        "preferred_sources": [VISION_SOURCE],
        "fallback_sources": [BEVFUSION_SOURCE],
        "fallback_policy": "use_bevfusion_anchor_then_targeted_visual_status",
    }
    mapping["COUNT_CANDIDATES"] = {
        "required_predicates": ["validated_count"],
        "auxiliary_predicates": ["proposal_count", "object_candidates", "object_geometry"],
        "preferred_sources": [VISION_SOURCE],
        "fallback_sources": [BEVFUSION_SOURCE],
        "fallback_policy": "use_bevfusion_candidates_then_visual_validation",
    }
    mapping["REFERENCE_RESOLUTION"] = {
        "required_predicates": ["reference_grounding"],
        "auxiliary_predicates": ["object_candidates", "object_geometry"],
        "preferred_sources": [VISION_SOURCE],
        "fallback_sources": [BEVFUSION_SOURCE],
        "fallback_policy": "use_bevfusion_candidate_then_visual_reference_grounding",
    }
    for group in ["FIND_OBJECTS", "CHECK_EXISTENCE", "CHECK_EGO_RELATION", "CHECK_OBJECT_OBJECT_RELATION"]:
        spec = mapping[group]
        spec["preferred_sources"] = [BEVFUSION_SOURCE]
    return mapping


def evidence_control_matrix(router: SourceAwareEvidenceRouter) -> list[dict[str, Any]]:
    rows = []
    for group in [
        "FIND_OBJECTS",
        "CHECK_EXISTENCE",
        "CHECK_EGO_RELATION",
        "CHECK_OBJECT_OBJECT_RELATION",
        "CHECK_STATUS",
        "COUNT_CANDIDATES",
        "REFERENCE_RESOLUTION",
        "COMPARE_COUNTS",
    ]:
        route = router.route({"operation_group": group}, [BEVFUSION_SOURCE, VISION_SOURCE])
        rows.append(
            {
                "operation_group": group,
                "selected_sources": route.get("selected_sources", []),
                "auxiliary_sources": route.get("auxiliary_sources", []),
                "missing_capabilities": route.get("missing_capabilities", []),
                "execution_mode": route.get("execution_mode"),
                "reason": route.get("reason"),
            }
        )
    return rows


def load_phase3b8_cases() -> list[dict[str, Any]]:
    payload = read_json(PHASE3B8_QUESTION_ANALYSIS, {})
    return list(payload.get("questions", []))


def select_representative_cases(rows: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("phase3b8_primary_category") == "B_missing_semantic_evidence"
        and row.get("phase3b8_primary_subtype") == "fine_attribute_required"
    ]
    preferred = []
    seen_questions = set()
    for row in selected:
        q = row.get("question")
        if q not in seen_questions:
            preferred.append(row)
            seen_questions.add(q)
        if len(preferred) >= max_cases:
            break
    return preferred


def unique_tokens(tokens: Any) -> list[str]:
    out = []
    for token in tokens:
        token = str(token)
        if token not in out:
            out.append(token)
    return out


def anchor_feasibility(adapter: BEVFusionSceneAdapter, sample_token: str) -> dict[str, Any]:
    objects = adapter.normalize_objects(sample_token)
    anchors = []
    for obj in objects[:20]:
        geom = obj.get("geometry") or {}
        anchors.append(
            {
                "object_id": obj.get("object_id"),
                "category": obj.get("category"),
                "center_xyz_present": bool(geom.get("center_xyz")),
                "size_wlh_present": bool(geom.get("size_wlh")),
                "yaw_present": geom.get("yaw") is not None,
                "confidence_present": obj.get("confidence") is not None,
                "status": obj.get("status"),
            }
        )
    required_ok = all(
        a["object_id"]
        and a["category"]
        and a["center_xyz_present"]
        and a["size_wlh_present"]
        and a["yaw_present"]
        and a["confidence_present"]
        for a in anchors
    )
    return {
        "sample_token": sample_token,
        "object_count": len(objects),
        "checked_anchor_count": len(anchors),
        "required_anchor_fields_present": required_ok,
        "example_anchor": anchors[0] if anchors else None,
    }


def project_case_anchors(adapter: BEVFusionSceneAdapter, cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resource_report = projection_resource_report()
    if not resource_report.get("resources_present") or not all(resource_report["resources_present"].values()):
        return [], {"valid": False, "reason": "projection_metadata_incomplete", **resource_report}
    idx = metadata_indices()
    records = []
    errors = []
    seen_tokens = []
    for case in cases:
        token = str(case["sample_token"])
        if token not in seen_tokens:
            seen_tokens.append(token)
    for token in seen_tokens:
        objects = adapter.normalize_objects(token)[:5]
        chans = idx["sample_data_by_sample"].get(token, {})
        if "LIDAR_TOP" not in chans or not all(cam in chans for cam in CAMERA_CHANNELS):
            errors.append({"sample_token": token, "error": "missing_lidar_or_camera_sample_data"})
            continue
        records.extend(project_objects_for_token(token, objects, idx))
    audit = summarize_projection(records, errors)
    if "total_centerpoint_objects" in audit:
        audit["total_bevfusion_objects_checked"] = audit.pop("total_centerpoint_objects")
    return records, audit


def project_objects_for_token(sample_token: str, objects: list[dict[str, Any]], idx: dict[str, Any]) -> list[dict[str, Any]]:
    by = idx["by_token"]
    chans = idx["sample_data_by_sample"][sample_token]
    lidar_sd = chans["LIDAR_TOP"]
    lidar_cal = by["calibrated_sensor"][lidar_sd["calibrated_sensor_token"]]
    lidar_pose = by["ego_pose"][lidar_sd["ego_pose_token"]]
    t_lidar_to_global = transform_matrix(lidar_pose["translation"], lidar_pose["rotation"]) @ transform_matrix(
        lidar_cal["translation"], lidar_cal["rotation"]
    )
    records = []
    for obj in objects:
        geom = obj.get("geometry") or {}
        center = geom.get("center_xyz")
        size = geom.get("size_wlh")
        yaw = geom.get("yaw")
        if not center or not size or yaw is None:
            continue
        corners_lidar = box_corners(center, size, float(yaw))
        corners_lidar_h = np.c_[corners_lidar, np.ones(len(corners_lidar))]
        for camera in CAMERA_CHANNELS:
            cam_sd = chans[camera]
            cam_cal = by["calibrated_sensor"][cam_sd["calibrated_sensor_token"]]
            cam_pose = by["ego_pose"][cam_sd["ego_pose_token"]]
            t_global_to_cam = transform_matrix(cam_cal["translation"], cam_cal["rotation"], inverse=True) @ transform_matrix(
                cam_pose["translation"], cam_pose["rotation"], inverse=True
            )
            cam_pts = (t_global_to_cam @ t_lidar_to_global @ corners_lidar_h.T).T[:, :3]
            records.append(build_projection_record(sample_token, obj, camera, cam_sd, cam_cal, cam_pts))
    return records


def build_projection_record(
    sample_token: str,
    obj: dict[str, Any],
    camera: str,
    cam_sd: dict[str, Any],
    cam_cal: dict[str, Any],
    cam_pts: np.ndarray,
) -> dict[str, Any]:
    in_front = cam_pts[:, 2] > 0.1
    uv = []
    if np.any(in_front):
        k = np.array(cam_cal["camera_intrinsic"], dtype=float)
        proj = (k @ cam_pts[in_front].T).T
        uv = proj[:, :2] / proj[:, 2:3]
    width, height = int(cam_sd.get("width") or 1600), int(cam_sd.get("height") or 900)
    valid = False
    bbox_un = bbox_cl = [None, None, None, None]
    area = trunc = 0.0
    if len(uv):
        x1, y1 = np.min(uv, axis=0)
        x2, y2 = np.max(uv, axis=0)
        bbox_un = [float(x1), float(y1), float(x2), float(y2)]
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(width - 1), x2), min(float(height - 1), y2)
        bw, bh = max(0.0, cx2 - cx1), max(0.0, cy2 - cy1)
        unclipped = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = bw * bh
        trunc = 1.0 - (area / unclipped) if unclipped > 0 else 1.0
        bbox_cl = [float(cx1), float(cy1), float(cx2), float(cy2)]
        valid = bool(area > 0 and bw >= 4 and bh >= 4)
    geom = obj.get("geometry") or {}
    return {
        "projection_record_id": f"{sample_token}_{obj['object_id']}_{camera}",
        "sample_token": sample_token,
        "object_id": obj["object_id"],
        "category": obj.get("category"),
        "camera": camera,
        "projection_valid": valid,
        "bbox_unclipped": bbox_un,
        "bbox_clipped": bbox_cl,
        "visible_corner_count": int(np.sum(in_front)),
        "projected_area": round(float(area), 4),
        "truncation_ratio": round(float(max(0.0, min(1.0, trunc))), 4),
        "depth_m": round(float(np.median(cam_pts[in_front, 2])) if np.any(in_front) else float(np.median(cam_pts[:, 2])), 4),
        "image_width": width,
        "image_height": height,
        "image_path": str(Path("data/nuscenes") / cam_sd["filename"]),
        "geometry": {
            "center_xyz": geom.get("center_xyz"),
            "size_wlh": geom.get("size_wlh"),
            "yaw": geom.get("yaw"),
            "confidence": obj.get("confidence"),
        },
        "source": "bevfusion_3d_projection_sanity",
    }


def build_case_analyses(
    cases: list[dict[str, Any]],
    anchor_checks: list[dict[str, Any]],
    projection_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchor_by_token = {row["sample_token"]: row for row in anchor_checks}
    valid_projection_by_token = Counter(row["sample_token"] for row in projection_checks if row.get("projection_valid"))
    analyses = []
    for row in cases:
        token = row["sample_token"]
        requested = requested_attribute(row.get("question", ""))
        analyses.append(
            {
                "eval_index": row.get("eval_index"),
                "sample_token": token,
                "question": row.get("question"),
                "template_type": row.get("template_type"),
                "question_requirement": row.get("evidence_requirements"),
                "requested_visual_semantics": requested,
                "bevfusion_already_provides": [
                    "object_id",
                    "category",
                    "3D center",
                    "3D size",
                    "yaw/orientation",
                    "confidence",
                    "coarse motion from velocity",
                ],
                "missing_evidence": [
                    "fine semantic status" if requested else "semantic attribute",
                    "traceable camera ROI confirming the attribute",
                ],
                "targeted_visual_request_feasible": bool(
                    anchor_by_token.get(token, {}).get("required_anchor_fields_present")
                    and valid_projection_by_token.get(token, 0) > 0
                ),
                "valid_projection_pairs_for_checked_anchors": int(valid_projection_by_token.get(token, 0)),
                "phase3b8_failure_subtype": row.get("phase3b8_primary_subtype"),
                "bevfusion_trace_summary": row.get("bevfusion"),
            }
        )
    return analyses


def requested_attribute(question: str) -> str | None:
    q = question.lower().replace("-", " ")
    for term in ["parked", "stopped", "standing", "not standing", "same status"]:
        if term in q:
            return term.replace(" ", "_")
    return None


def minimum_visual_interface() -> dict[str, Any]:
    return {
        "input": {
            "sample_token": "nuScenes sample token",
            "object_anchor": {
                "object_id": "BEVFusion object id",
                "category": "canonical object category",
                "center_xyz": "3D center in LIDAR_TOP frame",
                "size_wlh": "3D box size",
                "yaw": "box orientation",
                "confidence": "detector confidence",
            },
            "missing_predicate": "semantic_status | visual_attribute | visibility | validated_count | reference_grounding",
            "projection": "optional camera, bbox, image_path, crop_path if precomputed",
        },
        "structured_output": {
            "object_id": "same anchor id",
            "requested_attribute": "predicate name",
            "attribute_value": "normalized value or unknown",
            "confidence": "high | medium | low | unknown",
            "source": "targeted_visual_evidence_module",
        },
        "provenance_output": {
            "sample_token": "nuScenes sample token",
            "camera": "camera channel",
            "object_id": "same anchor id",
            "bbox": "2D ROI",
            "image_or_crop_ref": "path or artifact id",
            "visual_evidence_result": "raw/parsed module result",
            "confidence": "module confidence",
        },
    }


def architecture_compatibility_md(summary: dict[str, Any]) -> str:
    rows = [
        ("evidence requirement determination", "Partial", "Operation-predicate mapping already declares required and auxiliary predicates per atomic group."),
        ("source/tool selection", "Yes", "SourceAwareEvidenceRouter selects active sources from a registry."),
        ("availability/sufficiency checking", "Partial", "Static capability sufficiency exists; runtime candidate/attribute sufficiency needs a small policy hook."),
        ("conditional/fallback acquisition", "Partial", "Fallback sources are representable, but dependent BEVFusion-to-visual handoff is not first-class yet."),
        ("passing evidence into another tool", "Minor extension needed", "BEVFusion anchors exist, but the router currently returns source names rather than a handoff payload."),
        ("merging new evidence", "Minor extension needed", "scene_rows and SourceEvidenceResult can carry results, but semantic overlay merge needs an adapter."),
        ("forwarding to Atomic Executor", "Yes", "Executor consumes source-agnostic scene_rows."),
        ("provenance preservation", "Partial", "SourceEvidenceResult and projection records can preserve provenance; verifier integration is future work."),
    ]
    lines = [
        "# Phase 3B-10 Architecture Compatibility",
        "",
        f"Phase 3B-9 validation passed: `{summary['phase3b9_validation_passed']}`.",
        "",
        "| Mechanism | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {name} | {status} | {evidence} |" for name, status, evidence in rows)
    lines.extend(
        [
            "",
            "The existing architecture is compatible with hierarchical evidence acquisition. The missing piece is not a new Agent, but a narrow controller interface for carrying selected BEVFusion object anchors into a dependent visual tool and merging returned predicates back onto those same object IDs.",
        ]
    )
    return "\n".join(lines)


def evidence_sufficiency_analysis_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3B-10 Evidence Sufficiency Analysis",
        "",
        "The current system can compare required predicates with source capabilities through the operation-predicate mapping and source registry. This is enough to identify operations where BEVFusion is sufficient versus operations where visual/semantic evidence is required.",
        "",
        "| Operation group | Selected sources | Auxiliary sources | Missing capabilities | Control implication |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary["control_matrix"]:
        selected = ", ".join(row["selected_sources"]) or "none"
        aux = ", ".join(row["auxiliary_sources"]) or "none"
        missing = ", ".join(row["missing_capabilities"]) or "none"
        if missing != "none":
            implication = "unsupported or future policy needed"
        elif selected == BEVFUSION_SOURCE:
            implication = "BEVFusion sufficient"
        elif VISION_SOURCE in row["selected_sources"]:
            implication = "visual/dependent evidence needed"
        else:
            implication = "unsupported or future policy needed"
        lines.append(f"| `{row['operation_group']}` | {selected} | {aux} | {missing} | {implication} |")
    lines.extend(
        [
            "",
            "Static sufficiency is already representable. Runtime sufficiency still needs one small extension: after BEVFusion returns candidates, the controller should detect empty candidates, missing attributes, low confidence, or ambiguous bindings and build a targeted visual request from the surviving anchors.",
        ]
    )
    return "\n".join(lines)


def bevfusion_visual_handoff_md(summary: dict[str, Any]) -> str:
    audit = summary["projection_audit"]
    lines = [
        "# Phase 3B-10 BEVFusion Visual Handoff",
        "",
        "BEVFusion can act as the prerequisite structured source because normalized detections expose the anchor fields a future visual module needs.",
        "",
        "## Anchor Field Check",
        "",
        "| Sample token | Objects | Checked anchors | Required fields present | Example anchor |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in summary["anchor_checks"]:
        lines.append(
            f"| `{row['sample_token']}` | {row['object_count']} | {row['checked_anchor_count']} | "
            f"`{row['required_anchor_fields_present']}` | `{row['example_anchor']}` |"
        )
    lines.extend(
        [
            "",
            "## Projection Sanity Check",
            "",
            f"- Projection metadata valid: `{audit.get('valid', True)}`",
            f"- BEVFusion objects checked: `{audit.get('total_bevfusion_objects_checked')}`",
            f"- Object-camera pairs checked: `{audit.get('object_camera_pairs')}`",
            f"- Valid projection pairs: `{audit.get('valid_projection_pairs')}`",
            f"- Objects projected in at least one camera: `{audit.get('objects_projected_at_least_one_camera')}`",
            "",
            "The project already contains nuScenes sample, sample_data, calibrated_sensor, ego_pose, and sensor tables plus camera images. Existing transform/projection utilities can map LIDAR_TOP-frame 3D boxes into camera ROIs.",
        ]
    )
    return "\n".join(lines)


def representative_cases_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3B-10 Representative Phase 3B-8 Cases",
        "",
        f"Semantic evidence failure cases found: `{summary['semantic_failure_cases_total']}`.",
        "",
    ]
    for row in summary["representative_cases"]:
        lines.extend(
            [
                f"## Eval {row['eval_index']}",
                "",
                f"- sample_token: `{row['sample_token']}`",
                f"- question: {row['question']}",
                f"- requirement: `{row['question_requirement']}`",
                f"- BEVFusion provides: `{', '.join(row['bevfusion_already_provides'])}`",
                f"- missing evidence: `{', '.join(row['missing_evidence'])}`",
                f"- targeted visual request feasible: `{row['targeted_visual_request_feasible']}`",
                f"- valid projection pairs for checked anchors: `{row['valid_projection_pairs_for_checked_anchors']}`",
                "",
            ]
        )
    return "\n".join(lines)


def final_decision_md(summary: dict[str, Any]) -> str:
    interface = summary["minimum_next_interface"]
    answers = [
        ("Can the existing Evidence Controller support hierarchical evidence acquisition?", "Yes, with a minor extension for dependent handoff payloads and runtime sufficiency checks."),
        ("Can BEVFusion evidence act as the prerequisite/anchor for targeted visual evidence?", "Yes. It provides sample token, object id, category, 3D center, size, yaw, and confidence."),
        ("Can the controller invoke visual evidence only when BEVFusion evidence is insufficient?", "Conceptually yes through predicate sufficiency; implementation needs a small policy hook after BEVFusion candidate retrieval."),
        ("Can visual semantic evidence return to the existing Executor path?", "Yes, by merging returned semantic predicates onto the same object IDs in scene_rows or by preserving SourceEvidenceResult objects."),
        ("Can visual provenance be retained for later verification?", "Yes. Existing source result and projection record patterns can store camera, ROI, crop/image path, parsed result, confidence, and trace."),
        ("What is the minimum implementation needed next?", "A targeted visual evidence adapter, a BEVFusion-anchor handoff request schema, and a merge/provenance adapter. No Planner or Atomic Executor redesign is required."),
    ]
    lines = [
        "# Phase 3B-10 Final Decision",
        "",
        f"Decision: `{summary['final_decision']}`",
        "",
        "The existing architecture is fundamentally compatible with the requested hierarchical flow, but it is not fully ready as-is. The router can select sources by required predicates, and BEVFusion can provide structured anchors. The missing pieces are small controller/interface capabilities: construct a visual request from BEVFusion candidates, trigger it from insufficiency conditions, and merge/provenance-track returned semantic predicates.",
        "",
        "## Explicit Answers",
        "",
    ]
    lines.extend(f"- {q} {a}" for q, a in answers)
    lines.extend(
        [
            "",
            "## Minimum Visual Module Interface",
            "",
            "Input:",
            f"```json\n{json.dumps(interface['input'], indent=2)}\n```",
            "Structured output:",
            f"```json\n{json.dumps(interface['structured_output'], indent=2)}\n```",
            "Provenance output:",
            f"```json\n{json.dumps(interface['provenance_output'], indent=2)}\n```",
        ]
    )
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_BEVFUSION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    summary = run_feasibility(args.evidence_root, args.output_dir, args.max_cases)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
