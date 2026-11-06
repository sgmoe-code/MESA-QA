from __future__ import annotations

from step71_common import OUTPUT_DIR, write_json, write_md


def build_mapping() -> dict:
    return {
        "CHECK_EXISTENCE": {"required_predicates": ["object_candidates", "object_existence"], "preferred_sources": ["centerpoint_predicted_scene_graph"]},
        "CHECK_EGO_RELATION": {"required_predicates": ["object_geometry", "ego_spatial_relation"], "preferred_sources": ["centerpoint_predicted_scene_graph"]},
        "CHECK_OBJECT_OBJECT_RELATION": {"required_predicates": ["object_geometry", "object_object_spatial_relation"], "preferred_sources": ["centerpoint_predicted_scene_graph"]},
        "CHECK_STATUS": {"required_predicates": ["semantic_status"], "auxiliary_predicates": ["motion_status"], "preferred_sources": ["vision_predicate_placeholder"], "fallback_sources": ["centerpoint_predicted_scene_graph"], "fallback_policy": "motion_status_only_not_sufficient_for_semantic_status"},
        "COUNT_CANDIDATES": {"required_predicates": ["validated_count"], "auxiliary_predicates": ["proposal_count", "object_candidates"], "preferred_sources": ["vision_predicate_placeholder"], "fallback_sources": ["centerpoint_predicted_scene_graph"], "fallback_policy": "proposal_count_diagnostic_only"},
        "COMPARE_COUNTS": {"required_predicates": ["comparison", "validated_count"], "preferred_sources": ["executor_logical_comparison"], "fallback_policy": "unsupported_without_validated_counts"},
        "FIND_OBJECTS": {"required_predicates": ["object_candidates"], "auxiliary_predicates": ["object_category", "object_geometry"], "preferred_sources": ["centerpoint_predicted_scene_graph"], "future_sources": ["vision_predicate_placeholder"]},
        "REFERENCE_RESOLUTION": {"required_predicates": ["reference_grounding"], "preferred_sources": ["vision_predicate_placeholder"], "fallback_policy": "centerpoint_geometry_auxiliary_only"},
        "UNKNOWN": {"required_predicates": [], "preferred_sources": []},
    }


def main() -> None:
    mapping = build_mapping()
    write_json(OUTPUT_DIR / "step73_operation_predicate_mapping.json", mapping)
    write_md(OUTPUT_DIR / "step73_operation_predicate_mapping.md", "# Step 7.3 Operation Predicate Mapping\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in mapping.items()))


if __name__ == "__main__":
    main()
