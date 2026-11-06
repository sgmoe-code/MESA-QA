from __future__ import annotations

from step71_common import OUTPUT_DIR, write_json, write_md


def build_registry() -> dict:
    return {
        "centerpoint_predicted_scene_graph": {
            "active": True,
            "supports": ["object_existence", "object_category", "object_candidates", "object_geometry", "ego_spatial_relation", "object_object_spatial_relation", "distance", "motion_status", "proposal_count"],
            "does_not_support": ["semantic_status", "visual_attribute", "visibility", "validated_count", "comparison", "fine_grained_reference_grounding"],
        },
        "vision_predicate_placeholder": {
            "active": False,
            "future_support": ["semantic_status", "visual_attribute", "visibility", "validated_count", "reference_grounding"],
        },
        "oracle_kg_diagnostic_only": {"active": False, "diagnostic_only": True},
        "rag_context_non_scene_fact": {"active": False, "supports": ["reasoning_context", "exemplar_context"], "does_not_support": ["direct_scene_fact"]},
    }


def main() -> None:
    registry = build_registry()
    write_json(OUTPUT_DIR / "step73_source_capability_registry.json", registry)
    write_md(OUTPUT_DIR / "step73_source_capability_registry.md", "# Step 7.3 Source Capability Registry\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in registry.items()))


if __name__ == "__main__":
    main()
