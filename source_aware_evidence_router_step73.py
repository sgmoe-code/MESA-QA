from __future__ import annotations

from operation_predicate_mapping_step73 import build_mapping
from source_capability_registry_step73 import build_registry
from step71_common import OUTPUT_DIR, write_json, write_md


class SourceAwareEvidenceRouter:
    def __init__(self, registry: dict | None = None, mapping: dict | None = None):
        self.registry = registry or build_registry()
        self.mapping = mapping or build_mapping()

    def route(self, operation: dict, available_sources: list[str] | None = None) -> dict:
        group = operation.get("operation_group") or operation.get("group") or "UNKNOWN"
        spec = self.mapping.get(group, self.mapping["UNKNOWN"])
        available_sources = available_sources or list(self.registry)
        selected = []
        missing = []
        for pred in spec.get("required_predicates", []):
            source = self._find_source(pred, spec.get("preferred_sources", []) + spec.get("fallback_sources", []), available_sources)
            if source:
                selected.append(source)
            else:
                missing.append(pred)
        aux = []
        for pred in spec.get("auxiliary_predicates", []):
            source = self._find_source(pred, spec.get("fallback_sources", []) + spec.get("preferred_sources", []), available_sources)
            if source:
                aux.append(source)
        selected = sorted(set(selected))
        aux = sorted(set(aux) - set(selected))
        if missing:
            mode = "unsupported_with_auxiliary_motion_only" if aux else "unsupported_missing_capability"
            reason = f"no_active_source_supports_{'_and_'.join(missing)}"
        else:
            mode = "active_source_selected"
            reason = "all_required_predicates_supported"
        return {"operation_group": group, "required_predicates": spec.get("required_predicates", []), "auxiliary_predicates": spec.get("auxiliary_predicates", []), "selected_sources": selected, "auxiliary_sources": aux, "missing_capabilities": missing, "execution_mode": mode, "reason": reason, "routing_valid": True}

    def _find_source(self, predicate: str, preferred: list[str], available: list[str]) -> str | None:
        for src in preferred + available:
            info = self.registry.get(src) or {}
            if src in available and info.get("active") and predicate in info.get("supports", []):
                return src
        return None


def main() -> None:
    router = SourceAwareEvidenceRouter()
    tests = {g: router.route({"operation_group": g}, ["centerpoint_predicted_scene_graph", "vision_predicate_placeholder"]) for g in build_mapping()}
    write_json(OUTPUT_DIR / "step73_router_unit_tests.json", tests)
    write_md(OUTPUT_DIR / "step73_router_unit_tests.md", "# Step 7.3 Router Unit Tests\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in tests.items()))


if __name__ == "__main__":
    main()
