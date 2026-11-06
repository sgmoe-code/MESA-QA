# atomic_program_executor.py
# Deterministic compilation of normalized planner output into schema-parity atomic steps.

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from query_resolver import resolve_query
from scene_graph_query import (
    choose_primary,
    object_matches,
    relation_between,
    relation_matches,
    sort_objects,
)


@dataclass
class AtomicStep:
    op: str
    input: Dict[str, Any]
    output: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PlanCompiledAtomicExecutor:
    """
    Deterministic executor that compiles a normalized planner query into
    explicit low-level atomic steps while preserving schema-level semantics.
    """

    def __init__(self, tools):
        self.tools = tools

    def execute(
        self,
        plan: Dict[str, Any],
        scene_rows: List[Dict[str, Any]],
        eval_type: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        if self.tools is not None:
            self.tools.set_scene(scene_rows)
        return self.execute_plan(plan, scene_rows, eval_type=eval_type)

    def execute_plan(
        self,
        plan: Dict[str, Any],
        scene_rows: List[Dict[str, Any]],
        eval_type: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        compile_started = time.perf_counter()
        query = resolve_query(plan, eval_type=eval_type)
        operation = query["operation"]
        compiled_program = self.compile_program(query)
        compile_time_ms = _elapsed_ms(compile_started)

        execute_started = time.perf_counter()
        steps: List[Dict[str, Any]] = []

        target_filter = query["target_filter"]
        targets_all, step = self._find_objects(scene_rows, target_filter.get("category", "unknown"), role="target")
        steps.append(step)

        targets, step = self._filter_attributes(
            targets_all,
            target_filter.get("attributes", []),
            role="target",
        )
        steps.append(step)

        references: List[Dict[str, Any]] = []
        reference_filter = query.get("reference_filter")
        if reference_filter is not None:
            refs_all, step = self._find_objects(scene_rows, reference_filter.get("category", "unknown"), role="reference")
            steps.append(step)

            references, step = self._filter_attributes(
                refs_all,
                reference_filter.get("attributes", []),
                role="reference",
            )
            steps.append(step)

        relation_edges: List[Dict[str, Any]] = []
        if query.get("spatial_relations"):
            if not references:
                execute_time_ms = _elapsed_ms(execute_started)
                trace = self._make_trace(
                    query=query,
                    compiled_program=compiled_program,
                    steps=steps,
                    targets=targets,
                    references=references,
                    relation_edges=relation_edges,
                    candidate_answer=None,
                    compile_time_ms=compile_time_ms,
                    execute_time_ms=execute_time_ms,
                    failure_reason="no_reference_match",
                )
                return None, trace

            targets, relation_edges, step = self._spatial_match_all_refs(
                targets,
                references,
                query["spatial_relations"],
            )
            steps.append(step)

        if query.get("comparison_relations") and operation != "compare":
            ignored = query.get("comparison_relations", [])
            steps.append(AtomicStep(
                op="ignore_comparison_relations_for_non_compare",
                input={"comparison_relations": ignored, "operation": operation},
                output={"reason": "schema_parity"},
            ).to_dict())

        answer, final_targets, final_references, finalize_step, extra, failure_reason = self._finalize_answer(
            operation=operation,
            targets=targets,
            references=references,
            relation_edges=relation_edges,
            query=query,
        )
        steps.append(finalize_step)

        execute_time_ms = _elapsed_ms(execute_started)
        trace = self._make_trace(
            query=query,
            compiled_program=compiled_program,
            steps=steps,
            targets=final_targets,
            references=final_references,
            relation_edges=relation_edges,
            candidate_answer=answer,
            compile_time_ms=compile_time_ms,
            execute_time_ms=execute_time_ms,
            failure_reason=failure_reason,
            extra=extra,
        )
        return answer, trace

    def compile_program(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        target_filter = query["target_filter"]
        program = [
            {"op": "find_target_objects", "category": target_filter.get("category", "unknown")},
            {"op": "filter_target_attributes", "attributes": target_filter.get("attributes", []) or []},
        ]

        reference_filter = query.get("reference_filter")
        if reference_filter is not None:
            program.extend([
                {"op": "find_reference_objects", "category": reference_filter.get("category", "unknown")},
                {"op": "filter_reference_attributes", "attributes": reference_filter.get("attributes", []) or []},
            ])

        if query.get("spatial_relations"):
            program.append({"op": "spatial_match_all_refs", "relations": query.get("spatial_relations", [])})

        operation = query.get("operation")
        if query.get("comparison_relations") and operation != "compare":
            program.append({
                "op": "ignore_comparison_relations_for_non_compare",
                "comparison_relations": query.get("comparison_relations", []),
            })

        if operation == "compare":
            program.append({
                "op": "compare_objects",
                "comparison_relations": query.get("comparison_relations", []),
            })
        else:
            program.append({"op": "finalize_answer", "operation": operation})
        return program

    def _find_objects(
        self,
        scene_rows: List[Dict[str, Any]],
        category: str,
        role: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        objects = sort_objects([
            obj for obj in scene_rows
            if object_matches(obj, category=category, attributes=[])
        ])
        step = AtomicStep(
            op="find_objects",
            input={"role": role, "category": category},
            output={"ids": _object_ids(objects), "objects": _minify_objects(objects)},
        ).to_dict()
        return objects, step

    def _filter_attributes(
        self,
        objects: List[Dict[str, Any]],
        attributes: List[str],
        role: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        attributes = attributes or []
        if attributes:
            filtered = [
                obj for obj in objects
                if object_matches(
                    obj,
                    category=obj.get("category", "unknown"),
                    attributes=attributes,
                )
            ]
        else:
            filtered = list(objects)
        step = AtomicStep(
            op="filter_attributes",
            input={"role": role, "input_ids": _object_ids(objects), "attributes": attributes},
            output={"ids": _object_ids(filtered), "objects": _minify_objects(filtered)},
        ).to_dict()
        return filtered, step

    def _spatial_match_all_refs(
        self,
        targets: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        spatial_relations: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if not spatial_relations:
            step = AtomicStep(
                op="spatial_match_all_refs",
                input={
                    "target_ids": _object_ids(targets),
                    "reference_ids": _object_ids(references),
                    "relations": [],
                },
                output={"ids": _object_ids(targets), "edges": []},
            ).to_dict()
            return targets, [], step

        matched: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_ids = set()

        for target in targets:
            for reference in references:
                if target.get("id") == reference.get("id"):
                    continue
                actual = relation_between(target, reference)
                ok = any(relation_matches(actual, required) for required in spatial_relations)
                if ok:
                    edges.append({
                        "subject": target.get("id"),
                        "relation": actual,
                        "object": reference.get("id"),
                        "target_category": target.get("category"),
                        "reference_category": reference.get("category"),
                    })
                    if target.get("id") not in seen_ids:
                        matched.append(target)
                        seen_ids.add(target.get("id"))
                    break

        matched = sort_objects(matched)
        step = AtomicStep(
            op="spatial_match_all_refs",
            input={
                "target_ids": _object_ids(targets),
                "reference_ids": _object_ids(references),
                "relations": spatial_relations,
            },
            output={"ids": _object_ids(matched), "edges": edges},
        ).to_dict()
        return matched, edges, step

    def _finalize_answer(
        self,
        operation: str,
        targets: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        relation_edges: List[Dict[str, Any]],
        query: Dict[str, Any],
    ) -> Tuple[Optional[str], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Optional[str]]:
        if operation == "count":
            answer = str(len(targets))
            step = AtomicStep(
                op="finalize_answer",
                input={"operation": operation, "target_ids": _object_ids(targets)},
                output={"answer": answer, "num_support": len(targets)},
            ).to_dict()
            return answer, targets, references, step, {}, None

        if operation == "exist":
            answer = "yes" if len(targets) > 0 else "no"
            step = AtomicStep(
                op="finalize_answer",
                input={"operation": operation, "target_ids": _object_ids(targets)},
                output={"answer": answer, "num_support": len(targets)},
            ).to_dict()
            return answer, targets, references, step, {}, None

        if operation == "lookup_category":
            obj = choose_primary(targets)
            if obj is None:
                step = AtomicStep(
                    op="finalize_answer",
                    input={"operation": operation, "target_ids": []},
                    output={"answer": None, "failure_reason": "no_target_match"},
                ).to_dict()
                return None, targets, references, step, {}, "no_target_match"
            answer = obj.get("category", "unknown")
            step = AtomicStep(
                op="finalize_answer",
                input={"operation": operation, "target_ids": _object_ids(targets)},
                output={"chosen_id": obj.get("id"), "answer": answer},
            ).to_dict()
            return answer, [obj], references, step, {}, None

        if operation == "lookup_status":
            obj = choose_primary(targets)
            if obj is None:
                step = AtomicStep(
                    op="finalize_answer",
                    input={"operation": operation, "target_ids": []},
                    output={"answer": None, "failure_reason": "no_target_match"},
                ).to_dict()
                return None, targets, references, step, {}, "no_target_match"
            answer = obj.get("status", "unknown")
            step = AtomicStep(
                op="finalize_answer",
                input={"operation": operation, "target_ids": _object_ids(targets)},
                output={"chosen_id": obj.get("id"), "answer": answer},
            ).to_dict()
            return answer, [obj], references, step, {}, None

        if operation == "compare":
            return self._execute_compare_atomically(targets, references, relation_edges, query)

        step = AtomicStep(
            op="finalize_answer",
            input={"operation": operation, "target_ids": _object_ids(targets)},
            output={"answer": None, "failure_reason": "unsupported_operation"},
        ).to_dict()
        return None, targets, references, step, {}, "unsupported_operation"

    def _execute_compare_atomically(
        self,
        targets: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        relation_edges: List[Dict[str, Any]],
        query: Dict[str, Any],
    ) -> Tuple[Optional[str], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Optional[str]]:
        target_obj = choose_primary(targets)
        ref_obj = choose_primary(references)
        comp_rels = query.get("comparison_relations", [])
        compare_field = "category" if "same-category-as" in comp_rels else "status"

        if target_obj is None:
            step = AtomicStep(
                op="compare_objects",
                input={"target_id": None, "reference_id": ref_obj.get("id") if ref_obj else None, "compare_field": compare_field},
                output={"answer": None, "failure_reason": "no_compare_target"},
            ).to_dict()
            return None, targets, references, step, {"compare_field": compare_field}, "no_compare_target"

        if ref_obj is None:
            step = AtomicStep(
                op="compare_objects",
                input={"target_id": target_obj.get("id"), "reference_id": None, "compare_field": compare_field},
                output={"answer": None, "failure_reason": "no_compare_reference"},
            ).to_dict()
            return None, [target_obj], references, step, {"compare_field": compare_field}, "no_compare_reference"

        target_value = target_obj.get(compare_field)
        reference_value = ref_obj.get(compare_field)
        answer = "yes" if target_value == reference_value else "no"
        step = AtomicStep(
            op="compare_objects",
            input={
                "target_id": target_obj.get("id"),
                "reference_id": ref_obj.get("id"),
                "compare_field": compare_field,
            },
            output={
                "target_value": target_value,
                "reference_value": reference_value,
                "answer": answer,
            },
        ).to_dict()
        extra = {
            "compare_field": compare_field,
            "target_value": target_value,
            "reference_value": reference_value,
        }
        return answer, [target_obj], [ref_obj], step, extra, None

    def _make_trace(
        self,
        query: Dict[str, Any],
        compiled_program: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        relation_edges: List[Dict[str, Any]],
        candidate_answer: Optional[str],
        compile_time_ms: float,
        execute_time_ms: float,
        failure_reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        trace = {
            "source": "compiled_atomic_executor",
            "query": query,
            "compiled_program": compiled_program,
            "steps": steps,
            "supporting_objects": _object_ids(targets),
            "reference_objects": _object_ids(references),
            "relation_edges": relation_edges,
            "num_support": len(targets),
            "num_references": len(references),
            "candidate_answer": candidate_answer,
            "semantic_mode": "schema_parity_low_level_primitives",
            "model_calls": 0,
            "runtime": {
                "compile_time_ms": compile_time_ms,
                "execute_time_ms": execute_time_ms,
                "num_atomic_steps": len(steps),
                "model_calls": 0,
            },
            "semantic_flags": {
                "uses_all_references_for_spatial_relation": True,
                "uses_schema_object_matches": True,
                "comparison_relations_only_for_compare": True,
            },
        }
        if query.get("comparison_relations") and query.get("operation") != "compare":
            trace["comparison_relations_ignored_for_schema_parity"] = query.get("comparison_relations", [])
        if failure_reason:
            trace["failure_reason"] = failure_reason
        if extra:
            trace.update(extra)
        return trace


def execute_atomic_program(
    plan: Dict[str, Any],
    scene_rows: List[Dict[str, Any]],
    eval_type: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    executor = PlanCompiledAtomicExecutor(tools=None)
    return executor.execute_plan(plan, scene_rows, eval_type=eval_type)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _object_ids(objects: List[Dict[str, Any]]) -> List[Any]:
    return [obj.get("id") for obj in objects]


def _minify_objects(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": o.get("id"),
            "category": o.get("category"),
            "status": o.get("status", "unknown"),
            "raw_status": o.get("raw_status", "unknown"),
            "distance": o.get("distance"),
        }
        for o in objects
    ]


# Legacy JSON helpers are retained for compatibility with older debug imports.
def _parse_json_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "matches" in parsed and isinstance(parsed["matches"], list):
        return parsed["matches"]
    if isinstance(parsed, dict) and "error" in parsed:
        raise ValueError(parsed["error"])
    raise ValueError(f"Expected JSON list, got: {parsed}")


def _parse_json_obj(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if isinstance(parsed, dict):
        if "error" in parsed:
            raise ValueError(parsed["error"])
        return parsed
    raise ValueError(f"Expected JSON object, got: {parsed}")
