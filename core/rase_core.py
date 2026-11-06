"""Retrieval-Augmented Stateful Execution (RASE) core prototype.

The implementation is deliberately isolated from :mod:`atomic_program_executor`:
S0 remains reproducible, while this executor consumes the same normalized plan
and canonical executor rows.  Its important boundary is that retrieval only
selects facts already present in ``scene_rows``; it never completes evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from domain_schema import ANY_OBJECT_TYPES
from query_resolver import resolve_query
from scene_graph_query import relation_between, relation_matches, sort_objects


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    value: Any
    other_variable: str | None = None


@dataclass
class VariableState:
    name: str
    candidate_ids: list[str] = field(default_factory=list)
    satisfied: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    version: int = 0


@dataclass(frozen=True)
class EvidenceRequest:
    operation: str
    variable: str
    constraints: tuple[Constraint, ...]
    candidate_ids: tuple[str, ...] | None
    memory_version: int


@dataclass
class EvidenceResult:
    candidate_ids: list[str]
    rejected_ids: list[str]
    pending_ids: list[str]
    facts: list[dict[str, Any]]


@dataclass
class DerivedResult:
    id: str
    operation: str
    variable: str
    value: Any
    dependency_version: int
    dependencies: list[str]
    status: str = "SUPPORTED"
    revision: int = 0


class WorkingEvidenceMemory:
    """Mutable, QA-local bindings, constraints, provenance and computations."""

    def __init__(self) -> None:
        self.variables: dict[str, VariableState] = {}
        self.constraints: dict[str, list[Constraint]] = {}
        self.results: dict[str, DerivedResult] = {}
        self.events: list[dict[str, Any]] = []
        self.version = 0

    def state(self, variable: str) -> VariableState:
        return self.variables.setdefault(variable, VariableState(variable))

    def add_constraint(self, variable: str, constraint: Constraint) -> None:
        existing = self.constraints.setdefault(variable, [])
        if constraint not in existing:
            existing.append(constraint)
            self.state(variable).pending.append(constraint.id)
            self.version += 1
            self.events.append({"event": "constraint_added", "variable": variable,
                                "constraint": asdict(constraint), "memory_version": self.version})

    def update_candidates(self, variable: str, result: EvidenceResult) -> bool:
        state = self.state(variable)
        previous = list(state.candidate_ids)
        state.candidate_ids = list(result.candidate_ids)
        state.evidence_refs = list(dict.fromkeys(state.evidence_refs + [f["evidence_ref"] for f in result.facts]))
        for constraint in self.constraints.get(variable, []):
            if result.pending_ids:
                if constraint.id not in state.pending:
                    state.pending.append(constraint.id)
            else:
                if constraint.id in state.pending:
                    state.pending.remove(constraint.id)
                if constraint.id not in state.satisfied:
                    state.satisfied.append(constraint.id)
        for oid in result.rejected_ids:
            state.rejected.setdefault(oid, []).extend(
                c.id for c in self.constraints.get(variable, []) if c.id not in state.rejected[oid]
            )
        changed = previous != state.candidate_ids
        if changed:
            self.version += 1
            state.version = self.version
        self.events.append({"event": "candidates_updated", "variable": variable,
                            "before": previous, "after": state.candidate_ids,
                            "memory_version": self.version})
        return changed

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "variables": {k: asdict(v) for k, v in self.variables.items()},
            "constraints": {k: [asdict(c) for c in v] for k, v in self.constraints.items()},
            "results": {k: asdict(v) for k, v in self.results.items()},
            "events": list(self.events),
        }


class OperationGuidedRetriever:
    """Select operation-required rows/facts from immutable canonical evidence."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.rows = sort_objects(list(rows))
        self.by_id = {str(row.get("id")): row for row in self.rows}

    def retrieve(self, request: EvidenceRequest, memory: WorkingEvidenceMemory) -> EvidenceResult:
        pool = self.rows if request.candidate_ids is None else [
            self.by_id[x] for x in request.candidate_ids if x in self.by_id
        ]
        accepted, rejected, pending, facts = [], [], [], []
        for row in pool:
            verdicts: list[bool | None] = []
            row_facts: list[dict[str, Any]] = []
            for constraint in request.constraints:
                verdict, used = self._evaluate(row, constraint, memory)
                verdicts.append(verdict)
                row_facts.extend(self._facts(row, fields=used, operation=request.operation))
            oid = str(row.get("id"))
            if False in verdicts:
                rejected.append(oid)
            elif None in verdicts:
                pending.append(oid)
            else:
                accepted.append(row)
                facts.extend(row_facts)
        return EvidenceResult([str(x.get("id")) for x in sort_objects(accepted)], rejected, pending, facts)

    def _evaluate(self, row: dict[str, Any], constraint: Constraint,
                  memory: WorkingEvidenceMemory) -> tuple[bool | None, list[str]]:
        if constraint.kind == "category":
            return (True if constraint.value in ANY_OBJECT_TYPES else row.get("category") == constraint.value), ["category"]
        if constraint.kind == "attribute":
            status = row.get("status") or row.get("raw_status")
            if status in {None, "", "unknown"}:
                return None, ["velocity"]
            value = constraint.value
            if value == "not_standing":
                return status != "standing", ["velocity"]
            return status == value, ["velocity"]
        if constraint.kind in {"relation", "relation_any"}:
            other = memory.state(constraint.other_variable or "reference").candidate_ids
            if not other:
                return None, ["center", "coordinate_frame"]
            saw_valid = False
            for oid in other:
                ref = self.by_id.get(oid)
                if ref is None or ref.get("id") == row.get("id"):
                    continue
                if not _has_geometry(row) or not _has_geometry(ref) or row.get("coordinate_frame") != ref.get("coordinate_frame"):
                    continue
                saw_valid = True
                required = constraint.value if constraint.kind == "relation_any" else [constraint.value]
                if any(relation_matches(relation_between(row, ref), value) for value in required):
                    return True, ["center", "coordinate_frame"]
            return (False if saw_valid else None), ["center", "coordinate_frame"]
        if constraint.kind == "ids":
            return str(row.get("id")) in set(map(str, constraint.value)), ["id"]
        raise ValueError(f"unsupported constraint kind: {constraint.kind}")

    @staticmethod
    def _facts(row: dict[str, Any], fields: list[str], operation: str) -> list[dict[str, Any]]:
        base = row.get("feature_ref") or _feature_ref(row)
        values = {"id": row.get("id"), "category": row.get("category"),
                  "velocity": row.get("velocity_xy"),
                  "center": (row.get("geometry") or {}).get("center_xyz"),
                  "coordinate_frame": row.get("coordinate_frame")}
        return [{"object_id": str(row.get("id")), "field": f, "value": values.get(f),
                 "source": row.get("source"), "evidence_ref": f"{base}.{f}", "operation": operation}
                for f in fields]


class RASEExecutor:
    """Execute a frozen plan through retrieval -> memory -> atomic execution."""

    def execute_plan(self, plan: dict[str, Any], rows: list[dict[str, Any]],
                     eval_type: str | None = None) -> tuple[str | None, dict[str, Any]]:
        query = resolve_query(plan, eval_type=eval_type)
        operations: list[dict[str, Any]] = []
        target = query["target_filter"]
        operations.append({"op": "constrain", "variable": "target", "kind": "category", "value": target["category"]})
        for attr in target.get("attributes") or []:
            operations.append({"op": "constrain", "variable": "target", "kind": "attribute", "value": attr})
        ref = query.get("reference_filter")
        if ref:
            operations.append({"op": "constrain", "variable": "reference", "kind": "category", "value": ref["category"]})
            for attr in ref.get("attributes") or []:
                operations.append({"op": "constrain", "variable": "reference", "kind": "attribute", "value": attr})
        relations = query.get("spatial_relations") or []
        if relations:
            # The legacy query schema defines a relation list as alternatives.
            operations.append({"op": "constrain", "variable": "target", "kind": "relation_any",
                               "value": relations, "other_variable": "reference"})
        operations.append({"op": "compute", "operation": query["operation"], "variable": "target",
                           "reference_variable": "reference", "requires_reference": bool(ref and relations),
                           "comparison_relations": query.get("comparison_relations") or []})
        answer, trace = self.execute_program(operations, rows)
        trace["query"] = query
        return answer, trace

    def execute_program(self, operations: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
        memory = WorkingEvidenceMemory()
        retriever = OperationGuidedRetriever(rows)
        steps: list[dict[str, Any]] = []
        answer: str | None = None
        for index, op in enumerate(operations, 1):
            if op["op"] == "constrain":
                variable = op["variable"]
                constraint = Constraint(op.get("id", f"c{index:02d}"), op["kind"], op["value"], op.get("other_variable"))
                memory.add_constraint(variable, constraint)
                current = memory.state(variable).candidate_ids
                # Category starts a binding; later constraints refine it. A
                # deliberate second category may rebind from the full evidence.
                scope = None if constraint.kind == "category" else tuple(current)
                request = EvidenceRequest("apply_constraint", variable,
                                          tuple(memory.constraints[variable]), scope, memory.version)
                result = retriever.retrieve(request, memory)
                changed = memory.update_candidates(variable, result)
                revised = self._refresh_results(memory) if changed else []
                steps.append({"step_id": f"s{index:02d}", "op": "retrieve_and_update",
                              "request": _request_dict(request), "retrieval": asdict(result),
                              "revised_results": revised, "memory_version": memory.version})
            elif op["op"] == "compute":
                answer, result = self._compute(op, memory, retriever)
                memory.results[result.id] = result
                steps.append({"step_id": f"s{index:02d}", "op": "stateful_atomic_execute",
                              "input": dict(op), "output": asdict(result), "memory_version": memory.version})
            else:
                raise ValueError(f"unsupported RASE operation: {op['op']}")
        if memory.results:
            answer = self._render(next(reversed(memory.results.values())).value)
        final = next(reversed(memory.results.values())) if memory.results else None
        return answer, {"source": "rase_core", "steps": steps, "working_memory": memory.snapshot(),
                        "candidate_answer": answer, "final_status": final.status if final else "EXECUTION_FAILURE",
                        "ambiguity_candidates": (memory.state(final.variable).candidate_ids
                                                 if final and final.status == "AMBIGUOUS" else [])}

    def _compute(self, op: dict[str, Any], memory: WorkingEvidenceMemory,
                 retriever: OperationGuidedRetriever) -> tuple[str | None, DerivedResult]:
        variable = op.get("variable", "target")
        ids = memory.state(variable).candidate_ids
        operation = op["operation"]
        value: Any
        status = "SUPPORTED"
        reference_ids = (memory.state(op.get("reference_variable", "reference")).candidate_ids
                         if operation == "compare" or op.get("requires_reference") else [])
        if op.get("requires_reference") and not reference_ids:
            value, status = None, "EXECUTION_FAILURE"
        elif operation == "count": value = len(ids)
        elif operation == "exist": value = bool(ids)
        elif operation in {"lookup_category", "lookup_status"}:
            field = "category" if operation == "lookup_category" else "status"
            value = retriever.by_id[ids[0]].get(field) if len(ids) == 1 else None
            status = "AMBIGUOUS" if len(ids) > 1 else "EXECUTION_FAILURE" if not ids else "SUPPORTED"
        elif operation == "compare":
            refs = reference_ids
            if len(ids) != 1 or len(refs) != 1:
                value = None
                status = "AMBIGUOUS" if ids and refs else "EXECUTION_FAILURE"
            else:
                rels = op.get("comparison_relations") or []
                field = "category" if "same-category-as" in rels else "status"
                value = retriever.by_id[ids[0]].get(field) == retriever.by_id[refs[0]].get(field)
        else: value = None
        result_id = op.get("result", f"result_{len(memory.results)+1}")
        result = DerivedResult(result_id, operation, variable, value, memory.state(variable).version,
                               [variable] + ([op.get("reference_variable", "reference")] if operation == "compare" else []),
                               status=status)
        return self._render(value), result

    def _refresh_results(self, memory: WorkingEvidenceMemory) -> list[dict[str, Any]]:
        revised = []
        for result in memory.results.values():
            state = memory.state(result.variable)
            if state.version == result.dependency_version:
                continue
            old = result.value
            if result.operation == "count": result.value = len(state.candidate_ids)
            elif result.operation == "exist": result.value = bool(state.candidate_ids)
            else: continue
            result.dependency_version = state.version
            result.revision += 1
            event = {"result_id": result.id, "old_value": old, "new_value": result.value,
                     "revision": result.revision, "cause_version": state.version}
            memory.events.append({"event": "result_revised", **event})
            revised.append(event)
        return revised

    @staticmethod
    def _render(value: Any) -> str | None:
        if value is None: return None
        if isinstance(value, bool): return "yes" if value else "no"
        return str(value)


def _request_dict(request: EvidenceRequest) -> dict[str, Any]:
    result = asdict(request)
    result["constraints"] = [asdict(x) for x in request.constraints]
    return result


def _has_geometry(row: dict[str, Any]) -> bool:
    return row.get("x") is not None and row.get("y") is not None


def _feature_ref(row: dict[str, Any]) -> str:
    oid = str(row.get("id", ""))
    try: index = max(0, int(oid.rsplit("_", 1)[-1]) - 1)
    except ValueError: index = 0
    return f"{row.get('sample_token', 'unknown')}.json:objects[{index}]"
