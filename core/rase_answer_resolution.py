"""Constraint-aware final resolution and aligned IR for RASE Phase 3."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from query_resolver import resolve_query
from rase_core import RASEExecutor
from scene_graph_query import relation_between, relation_matches, select_objects, select_related


@dataclass(frozen=True)
class ResolutionDecision:
    status: str
    answer: str | None
    strategy: str
    candidate_answers: dict[str, str]
    chosen_ids: list[str]
    ranking: list[dict[str, Any]]
    reason: str


def compile_aligned_ir(plan: dict[str, Any]) -> dict[str, Any]:
    """Compile the flat Planner plan into an explicit variable/constraint IR."""
    query = resolve_query(plan)
    constraints, cid = [], 0
    def add(variable, kind, value, other=None):
        nonlocal cid
        cid += 1
        constraints.append({"id": f"c{cid:02d}", "variable": variable, "kind": kind,
                            "value": value, "other_variable": other})
    add("target", "category", query["target_filter"]["category"])
    for value in query["target_filter"].get("attributes") or []: add("target", "attribute", value)
    if query.get("reference_filter"):
        add("reference", "category", query["reference_filter"]["category"])
        for value in query["reference_filter"].get("attributes") or []: add("reference", "attribute", value)
    if query.get("spatial_relations"):
        add("target", "relation_any", query["spatial_relations"], "reference")
    return {"version": "rase_phase3.aligned_ir.v1", "variables": {
                "target": {"role": "answer_subject"},
                **({"reference": {"role": "relation_anchor"}} if query.get("reference_filter") else {})},
            "constraints": constraints,
            "answer": {"operation": query["operation"], "target_variable": "target",
                       "reference_variable": "reference" if query.get("reference_filter") else None,
                       "comparison_relations": query.get("comparison_relations") or [],
                       "depends_on": [x["id"] for x in constraints]},
            "query": query}


class ConstraintAwareAnswerResolver:
    """Resolve final candidates after all IR constraints have run."""

    def resolve(self, query: dict[str, Any], memory: dict[str, Any],
                rows: list[dict[str, Any]], strict_answer: str | None,
                strict_status: str) -> ResolutionDecision:
        variables = memory.get("variables", {})
        target_ids = (variables.get("target") or {}).get("candidate_ids", [])
        reference_ids = (variables.get("reference") or {}).get("candidate_ids", [])
        by_id = {str(row.get("id")): row for row in rows}
        if strict_status == "SUPPORTED":
            return ResolutionDecision("SUPPORTED", strict_answer, "already_determined", {},
                                      target_ids[:1] + (reference_ids[:1] if query["operation"] == "compare" else []),
                                      [], "execution_already_produced_a_definite_answer")
        if strict_status != "AMBIGUOUS":
            return ResolutionDecision(strict_status, None, "unresolved", {}, [], [],
                                      "required_binding_or_evidence_is_missing")
        op = query["operation"]
        if op in {"lookup_category", "lookup_status"}:
            field = "category" if op == "lookup_category" else "status"
            answers = {oid: _text((by_id.get(oid) or {}).get(field)) for oid in target_ids}
            usable = {k: v for k, v in answers.items() if v is not None and v != "unknown"}
            return self._resolve_answers(usable, target_ids, reference_ids, query, by_id)
        if op == "compare":
            field = "category" if "same-category-as" in query.get("comparison_relations", []) else "status"
            pair_answers: dict[str, str] = {}
            legal_pairs: list[tuple[str, str]] = []
            for tid in target_ids:
                for rid in reference_ids:
                    if tid == rid or tid not in by_id or rid not in by_id: continue
                    if query.get("spatial_relations") and not any(
                        relation_matches(relation_between(by_id[tid], by_id[rid]), required)
                        for required in query["spatial_relations"]):
                        continue
                    left, right = by_id[tid].get(field), by_id[rid].get(field)
                    if left in {None, "", "unknown"} or right in {None, "", "unknown"}: continue
                    key = f"{tid}|{rid}"; legal_pairs.append((tid, rid))
                    pair_answers[key] = "yes" if left == right else "no"
            return self._resolve_answers(pair_answers, target_ids, reference_ids, query, by_id, legal_pairs)
        return ResolutionDecision("AMBIGUOUS", None, "unresolved", {}, [], [],
                                  "operation_has_no_multi_candidate_resolution_rule")

    def _resolve_answers(self, answers: dict[str, str], target_ids: list[str], reference_ids: list[str],
                         query: dict[str, Any], by_id: dict[str, dict[str, Any]],
                         legal_pairs: list[tuple[str, str]] | None = None) -> ResolutionDecision:
        if not answers:
            return ResolutionDecision("AMBIGUOUS", None, "unresolved", {}, [], [],
                                      "candidates_do_not_expose_the_required_answer_field")
        values = set(answers.values())
        if len(values) == 1:
            return ResolutionDecision("SUPPORTED", next(iter(values)), "answer_aggregation", answers,
                                      sorted({x for key in answers for x in key.split("|")}), [],
                                      "all_constraint_consistent_candidates_imply_the_same_answer")
        ranking = []
        if legal_pairs is None:
            for oid in answers:
                compatible = [rid for rid in reference_ids if rid in by_id and rid != oid and
                              (not query.get("spatial_relations") or any(
                                  relation_matches(relation_between(by_id[oid], by_id[rid]), required)
                                  for required in query["spatial_relations"]))]
                if compatible and query.get("spatial_relations"):
                    choices = []
                    for rid in compatible:
                        score, components = _candidate_score(by_id[oid], by_id[rid], query)
                        choices.append((score, rid, components))
                    score, rid, components = sorted(choices, key=lambda x: (-x[0], x[1]))[0]
                    ranking.append({"key": f"{oid}|{rid}", "ids": [oid, rid], "answer": answers[oid],
                                    "score": score, "components": components})
                else:
                    score, components = _candidate_score(by_id[oid], None, query)
                    ranking.append({"key": oid, "ids": [oid], "answer": answers[oid],
                                    "score": score, "components": components})
        else:
            for tid, rid in legal_pairs:
                key = f"{tid}|{rid}"
                score, components = _candidate_score(by_id[tid], by_id[rid], query)
                ranking.append({"key": key, "ids": [tid, rid], "answer": answers[key],
                                "score": score, "components": components})
        ranking.sort(key=lambda x: (-x["score"], x["key"]))
        if not ranking:
            return ResolutionDecision("AMBIGUOUS", None, "unresolved", answers, [], [], "no_legal_ranked_binding")
        top = ranking[0]
        return ResolutionDecision("SUPPORTED", top["answer"], "constraint_consistent_ranking", answers,
                                  top["ids"], ranking,
                                  "conflicting_answers_ranked_after_all_constraints_using_confidence_relation_margin_and_distance")


class ConstraintAwareRASEExecutor:
    def __init__(self) -> None:
        self.core = RASEExecutor()
        self.resolver = ConstraintAwareAnswerResolver()

    def execute_plan(self, plan: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
        ir = compile_aligned_ir(plan)
        strict_answer, trace = self.core.execute_plan(plan, rows)
        decision = self.resolver.resolve(ir["query"], trace["working_memory"], rows,
                                         strict_answer, trace["final_status"])
        consumed = {c["id"] for values in trace["working_memory"]["constraints"].values() for c in values}
        expected = {c["id"] for c in ir["constraints"]}
        trace = deepcopy(trace)
        trace["aligned_ir"] = ir
        trace["ir_alignment"] = {"expected_constraint_ids": sorted(expected),
                                 "consumed_constraint_ids": sorted(consumed),
                                 "missing_constraint_ids": sorted(expected-consumed),
                                 "expected_answer_operation": ir["answer"]["operation"],
                                 "consumed_answer_operation": trace["query"]["operation"],
                                 "expected_comparison_relations": ir["answer"]["comparison_relations"],
                                 "consumed_comparison_relations": trace["query"]["comparison_relations"],
                                 "all_constraints_consumed": (expected == consumed and
                                     ir["answer"]["operation"] == trace["query"]["operation"] and
                                     ir["answer"]["comparison_relations"] == trace["query"]["comparison_relations"])}
        trace["strict_candidate_answer"] = strict_answer
        trace["strict_final_status"] = trace["final_status"]
        trace["resolution"] = asdict(decision)
        trace["candidate_answer"] = decision.answer
        trace["final_status"] = decision.status
        return decision.answer, trace


class StatelessConstraintAwareExecutor:
    """Equivalent final resolver over local lists; retrieval/memory ablation."""

    def __init__(self) -> None:
        self.resolver = ConstraintAwareAnswerResolver()

    def execute_plan(self, plan: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
        query = resolve_query(plan)
        targets = select_objects(rows, query["target_filter"])
        references = select_objects(rows, query["reference_filter"]) if query.get("reference_filter") else []
        if query.get("spatial_relations"):
            if not references:
                return None, {"source": "stateless_constraint_aware", "final_status": "EXECUTION_FAILURE",
                              "resolution": {"strategy": "unresolved", "reason": "no_reference_match"}}
            targets, _ = select_related(targets, references, query["spatial_relations"])
        op = query["operation"]
        if op == "count": strict_answer, strict_status = str(len(targets)), "SUPPORTED"
        elif op == "exist": strict_answer, strict_status = ("yes" if targets else "no"), "SUPPORTED"
        elif op in {"lookup_category", "lookup_status"} and len(targets) == 1:
            strict_answer = str(targets[0]["category" if op == "lookup_category" else "status"]); strict_status = "SUPPORTED"
        elif op == "compare" and len(targets) == len(references) == 1:
            field = "category" if "same-category-as" in query.get("comparison_relations", []) else "status"
            strict_answer = "yes" if targets[0].get(field) == references[0].get(field) else "no"; strict_status = "SUPPORTED"
        else:
            strict_answer = None
            strict_status = "AMBIGUOUS" if targets and (op != "compare" or references) else "EXECUTION_FAILURE"
        memory = {"variables": {"target": {"candidate_ids": [str(x["id"]) for x in targets]},
                                "reference": {"candidate_ids": [str(x["id"]) for x in references]}}}
        decision = self.resolver.resolve(query, memory, rows, strict_answer, strict_status)
        return decision.answer, {"source": "stateless_constraint_aware", "query": query,
                                 "final_status": decision.status, "resolution": asdict(decision),
                                 "candidate_answer": decision.answer}


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _candidate_score(target: dict[str, Any], reference: dict[str, Any] | None,
                     query: dict[str, Any]) -> tuple[float, dict[str, float]]:
    tc = float(target.get("confidence") or 0.0)
    rc = float(reference.get("confidence") or 0.0) if reference else 0.0
    relation_margin = _relation_margin(target, reference, query.get("spatial_relations") or []) if reference else 0.0
    # Distance is deliberately a small last-order term; unlike S0 it cannot
    # override confidence or relation consistency.
    distance_penalty = min(float(target.get("distance") or 1e6), 1e6) * 1e-6
    score = tc + rc + 0.01 * relation_margin - distance_penalty
    return score, {"target_confidence": tc, "reference_confidence": rc,
                   "relation_margin": relation_margin, "distance_penalty": distance_penalty}


def _relation_margin(target: dict[str, Any], reference: dict[str, Any] | None,
                     relations: list[str]) -> float:
    if reference is None or not relations: return 0.0
    dx = float(target.get("x", 0.0))-float(reference.get("x", 0.0))
    dy = float(target.get("y", 0.0))-float(reference.get("y", 0.0))
    margins = []
    for relation in relations:
        parts = relation.split("-"); values = []
        if "front" in parts: values.append(dx-2.0)
        if "back" in parts: values.append(-dx-2.0)
        if "left" in parts: values.append(dy-2.0)
        if "right" in parts: values.append(-dy-2.0)
        if relation == "near": values.append(2.0-max(abs(dx), abs(dy)))
        if values: margins.append(min(values))
    return max(margins) if margins else 0.0
