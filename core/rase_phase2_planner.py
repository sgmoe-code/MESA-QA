"""Question-grounded Planner/RASE feedback interface for Phase 2.

This is a deterministic IR repair controller, not a replacement MLLM.  It may
read the question, frozen plan and RASE state summary; it never reads labels.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class PlannerRASEController:
    VERSION = "rase_phase2.question_grounded_feedback.v1"

    def enhance(self, question: str, frozen_plan: dict[str, Any],
                feedback: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        plan = deepcopy(frozen_plan)
        text = " ".join(question.lower().replace("_", " ").split())
        repairs: list[dict[str, Any]] = []
        relations = list(plan.get("relations") or [])

        # The old flat schema frequently classifies explicit comparisons as a
        # status/object lookup. Preserve the spatial relation and add the
        # comparison operator expressed verbatim by the question.
        if any(phrase in text for phrase in ("same status", "same state")):
            if plan.get("question_type") != "comparison":
                repairs.append({"field": "question_type", "before": plan.get("question_type"),
                                "after": "comparison", "source": "question_explicit_same_status"})
                plan["question_type"] = "comparison"
                plan["answer_type"] = "yes_no"
            if "same-status-as" not in relations:
                relations.append("same-status-as")
                repairs.append({"field": "relations", "added": "same-status-as",
                                "source": "question_explicit_same_status"})
        if "same categor" in text or "same type" in text:
            if plan.get("question_type") != "comparison":
                repairs.append({"field": "question_type", "before": plan.get("question_type"),
                                "after": "comparison", "source": "question_explicit_same_category"})
                plan["question_type"] = "comparison"
                plan["answer_type"] = "yes_no"
            if "same-category-as" not in relations:
                relations.append("same-category-as")
                repairs.append({"field": "relations", "added": "same-category-as",
                                "source": "question_explicit_same_category"})
        plan["relations"] = relations

        feedback = feedback or {}
        request = None
        if feedback.get("final_status") == "AMBIGUOUS":
            request = {"type": "need_additional_question_constraint",
                       "candidate_ids": feedback.get("ambiguity_candidates") or [],
                       "legal_sources": ["question_text", "current_real_evidence"],
                       "resolved": bool(repairs)}
        return plan, {"controller_version": self.VERSION, "repairs": repairs,
                      "feedback_request": request, "model_calls": 0,
                      "label_access": False, "new_evidence_access": False}
