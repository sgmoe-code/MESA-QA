# agent_question_planner.py
# Step 1: 问题解析与受约束推理路径生成模块

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


CATEGORY_ALIASES = {
    "cars": "car", "car": "car", "vehicle": "car", "vehicles": "car",
    "truck": "truck", "trucks": "truck",
    "bus": "bus", "buses": "bus",
    "trailer": "trailer", "trailers": "trailer",
    "construction vehicle": "construction_vehicle",
    "construction vehicles": "construction_vehicle",
    "pedestrian": "pedestrian", "pedestrians": "pedestrian",
    "person": "pedestrian", "people": "pedestrian", "human": "pedestrian",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle", "motorbike": "motorcycle",
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle",
    "traffic cone": "traffic_cone", "traffic cones": "traffic_cone",
    "cone": "traffic_cone", "cones": "traffic_cone",
    "barrier": "barrier", "barriers": "barrier",
}

STATUS_ALIASES = {
    "moving": "moving", "move": "moving", "driving": "moving",
    "walking": "moving", "running": "moving",
    "parked": "parked", "parking": "parked",
    "stopped": "stopped", "stop": "stopped",
    "standing": "stopped", "stationary": "stopped",
    "with rider": "with_rider", "without rider": "without_rider",
}

RELATION_PATTERNS = [
    ("front-left", ["front left", "front-left", "left front"]),
    ("front-right", ["front right", "front-right", "right front"]),
    ("back-left", ["back left", "back-left", "rear left", "left back"]),
    ("back-right", ["back right", "back-right", "rear right", "right back"]),
    ("front", ["in front of", "to the front of", "front of", "front"]),
    ("back", ["behind", "to the back of", "back of", "rear of", "back"]),
    ("left", ["to the left of", "left of", "left"]),
    ("right", ["to the right of", "right of", "right"]),
]


@dataclass
class QuestionPlan:
    question: str
    question_type: str
    answer_type: str
    hop: str
    target_entity: Dict[str, Any]
    reference_entity: Optional[Dict[str, Any]]
    relations: List[str]
    need_modules: List[str]
    reasoning_path: List[str]
    parse_confidence: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GuidedQuestionPlanner:
    """
    受约束 Planner V1：
    1. 先用 benchmark template_type 做稳定锚点；
    2. 再从 question 中抽取疑问词、实体、状态、空间关系；
    3. 输出固定 JSON-like plan；
    4. 后续可把内部规则替换成 MLLM Planner，但下游接口不变。
    """

    def plan(self, question: str, template_type: str = "") -> QuestionPlan:
        q = self._normalize(question)
        template_main = self._template_main(template_type)

        question_type = self._infer_question_type(q, template_main)
        answer_type = self._infer_answer_type(question_type)
        categories = self._extract_categories(q)
        statuses = self._extract_statuses(q)
        relations = self._extract_relations(q)
        hop = "one-hop" if self._is_one_hop(q, categories, relations) else "zero-hop"

        target_entity = {
            "category": categories[0] if categories else "unknown",
            "attributes": statuses[:1] if statuses else [],
        }

        reference_entity = None
        if len(categories) >= 2 and (relations or "same as" in q):
            reference_entity = {
                "category": categories[1],
                "attributes": statuses[1:2] if len(statuses) > 1 else [],
            }

        need_modules, reasoning_path = self._select_modules(
            question_type=question_type,
            hop=hop,
            relations=relations,
            q=q,
        )

        confidence = self._confidence(template_main, question_type, categories, relations)
        notes = f"template={template_main or 'NA'}; type={question_type}; hop={hop}; relations={relations}"

        return QuestionPlan(
            question=question,
            question_type=question_type,
            answer_type=answer_type,
            hop=hop,
            target_entity=target_entity,
            reference_entity=reference_entity,
            relations=relations,
            need_modules=need_modules,
            reasoning_path=reasoning_path,
            parse_confidence=confidence,
            notes=notes,
        )

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = text.replace("?", " ?")
        return re.sub(r"\s+", " ", text)

    def _template_main(self, template_type: str) -> str:
        if not template_type:
            return ""
        return template_type.lower().split("_")[0]

    def _infer_question_type(self, q: str, template_main: str) -> str:
        if template_main in {"exist", "count", "object", "status", "comparison"}:
            return template_main
        if q.startswith("how many") or "what number of" in q or "number of" in q:
            return "count"
        if q.startswith("is there") or q.startswith("are there") or q.startswith("do you see"):
            return "exist"
        if "same as" in q or "more than" in q or "less than" in q or "fewer than" in q:
            return "comparison"
        if "status" in q or "moving" in q or "parked" in q or "stopped" in q:
            return "status"
        if q.startswith("what") or "what object" in q or "what kind" in q:
            return "object"
        return "object"

    def _infer_answer_type(self, question_type: str) -> str:
        if question_type == "count":
            return "number"
        if question_type in {"exist", "comparison"}:
            return "yes_no"
        if question_type == "status":
            return "status"
        if question_type == "object":
            return "category"
        return "text"

    def _extract_categories(self, q: str) -> List[str]:
        found = []
        for alias in sorted(CATEGORY_ALIASES.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", q):
                cat = CATEGORY_ALIASES[alias]
                if cat not in found:
                    found.append(cat)
        return found

    def _extract_statuses(self, q: str) -> List[str]:
        found = []
        for alias in sorted(STATUS_ALIASES.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", q):
                status = STATUS_ALIASES[alias]
                if status not in found:
                    found.append(status)
        return found

    def _extract_relations(self, q: str) -> List[str]:
        found = []
        for rel, pats in RELATION_PATTERNS:
            for pat in pats:
                if re.search(r"\b" + re.escape(pat) + r"\b", q):
                    if rel not in found:
                        found.append(rel)
                    break
        return found

    def _is_one_hop(self, q: str, categories: List[str], relations: List[str]) -> bool:
        markers = [
            "of the", "of a", "of an", "behind the", "behind a",
            "to the left of the", "to the right of the",
            "front of the", "back of the", "same as the",
        ]
        if len(categories) >= 2 and any(m in q for m in markers):
            return True
        if len(categories) >= 2 and relations:
            return True
        return False

    def _select_modules(self, question_type: str, hop: str, relations: List[str], q: str):
        need = []
        path = []

        if question_type in {"count", "comparison"} or hop == "one-hop" or relations:
            need.append("scene_graph_retriever")
            path.append("scene_graph_retriever")

        if question_type in {"count", "comparison"}:
            need.extend(["symbolic_executor", "evidence_verifier"])
            path.extend(["symbolic_executor", "evidence_verifier"])
        elif question_type == "status":
            need.extend(["visual_tool", "result_parser"])
            path.extend(["visual_tool", "result_parser"])
        elif question_type == "exist":
            need.extend(["visual_tool", "result_parser"])
            path.extend(["visual_tool", "result_parser"])
            if relations:
                need.append("evidence_verifier")
                path.append("evidence_verifier")
        else:
            need.extend(["visual_tool", "result_parser"])
            path.extend(["visual_tool", "result_parser"])

        if any(k in q for k in ["safe", "risk", "danger", "should", "evacuate", "rule"]):
            need.append("open_rag")
            path.insert(0, "open_rag")

        need = list(dict.fromkeys(need))
        path = list(dict.fromkeys(path))
        return need, path

    def _confidence(self, template_main, question_type, categories, relations):
        score = 0.55
        if template_main:
            score += 0.2
        if question_type:
            score += 0.1
        if categories:
            score += 0.1
        if relations:
            score += 0.05
        return min(score, 0.95)
