"""Train-split QA exemplar retrieval for NuScenes-QA."""

import json
import os
import re
from typing import Any, Dict, List, Set

from agent_question_planner import CATEGORY_ALIASES, GuidedQuestionPlanner, RELATION_PATTERNS, STATUS_ALIASES


SOURCE = "nuscenes_qa_train"
STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "both", "do", "does", "for", "how",
    "in", "is", "it", "me", "of", "or", "that", "the", "there", "to", "what", "which",
}
CATEGORY_EXTRA = {"trafficcone": "traffic_cone"}
STATUS_EXTRA = {
    "not standing": "not_standing",
    "not_standing": "not_standing",
    "sitting lying down": "not_standing",
}


def _as_plan(plan: Any) -> Dict[str, Any]:
    if hasattr(plan, "to_dict"):
        return plan.to_dict()
    return plan if isinstance(plan, dict) else {}


def _norm_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    for src, dst in [
        ("front left", "front-left"),
        ("front right", "front-right"),
        ("back left", "back-left"),
        ("back right", "back-right"),
    ]:
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()


def _norm_category(value: str) -> str:
    aliases = dict(CATEGORY_ALIASES)
    aliases.update(CATEGORY_EXTRA)
    value = _norm_text(value)
    return aliases.get(value, value.replace(" ", "_"))


def _norm_status(value: str) -> str:
    aliases = dict(STATUS_ALIASES)
    aliases.update(STATUS_EXTRA)
    value = _norm_text(value)
    return aliases.get(value, value.replace(" ", "_"))


def _find_aliases(text: str, aliases: Dict[str, str]) -> List[str]:
    padded = f" {_norm_text(text)} "
    found = []
    for alias in sorted(aliases, key=len, reverse=True):
        needle = f" {alias.replace('_', ' ')} "
        if needle in padded:
            value = aliases[alias]
            if value not in found:
                found.append(value)
    return found


def _extract_categories(text: str) -> List[str]:
    aliases = dict(CATEGORY_ALIASES)
    aliases.update(CATEGORY_EXTRA)
    return _find_aliases(text, aliases)


def _extract_statuses(text: str) -> List[str]:
    aliases = dict(STATUS_ALIASES)
    aliases.update(STATUS_EXTRA)
    return _find_aliases(text, aliases)


def _extract_relations(text: str) -> List[str]:
    padded = f" {_norm_text(text)} "
    found = []
    for relation, patterns in RELATION_PATTERNS:
        for pattern in patterns:
            needle = f" {_norm_text(pattern)} "
            if needle in padded:
                if relation not in found:
                    found.append(relation)
                break
    return found


def _keywords(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", _norm_text(text))
    tokens = []
    for word in words:
        token = _norm_status(_norm_category(word))
        if token not in STOPWORDS and len(token) > 1:
            tokens.append(token)
    tokens.extend(_extract_categories(text))
    tokens.extend(_extract_statuses(text))
    tokens.extend(_extract_relations(text))
    return sorted(set(tokens))


def _entity(plan: Dict[str, Any], key: str) -> Dict[str, Any]:
    entity = plan.get(key) or {}
    if not isinstance(entity, dict):
        return {}
    out = dict(entity)
    if out.get("category"):
        out["category"] = _norm_category(str(out["category"]))
    attrs = out.get("attributes") or []
    if isinstance(attrs, str):
        attrs = [attrs]
    out["attributes"] = [_norm_status(str(attr)) for attr in attrs]
    return out



def _cheap_plan(question: str, template_type: str, num_hop: Any = None) -> Dict[str, Any]:
    categories = _extract_categories(question)
    statuses = _extract_statuses(question)
    relations = _extract_relations(question)
    qtype = (template_type or "").lower().split("_")[0]
    if qtype not in {"exist", "count", "object", "status", "comparison"}:
        q = _norm_text(question)
        if q.startswith("how many") or "number of" in q:
            qtype = "count"
        elif q.startswith("is there") or q.startswith("are there"):
            qtype = "exist"
        elif "same as" in q or "more than" in q or "less than" in q or "fewer than" in q:
            qtype = "comparison"
        elif "status" in q or statuses:
            qtype = "status"
        else:
            qtype = "object"
    answer_type = {"count": "number", "exist": "yes_no", "comparison": "yes_no", "status": "status", "object": "category"}.get(qtype, "text")
    if num_hop is not None:
        try:
            hop = "one-hop" if int(num_hop) > 0 else "zero-hop"
        except (TypeError, ValueError):
            hop = "one-hop" if len(categories) >= 2 and relations else "zero-hop"
    else:
        hop = "one-hop" if len(categories) >= 2 and relations else "zero-hop"
    return {
        "question_type": qtype,
        "answer_type": answer_type,
        "hop": hop,
        "target_entity": {"category": categories[0] if categories else "unknown", "attributes": statuses[:1]},
        "reference_entity": {"category": categories[1], "attributes": statuses[1:2]} if len(categories) >= 2 and relations else None,
        "relations": relations,
        "need_modules": [],
    }

def _question_type(template_type: str, plan: Dict[str, Any]) -> str:
    return plan.get("question_type") or (template_type or "").lower().split("_")[0]


def _hop(item: Dict[str, Any], plan: Dict[str, Any]) -> str:
    if "num_hop" in item:
        try:
            return "one-hop" if int(item["num_hop"]) > 0 else "zero-hop"
        except (TypeError, ValueError):
            pass
    return plan.get("hop", "")


def _reasoning_program(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    target = _entity(plan, "target_entity")
    reference = _entity(plan, "reference_entity")
    relations = plan.get("relations") or []
    if isinstance(relations, str):
        relations = [relations]
    program = [{"op": "find_objects", "category": target.get("category", "unknown")}]
    if target.get("attributes"):
        program.append({"op": "filter_attributes", "attributes": target["attributes"]})
    if reference:
        program.append({"op": "find_references", "category": reference.get("category", "unknown")})
        if reference.get("attributes"):
            program.append({"op": "filter_reference_attributes", "attributes": reference["attributes"]})
    if relations:
        program.append({"op": "spatial_match_all_refs", "relations": [_norm_text(rel) for rel in relations]})
    qtype = plan.get("question_type", "object")
    program.append({
        "count": {"op": "count"},
        "exist": {"op": "exist"},
        "status": {"op": "lookup_status"},
        "comparison": {"op": "compare"},
    }.get(qtype, {"op": "lookup_category"}))
    return program


class QAExemplarIndex:
    def __init__(
        self,
        train_path: str,
        index_path: str = None,
        topk: int = 3,
        use_answer: bool = False,
        use_template: bool = True,
        use_plan: bool = True,
        use_reasoning_program: bool = True,
        rebuild_index: bool = False,
        fallback_planner=None,
        debug: bool = False,
    ):
        self.train_path = train_path
        self.index_path = index_path
        self.topk = topk
        self.use_answer = use_answer
        self.use_template = use_template
        self.use_plan = use_plan
        self.use_reasoning_program = use_reasoning_program
        self.rebuild_index = rebuild_index
        self.fallback_planner = fallback_planner or GuidedQuestionPlanner()
        self.debug = debug
        self.items: List[Dict[str, Any]] = []

    def build_or_load(self):
        if self.items:
            return self
        if self.index_path and os.path.exists(self.index_path) and not self.rebuild_index:
            self.items = self._load_index(self.index_path)
            return self
        self.items = self._build_index()
        if self.index_path:
            parent = os.path.dirname(self.index_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = self.index_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for item in self.items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self.index_path)
        return self

    def retrieve(self, question: str, plan: dict = None, template_type: str = "", topk: int = None) -> dict:
        self.build_or_load()
        query = self._features(question, plan, template_type)
        scored = [(self._score(query, item), item.get("example_id", ""), item) for item in self.items]
        scored.sort(key=lambda row: (-row[0], row[1]))
        effective_topk = self.topk if topk is None else max(0, int(topk))
        return {
            "kb_type": "qa_exemplar",
            "mode": "qa_exemplar",
            "query": {
                "question": question,
                "template_type": template_type,
                "question_type": query.get("question_type", ""),
                "target_category": query.get("target_category", ""),
                "reference_category": query.get("reference_category", ""),
                "relations": query.get("relations", []),
            },
            "retrieved_exemplars": [
                self._public_item(item, rank, score)
                for rank, (score, _, item) in enumerate(scored[:effective_topk], start=1)
            ],
            "retrieval_stats": {
                "num_indexed_examples": len(self.items),
                "topk": effective_topk,
                "use_answer": self.use_answer,
                "index_path": self.index_path,
            },
        }

    def _build_index(self) -> List[Dict[str, Any]]:
        if not self.train_path or not os.path.exists(self.train_path):
            raise FileNotFoundError(f"QA_EXEMPLAR_TRAIN_PATH not found: {self.train_path}")
        if any(part in os.path.basename(self.train_path).lower() for part in ("val", "test")):
            raise ValueError(f"QA exemplar index refuses non-train path: {self.train_path}")
        with open(self.train_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("QA exemplar train file must be a list or contain a questions list")
        items = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict) or str(row.get("split", "train")).lower() != "train":
                continue
            question = row.get("question", "")
            template_type = row.get("template_type", "")
            plan = _cheap_plan(question, template_type, row.get("num_hop"))
            relations = plan.get("relations") or _extract_relations(question)
            if isinstance(relations, str):
                relations = [relations]
            items.append({
                "example_id": str(row.get("question_id") or row.get("id") or f"train_{idx:07d}"),
                "question": question,
                "answer": str(row.get("answer", "")),
                "template_type": template_type,
                "question_type": _question_type(template_type, plan),
                "answer_type": plan.get("answer_type", ""),
                "hop": _hop(row, plan),
                "target_entity": _entity(plan, "target_entity"),
                "reference_entity": _entity(plan, "reference_entity"),
                "relations": [_norm_text(rel) for rel in relations],
                "need_modules": plan.get("need_modules", []),
                "reasoning_program": _reasoning_program(plan),
                "keywords": _keywords(question),
                "source": SOURCE,
            })
        return items

    def _load_index(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _features(self, question: str, plan: dict, template_type: str) -> Dict[str, Any]:
        plan = _as_plan(plan) or _as_plan(self.fallback_planner.plan(question, template_type=template_type))
        target = _entity(plan, "target_entity")
        reference = _entity(plan, "reference_entity")
        relations = plan.get("relations") or _extract_relations(question)
        if isinstance(relations, str):
            relations = [relations]
        return {
            "template_type": template_type,
            "question_type": _question_type(template_type, plan),
            "hop": plan.get("hop", ""),
            "target_category": target.get("category", ""),
            "reference_category": reference.get("category", ""),
            "target_attributes": set(target.get("attributes", [])),
            "reference_attributes": set(reference.get("attributes", [])),
            "relations": [_norm_text(rel) for rel in relations],
            "keywords": set(_keywords(question)),
        }

    def _score(self, query: Dict[str, Any], item: Dict[str, Any]) -> float:
        score = 0.0
        if self.use_template and query.get("template_type") and query["template_type"] == item.get("template_type"):
            score += 8
        if query.get("question_type") and query["question_type"] == item.get("question_type"):
            score += 5
        if query.get("hop") and query["hop"] == item.get("hop"):
            score += 4
        if query.get("target_category") and query["target_category"] == (item.get("target_entity") or {}).get("category"):
            score += 4
        if query.get("reference_category") and query["reference_category"] == (item.get("reference_entity") or {}).get("category"):
            score += 4
        score += 3 * len(set(query.get("relations", [])) & set(item.get("relations", [])))
        item_attrs = set((item.get("target_entity") or {}).get("attributes", []))
        item_attrs |= set((item.get("reference_entity") or {}).get("attributes", []))
        query_attrs: Set[str] = set(query.get("target_attributes", set())) | set(query.get("reference_attributes", set()))
        score += 3 * len(query_attrs & item_attrs)
        score += len(query.get("keywords", set()) & set(item.get("keywords", [])))
        if query.get("hop") == item.get("hop") == "one-hop":
            score += 1
        if query.get("question_type") == item.get("question_type") == "comparison":
            score += 1
        return score

    def _public_item(self, item: Dict[str, Any], rank: int, score: float) -> Dict[str, Any]:
        out = {
            "rank": rank,
            "score": round(float(score), 4),
            "example_id": item.get("example_id", ""),
            "question": item.get("question", ""),
            "template_type": item.get("template_type", ""),
            "question_type": item.get("question_type", ""),
            "answer_type": item.get("answer_type", ""),
            "hop": item.get("hop", ""),
            "target_entity": item.get("target_entity", {}),
            "reference_entity": item.get("reference_entity", {}),
            "relations": item.get("relations", []),
            "source": item.get("source", SOURCE),
            "answer_included": bool(self.use_answer),
        }
        if self.use_plan:
            out["need_modules"] = item.get("need_modules", [])
        if self.use_reasoning_program:
            out["reasoning_program"] = item.get("reasoning_program", [])
        if self.use_answer:
            out["answer"] = item.get("answer", "")
        return out


def _program_line(step: Dict[str, Any]) -> str:
    args = []
    for key in ("category", "attributes", "relations"):
        if key in step:
            value = step[key]
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            args.append(str(value))
    return f"{step.get('op', '')}({', '.join(args)})" if args else str(step.get("op", ""))


def format_qa_exemplars_for_prompt(rag_trace: dict, max_examples: int = 3, include_answer: bool = False) -> str:
    qa_trace = rag_trace.get("qa_exemplar", rag_trace) if isinstance(rag_trace, dict) else {}
    exemplars = qa_trace.get("retrieved_exemplars", []) if isinstance(qa_trace, dict) else []
    lines = ["[Retrieved QA Exemplars]"]
    for idx, ex in enumerate(exemplars[:max_examples], start=1):
        target = ex.get("target_entity") or {}
        reference = ex.get("reference_entity") or {}
        attrs = ", ".join(target.get("attributes", []))
        target_text = target.get("category", "unknown")
        if attrs:
            target_text = f"{target_text} [{attrs}]"
        lines.extend([
            f"Example {idx}:",
            f"Question: {ex.get('question', '')}",
            f"Template: {ex.get('template_type', '')}",
            "Plan:",
            f"- question_type: {ex.get('question_type', '')}",
            f"- target: {target_text}",
        ])
        if reference:
            lines.append(f"- reference: {reference.get('category', 'unknown')}")
        lines.append(f"- relations: {', '.join(ex.get('relations', []))}")
        if ex.get("reasoning_program"):
            lines.append("Reasoning Program:")
            for line_no, step in enumerate(ex["reasoning_program"], start=1):
                lines.append(f"{line_no}. {_program_line(step)}")
        if include_answer and ex.get("answer_included") and "answer" in ex:
            lines.append(f"Answer: {ex.get('answer', '')}")
    return "\n".join(lines)
