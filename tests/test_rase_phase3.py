from rase_answer_resolution import ConstraintAwareRASEExecutor, compile_aligned_ir


def row(oid, category="car", x=1.0, status="moving", confidence=.5):
    return {"id": oid, "category": category, "x": x, "y": 0.0, "distance": abs(x),
            "status": status, "raw_status": status, "confidence": confidence,
            "velocity_xy": [1.0, 0.0], "coordinate_frame": "lidar",
            "geometry": {"center_xyz": [x, 0.0, 0.0]}, "source": "bevfusion", "sample_token": "s"}


def plan(kind="status", target="car", reference=None, relations=None):
    return {"question_type": kind, "target_entity": {"category": target},
            "reference_entity": {"category": reference} if reference else None,
            "relations": relations or []}


def test_unanimous_multi_candidate_answer_is_aggregated():
    answer, trace = ConstraintAwareRASEExecutor().execute_plan(
        plan(), [row("a", status="moving"), row("b", x=2, status="moving")])
    assert answer == "moving"
    assert trace["strict_final_status"] == "AMBIGUOUS"
    assert trace["resolution"]["strategy"] == "answer_aggregation"


def test_conflicting_answers_use_post_constraint_ranking_not_nearest():
    rows = [row("near", x=1, status="moving", confidence=.2),
            row("far", x=8, status="stationary", confidence=.9)]
    answer, trace = ConstraintAwareRASEExecutor().execute_plan(plan(), rows)
    assert answer == "stationary"
    assert trace["resolution"]["chosen_ids"] == ["far"]
    assert trace["resolution"]["strategy"] == "constraint_consistent_ranking"


def test_comparison_uses_only_relation_consistent_pairs():
    rows = [row("front_car", "car", 6, "moving", .8), row("back_car", "car", -6, "stationary", .9),
            row("bus", "bus", 0, "moving", .8)]
    p = plan("comparison", "car", "bus", ["front", "same-status-as"])
    answer, trace = ConstraintAwareRASEExecutor().execute_plan(p, rows)
    assert answer == "yes"
    assert set(trace["resolution"]["chosen_ids"]) == {"front_car", "bus"}


def test_ir_alignment_consumes_every_explicit_constraint():
    p = plan("comparison", "car", "bus", ["front", "same-status-as"])
    ir = compile_aligned_ir(p)
    assert [x["kind"] for x in ir["constraints"]] == ["category", "category", "relation_any"]
    _, trace = ConstraintAwareRASEExecutor().execute_plan(p, [row("car", "car", 6), row("bus", "bus", 0)])
    assert trace["ir_alignment"]["all_constraints_consumed"]


def run_all():
    tests=[v for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests: test()
    print(f"PASS: {len(tests)} RASE Phase 3 deterministic CPU tests")


if __name__ == "__main__": run_all()
