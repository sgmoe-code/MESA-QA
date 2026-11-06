from run_simple_reasoner_selection import answer_type, question_flags, make_model, NUMERIC, CATEGORICAL


def test_answer_type():
    assert answer_type("") == "missing"
    assert answer_type("yes") == "yes_no"
    assert answer_type("12") == "number"
    assert answer_type("traffic cone") == "category"


def test_question_flags_are_deterministic():
    q = "How many moving cars are in front of the bus?"
    a = question_flags(q)
    assert a == question_flags(q)
    assert a["q_count"] == a["q_motion"] == a["q_spatial"] == a["multi_entity"] == 1


def test_pipeline_has_required_fixed_model():
    model = make_model(20261103)
    clf = model.named_steps["classifier"]
    assert clf.C == 1.0 and clf.penalty == "l2" and clf.random_state == 20261103
    assert len(NUMERIC) == 18 and len(CATEGORICAL) == 4
