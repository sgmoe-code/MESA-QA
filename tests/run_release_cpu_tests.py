#!/usr/bin/env python3
"""Run deterministic CPU unit checks without pytest."""
from __future__ import annotations
import sys
from pathlib import Path

RELEASE=Path(__file__).resolve().parents[1]
for rel in ('core','domains/nuscenes_qa','selection','tests'):
    sys.path.insert(0,str(RELEASE/rel))
import test_rase_phase3
import test_simple_reasoner_selection
from rase_phase2_planner import PlannerRASEController

def test_p2_question_grounded_repair():
    plan={'question_type':'status','answer_type':'status','target_entity':{'category':'truck'},'reference_entity':{'category':'bus'},'relations':['back']}
    fixed,audit=PlannerRASEController().enhance('Is the truck the same status as the bus?',plan,{'final_status':'AMBIGUOUS'})
    assert fixed['question_type']=='comparison'
    assert fixed['relations']==['back','same-status-as']
    assert audit['feedback_request']['resolved'] and not audit['label_access']

def main():
    # Core imports are an explicit release gate.
    from rase_answer_resolution import ConstraintAwareRASEExecutor
    from rase_phase2_planner import PlannerRASEController
    from bevfusion_scene_adapter import BEVFusionSceneAdapter
    assert ConstraintAwareRASEExecutor and PlannerRASEController and BEVFusionSceneAdapter
    tests=[test_p2_question_grounded_repair]
    tests += [v for k,v in vars(test_rase_phase3).items() if k.startswith('test_') and callable(v)]
    tests += [v for k,v in vars(test_simple_reasoner_selection).items() if k.startswith('test_') and callable(v)]
    for test in tests:test()
    print(f'PASS: {len(tests)} release CPU tests and core runtime imports')
if __name__=='__main__':main()
