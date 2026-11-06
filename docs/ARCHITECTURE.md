# MESA-QA architecture

## Core and domain boundary

The domain adapter converts frozen perception outputs into a normalized Evidence State and the scene-row contract used by the executor. Core schema, Planner feedback, query resolution, deterministic execution and answer normalization do not depend on the NuScenes data loader. A future adapter can translate another domain's evidence to these interfaces; DisasterVQA is a planned example, with no completed experiment or result in this release.

## Frozen NuScenes-QA path

1. BEVFusion detections are normalized by the NuScenes-QA adapter. Evidence provenance and uncertainty remain explicit.
2. The Direct baseline uses the frozen Qwen3.5-9B text-only runtime over serialized current evidence.
3. The Structured path uses domain RAG and the frozen Planner prompt to produce a normalized plan. P2 applies question-grounded constraint repair and executes candidates with the Atomic Executor and deterministic scene tools.
4. The Simple Post-Selection Logistic Regression chooses among existing Direct/Structured answers using five physical-scene OOF folds. It does not regenerate an answer.

Core files are in `core/`, P2/Atomic runtime and Qwen integrations are kept at repository root to preserve their tested Python imports, the domain adapter is in `domains/nuscenes_qa/`, and selection code is in `selection/`. The frozen CPU reproduction uses existing predictions and deterministic CPU operations only; it does not invoke a model.
