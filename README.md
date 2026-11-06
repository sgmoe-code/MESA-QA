# MESA-QA

MESA-QA is a modular, evidence-grounded question-answering research framework. It separates domain evidence extraction from domain-independent planning, deterministic execution and answer selection. The current implemented adapter targets NuScenes-QA; additional adapters such as DisasterVQA are planned, not evaluated here.

## Research motivation

Driving-scene QA combines perception uncertainty, structured reasoning and answer generation. MESA-QA represents normalized evidence explicitly, lets a Planner propose structured operations, executes those operations against a deterministic domain interface, and compares Direct and Structured answers with a learned post-selector. This makes evidence support and execution traces auditable.

## Architecture

```text
Frozen perception -> normalized Evidence State -> NuScenes-QA/BEVFusion adapter
Question -> domain RAG + Qwen3.5-9B Planner -> normalized plan
          -> P2 constraint repair + Atomic Executor -> Structured answer
Direct Qwen3.5-9B answer --------------------------> Simple Post-Selection LR
```

The release contains the domain-independent P2/RASE core, schema/query resolution, Atomic Executor and evidence-state interfaces, Qwen3.5 Direct/Structured runners, BEVFusion/NuScenes adapter, retrieval and Planner integrations, and the frozen Logistic Regression post-selector. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Verified development results

| Method | Correct | Accuracy |
|---|---:|---:|
| Direct | 396/1000 | 39.60% |
| Structured P2 | 311/1000 | 31.10% |
| P2 Post-Selection | 428/1000 | 42.80% |

These are retrospective development results on the same 1,000-QA cohort, not independent external validation. The five scene-fold OOF fits are evaluation artifacts, not a separately trained production model. The active frozen backbone is **Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Historical results retain their original recorded model identities.

## Layout

- `core/`: validated P2/RASE logic, schema and deterministic execution.
- `domains/nuscenes_qa/`: BEVFusion evidence adapter.
- `selection/`: Simple Post-Selection implementation.
- `experiments/`: frozen CPU replay runner and its source snapshot.
- `tests/`: CPU regression tests.
- `configs/`, `results/`, `docs/`: protocol, dependency/provenance manifests, result summaries and reproduction guide.
- Top-level `run_train20_qwen35_*.py` and supporting modules preserve the frozen inference imports.

## Environment and external assets

CPU replay uses Python 3.12 with NumPy, SciPy and scikit-learn; see `requirements-replay.txt`. Model inference was frozen with PyTorch 2.8.0+cu128, Transformers 5.9.0, Accelerate 1.13.0, CUDA 12.8, BF16 and SDPA; see `requirements-inference.txt`. No model weights, dataset, evidence cache, RAG index or predictions are included. Configure paths with `.env.example` or process environment variables. See `configs/external_artifacts.json` for acquisition notes and known checksums.

## Reproduction

Run the self-contained deterministic CPU tests:

```bash
python tests/run_release_cpu_tests.py
```

The frozen cached-artifact reproduction additionally requires access-controlled cohort, P2 predictions/traces, evidence and fold artifacts outside this repository:

```bash
python experiments/reproduce_nuscenes_qa.py --artifact-root /path/to/authorized-artifacts
```

A GitHub-only checkout cannot reproduce the full benchmark without those external artifacts. The Direct and Structured inference runners also require Qwen3.5-9B weights and licensed NuScenes-QA assets; they do not download either automatically. Full instructions and caveats are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
