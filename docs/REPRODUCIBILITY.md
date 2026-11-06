# Reproducibility

## CPU validation

```bash
python tests/run_release_cpu_tests.py
python experiments/reproduce_nuscenes_qa.py --artifact-root /path/to/authorized-artifacts
```

The CPU suite checks P2 semantics, imports and selector behavior. The cached-artifact runner independently replays P2 and refits the five-fold selector, checks cohort/fold identity, model ID/revision metadata and frozen artifact hashes. The local release validation passed 8 tests, P2 exact replay at 311/1000, Direct at 396/1000, and selector at 428/1000 with zero route/correctness mismatches and zero probability delta. It verified six frozen P2 input hashes and 21 Structured freeze artifacts. No GPU inference was run.

## External artifact root

`--artifact-root` must contain the documented relative paths in `configs/external_artifacts.json`, including frozen Direct/S0 inputs, historical P2 evaluation, normalized evidence, selector fold manifest and frozen five-fold predictions/metrics. These files are not distributed here because they contain restricted QA/evidence or large generated artifacts. Provide an authorized artifact root; do not substitute regenerated predictions.

## Inference identity

Model: Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Frozen runtime: Python 3.12, PyTorch 2.8.0+cu128, Transformers 5.9.0, Accelerate 1.13.0, CUDA 12.8, BF16, SDPA. Model files are external; set `MESA_QA_MODEL_PATH`. Dataset root is supplied using `NUSCENES_ROOT`.

The reported 1,000-QA scores are retrospective development evaluations, not a blinded or independent external test. The five OOF fits are not a production selector model.
