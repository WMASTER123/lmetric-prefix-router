# lmetric-prefix-router

Experimental prefix-aware placement for homogeneous vLLM-HUST serving replicas.
The unit of placement is a model replica, which can span multiple Ascend devices.
The current package implements the pure combined decision and a discovery-only
descriptor. The Manager rejects activation until the target runtime is qualified.

## Installation and tests

From the repository root, run the following in a clean Python environment:

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests
vllm-hust-ext extension inspect io.github.wmaster123.lmetric-prefix-router
```

## Target and provenance

The target is Qwen3.5-35B-A3B BF16 on the frozen HUST r10 release
(`2026.11.21.10`). Its model admission explicitly requires TP2/DP1.
The prior 0.23 shared internal-DP tests do not qualify this release.
A deployment consisting of two independent TP2 endpoints requires a separately
identified external dispatch policy for equal hits and misses.

- Scaffold: [HUST template at 4163473](https://github.com/vLLM-HUST/extension-template/tree/416347398ba197a4db5e4360db1aaecc64355b79).
- Decision semantics: WMASTER123's lmetric-combined routing experiments.
- Target runtime: [r10 frozen artifacts](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/engine-wheels-r10-v1).
- External router interfaces and reliability reference: [HUST Prefix Router](https://github.com/vLLM-HUST/vllm-hust-prefix-router).
