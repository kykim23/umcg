# PyTorch 2.11 AOT literal-output compatibility patch

## Trigger

- Environment: `umcg`, PyTorch `2.11.0+cu128`, two H100 GPUs, DDP, `torch.compile`
- Workload: 350M Russian Roulette training, first 2048-token forward
- Failure: `FakifiedOutWrapper.pre_compile()` assumed every split-graph output was an `fx.Node` and raised `AttributeError: 'int' object has no attribute 'meta'` for a literal integer output.
- Failed run artifacts: `rr_3000/` and `logs/rr_3000/`; no optimizer update or metrics row was produced.

## Patch

Patched environment file:

`/home/ubuntu/keunyoung/miniconda3/envs/umcg/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py`

```diff
- n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])
+ n.meta["val"] if isinstance(n, fx.Node) else n
+ for n in (list(fw_module.graph.nodes)[-1].args[0])
```

The subsequent `_compute_output_meta_with_inductor_strides()` implementation already skips non-Tensor outputs. The compatibility change therefore preserves literal outputs instead of trying to read node metadata from them. It does not disable `torch.compile`, DDP graph partitioning, or communication overlap.

## Integrity

- Original file SHA-256: `6101332cb8d6549c3e662f518a7bb4dc932c3648c622a7c9b774c5ee59bdb82e`
- Patched file SHA-256: `e268d4e18bad02ba2312cbb3ebf179d2622a3df6782a2e35163f2095a64610a9`

## Outcome and rollback

The patch passed the original literal-output failure, but compilation later failed in another DDP-split subgraph with:

`AssertionError: For Min(4096, s7) - 3968, expected [s7] to have been codegen-ed.`

No optimizer update or metrics row was produced. The preflight and logs are preserved as `rr_350m_patch_preflight_failed_inductor_symbol/` and `logs/rr_350m_patch_preflight_failed_inductor_symbol/`.

The environment file was restored byte-for-byte to its original SHA-256, `6101332cb8d6549c3e662f518a7bb4dc932c3648c622a7c9b774c5ee59bdb82e`. This document records an attempted diagnostic patch, not an active environment modification. The next gate uses PyTorch's documented `torch._dynamo.config.optimize_ddp = False` workaround: it retains `torch.compile` but disables only Dynamo's DDP graph splitting.
