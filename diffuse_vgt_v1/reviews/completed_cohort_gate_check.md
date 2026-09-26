# Actual approval gate check

updated: 2026-09-26; agent: codex (independent reviewer)

After publishing the reviewer-authored gate and report to the execution host, imported the actual `evaluate_frozen.py` by importlib under a unique name, called `dependencies()` then `approved_context(r, dg)` only, and asserted both `not torch.cuda.is_initialized()` and absence of `results/`. No `execute`, `main`, live metadata, camera rows, data loading or prediction function was called. Process exited 0.

Observed output:

```text
APPROVED_CONTEXT_PASS_NO_GPU_NO_GT
approval 275166aefc4bad8df271cd19cd15d319a46e4defa5fdc474936a820bbcee5404
review JSON 479df1b1a01c3f3f52e80cebd28dc52830f5292a215535052aba8976f8be8292
review Markdown d272353db16f98e94ff65ca4678cbb4eff86555ab51d61cdf0b8ea7972209e55
CPU audit 726468f0825a38e1201738d82a45847f32bda3f20725fbbaab1425addeae59e6
```

All four remote digests equal the locally checked artifacts. This confirms the actual current approval schema accepts the bounded cohort and current artifacts before real scoring. It does not verify scoring results or initialize GPU execution.
