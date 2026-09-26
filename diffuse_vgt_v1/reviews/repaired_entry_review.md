# Repaired runner and render-preflight entry review

updated: 2026-09-26; agent: codex (fresh read-only reviewer)

**Artifact verdict: PASS. Contract verdict: ADEQUATE for code freeze and a no-optimizer, full-668 full/tail render diagnostic, conditional on completed cache passing existing provenance checks and the GPU prior job ending.** Independent validation passed for that limited entry scope (S1). This is not calibration or formal-training approval, not a measured GPU-memory pass, and not performance evidence.

Scope: original USER_TASK.txt, REVISION_1.md, prior runner_code_review.md, executor repair evidence, current runner/synthesis/preflight/tests and actual inherited dg_synthesis implementation. Only this review file was written; implementation, thresholds and data were not changed. No image inference, optimizer update, development or test GT access was performed. Fresh context and separate test processes, but shared filesystem/model: isolation is limited.

## Findings resolved

1. `check_cache` now rejects missing or unequal prior-manifest and producer-code SHA256 values. It equates ordered scene IDs and per-file source hashes to the inherited DG frozen inventory, then verifies every cache asset/input hash. Regression tests independently passed for stale/missing provenance and changed parent source identity.
2. Calibration now supplies explicit `[0]` and `[127]` optimizer-batch indices. Consumption is mapped back through parent order before adaptation and endpoint-ledger accumulation. History, diagnostic and completion represent eight consumed views; recovery cursor is selected batch plus one. Tests reject the former full-cycle consumption and independently verify one-batch endpoint mass. The full rendered cycle remains a memory/integration probe, not claimed optimizer consumption.
3. Loader retains shading-valid and normal-valid masks and verifies their intersection against producer validity. Resampling transports the two masks separately and uses local shading validity plus local/rotated normal validity. The shading-only-invalid regression verifies that rotating shading rejection is no longer introduced.

## Preflight and memory boundaries

- Uses actual `runner.runtime()` modules, `load_inputs`, inherited `dg.synth.render_cycle(..., mode='O', capture=...)` and the candidate adapter. Parent source confirms the captured keys, original image/GT processing, seven-result return and crop/geometry draw sequence. This is not a mock parent API in the proposed full-cohort run.
- Both cycle 0 / 167-step and cycle 416 / 128-step prefixes render all 1,336 views. It checks source/endpoint cohort sizes, draws/order/identity, identical A/B/C acceptance/consumption ledgers, A/B GT equality, matched crop/lamps and finite nonzero normalized intervention probes. C is correctly permitted different GT. Probe RMS covers first 16 mixed ordered views; full mixed streams are also hashed. A positive probe is only an engineering signal, not mechanism or accuracy evidence.
- Arms run sequentially, release result/capture before the next arm, and keep only CPU summaries/probes. Hashing device copies is chunked at 32; counterfactual rendering is chunked at 16. It does not retain three whole-cohort counterfactual bundles concurrently. Source/prior arrays, parent capture, resampling buffers and whole output tensors still exist; CPU tests cannot establish 32GB sufficiency. The authorized diagnostic records peak allocated/reserved memory and fails visibly if resources are insufficient. Run only after the active cache job has ended; do not overlap GPU use.
- `preflight.py` binds its own SHA256 in its output, while runner's freeze hashes bind the training/renderer/cache code. Its hash is listed below for this entry decision. A later altered preflight is not covered by this review.
- No optimizer or model creation/update occurs in preflight. It does not create training approvals. Existing real-QC/input-sensitivity/calibration gates remain in force. Complete live cache loading and actual GPU execution remain unperformed by this reviewer.

## Independent executed tests

Remote working directory: diffuse_vgt_v1 on the authorized instance. Environment `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:pydeps:../pydeps:../code` and `/root/miniconda3/bin/python`.

1. `python -m unittest discover -s tests -v`: 24 available tests passed, 1.571 seconds (7 synthesis, 7 renderer, 10 runner). The accompanying shell hash command initially exited 1 solely because preflight had not yet been synchronized; the tests themselves exited successfully.
2. After synchronization, `python -m unittest discover -s tests -p test_preflight.py -v`: 4 tests passed, 0.001 seconds. Independently rerun locally with `python3` as well: 4 passed.
3. Local `shasum -a 256` and remote `sha256sum` match for all files below. Total independent test coverage: 28 tests. These tests do not replace actual full-cohort memory, optimizer or downstream evaluation.

| File | SHA256 |
|---|---|
| train_arms.py | 6e4f3658c8ff2c2b1df06d36b093c0f52c23c5a78dedaccb280f386e501cf40a |
| diffuse_synthesis.py | 74752e9308044ca34a9b45b9f2b52068d65dab44fc6c3d3ef6a0d5b0aa62e340 |
| preflight.py | 936ce526233083eb6198e2c01c6543dd17759c68fd7bafed3f4737e1de1f89a3 |
| tests/test_train_arms.py | d35fa2ca9b46f308a7d47b684fdcc7f61b5cddf45d577befc67d0bc0c1739c6e |
| tests/test_diffuse_synthesis.py | ec547952406dc18cb5a7adb8ab777504db52322db586360fcbbef88eb7a50792 |
| tests/test_renderer.py | 15e5898ba0c38261ab3a8fc367bb3b5289f8ee4cf213394dc030e8a4e1eb27f0 |
| tests/test_preflight.py | 7ea78dcf59a7664a75c5e9ced49488ed11dd6bf0917e528866258098074282b9 |

Correctness: three prior findings repaired, no further blocking issue found within diagnostic scope. Evidence: code/CPU tests and exact hashes, not GPU measurements. Reproducibility: source/cache/freeze checks fail closed. Task fit: bounded no-update diagnostic preserves original source restrictions and precedes separate optimizer calibration. No extra variant, threshold or training authorization was introduced.
