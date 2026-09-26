# Diffuse cohort: independent final metrics audit preparation

Updated: 2026-09-26; agent: codex independent reviewer.

Status: **PREPARED ONLY**. No final approval, accuracy result, or performance verdict. Read only evaluator source and frozen manifest during preparation; no results/real GT opened. Wait for explicit controller notification that final scoring completed. Do not poll training.

## Frozen scope

Five models: baseline, O, A, B, C. Independently recompute 90 means (5 × 18): val/test × all/single/multi × patch-pooled/patch-image-balanced/pixel-image-balanced. User-facing test report must retain single and multi rows, not only all-image aggregate. Lower angular error is better.

## Bounded execution after notification

1. Read `development_selection.json`, frozen manifest, evaluation approval and bound independent review, `results/evaluation_start.json`, `evaluation_manifest.json`, `summary.json`. Verify closed cohort/tie order O,A,B,C, checkpoint identities and hashes; recompute development formula independently from saved component means and frozen reference means: `.5*real_single/reference + .125*sum(global,sigmoid,blob,lowfreq ratios)`. Verify winner is unchanged strict minimum/tie order and selection timestamp precedes evaluation start. This verifies recorded chronology, not absence of historical test exposure.
2. Hash all 30 model/split summary/CSV/NPZ files and require exact completed-manifest inventory. Verify all bindings, implementation hashes and paired CSV hashes. Inspect frozen source metadata hash bindings without reading image GT. Completion/optimizer audit is a separate prerequisite, not inferred from final metrics.
3. Read saved NPZ angular errors and valid patch masks with `allow_pickle=False`. For each image, independently calculate `math.fsum(valid errors)/valid count`. Recompute pooled subgroup means from summed errors/counts and image-balanced subgroup means from those image means. Do not import/call production `aggregate` as evidence. Compare final values within 1e-12 degrees and serialized float32 CSV/summary patch means within the unchanged 2e-5-degree parent tolerance. Invalid support or discrepancy stops acceptance; do not alter thresholds after results.
4. Reaggregate saved CSV pixel means using float64/math.fsum for equal-image subgroup means, and separately check whole-split valid-pixel-weighted summary. Explicit boundary: not independently reconstructing pixel angular errors from raw predictions and pixel GT. Do not open new raw GT or run GPU.
5. Exact IDs/order, unique IDs, official reference identity (394 val, 204 test), scene-disjoint splits, NPZ/CSV agreement, valid patch positions, image light counts, valid-pixel counts, and subgroup support must agree across all five models. Require nonempty single/multi groups and finite nonnegative errors/counts. Check light counts in {1,2,3}.
6. Baseline and O are rescored in this evaluator, not byte-copied. Verify frozen checkpoint hashes (baseline `94d1cbf18b550156d46b6a201f87cc5d05619ab7e73c19983d7785bd382d39bb`, O `7670d875b96331df7e3c937748aee40d006e8147710ae77ad4962eabb7291351`) and matching official support. Compare with historical round3 numerical metrics; report exact differences. Do not demand byte identity for timestamp-bearing fresh summaries or silently excuse unexplained numeric drift. Any drift outside established float serialization tolerances requires explanation before an agreement claim.
7. Independently calculate all 18 candidate-minus-baseline deltas and strict `<0` pass counts; verify saved gate and outcome. Performance acceptance is separate from numerical artifact correctness, and O cannot become a new-arm success.
8. Independently recompute B-A and B-C for all 18 means, each per-image CSV delta and each per-scene equal-composition mean. Negative delta favors B. Group compositions by scene; do not treat multiple compositions as independent scene replicates. Scene means are descriptive and differ in weighting from all-image means. No significance or causal superiority claim from one seed. Do not use test subgroup improvements to change frozen winner.

## Deliverable and boundaries

Write only dedicated reviewer report/JSON/checker under `reviews/`; no production source, contract, threshold or data edits. Record actual commands, input/output SHA256, numerical tolerances, omissions, separate artifact/contract verdicts and performance outcome only after execution. Shared disk/same model imply limited isolation. Final report must separate test all/single/multi, patch/pixel definitions, baseline/O comparisons and B-A/B-C descriptive outcomes. No additional training or parameter selection authorized by this review.
