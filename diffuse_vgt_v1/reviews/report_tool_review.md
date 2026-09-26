# Report renderer independent review

updated: 2026-09-26; agent: codex

Artifact verdict: **FAIL**. Contract verdict: **ADEQUATE** for the bounded renderer review. This is not approval of final metrics, evaluation access, or performance. 隔离有限：fresh agent context and separate Python process, shared filesystem and model family; no OS-enforced read-only isolation.

## Scope and hashes

Only implementation, original task, revision, evaluator schema, and synthetic temporary fixtures were read. No real GT, development/test inputs, actual results, checkpoints, or training histories were accessed. No implementation was modified. The only persistent output is this review.

| File | SHA256 |
| --- | --- |
| build_report.py | b00732e13e5618dd0db70d9584a25579ea46db435fbe1e0ae9a74f1b7d15465b |
| tests/test_build_report.py | 10b72042e2b4fbc13b0c56771d9914bb8508b9e248e797dad5e2cf59a74a0b98 |
| USER_TASK.txt | 7488b5be071894e25144554ffee225188872e5656270b42eba501f699942424f |
| REVISION_1.md | ad6b4f6da46b2a8ed72a548320a09d5a4ff85a71c8b334097e564e2f6325a176 |
| evaluate_frozen.py | c57bb89a506afde4cbe9cf107bd69b88d96430846636587585ad775a59603fd5 |

## Blocking findings and exact repair scope

1. **Training consumption can be rendered from stale evidence.** `load()` verifies only `completion.json` from `completion_evidence_sha256`; `render()` subsequently consumes `history.jsonl` for timing and all four actual consumption fields without verifying its evaluator-bound hash. In a temporary fixture I added correct history hashes to all three summary bindings, then changed A's first `consumed_views` to 999999999 without updating the binding. `render()` returned successfully despite the confirmed history hash mismatch. This permits unrelated or edited histories to be presented as the scored cohort's actual consumption. Require the existing evaluator history binding before using it, preferably at the same validation entry used by `--check-only`; preserve the direct-count semantics and add a rejecting regression test. No evaluator, training, or metric protocol changes are needed.

2. **Private review text is copied verbatim into the deliverable.** Lines 170–175 embed any `reviews/final_metrics_review.md` contents in both Markdown and HTML. A temporary mock review containing `/Users/mockperson/private/project` and `mockuser@private.example` was reproduced in the HTML. Escaping protects markup but does not protect private paths or hosts. The generator itself does not upload anything; the issue is an unsanitized report that may later be shared. Define a public-safe review excerpt or explicitly redact/reject private content before report creation, while retaining the local original as evidence. Apply the same policy to free-text prior/preflight fields, and test it. Do not claim that HTML escaping addresses privacy.

3. **Paired mechanism evidence links can point to missing or changed artifacts.** The renderer always links `results/per_image_deltas.csv` and `results/per_scene_deltas.csv`, without existence/hash checks against `paired_artifacts_sha256`. The existing mock fixture has neither file and still renders. The evaluator explicitly provides those hashes. Verify both before linking and add missing/stale-file tests; this does not require recomputing metrics or reading GT.

## Correct behavior observed

- Baseline is described as the project's frozen modified One-Net baseline, with an explicit boundary against claiming the author's full reproduced protocol.
- All 18 cells are displayed; selection uses fixed development order O/A/B/C; outcome requires a preselected new arm to strictly beat baseline in every cell, and never claims the multi-seed goal complete.
- B-A and B-C arithmetic and signs are checked/displayed; B-O is not misattributed to a single mechanism. Single-seed and scene/composition limitations are explicit.
- Generated slots and four directly summed actual-consumption fields are separated; missing consumption raises an error. Online means use actual update-count weights. Slot counts are not called independent observations.
- REVISION_1's unweighted parent patch labels and accepted but unverified TIFF lineage are disclosed, alongside external priors and unperformed physical validation.
- HTML has eight correctly closed sections; an independent HTMLParser stack check found no mismatched or unclosed tags. Escaping is used for tabular/free-text content. No browser visual/layout acceptance was performed.

## Tests and observations

Ran `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_build_report.py -v`: 3 tests PASS, all artifacts temporary.

Ran an independent inline Python process importing only the synthetic fixture and renderer, using `TemporaryDirectory(prefix='report-independent-')`. Operations: bind fixture histories, modify one count after hashing, call render; add mock private review text, call render; check linked CSV existence; parse emitted HTML with a stack-based HTMLParser. Observed: stale history accepted (hash mismatch true); private review text copied true; missing paired evidence linked true; markup errors empty and remaining tags empty. The diagnostic substring check for `999` was false because summation changes the printed total; acceptance is evidenced by `render()` returning without error despite the independently confirmed mismatched hash, not by that substring check.

Correctness: outcome and contrast arithmetic checked, provenance checks incomplete. Evidence: synthetic counterexamples establish the failures; actual final metrics unreviewed. Reproducibility: existing tests and described fixture mutations reproduce the issues. Task fit: report content mostly matches requirements, but traceability/privacy defects block acceptance. S1 is not granted for this artifact version; final scientific performance maturity is outside scope.

Final status: **未通过（附失败证据）**. After author fixes these bounded issues, use a fresh reviewer and updated hashes. This review grants no permission to read final GT or run final scoring.
