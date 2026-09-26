# Completed diffuse cohort and final evaluation entry review

updated: 2026-09-26; agent: codex (fresh independent reviewer)

Artifact verdict: PASS. Contract verdict: ADEQUATE. 独立验证通过 at S1 for completed frozen training, development selection bookkeeping, and entry to exactly one unified baseline/O/A/B/C real scoring. This is not a performance pass and authorizes no new training, seeds, variants, or test-driven reselection.

Executed `CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:pydeps:../pydeps:../code:../analysis python reviews/audit_completed_cohort.py`, exit 0. The independent CPU process rehashed the frozen code/assets, all 668 caches and completion-bound actual artifacts; loaded actual final and recovery states; checked the existing entry gates; and independently recomputed the five-component development formula. Evidence is `reviews/completed_cohort_cpu_audit.json`; the reproducible audit is `reviews/audit_completed_cohort.py`. No GPU initialized, real mixed images or GT were opened, training ran, or real evaluation entry executed during this review.

All three runs contain exactly 69,600 successful updates over 417 cycles: 416 cycles of 167 steps, then 128 steps. Every one of the 18 actual Adam parameter states has step 69,600, finite moments and nonnegative second moments. Actual recovery has cycle 416 and next batch 128. Every final tensor equals its recovery tensor and is finite; each final differs from the frozen initialization. Per-update losses, positive finite gradient norms and parameter changes, 512 candidates, valid-patch sums and consumed views agree with history/diagnostics. Completion hashes and recovery/NPZ ledgers agree. All matched sampling, branch, technical validity, consumed mass fields and every endpoint ledger array are identical across A/B/C.

| Arm | Final L2 distance from initial | Recorded training seconds | Peak allocated bytes |
| --- | ---: | ---: | ---: |
| A | 39.2725233081 | 1405.723369 | 19186913280 |
| B | 37.2819446527 | 1415.929714 | 19186913280 |
| C | 38.4719657071 | 1478.724807 | 19186913280 |

These distances establish actual changed parameters, not useful predictive performance. Timing is recorded runtime, not an independent stopwatch measurement or a promise about future runs.

The frozen score is 0.5 times the real-single mean/reference ratio plus 0.125 times each of four synthetic mean/reference ratios; lower is preferred. The checked five means and reference hashes are preserved in the evidence. Candidate order and exact-tie rule are O, A, B, C. Scores are O=0.8573986570311704, A=0.9253805212050032, B=0.9655792014703064, C=0.9614752020351813. The correct preselected winner is O. Selection time 1790393392.9745696 is after freeze and before this review; selection binds all final checkpoint hashes. This CPU review recomputed scoring arithmetic, not the GPU predictions used to obtain the recorded development means.

The evaluation implementation remains the version accepted in `reviews/evaluation_repair_review.md`: its SHA256 is c57bb89a506afde4cbe9cf107bd69b88d96430846636587585ad775a59603fd5. Its actual `approved_context` requires the complete frozen cohort and externally authored approval before real data access; the issued JSON binds current code, evaluator dependencies, selection, baseline/O/A/B/C checkpoints, all completion evidence, and official reference manifest. Live split/meta checks occur only after this gate and are required before and after real scoring. The fixed results directory remains absent at review time. The approval permits only the existing once-only scoring entry. The 18-cell comparisons, parent support, frozen winner and all negative results must be retained. Since O is selected, the current implementation cannot declare a new A/B/C seed0 target achieved by posthoc choosing another arm.

Correctness: actual completed optimizer/state/accounting and selection arithmetic passed. Evidence: actual CPU-loaded states, independently recomputed hashes and detailed audit. Reproducibility: frozen inputs/code, audit script and hash-bound approval. Task fit: bounded A/B/C and unchanged five-item development selection, followed only by the authorized one-shot scoring. Real downstream accuracy and final 18-cell outcome are not yet verified. No multi-seed or causal mechanism conclusion is established. The accepted TIFF lineage assumption and external prior/domain/physical limitations remain; real relighting physical validation is NOT_PERFORMED.

Isolation is limited (隔离有限): a fresh reviewer context and separate CPU process share the model and filesystem; no operating-system read-only isolation is claimed. Approval is evidence-bound and becomes stale if any bound artifact changes.
