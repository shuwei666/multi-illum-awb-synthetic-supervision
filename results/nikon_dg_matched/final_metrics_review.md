# DG revision 1 final numerical review

Updated: 2026-09-25; agent: codex independent reviewer.

Artifact verdict: **PASS** for numerical aggregation and the stated artifact bindings. Contract verdict: **ADEQUATE** for this bounded seed0 comparison. 独立验证通过；共享磁盘且同模型，隔离有限。Performance acceptance: **FAIL**; these are separate judgments.

## Evidence and execution

Executed `python3 dg_matched/reviews/independent_final_metrics.py <ROOT>` in a separate CPU process; exit 0. Checker imports neither production evaluator nor production aggregator. It uses float64 and `math.fsum` / explicit counts. Machine-readable detailed result: `final_metrics_independent.json`.

- Recomputed all 72 means: four models × 18 cells (val/test, all/single/multi, patch-pooled/patch-image-balanced/pixel-image-balanced).
- All 24 summary/CSV/NPZ hashes match both completed evaluation manifest and strict comparison. All 12 baseline/O artifacts are byte-identical to historical round3 files.
- Exact image ordering, unique IDs, 394 validation and 204 test images, patch validity masks, image light counts, valid-pixel counts, support hashes, summary checkpoint bindings, and disjoint val/test scene IDs checked.
- NPZ patch means independently recomputed; largest per-image difference from serialized CSV float32 means is 0.000001420267 degrees, within the frozen 0.00002 tolerance. Final 72 aggregate comparisons agree within 1e-12 degrees.
- Pixel-image-balanced means and overall pixel-pooled summaries independently reaggregated from CSV. This does **not** independently recreate pixel angular errors from predictions and raw pixel GT.
- Development JSON hashes checked separately. Recomputed `.5 * real_single/reference + .125 * sum(four synthetic/reference)` gives O 0.8573986570311704, DG-C 0.9150515595023583, DG-CS 0.90027755987137. Strict minimum selects O. Selection hash and recorded `selected_at < evaluation.created <= evaluation.completed` checked; frozen selection did not use these final test scores. Historical test exposure is not erased by this chronology check.

## Observed results

Angles in degrees; lower is better. Patch is pooled across valid patches; pixel is equal-image averaged.

| Model | Test all patch | Test all pixel | Test multi patch | Test multi pixel | Strict cells better than baseline |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.969242 | 2.167129 | 2.533778 | 2.910994 | reference |
| O | 2.423858 | 2.537886 | 3.242606 | 3.459937 | 0/18 |
| DG-C | 2.495608 | 2.624675 | 3.220908 | 3.466917 | 0/18 |
| DG-CS | 2.460909 | 2.587955 | 3.162417 | 3.404579 | 0/18 |

Both new arms have worse overall test error than O and baseline. DG-CS improves test multi patch/pixel relative to O and improves overall test errors relative to DG-C, but does not win the frozen development criterion and does not meet baseline acceptance. One seed cannot establish robust benefit or significance; do not select DG-CS using the test subgroup advantage.

## Scope and maturity

Correctness: numerical reduction and comparisons PASS. Evidence: hashed closed inventory and reused-byte identity PASS. Reproducibility: saved-error CPU reaggregation PASS, not an independent model retraining. Task fitness: experiment answers this bounded comparison, but requested performance improvement is not achieved. S1 traceable experimental result; no S2 performance-success claim.

This review does not newly certify training optimizer steps, renderer physics, TIFF black-level/linearity provenance, or raw per-pixel error computation. User-authorized R1 retains that preprocessing assumption. Those boundaries must remain disclosed.

Strict comparison SHA256: `24e6c8219fc8b16d6eccc6d1b511b49bf3a593bac5f7bb26fe819bd042b963f1`.
