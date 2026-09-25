# Evidence scope

Updated: 2026-09-25 · Agent: codex

This update publishes documentation and compact numeric summaries, not an independently runnable release. Private dataset files, checkpoints, raw visualizations, credentials, and deployment details are excluded.

| Public content | Evidence used for this update |
|---|---|
| Nikon train-only single-source rule and four synthesis arms | Frozen `EXPERIMENT.md`, source manifest and trainer review |
| Successful 69,600 updates per arm | G/R/M/P `completion.json` and training histories |
| Test patch and pixel metrics | The five `test_*_summary.json` files in frozen `final_001` evaluation |
| Identical evaluation image identities | Evaluation manifest and independent comparison of per-image outputs |
| Model, evaluator and checkpoint identity | SHA256 fields retained in the public summary |
| Second-round configurations | Approved `NEXT_EXPERIMENT_PLAN.md`; execution authorization follows the proposal |
| Second-round completed C/A/AC/L/S training | Per-arm completion records: 69,600 updates and 417 cycles; M and baseline checkpoints reused |
| Second-round development-only final selection | Frozen development JSONs, selection/checkpoint hashes and selection timestamp; AC selected before final mixed-GT scoring |
| Second-round 18-cell performance failure | Re-aggregation of saved patch/per-image evidence: every candidate passes 0/18 cells; public values and hashes in `results/nikon_round2/summary.json` |
| Second-round interrupted execution and recovery | Historical BrokenPipe failure preserved; recovery completed remaining development/training, selection and final evaluation without changing frozen training code |
| Third-round completed training and selection | O/B/ER/EP each completed 69,600 updates/417 cycles; AC reused; frozen development selected O before real scoring and the combination rule reused O |
| Third-round final performance | All five candidates pass 0/18 strict cells; completed final audit: performance FAIL, contract ADEQUATE, operational correctness PASS; see `results/nikon_round3/summary.json` |
| Conditional AN/PS preparation | Contract predeclared before Round3 real results; CPU helper and toy renderer independently checked; full trainer, actual Nikon calibration and training remain pending at this snapshot |
| Pixel-wise map and single-light relit explanation | First-round synthesis and patch-target construction |

The first-round local independent review checked checkpoint hashes, increasing successful-update counts, finite histories, common source identities, identical evaluation image orders, disjoint val/test image identities, and re-aggregation of saved patch/per-image results. Its verdict distinguished a traceable completed experiment from the failed performance target. Initial weights were not separately archived in round one: matching initialization is inferred from seed and construction order, not verified from initial-weight files. The reviewer shared the filesystem with the implementer, so isolation was limited.

The separate round-two review independently re-aggregated all seven models and 18 cells, recalculated the development scores, checked source/checkpoint/code hashes and training histories, and checked selection-before-scoring provenance. Two scoring fixtures passed. Twelve sampled pixel checks covering baseline/AC, val/test and one/two/three illuminants matched saved means to within 2.22e-16 degrees. It found no error changing the 0/18 conclusion: artifact verdict FAIL relative to the performance contract, contract ADEQUATE, supporting S1 traceable failed experimentation. It did not retrain, rerun all network predictions or fully recheck every original patch GT; source-code inspection is not an operating-system audit of every historical file access. Shared filesystem and the same model family limit isolation. This is evidence of a valid negative result, not S2 performance success.

The public JSON is a transcription of frozen numeric evidence. Checkpoint hashes identify internal artifacts but do not make those artifacts publicly available. Re-running independently requires the dataset, implementation, environment and checkpoint access, which are not supplied by this documentation-only update.

The fresh round-three final audit independently aggregated all six models and all 18 cells (108 means) using float64 and an independent summation path; maximum disagreement was 8.89e-16 degrees. It checked 40 local/remote artifact matches and rehashed 70 previously audited completed-cohort artifacts, confirming unchanged baseline/AC predictions and errors between rounds. Twenty-four sampled native-grid pixel checks covered six models and the first single/multi image in each split, with maximum disagreement 3.25e-14 degrees. Selection, approval and scoring hashes/timestamps were checked; timestamps are provenance evidence, not tamper-proof attestation. This review did not retrain, rerun network inference or fully recompute every original patch GT. Source claims use frozen code/manifests and the prior training audit, not historical system-call tracing. Shared filesystem/model family limits isolation. These findings support S1 traceable failed experimentation, not S2 performance success.

Round-two values use float64 re-aggregation of saved float32 patch errors and equal-image pixel means. With 256 valid patches per image in these evaluations, patch-pooled and patch-image-balanced means coincide; both remain in the 18-cell contract, which is not 18 independent statistical tests. Original float32 summary reductions can differ in the final digits. Acceptance uses unrounded values against the same recomputed historical seed0 baseline. The frozen AC candidate fails all 18 cells; completed execution is not a passed performance goal, and startup PASS/ADEQUATE does not establish performance validity or success.

The current contract requires the development-selected seed0 candidate to improve strictly in every val/test × all/single/multi × patch-pooled/patch-image-balanced/pixel-image-balanced cell. Only then is its configuration frozen for seeds1/2; the arithmetic three-seed mean must also pass every cell, with each seed disclosed. Neither round two nor round three met the seed0 gate; no round-three extra-seed stage is authorized and the multi-seed goal remains incomplete. The historical baseline used real mixed-light training and is a fixed single-seed reference, not a matched multi-seed causal comparison.

All train global Light1/2/3 endpoint colors, including those of multi-light scenes, are permitted. Real mixed images, dense GT, mixture maps and derived information are prohibited from candidate construction, training, development selection and tuning. Synthetic spatial supervision is permitted. Development uses frozen official val single-light RAW/Light1 and synthetic data; true mixed GT is accessed only for scoring after final selection is frozen. Earlier test outcomes have already been viewed, so these iterations are exploratory, not fresh blind tests. No result-dependent winner replacement is allowed.

Third-round protocol approval, code verification, calibration, training completion and performance acceptance are distinct stages. Its completed performance result is negative: O's improved development score does not establish test success, and no posthoc winner was substituted. All models share the fixed support; patch-pooled and image-balanced means coincide because each image has 256 valid patches. The 18 cells are not independent statistical tests. Unrounded values determine acceptance.

AN/PS was predeclared at 2026-09-25 14:05:54 +08:00 before Round3 real scoring. The binary failure outcome is disclosed activation feedback, not a new blind test. The parent is development-selected O, not a test-selected or pretrained model. Only CPU helper/toy-renderer behavior has independent evidence at this snapshot; full integration, actual Nikon calibration and training are still pending. No new-method or performance claim is made, and the protocol was not parameter-designed from Round3 real scores.

Historical Sony evidence is preserved separately in [sony_historical_evidence.md](docs/sony_historical_evidence.md). Its claims and limitations do not transfer automatically to the Nikon protocol.
