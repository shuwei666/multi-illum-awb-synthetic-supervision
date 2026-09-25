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
| Pixel-wise map and single-light relit explanation | First-round synthesis and patch-target construction |

The local independent review checked checkpoint hashes, increasing successful-update counts, finite histories, common source identities, identical evaluation image orders, disjoint val/test image identities, and re-aggregation of saved patch/per-image results. Its verdict distinguished a traceable completed experiment from the failed performance target. Initial weights were not separately archived in round one: matching initialization is inferred from seed and construction order, not verified from initial-weight files. The reviewer shared the filesystem with the implementer, so isolation was limited.

The public JSON is a transcription of frozen numeric evidence. Checkpoint hashes identify internal artifacts but do not make those artifacts publicly available. Re-running independently requires the dataset, implementation, environment and checkpoint access, which are not supplied by this documentation-only update.

Historical Sony evidence is preserved separately in [sony_historical_evidence.md](docs/sony_historical_evidence.md). Its claims and limitations do not transfer automatically to the Nikon protocol.
