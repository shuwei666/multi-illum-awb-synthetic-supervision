# Diffuse Virtual GT: completed seed0 batch

updated: 2026-09-26; agent: codex

Read [FINAL_REPORT.md](FINAL_REPORT.md) or download [FINAL_REPORT.html](FINAL_REPORT.html). The experiment is complete and the final target was **not met**. All three new runs used 69,600 successful updates; fixed development selected O. All O/A/B/C candidates failed the 18-cell baseline criterion. No new runs are started by this archive.

## Publication scope

This branch is a static, separately reviewed publication snapshot based on commit `3122dc79462018ecec6af965929f594ceb691ee7`. It does not include the unpublished local commit ancestry. Original report and numerical evidence bytes are retained. Historical mentions of publication being held describe the previous local-delivery stage; this README and the publication review describe the subsequent user-authorized static snapshot.

The generic report exporter still has its documented privacy-filter failure. Publishing this particular inspected snapshot does not certify or fix that exporter. Fabricated private paths/hosts in its regression tests and failure reviews are test examples, not actual credentials or infrastructure.

The machine-specific `audit_inputs.py` is omitted from this public snapshot; its original remains in the private experiment archive. Any recorded hashes referring to it identify that original, not a replacement. RAW, model weights, prior caches, checkpoint/recovery files, patch NPZs, screenshots and full private runtime are not distributed here. Missing private dependencies mean this is **not a standalone reproduction package**; see [ARCHIVE_SCOPE.md](ARCHIVE_SCOPE.md).

## Next experiment preparation

- Reuse the frozen Nikon split, same baseline checkpoint, initialization protocol, consumer, budget and 18-cell evaluator. Only 668 single-light train images may supply image content; all allowed train global white points may supply light colors. No real mixed image/mixture-map/dense-GT source for construction, QC or development.
- Retain the completed batch unchanged. Do not rerun its fixed `runs/` or `results/` destinations: these are intentionally exclusive. A future authorized experiment needs a new version/output directory and new frozen manifest.
- Before another launch, verify the mounted data identity, source/cache hashes, remote authentication/GPU ownership and available memory. Prior successful GPU checks are historical, not a current readiness guarantee.
- Freeze the next hypothesis and matched control before implementation. This batch gives no stable de-shading benefit and no scene-alignment advantage; it does not authorize another parameter sweep or selecting C after seeing test.
- After an approved change: numerical tests → full/tail real-input preflight → independent fresh optimizer calibration → independent entry check → fixed-budget training → frozen development selection → once-only scoring. Keep actual view/patch consumption records and single/multi metrics.
- Current state: code, results and protocol are archived; **no new experimental hypothesis/configuration or GPU run is launched**. Runtime checks and a specific next-run configuration remain required.
