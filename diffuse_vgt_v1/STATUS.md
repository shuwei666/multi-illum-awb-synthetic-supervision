# Diffuse Virtual GT v1

updated: 2026-09-26; agent: codex

**FINAL_TARGET_NOT_MET — training and unified evaluation complete; independent numerical audit PASS/ADEQUATE (limited isolation). Public export remains held.**

Public-report export is on hold: two successive independent report-tool reviews failed privacy filtering. The second review confirms history/CSV binding repairs but demonstrates an unfiltered bare SSH hostname. Automatic repair iterations have stopped; the user has been notified and asked about local-only delivery. This is not a training/evaluation failure. No push has been performed in this batch; see `reviews/report_tool_review.md` and `reviews/report_tool_repair_review.md`.

## Revision 1 progress

- User accepts inherited photometry assumptions and retains the parent unmasked patch-label convention; see `REVISION_1.md`. These are assumptions, not verified black-level provenance.
- GPU connection restored. Fixed 24 sources completed prior inference in 37.73 seconds, peak allocated GPU memory 2,126,633,472 bytes.
- Independent artifact review permits all-source offline caching only. Technical validity was complete for these 24; material, normal accuracy, dark noise and physical relighting remain unverified. See `reviews/qc_artifact_review.md`.
- Three training-entry issues were repaired and independently rechecked: stale-cache provenance, calibration consumption accounting, and separate shading/normal validity. At that earlier gate, 28 CPU tests passed and formal training had not begun; subsequent formal progress is recorded below. No new test scoring has begun.
- All668 cache completed with process exit0: 824.238seconds, 3,011,816,286bytes, unusable source fraction0. Manifest SHA807bf9573ecd5cef4d01e5e96502461797f242dcd20023a8b54af4106cc00ae0.
- Frozen manifest SHA4886059347f1ddf4f9d9f285cc21cb8a2be9a47d948ec940d98a71e1365b7803. Source/parent/producer/cache identity checks passed at freeze.
- Full668 GPU preflight exited0, assertions PASS: full cycle0 and tail416, matched sampling/mask/ledgers, unchanged identity branch, equal A/B GT, nonzero normalized input interventions. Total106.318seconds; peak allocated19,186,227,200bytes. See `preflight_001.json`; this is not optimizer or performance evidence.
- Final evaluator live split/meta protections were added and independently tested; no real scoring is authorized until the complete cohort and development selection are frozen.
- All six actual calibration updates passed independent CPU checkpoint/Adam/ledger audit. Formal entry PASS/ADEQUATE permits exactly three sequential fresh runs, not a performance claim. See `reviews/formal_entry_review.md`.
- Formal queue started2026-09-26T02:16:51Z. It stops on any failure; after all three runs it performs only frozen development. Each run writes heartbeat/history/recovery/completion. Current task monitors through completion and later scoring gate.
- Local DATA_4T copy of all668 cached NPZs independently rehashed by the main executor after transfer; 3,011,816,286bytes match the manifest.
- A completed69,600updates/417cycles in1,405.723seconds; its final checkpoint/history/recovery/counts/completion have been copied locally. B started from frozen initialization. No development or real evaluation yet.
- B completed69,600updates/417cycles in1,415.930seconds; its full output has been copied to DATA_4T and all five completion-bound file hashes match. C is loading from the same frozen initialization. No development or real evaluation yet.
- C completed69,600updates/417cycles in1,478.725seconds. All three runs are copied locally with matching completion-bound output hashes; serial training/development process exited0. Frozen development scores: O0.8573986570, A0.9253805212, B0.9655792015, C0.9614752020; O selected. Completed-cohort independent audit is in progress before any real scoring.
- Completed-cohort actual CPU audit and actual `approved_context` gate passed independently (limited isolation). Approval SHA275166aefc4bad8df271cd19cd15d319a46e4defa5fdc474936a820bbcee5404. Exactly one unified baseline/O/A/B/C real scoring started; no additional training or reselection.
- Unified evaluation exited0; all30 result files and summary copied locally. Frozen winner remainsO. All O/A/B/C pass0/18 baseline comparisons. Test/all patch pooled: baseline1.9692417244, O2.4238576383, A2.4608197267, B2.4198671548, C2.3456805275. These current outputs await independent numerical acceptance. Training/scoring processes have ended; GPU0MiB/0% at03:37UTC. No new seeds or parameter changes.
- Final independent numerical review completed:24,276 assertions,90 means (max discrepancy1.33e-15degrees),30 file hashes, baseline/O historical agreement,1196 image contrasts and1722 scene contrasts passed. Pixel verification is CSV reaggregation, not independent raw-GT native-pixel reconstruction. B-A improves all test cells but worsens all val cells; B-C worsens all18. Single seed supports no stable or significant benefit claim. See `reviews/final_metrics_review.md`.

## Historical stage 0 (superseded blockers retained for traceability)

- Read task and inspected parent/official preprocessing code; hashed 668 allowed source TIFFs and enumerated 1,353 endpoint entries.
- Core A/B/C idea is suitable for a bounded test, not demonstrated efficacy. Initial independent contract verdict: PROVISIONAL.
- Corrected proposed algebra-test interpretation: common lamp power cancels after median normalization.
- Blocking: actual TIFF conversion/black-level provenance is not closed; remote port 21602 refuses connection; masked-pixel labels conflict with the parent's unmasked patch mean and require a decision.
- No external model inference, 24-image visual QC, full cache, optimizer update or test scoring performed.
- See `00_audit.md`, `source_manifest.json`, `source_access_policy.json`, `environment.lock.txt`, `test_report.json` and independent review.
