# Seven-arm execution status

Updated: 2026-09-26. Agent: codex. Implementation snapshot, not a result.

Upstream specification2c52cbe has been fetched. New branch: codex/diffuse-vgt-es-plus3. Existing diffuse A/B/C and low-frequency pair artifacts are preserved; no historical checkpoint substitutes for the new matched controls.

Completed checks:6 actual server CPU formula tests; full and tail seven-arm production preflight; fixed24 permitted Nikon scene previews and whole-image/16patch intervention diagnostics;24 known-truth synthetic-prior observations. Independent metric recomputation for the prior diagnostic matched within7.1e-15. Actual full/tail normalized global/local input RMS is nonzero for all interventions versus ES00. All seven retain matching sampling, masks, consumption ledgers and unchanged original O branch inputs/labels.

Preflight full-cycle consumption:1336views,84983 accepted patches. Tail:1024views,65136 accepted patches. These counts are technical support checks, not a performance improvement. Known-truth scenes are simplified domain-external diagnostic, not physical Nikon truth.

Fresh full/tail optimizer calibration completed: two independently initialized single updates per arm, fourteen total. Independent saved-state verification passed: first-step AdamW states, finite parameters/moments, parameter changes, gradient norms, output identities and consumption ledgers. All seven formal entry gates passed. The supervisor and first formal arm ES00 are RUNNING; no seven-arm final performance result yet.

Declared formal order: ES00, ES11, ES10, ES01, KEEP_OLD_R, SHAM_OLD_0, NO_NEW_S. Each69600successful updates; total487200maximum. Fresh initialization/AdamW, model, loss, sampling and label rule unchanged. Shared mask includes rotated shading validity; exposure covers allseven plus historical A. No test-based tuning or appended runs. Exact interrupted-run recovery is not implemented: failures stop for review, never silently restart.

The finite supervisor runs the approved queue then freezes development selection. A quiet thread heartbeat checks every30minutes and triggers independent completed-cohort acceptance, one-shot real scoring and final sanitized result/Obsidian delivery when ready. It stops after this batch. A monitoring setup is not a claim that training is complete.

Source entry points: queue_train.py, seven_synthesis.py, test_seven_synthesis.py, seven_qc.py, seven_signal.py, prior_calibration.py, calibrate_queue.py, supervise_queue.py. Private source/model/cache assets are not in this public folder. Set DIFFUSE_PARENT to the existing reviewed producer and ES_SPEC_DIR to this specification directory; ordinary public checkout alone is not a ready-to-train dataset/environment.
