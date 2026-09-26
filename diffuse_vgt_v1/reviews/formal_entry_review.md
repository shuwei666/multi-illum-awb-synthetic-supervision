# Formal training entry review

updated: 2026-09-26; agent: codex (fresh independent reviewer)

Artifact PASS; contract ADEQUATE; S1 engineering entry. Approve exactly three sequential fresh seed0 runs A/B/C, each 69,600 successful optimizer updates, using the unchanged frozen model, AdamW and schedule. Calibration checkpoints and optimizer states must not initialize formal training. This approves entry, not completed training or downstream efficacy.

Independently executed `CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:pydeps:../pydeps:../code /root/miniconda3/bin/python reviews/audit_formal_entry.py` on the authorized remote host, exit 0. The script performs no GPU or training work and reads no real mixed GT. It rechecked frozen code/assets and all 668 cache hashes, then loaded all six actual full/tail checkpoints on CPU. Each Adam state has step exactly 1, finite parameters/moments and nonzero change from the frozen initial weights. Recomputed double-precision parameter deltas agree with recorded deltas. Recovery equals the actual tail model, with cycle416/next batch128, two accumulated successful calibration updates but fresh tail optimizer step1. All completion output hashes pass; independently rehashed local copies match all remote completion hashes.

|Arm|Full delta|Tail delta|Total calibration seconds|
|---|---:|---:|---:|
|A|1.79647884e-5|2.28778865e-5|48.9614|
|B|1.79527356e-5|2.28041919e-5|48.6613|
|C|1.78712032e-5|2.29132418e-5|50.4761|

All three full cases consume cycle0/batch0: eight views, 512 candidates and 505 valid patches. All tail cases consume cycle416/batch127: eight views, 512 candidates and 512 valid patches. History, completion, diagnostic counts and actual recovery endpoint ledgers agree. The three arms have identical matched sampling hashes, technical rejection, valid/mixed consumption and every endpoint ledger array. Recorded finite gradient norms span13.6222–20.0775. Full/tail prefix match and input intervention survival remain corroborated by unchanged preflight_001, including A/B GT equality and correctly distinct C GT.

Measured peak allocated memory is19,186,398,720 bytes for every calibration arm. These are actual optimizer executions, not a CPU memory prediction. Total times include one-time input loading; per-update history includes a whole-cohort synthesis despite only one calibration update. Neither is a valid direct multiplier for69,600 formal updates. Sequential formal execution is required by this entry scope. It does not guarantee future hardware availability or absence of later failures.

Freeze SHA256:4886059347f1ddf4f9d9f285cc21cb8a2be9a47d948ec940d98a71e1365b7803. Initial weights SHA256:a81e752797ae353ce8e8081013405906f9d8975a6f5899c9150df2806033336a. Completion A:1c90f148bb95fbf0e2e69c7e85647526e63e96abe373c9fa8c6e7d85df58c4e9; B:d5139d45b06ea5e792a8de84383a93a47d140b5a3b9b7dc32f2ef148b09ca0dc; C:2d65eccac4af1fc25ecf82d5458666295d3c26edcfba200c4a950186c3281e74.

Prior QC visual examination is reused with its unchanged evidence, not claimed as repeated here. Texture leakage, dark-preview limitations, approximate normals/intrinsics, non-Lambertian materials, camera-RGB proxy domain approximation and unknown overlap with external prior training data remain. This is external intrinsic/geometric-prior-assisted single-source Virtual GT. Unverified TIFF conversion/black-level lineage remains the explicitly accepted REVISION_1 engineering assumption; no physical certification is inferred. Real relighting physical validation: NOT_PERFORMED. No performance or mechanism benefit is established by these six updates.

Correctness: actual bounded updates, state and accounting passed. Evidence: executable CPU audit, local/remote hashes and unchanged preflight/QC chain. Reproducibility: gates bind frozen code and actual evidence. Task fit: exactly authorized A/B/C, fresh initialization, fixed budget, no new variant or real mixed GT training access. Isolation is limited: fresh reviewer context and CPU process, shared filesystem and model. Independent validation passed only for formal training entry.
