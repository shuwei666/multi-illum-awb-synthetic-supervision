# Diffuse Virtual GT v1 — user-authorized revision 1

updated: 2026-09-26; agent: codex

User explicitly accepted inherited TIFF black-level assumptions, retained the original label convention, and restarted the GPU. This revision changes only those previously reported gates, not the maximum three formal seed0 runs, source restrictions, consumer, fixed development/evaluation or QC requirements.

1. Retain unchanged camera TIFF pipeline; no guessed black subtraction, gamma correction or new baseline. Conversion lineage remains UNVERIFIED, accepted as an engineering assumption rather than physical certification.
2. Patch label = unmasked mean of G=1 per-pixel E over the entire original 16×16 patch, then the parent's L2 normalization. New technical validity only rejects patches consistently across A/B/C; it never weights GT pixels. Parent identity image/label construction is retained; the common technical rejection applies to all arms.
3. Before and after median normalization are tested separately: shared lamp power scales raw H/T; normalized H/I/J/E are invariant. Additional common exposure f scales I/J but not E.
4. Pre-QC conservative technical conventions: finite positive shading and finite nonzero normals define valid prior pixels. Fill invalid s0 with 1 and invalid normals with the verified camera-facing normal solely to keep context finite. Invalid support is never rescued for supervision. Interpolate validity conservatively; accept a patch only when every contributing technical-valid pixel is valid. For all A/B/C use the intersection of local shading/normal validity and rot180 normal validity, plus parent patch-valid. This prevents C transporting undefined fields into accepted patches. Fill choices and rejection rates must be recorded; >20% unusable sources still stops the batch.
5. Source/crop/patch/state/endpoint draws retain parent generators; new directional/power draws use a dedicated stream. Model inference consumes no training RNG. The current parent uses one stream for geometry and source ordering; retain that draw sequence rather than silently refactoring it.
6. Shared power-of-two exposure protects the actual /4 half-storage boundary using max of A/B/C counterfactual RGB and target60000. E unchanged; corresponding J scaled. Existing finite/mask guards retained. No claim of bitwise equivalence to old O.

This is a pre-QC contract; model coordinates, dependencies/weights/license, real24-source QC, full input sensitivity, full/tail optimizer calibration and independent entry approval must still pass before training. Original stage-zero blocked record remains historical, not retroactively passed.
