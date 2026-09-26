# Diffuse Virtual GT v1 — stage-zero audit

updated: 2026-09-26; agent: codex

Revision1 note: the following is the preserved stage-zero record. The user subsequently accepted inherited photometry assumptions, retained unmasked parent labels and restored GPU access. Current progress is in `STATUS.md` and `REVISION_1.md`; this historical block is not the current execution status. Photometric lineage remains unverified.

Status: **BLOCKED_INPUT_OR_PHOTOMETRY**. No prior inference, calibration or formal training started. This is a bounded input audit, not a completed experiment or physical validation.

## Verified inputs

`audit_inputs.py` reads only Nikon train `_1.tiff` and split/global whitepoint metadata. Actual inventory: 668 sources, 1,353 allowed endpoint entries; every TIFF is 512×512×3 uint16. Observed aggregate range is 0–15,520; this is not a verified saturation level. Full per-file SHA256 and Light1 are recorded in `source_manifest.json`. The deterministic 24-source list is frozen using SHA256 of `diffuse-vgt-v1:` plus scene_id. No real mixed image/map/GT was read by this audit.

Official LSMI source snapshot `0de53b82b76ff1c2d8e62e6c188768f7dd836d23`: `0_cvt2tiff.py` invokes dcraw `-h -D -4 -T`; `2_preprocess_data.py:61–63` subtracts a camera-template black level and clips to the template white level minus black. It subsequently masks MCC in training, center-crops and resizes. This establishes an available reference process, NOT proof that the current TIFF files were produced by it. No per-scene Nikon NEF, conversion script or conversion log exists within the inspected `raw_backup/512x512_lsmi/nikon_512` tree. The vendor Nikon DNG template is not a source-linked NEF for these 668 files. Wider storage has not been exhaustively searched.

Prior DG R1 explicitly accepted unverified conversion lineage for its two old runs only. The new task reinstates the photometry gate. No second black subtraction, gamma change or baseline retraining was performed.

## Parent implementation

`code/round2.py:49–62` is O/DG's actual source loader: OpenCV unchanged BGR read → RGB float32 → divide by Light1 and 4 → half storage, retaining 512×512 native and INTER_AREA-resized 256×256 copies. `round2.py:72–90` chooses half of the two-per-source views for crops of integer size 384–512, bilinear grid_sample at 512 then 2× average pooling. No additional black subtraction or gamma occurs here. Pixel observations alone cannot establish linearity or black-level subtraction.

`code/frozen/domislovic_v2.py:303–306` computes a single per-sample z-score over C,H,W, not separately per color channel. Thus the specific per-channel cancellation premise is not supported by this source inspection. Full actual-input forward sensitivity testing remains NOT_RUN because this local interpreter lacks Torch and SSH port 21602 returned connection refused. No conclusion from a substitute NumPy pipeline is presented as actual network validation.

Important compatibility issue: `code/round3_synthesis.py:131–135` takes E's unmasked mean across every 16×16 region; validity is patch-level nonblack and GT-component threshold. `dg_synthesis.py` inherits this. There is no existing per-pixel validity mask in that mean. Task §4.4's masked mean plus new technical masks is therefore a label-definition change, not pure parent inheritance. Before training, choose explicitly whether to retain full-grid means and reject technically invalid patches, or authorize masked-region labels. No choice silently implemented.

## Contract clarification and remaining gates

An independent fresh-context review judged the initial contract PROVISIONAL. Its explicit algebra correction: before median normalization, common lamp power scales H_raw/T_raw; after normalization H/I/J/E are all invariant; an additional shared exposure scalar scales I/J but not E. This corrects the test wording without changing the rendering equations. Four synthetic algebra tests exercise this distinction and basic three-arm neutrality; they do not validate Intrinsic decoding against its installed dependency or DSINE coordinates.

Undefined normal/shading fill values, rotated technical validity for C, exact common-mask/patch policy, storage exposure bound and random-stream integration remain unfrozen. No A/B/C cache or trainer is claimed complete. External priors not downloaded/inferred; license/training-overlap and immutable commit/weight hashes still require inspection before stage 1.

Server check: existing authorized SSH port 21602 returned `Connection refused`. No old server substituted; no credentials recorded in this artifact. Work stops at stage 0 under the user's own gate. Required next inputs: verifiable RAW-to-TIFF lineage (or an explicitly revised batch contract), restored server access, and resolution of parent-label compatibility.

## Version and evidence

Starting repository HEAD: `3122dc79462018ecec6af965929f594ceb691ee7`; clean before new branch `codex/diffuse-vgt-v1`. User task archived verbatim in `USER_TASK.txt`. Original modified sync directory was not touched. No remote publication at this stage; only lightweight audit/code may be committed on the new branch. Source files, existing results and checkpoints unchanged.
