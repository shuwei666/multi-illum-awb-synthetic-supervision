# Local report content acceptance

updated: 2026-09-26; agent: codex independent reviewer

Artifact verdict: **PASS**. Contract verdict: **ADEQUATE**. **独立验证通过，仅限以下哈希绑定的本地报告内容与证据链接。** Publication/export remains **HELD**. This does not approve the report generator, its privacy filter, public release, scientific success, or browser layout.

## Contract and isolation

Read-only acceptance of actual FINAL_REPORT.md and FINAL_REPORT.html against USER_TASK.txt, REVISION_1.md, development_selection.json, results/summary.json, final_metrics_review.md, and actual completion/history/cache/preflight/calibration records. Hard requirements: all 90 final metric values preserved; correct frozen baseline identity and O selection; no post-hoc promotion of C; faithful B-A/B-C interpretation; actual consumption and cost records; explicit evidence limitations; matching source hashes and resolving local evidence links. No new RAW/GT access, GPU, training, implementation changes, or publication. Only this review is written by the reviewer.

隔离有限：fresh agent context, separate standard-library Python process, shared model family and filesystem; read-only behavior is procedural, not OS-enforced. Browser 1400/768/375 rendering acceptance belongs to the main agent and is not independently certified here.

## Hash binding

| Artifact | SHA256 |
| --- | --- |
| FINAL_REPORT.md | d854c65321f1a4110dab134e67a069a75a4a8132f3bd775ba1abe0a05201fef0 |
| FINAL_REPORT.html | 74afca7c4c432f596a37da6658c07aa21b09e0d0e5131b32f1379a3353877c31 |
| USER_TASK.txt | 7488b5be071894e25144554ffee225188872e5656270b42eba501f699942424f |
| REVISION_1.md | ad6b4f6da46b2a8ed72a548320a09d5a4ff85a71c8b334097e564e2f6325a176 |
| results/summary.json | d2f72d42dbe50941062ea38ddf3f3866d4d650836a7cd0edda05206efcc7fc36 |
| development_selection.json | 674ede698fb96310b3ec985c3453e3513657e5737c3a2afaa34dae05e1d73081 |
| reviews/final_metrics_review.md | 07e035c5f6af5f1834d8502f729cdef376b8666a84dd23af842c67074d2b8439 |
| run_manifest.json | af5371a9efa15564074aa8cc1264fd95469a77fe900b4383fa6fdb8c75b7442f |
| build_report.py | 96a3a3eb755239621b5769d5f17590304a979387bb1681ea41b29310365773c9 |
| reviews/report_tool_repair_review.md | 48705d548dc311acb642c2327585f7ec740f2284f8642614e1ea61fd59f874dd |

## Executed checks and findings

Used inline `python3` processes with pathlib/json/re/hashlib/html.parser, without importing the renderer or evaluator. Parsed the Markdown rows and HTML table cells independently: all 90 metric values in each artifact exactly match serialized summary values; all 36 Markdown mechanism deltas match summary. Twenty development submetrics and four scores occur unchanged in both formats. All 13 entries in the report source-hash table match current files.

The baseline is explicitly this project's frozen modified One-Net, not a claimed full author-protocol reproduction. Development selects O with 0.8573986570311704; A/B/C scores are higher. C's lower test errors among these candidates do not change the selected winner. All candidates lose to baseline in all 18 cells, and FINAL_TARGET_NOT_MET, false seed0 gate, and incomplete multi-seed goal are retained. B-A is negative in 9 test cells and positive in 9 val cells; B-C is positive in all 18. Neither is promoted into statistical significance or proof of the physical mechanism.

Recomputed each arm's actual history consumption, with history SHA256 matching both summary binding and completion output binding: 417 cycles, 69,600 updates; 557,112 generated view slots, 556,800 consumed view slots, 35,423,630 valid patch slots, 278,403 mixed view slots, 17,712,121 mixed valid patch slots. All report counts match. These are explicitly repeated slots, not independent photographs/scenes.

Recomputed update-weighted online seconds/update and min/max from all cycle records: A 0.01928009741661278 [0.018885339403937676, 0.021792019644897142]; B 0.019426129315819204 [0.01911699664806891, 0.0217816864926658]; C 0.020218280044480643 [0.018947694651380985, 0.022442091547418386]. Formal seconds and peak allocated bytes match completion records; calibration states, two-update counts, seconds and peak bytes match separate calibration completions. Cache seconds 824.2384779453278 and 3,011,816,286 bytes match cache_manifest; peak 2,126,633,472 bytes matches cache_progress. Preflight seconds/peak match its record and are correctly described as rendering only. No cost figure is represented as historical throughput or entire-card usage.

The report preserves external learned prior/resource differences, unknown Nikon overlap, approximate camera proxy/intrinsics, unverified TIFF lineage accepted under REVISION_1, non-diffuse context risks, NOT_PERFORMED physical relighting, historical test exposure, single-seed uncertainty, and weak-intervention limitations. The copied numerical review explicitly limits pixel validation to saved per-image CSV reaggregation and patch validation to saved angular-error NPZ; no new GT reconstruction is implied.

Initial Markdown had six broken embedded-review relative links. The main agent mechanically rebased them; `diff -u report_drafts/before_link_fix.md FINAL_REPORT.md` contains only those six URL changes. The old Markdown SHA256 was 504612bc6644d6be9ef492cc8a284799b64503a32ceebec16c552cb5331da9c0. Rechecked all 22 current Markdown links: all resolve. All 16 actual local HTML hrefs resolve; eight other hrefs are in-page section anchors. HTML's copied review appears as preformatted literal Markdown, so its internal literal paths are not active broken hyperlinks. HTML itself remained unchanged.

## Status boundaries

Correctness: report values, counts, costs, selection and conclusions match their saved sources. Evidence: matching hashes and resolving local active links. Reproducibility: deterministic standard-library parsing and direct history sums; the already accepted independent metric audit remains the source for numerical GT/error-array scope. Task fit: complete local negative-result reporting, S1 traceability; no performance S2 or physical certification.

The separate report-tool repair review is the second consecutive FAIL and identifies an unhandled `ssh hostname` privacy case. It remains in force. No privacy repair was attempted or accepted here, and no export, upload, publication or external-delivery approval follows from this local-content PASS. Rebuilding with the generator may overwrite the manual Markdown link correction and requires checking the resulting artifact hashes again.
