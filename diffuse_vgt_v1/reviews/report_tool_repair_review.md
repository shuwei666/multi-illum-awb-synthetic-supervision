# Report renderer repair acceptance

updated: 2026-09-26; agent: codex

Artifact verdict: **FAIL**. Contract verdict: **ADEQUATE** for this bounded tool repair review. Final status: **未通过（附失败证据）**.

隔离有限：fresh agent context and separate Python process, shared filesystem and model family; no OS-enforced read-only isolation. This is not a scientific performance verdict, authorization for evaluation, or approval of a final report.

## Scope and frozen artifacts

Read USER_TASK.txt, REVISION_1.md, previous report-tool review, renderer and synthetic tests, and relevant evaluate_frozen.py schema code. No real GT, actual metrics/development outputs, checkpoints, or real histories were accessed. Only synthetic TemporaryDirectory fixtures were executed. No implementation or tests were edited; this review is the sole persistent output.

| File | SHA256 |
| --- | --- |
| build_report.py | 96a3a3eb755239621b5769d5f17590304a979387bb1681ea41b29310365773c9 |
| tests/test_build_report.py | 4f1ab4d924e91e2704aa9f5edd57a367af119fc90fe6de1cf0c69c5ef2de0b35 |
| USER_TASK.txt | 7488b5be071894e25144554ffee225188872e5656270b42eba501f699942424f |
| REVISION_1.md | ad6b4f6da46b2a8ed72a548320a09d5a4ff85a71c8b334097e564e2f6325a176 |
| evaluate_frozen.py | c57bb89a506afde4cbe9cf107bd69b88d96430846636587585ad775a59603fd5 |

## Confirmed repairs

- load() now requires actual history SHA256 to equal both the evaluator summary binding and the training completion outputs binding. This also applies to --check-only. The suite rejects changed/missing histories and independently changed completion bindings. A separate reviewer counterexample changed only the summary history hash to `stale`; load() rejected it with `Changed history binding`.
- Both paired CSV files must exist and match paired_artifacts_sha256 before links are emitted. Regression cases cover missing and modified files separately for each CSV. These keys match the evaluator schema.
- Source review text and dynamic paragraph/table fields pass through public_safe before rendering. Existing tests reject the tested personal paths, user@host strings, SSH URLs and IPv4 literals, preserving source files and avoiding final-report writes.
- Direct consumption sums, weighted timing, full 18-cell table, outcome checks and single-seed limitations remain covered by the synthetic suite. No real outcome has been checked.

## Remaining blocking finding

The private-host rejection is incomplete for an ordinary SSH command containing an unqualified host. A synthetic final_metrics_review.md containing exactly:

```text
Verdict: FAIL
Private host: ssh gpu-worker-07
```

was accepted by render(), and `Private host: ssh gpu-worker-07` occurred verbatim in both returned Markdown and HTML. This is a fabricated host used only for the test. The regex checks ssh://, user@host, IP addresses and selected domain suffixes, but not the common `ssh hostname` form. Because the original task explicitly excludes publishing hosts and the previous failure requested a public-safe review/field boundary, this remains a concrete gap in that same repair, not a new scientific requirement. The renderer does not itself publish; the defect is the emitted shareable report content.

Reproduction: import tests/test_build_report.py with importlib.util; create TemporaryDirectory; call fixture(root); write the above two-line string to root/reviews/final_metrics_review.md; call report.render(root); check the payload substring in each returned string. Observed: accepted; Markdown=True; HTML=True. The same guard protects free-text fields, so its detection boundary is shared. No actual private host was read or exposed by this review.

## Execution and coverage

Ran `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_build_report.py -v`: **6 tests PASS**, temporary artifacts only. Ran a separate inline Python process with the above counterexample and summary-only stale binding. Also tried fabricated `/var/home/mockperson/project` and `https://gpu-worker-07:8443` inputs; both were rejected. These do not negate the accepted SSH-command counterexample.

Correctness: provenance repairs pass; privacy boundary retains a reproducible gap. Evidence: source inspection plus independent synthetic execution, no real evaluation evidence. Reproducibility: deterministic fixture mutations and exact rejected/accepted payloads recorded above. Task fit: bounded report-tool requirements remain unmet; S1 is not granted for this artifact version. Browser layout, final scientific report, actual metrics, GT access and training are outside this acceptance scope.

This is the second consecutive FAIL for this tool after report_tool_review.md. The independent-validation workflow requires stopping the automatic repair/review loop and explaining the remaining issue to the user; this reviewer performs no further implementation changes. No training or evaluation decision follows from this report-tool verdict.
