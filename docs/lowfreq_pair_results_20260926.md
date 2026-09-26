# Full versus low-frequency de-shading: matched Nikon One-Net results

Updated: 2026-09-26. Agent: codex.

## Conclusion

Both completed training-data construction variants remain worse than the original baseline on real Nikon test. Low-frequency de-shading won the frozen development comparison but did not improve real test over full de-shading. This single-seed comparison does not support the claim that retaining more spatial detail necessarily creates better training data. Do not select full retrospectively based on test, or call this physical reflectance recovery.

## Protocol

Both arms: fresh frozen seed0 weights/AdamW,69,600 successful updates,417 cycles; matched scene/endpoint/crop/patch draws, labels, masks and common exposure. Training construction uses only668 permitted single-light train images, plus permitted global Nikon train whitepoints (including individual light endpoints from multi-light scenes). No real multi-light source image or dense mixture GT is used for construction. Evaluation may use real dense GT.

Full divides the white-balanced source by bounded estimated shading. Low divides by exp(GaussianBlur(log shading,sigma16 on native512)), weighted by shading_valid support. This support differs from nonblack-weighted diagnostic visualization; sigma16 was fixed for exploration, not optimized on test. Both add the same normal-derived virtual lighting. Their common exposure covers original ABC plus low-shading ABC counterfactuals, so full is a new matched control, not the historical B checkpoint.

Frozen development normalized scores (lower better): full0.94529719, low0.93095047. Single-light development angular means: full1.28091657 degrees, low1.25114000 degrees. Winner low was saved before real scoring.

## Real test results

Angular means in degrees, lower is better. Patch means pool all accepted patches; pixel means average each image's valid-pixel angular mean equally. Test204 images:97 single-light and107 multi-light. Evaluation support matches exactly across models. Patch pooled and equal-image patch means coincide here because every image contributes256 valid patches.

| Model | All patch | Single patch | Multi patch | All pixel | Single pixel | Multi pixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 1.969242 | 1.346506 | 2.533778 | 2.167129 | 1.346577 | 2.910994 |
| Full de-shading | 2.368493 | 1.509611 | 3.147106 | 2.495399 | 1.509690 | 3.388986 |
| Low-frequency de-shading | 2.398934 | 1.558680 | 3.160660 | 2.528580 | 1.558745 | 3.407777 |

Low minus full: all-patch+0.030441 degrees, single-patch+0.049068, multi-patch+0.013554; all-pixel+0.033181. Low minus baseline: all-patch+0.429692, multi-patch+0.626882. Thus the baseline target is not met. This does not isolate which remaining factor causes the gap; source diversity, illumination distribution and geometry priors are alternative explanations, not established causes. Do not enlarge the experiment into a test-tuned sigma search.

## Reproducibility and evidence limits

Full checkpoint SHA256:1a1e9f20b52c6a637bc1a28ae653df18b72aad2334075f5fddcc03165b652451.

Low checkpoint SHA256:8fe30b60520a67cca2fcd099b9ce802e902607c5ef93c6205e1e8dd82fc51e45.

Baseline checkpoint SHA256:94d1cbf18b550156d46b6a201f87cc5d05619ab7e73c19983d7785bd382d39bb.

Entry scripts: lowfreq_pair.py, lowfreq_supervise.py, lowfreq_score.py. Full raw outputs include per-image CSV, saved patch errors, val/test summaries and pre-test development selection binding. Final scoring verifies frozen parent code/assets, exact checkpoint identities,69600 update records,417 matched sampling histories and endpoint ledgers. The first scoring attempt was interrupted before saved results, preserved separately; a subsequent empty-output collision was preserved, not silently overwritten. Final scoring completed with the strengthened guards.

Single seed, historical test has already been viewed in this project. Pixel statistics are evaluator output/CSV means; subsequent numerical reaggregation is not independent reconstruction from native RAW pixel predictions. Independent numerical review status is recorded separately; no statistical-significance or physical-accuracy claim.
