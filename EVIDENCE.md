# Internal-to-public evidence map

This repository publishes a research report, not the raw dataset or checkpoints.

| Public claim | Archived evidence |
|---|---|
| Sony single-sensor scope and result chain | project `README.md`, lines 21–59 |
| NUS versus LSMI illuminant pool | `experiments.md`, LSMI-light experiment; `build_synth_lsmi.py`, pool construction |
| `_1` reflectance source | `build_synth_lsmi.py` module documentation and sample recovery |
| `_1 + _12` expansion | `build_t3.py`, expanded reflectance block; `experiments.md`, Phase 1 |
| Real validation checkpoint selection | `train_direct.py`, validation loop and `best.pt` save condition |
| t3 50-epoch settings | `run_t3.sh` |
| t4 initialization and settings | archived checkpoint metadata |
| 3.225 degree result | `server_artifacts/eval_lsmi_exp.json` |
| 2.834 degree distribution | `viz/test_stats.json` and archived checkpoint |

The checkpoint supporting the best pilot has SHA256:

```text
f59d53302550a8e59da3bbc58c6dca7d2971ba18b485f3e684932cd6040c829f
```

Known evidence gaps are deliberately retained in the report: `_12` reflectance preprocessing provenance, final per-image t4 evaluation archive, multi-seed stability, fair-budget baselines, and a truly sealed test.

The complete runner scripts for the historical NUS-pool baseline (stage A) and
the `_1 + _12` expansion run (stage C) are not in the current archive. Their
checkpoint-selection policy and any unrecorded hyperparameters are therefore
reported as unverified rather than inferred from the later training script.
