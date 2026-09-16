# Robot experiments: read this first

Status: archive migration completed, 2026-09-07. Inactive run artifacts are in
`deprecated/results/`; historical downloaded models in `deprecated/checkpoints/`.
Old paths have compatibility symlinks. Deleted 24 audited intermediate saves
(454.86 GiB), retaining evaluated saves and first/last saves. No GitHub push yet.
See [the execution journal](research/archive_execution_20260907.json).
The upstream [README](README.md) remains the general RLinf documentation.

## Scope

1. LIBERO-Spatial PPO with seven RL-training tasks and three held-out tasks;
   potentially extend evaluation to LIBERO-Plus with an explicit task mapping.
2. ManiSkill3 PPO on a small, explicitly enumerated task set; later one or two
   digital-cousin pick-and-place tasks. Those new environments are not implemented
   or selected yet. The existing 25Main run is a reference, not that experiment.

Start with [the cleanup proposal](research/CLEANUP_PLAN.md) and
[the result/artifact registry](research/RESULTS_AND_CHECKPOINTS.md).
Historical CSD experiments remain evidence, but are not the current default.

## Current LIBERO experiment

- Launcher: `scripts/run_spatial73_20260907.py`.
- Config: `examples/embodiment/config/libero_spatial_7train3heldout_pi05_20260907.yaml`.
- Run directory: `logs/20260907-spatial73/`; task names: `split.json`.
- Model: `checkpoints/RLinf-Pi05-LIBERO-SFT`, pi0.5; frozen VLM, trainable
  action expert and critic; PPO without CSD or LoRA.
- Train task IDs: 0, 2, 3, 4, 5, 6, 7. Held-out IDs: 1, 8, 9.
- This is the existing seed-42 random split, NOT a deliberately selected
  compositional split. Held-out means unseen in RL, not necessarily unseen in SFT.
- There is NO 25/25 train/evaluation initial-state partition in this run.
- Budget 150 iterations; evaluate held-out tasks every 10; save every 50.
- Evaluation: 3 tasks x 50 fixed trials = 150 episodes, excluding padding.
- Global batch 2048; one PPO epoch; actor/critic LR 5e-6/1e-4; gradient clip 1;
  Adam epsilon 1e-8; generation/execution 10/5; three denoising steps;
  horizon 240; 64 environments x eight rollout rounds; Flow-SDE noise 0.5.
- Runtime adaptations: microbatch 16 and rollout offload. The budget and split
  differ from the paper. See [original handoff](SPATIAL73_20260907.md).

The existing local readiness check (does not launch training) is:

```bash
cd /data/minjaeoh/RLinf
.venv/bin/python scripts/run_spatial73_20260907.py
```

Do not rerun `--launch` for the existing experiment. It uses fixed output paths
and GPUs 0-3. A new experiment needs a separate run ID and reviewed placement.
The local launchers are not yet portable public entrypoints: see cleanup gates.

## Evaluation rules for the next LIBERO iteration

- Preserve the current split and run unchanged for comparability.
- Report seen-task and held-out-task results separately. If adding an init-state
  split, make it a NEW protocol/run and save explicit trial indices.
- Evaluate the same SFT initialization alongside the trained checkpoint under
  exactly the same protocol. Do not call the held-out set SFT-unseen.
- For Plus, explicitly map each variant to its source task, retain seen/held-out
  grouping, and record variants, exclusions, trials, seeds and denominators.
- Log both `success_once` and `success_at_end`. Do not pool different horizons,
  denoising counts, Plus/PRO versions or partial evaluations into one score.
- Frequent held-out evaluation makes it a validation curve if used for selection;
  do not also present it as an untouched final test.

## ManiSkill reference and next small-task experiment

Existing reference: `logs/maniskill_pi05_memsafe`, initialized from
`checkpoints/RLinf-Pi05-ManiSkill-25Main-SFT`. Training was stopped; only step 50
is saved. Its ID and 12-setting OOD evaluations have finished.

- Preparation/runtime: `scripts/prepare_maniskill_pi05.py` and
  `.venv-maniskill-prep`; assets at `/data/minjaeoh/.maniskill`.
- Evaluation implementation: `scripts/eval_maniskill_cut_20260907.py` and
  `scripts/queue_maniskill_ood_20260907.py`.
- Existing protocol: 320 episodes per cell, horizon 80, generation/execution 8/5,
  four denoising steps, ODE evaluation, seed 0, four GPUs per evaluation.
- This uses the 25Main environment and mostly one receptacle, not exhaustive
  paper coverage. See [protocol caveats](MANISKILL_OOD_20260907.md).
- [ManiSkill preparation notes](MANISKILL_NEXT_RUN.md) contain historical stages;
  their original "not launched" heading is not the current status.

Before a new one/two-task run: approve task/environment IDs, assets, robot/action
space, compatible SFT normalization, train/eval object and scene splits, reward,
episode horizon and budget. Smoke-test reset, rendering, action execution and
SFT success before PPO. Do not silently treat 25Main or a distractor variant as
a digital cousin. Save a resolved config and best/latest checkpoint policy.

## Storage, reproducibility and GitHub

Keep code, small curated result tables, protocol manifests and documentation in
Git. Keep weights, optimizer states, raw videos, datasets, caches, virtualenvs,
credentials and raw W&B directories out of Git. A manifest must identify the
source model/revision, code commit plus local patch state, resolved config,
split, checkpoint step, metric definitions and artifact locations.

Historical shell launchers, configs and notes have moved with compatibility
symlinks. Python helpers remain in place pending a live-import refactor. Moving
within this filesystem does not free space; the approved checkpoint deletion did.
Any further deletion needs its own scoped approval.
Never delete a run's only evaluated weights merely because its training ended.

Proposed publication remote: existing `origin`, Riasok/RLinf. Branch currently
`ppo-csd-training-20260827`. Review a clean publication branch/diff before any
push; do not push the dirty worktree wholesale. See the cleanup plan for gates.
