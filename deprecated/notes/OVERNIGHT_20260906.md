# Overnight PPO controls — September 6

Started 01:16:50 KST; supervisor deadline 13:16:50 KST (12 hours).
Persistent tmux session: `overnight-ppo-controls`.

## Question

Can fresh single-suite PPO learn with the shorter execution length required for
our H10/K5 CSD comparison? Previous joint training degraded even without CSD.
The completed 500-trial checks confirmed Long step-10 success of 8.4% (PPO) and
9.2% (CSD) at K5/ns3, versus the historical SFT baseline of 29.2%.
Evaluating those trained checkpoints at K10 recovered only 15.6% and 17.0%.
Fresh SFT at K10/ns5 reached 48.2%. Therefore this overnight experiment prioritizes
plain-PPO controls rather than another unvalidated CSD beta sweep.

## Runs

| GPUs | Initialization | Suite | Generated / executed | Denoising | CSD |
| --- | --- | --- | --- | --- | --- |
| 4–5 | RLinf-Pi05-LIBERO-SFT | libero_10 (Long) | 10 / 10 | 5 | off |
| 6–7 | Same | Same | 10 / 5 | 5 | off |

Both use seed 42, 64 training environments, 8 rollout epochs, 480 training
steps per rollout epoch, four PPO update passes, global batch 2048, microbatch64,
actor LR 5e-6, value LR 1e-4, and a 1000-iteration cosine schedule. This follows
the bundled Long recipe except microbatch size, evaluation/logging, and bounds.
K changes the number of action chunks and therefore optimizer minibatches per
iteration; compare environment interactions and wall time as well as iterations.
This two-arm experiment does not itself isolate CSD's effect or fully explain
the earlier joint-run degradation. It is a single-seed diagnostic, not a final
benchmark result.

Evaluate every 5 iterations on the first 10 fixed trials for each of 10 Long
tasks (100 episodes), ODE inference, horizon520. Train horizon480 follows the
reference recipe. Screened SFT baselines on these 100 trials: K10/ns5 48%,
K5/ns5 34% (`logs/20260905-csd-diagnosis`). Do not compare training reward
directly to deterministic evaluation success.

## Persistence and limits

Checkpoints every 10 iterations; at most60 iterations per lane. The supervisor
stops its own process trees at the deadline, or earlier if free disk falls below
60 GiB. Existing user checkpoints are not deleted. A cutoff can discard progress
since the last checkpoint; an interrupted checkpoint must not be used without
validation. No automatic retry/resume is performed, to avoid hiding failures or
silently changing optimizer state. A failed lane's traceback remains in its log.
Logging is local TensorBoard, not W&B. GPUs0–3 are not assigned to these runs.

Artifacts: `logs/20260906-overnight-ppo-controls/`:

- `plan.json`, `status.json`, and eventual `stopped.json`.
- `long_k10/launcher.log` and `long_k5/launcher.log`.
- Per-lane exact `command.json`, resolved configs, TensorBoard events, checkpoints.

Launch implementation: `scripts/run_overnight_20260906.py` and
`examples/embodiment/config/libero_10_overnight_controls.yaml`.

Validation: Hydra config composition succeeded; regression tests for online CSD,
LIBERO trial filtering, and rollout seeding: **17 passed**. Startup verification
is separate from successful completion of a full PPO update.

Tomorrow: inspect finite losses/gradient norms, PPO clipping/approximate KL,
100-trial success trajectories, and checkpoint completeness. If a control learns,
then run a matched fixed-code CSD ablation at that setting and repeat selected
checkpoints on 500 trials. If both collapse, investigate the PPO update and
training/evaluation interface before adding CSD.
