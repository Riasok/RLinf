# Seven-task BC versus PPO: forgetting comparison

User requested BC from the existing few-shot RLinf-Pi05-LIBERO-SFT, NOT the
general pi05 base and NOT the full-shot 130-task release. PPO was stopped by
request on 2026-09-08. General-base download/conversion is not part of this run.

All available demonstrations in physical-intelligence/libero for the same seven
Spatial instructions are eligible: 296 episodes / 35,368 frames (counts in the
generated split manifest are authoritative). Simulator task IDs 0,2,3,4,5,6,7
are matched by full instruction, not mistaken for dataset task indices. Held-out
IDs 1,8,9 are excluded from additional BC; they may have appeared in initial SFT.
Use the original checkpoint normalization, unchanged, to match the PPO policy.

This is full-data, expert-only BC, not full-parameter SFT or an exact reproduction
of the full-shot release. Frozen VLM, no LoRA, no critic loss, no CSD. LR5e-6,
Adam(.9,.95), epsilon1e-8, weight decay.01, grad clip1, globalbatch2048, micro16,
constant LR, seed42. These match the prior PPO actor settings. BC uses standard
flow-matching loss on 10-action demonstration chunks; PPO used policy gradients.

Reference: pi_RL https://arxiv.org/html/2510.25889 Appendix J Table11 specifies
Spatial pi0.5 PPO batch2048, one update epoch, LR5e-6, horizon240,64envs x8rollouts,
prediction10/execution5 and denoise3. These are PPO settings, not a prescribed BC
recipe. The released OpenPI pi05_libero SFT recipe instead uses batch256/LR5e-5
and warmup. We deliberately use the matched-PPO actor controls for this ablation.

One prior PPO iteration contains 64 x 8 x (240/5) = 24,576 action chunks, or
12 optimizer updates of globalbatch2048. One BC comparison iteration similarly
contains 12 optimizer updates. Budget150 comparison iterations =1800 BC updates.
This matches update count/chunk presentations, NOT environment interactions,
unique data, action-target count, objective, or wall-clock compute.

Baseline at iteration0; save/evaluate at10,20,...150. Same ODE evaluation as PPO:
prediction10/execution5,denoise3,horizon240,50fixed trials per task. Log seen7
(350 episodes) and heldout3 (150 episodes) separately, success_once and
success_at_end. PPO's existing heldout curve can be compared directly; its
training-rollout success is NOT the same as a seen-task ODE evaluation curve.
PPO early unsaved checkpoints cannot be retrospectively evaluated.

Run: scripts/run_spatial73_bc_20260908.py --launch, local .venv, GPUs0-3 only.
No --launch prints the task manifest without starting training. Logs:
logs/20260908-spatial73-bc. Checkpoint directory global_step_N uses COMPARISON
iteration N; budget.json records underlying optimizer step. Do not resume it
through the stock SFTRunner, which interprets that directory number differently.
Trainer stays resident; parameters/optimizer offload during external evaluations.
Only child-owned processes are cleaned. Disk guard80GiB, RAM guard40GiB.
W&B project rlinf-ppo-csd; consolidated evaluation curves use comparison iteration.
