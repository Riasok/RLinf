# Queued Spatial PPO+CSD beta 0.5

Wait for BC iteration150, its final checkpoint, and both final seen/held-out
evaluation JSONs with complete episode counts. Then wait until physical GPUs0-3
are released. BC failure blocks the queue; no restart or partial-result bypass.
Requires235GiB free to budget8 retained checkpoints plus operational headroom.

Start fresh from RLinf-Pi05-LIBERO-SFT, NOT the BC-trained model. Same split,
80iteration budget (updated by user), optimizer, actor trainability and rollout parameters as
the original seven/three PPO. Plain CSD (constant gate), beta0.5, offset5,
CSD microbatch2. Save every10; evaluate heldout3 every10 including final80.
The inherited PPO runner does not add seen-task standalone evals to this run.
This is NOT reward-gated CSD. W&B project rlinf-ppo-csd.

Queue: scripts/queue_spatial73_csd_20260908.py --launch.
Dry-run without --launch validates the composed config only.
Status: logs/20260908-spatial73-csd-b05/queue_status.json.
Training is supervised with own-descendant cleanup and disk guards; no other
GPU jobs are stopped. No change to the ongoing BC run.
