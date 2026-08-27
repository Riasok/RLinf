# π0.5 LIBERO baseline setup

This checkout is pinned to upstream commit `5bf56eb6d2783f6c534abb88e28e7b50135b56ef`.
It targets GPUs 0 and 1 for continued supervised fine-tuning on all standard
LIBERO demonstrations, with LIBERO-Plus PPO/GRPO launch configurations retained
for later experiments.

## Provenance and limitations

- Starting checkpoint: `RLinf/RLinf-Pi05-LIBERO-SFT` (40 trajectories, one per task).
- Dataset: `physical-intelligence/libero` (1,693 episodes, 273,465 frames, 40 tasks).
- Continued SFT freezes the VLM (`train_expert_only: true`) and trains the action
  expert. It is therefore not an exact reproduction of a full-model/full-data
  paper baseline.
- `full_shard` is incompatible with π0.5's internal Gemma activation
  checkpointing in this Torch/OpenPI stack (`setStorage ... storage size of 0`).
  The validated configuration uses `no_shard` (about 18 GiB per H100).

## Local layout

```
checkpoints/RLinf-Pi05-LIBERO-SFT
datasets/physical-intelligence-libero
.runtime_cache/{huggingface,ray,tmp}
```

These directories, along with `.env`, logs, and native dependencies, must stay
untracked.

## Installation and downloads

```bash
bash requirements/install.sh embodied --model openpi --env libero
source .venv/bin/activate
mkdir -p checkpoints datasets .runtime_cache/{huggingface,ray,tmp}
export HF_HOME="$PWD/.runtime_cache/huggingface"
export HF_LEROBOT_HOME="$PWD/datasets"
export RAY_TMPDIR="$PWD/.runtime_cache/ray"
export TMPDIR="$PWD/.runtime_cache/tmp"
hf auth login
hf download RLinf/RLinf-Pi05-LIBERO-SFT \
  --local-dir checkpoints/RLinf-Pi05-LIBERO-SFT
hf download physical-intelligence/libero --repo-type dataset \
  --local-dir datasets/physical-intelligence-libero
```

Verify `meta/info.json` reports 1,693 episodes, 273,465 frames, and 40 tasks,
and that exactly 1,693 parquet episode files exist.

## Validation

```bash
bash -n run_pi05_libero_sft_2gpu.sh run_liberoplus_pi05_2gpu.sh
./run_pi05_libero_sft_2gpu.sh \
  runner.max_steps=2 runner.save_interval=100 \
  actor.optim.total_training_steps=2 actor.optim.lr_warmup_steps=1
```

A prior two-step validation completed forward, backward, optimizer, and save;
losses were approximately 0.0160 and 0.0587 and steady-state steps took about
3 seconds. A later 1,000-step run was manually stopped at step 187 before its
first save at step 250, so it has no resumable checkpoint. Do not start another
full run without explicit approval.

For later LIBERO-Plus work, use `./run_liberoplus_pi05_2gpu.sh ppo` or `grpo`.
