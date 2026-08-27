# Spatial PPO/PPO-CSD Run Settings

Three matched online-RL runs on standard LIBERO-Spatial. Only the CSD coefficient and GPU assignment differ.

## Runs

| Arm | GPUs | CSD β | W&B |
|---|---:|---:|---|
| PPO | 0–1 | 0 | [slk3n2sq](https://wandb.ai/riasok/rlinf-ppo-csd/runs/slk3n2sq) |
| PPO + CSD | 2–3 | 0.05 | [i7p525wr](https://wandb.ai/riasok/rlinf-ppo-csd/runs/i7p525wr) |
| PPO + CSD | 4, 7 | 0.5 | [c64tqh0p](https://wandb.ai/riasok/rlinf-ppo-csd/runs/c64tqh0p) |

## Shared setup

| Setting | Value |
|---|---|
| Benchmark | Standard LIBERO-Spatial; one joint policy over all 10 tasks |
| Base policy | π0.5 |
| Starting checkpoint | `checkpoints/RLinf-Pi05-LIBERO-SFT` |
| Prediction horizon | 10 actions |
| Execute/replan horizon | 5 actions |
| Temporal overlap | 5 actions |
| Flow sampling | Flow-SDE; 3 denoising steps; noise level 0.5 |
| Seed | 42 |
| Trainable | Action expert, action/time projections, PPO value head |
| Frozen | SigLIP + PaliGemma/Gemma VLM; no LoRA |

## PPO and rollout

| Setting | Value |
|---|---|
| Objective | Actor-critic PPO; chunk-level reward and log-probability |
| Advantages | GAE; normalized |
| Update epochs | 1 |
| Batch sizes | 2,048 global; 64 microbatch per actor |
| CUDA allocator | `expandable_segments:True` |
| Collection | 64 parallel environments × 8 rollout epochs |
| Episode/rollout horizon | 240 low-level environment steps |
| Discounting | γ=0.99; GAE λ=0.95 |
| Clipping | Policy=0.2; value=0.2; gradient norm=1.0 |
| KL / entropy | β_KL=0; entropy bonus=0 |
| Learning rates | Actor=5e-6; value head=1e-4 |
| Training budget | 100 optimizer steps |
| Checkpointing | Every 10 optimizer steps |

`runner.max_steps=100` counts optimizer updates, not simulator steps.

## CSD

\[
\mathcal{L} = \mathcal{L}_{\mathrm{PPO}} + \beta_{\mathrm{CSD}}\mathcal{L}_{\mathrm{CSD}}.
\]

For consecutive chunks, current suffix `[5:10]` is aligned with next-chunk prefix `[0:5]`. The next chunk is a detached, episode-safe teacher; CSD applies native conditional flow-matching velocity MSE to the student only. CSD microbatch size is 2.

## Artifacts

- PPO log: `logs/20260823-15:04:18-libero_spatial_ppo_pi05_gpu01_100steps`
- CSD β=0.05 log: `logs/20260823-15:04:18-libero_spatial_ppo_csd_pi05_gpu23_100steps`
- CSD β=0.5 log: `logs/20260823-15:04:16-libero_spatial_ppo_csd_b0_5_pi05_gpu47_100steps`

Config files:

- `examples/embodiment/config/libero_spatial_ppo_pi05_gpu01_100steps.yaml`
- `examples/embodiment/config/libero_spatial_ppo_csd_pi05_gpu23_100steps.yaml`
- `examples/embodiment/config/libero_spatial_ppo_csd_b0_5_pi05_gpu47_100steps.yaml`
