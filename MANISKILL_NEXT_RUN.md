# ManiSkill pi0.5 preparation — no training launched

The active Long run and official LIBERO-Plus evaluations are independent.

## Official sources

- Latest pi_RL paper: https://arxiv.org/html/2510.25889 Appendix J Table12.
- Upstream runnable recipe:
  https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml
- Local package pin from requirements/install.sh: haosulab/ManiSkill v3.0.0b22.
- SFT: RLinf/RLinf-Pi05-ManiSkill-25Main-SFT,
  revision eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030.
- Assets: RLinf/maniskill_assets,
  revision c23fc1880ed7861686d4f995360101eaee4d18a0.

## Paper versus runnable upstream (pi0.5 multitask)

| Parameter | Paper Table12 | Upstream YAML |
|---|---|---|
| RL iterations |150|1000|
| Global batch |5120|5120|
| PPO update epochs |5|5|
| Actor / critic LR |7.91e-6 / 1.55e-4|same|
| Scheduler |off|none configured|
| gamma / GAE lambda / PPO clip |.99 / .95 / .2|same|
| Parallel envs / rollout epochs |320 /1|same|
| Interaction horizon |48|80|
| Prediction / execution horizon |8 /5|same|
| Denoise steps |4|4|
| Sampler |Flow-SDE or Flow-Noise|Flow-Noise|
| Flow-SDE noise |.5|model default .5, inactive for Flow-Noise|
| Flow-Noise bounds, labeled log-var in paper |.04–.10|model default .08–.16|
| Flow-Noise entropy coefficient |.005|.005|

Important: YAML noise_params=[.16,.12,200] is an annealing parameter, NOT the
Flow-Noise bounds. The latter are noise_logvar_range; source comments label
them min_std/max_std. Do not claim these two settings are interchangeable.
The runner integer-divides interaction horizon by execution horizon, so setting
48 with execution5 gives9 chunks/45 physical steps, not exactly48. This requires
an explicit reproduction choice before calling a modified config paper-exact.

Additional code settings: seed0, microbatch32, gradient clip1, Adam(.9,.95),
eps1e-8, frozen VLM, trainable expert + VLM critic, joint_logprob=true,
bootstrap_type=always, chunk-level reward/logprob/entropy, no CSD/LoRA,
one camera, WidowX Bridge policy, GPU simulation, use_multiple_plates=false.
Environment: PutOnPlateInScene25Main-v3. Eval every10, save every50.

SIMPLER single-task pi0.5 differs: actor5.6e-6, critic1.1e-4, batch2560,
256envs,4 PPO passes;40 iterations for Eggplant/Carrot/Spoon,70 for Cube.
Prediction8/execution5,4 denoise steps,48 interactions,1 rollout epoch.
These are NOT drop-in alternatives for the25Main model/task without matching
single-task environment and initialization.

## Prepared entrypoint

Isolated environment: .venv-maniskill-prep. It reuses existing packages read-only
via a .pth and installs ManiSkill/dependency additions only into the new venv.
Default action is CPU-only readiness checking:

```
.venv-maniskill-prep/bin/python scripts/prepare_maniskill_pi05.py
```

When explicitly requested, --launch uses the official YAML on
GPUs4–7; --steps can bound it. It checks GPUs are idle before launching.
Checkpoint paths, GPU placement, local logs/W&B and video saving are changed
from upstream defaults. It does not silently switch from Flow-Noise to Flow-SDE.
The launcher also overrides both control_mode fields to YAML null: upstream's
bare None is parsed as a string and fails ManiSkill's supported-mode assertion.
Any paper-aligned150/48/Flow-SDE experiment should be named a separate profile.
GPU simulation/model-loading smoke test is deferred until GPUs are available;
passing the CPU check alone is not proof of end-to-end readiness.

Preparation completed:7.47GB SFT weights plus normalization stats, all140 RLinf
task asset files, bridge_v2_real2sim scene assets and WidowX250S v0.2.0 assets.
MS_ASSET_DIR is pinned to /data/minjaeoh/.maniskill. ManiSkill v3.0.0b22 resolves
to commit33967b9e3ead1f841eec57cc9f31d0d8b8cf0907. Isolated additions include
pytorch-kinematics0.7.6, fast-kinematics0.2.2, mplib0.1.1 and their required
import helpers. No package changes to the active .venv.
Validation passed: Hydra composition and core recipe assertions;812 checkpoint
tensors; normalization and robot/scene asset paths; imports and registration of
PutOnPlateInScene25Main-v3; Ruff. No GPU simulator/model smoke test or training.

## Authorized launch on 2026-09-06

Long training was stopped by request (requested_stop at17:58:47 KST); its
automatic step10 evaluation was canceled. Both official Plus-Spatial evaluations
finished: PPO13045.253956%, fullshot SFT54.704416%,2402 trials each.

GPU smoke passed on physical GPU4: exact25Main environment with two parallel
envs, reset and three sampled-action steps, GPU physics and RGB rendering.
The smoke required the null control_mode correction above. PhysX GPU library
was downloaded by SAPIEN's own enable_gpu bootstrap.

Launched in tmux maniskill-ppo with --launch --gpus4-7 --steps1000.
Logs/status/resolved config: logs/maniskill_pi05_next. Supervisor tracks only
its descendants, cleans up on signals/failure, and stops if data free space<80GiB,
root/tmp free space<5GiB, or available system RAM<40GiB. Ray spill uses data
volume. These guards can stop the run before its configured iteration budget;
they never delete checkpoints. W&B project is rlinf-ppo-csd, online mode.

## Memory-safe retry

The first attempt finished rollout but failed in actor.run_training on its first
microbatch. GPU consumers were approximately25GiB simulator,21GiB rollout,
33GiB actor, exceeding the80GiB capacity. No completed PPO update was reported.

The authorized retry uses --memory-safe: actor microbatch8 instead of32, actor
and rollout CPU offload between phases. Global batch5120,320envs,5PPO passes,
learning rates, precision, horizons, noise method, and1000-iteration budget stay
unchanged. Gradient accumulation preserves the effective batch (not guaranteed
bitwise identical due to microbatch ordering). No unsupported gradient checkpointing.
Fresh output logs/maniskill_pi05_memsafe preserves the failed attempt's evidence.
The user canceled both local Plus-Spatial lanes and the queued few-shot baseline;
their partial artifacts are retained. GPUs0-3 are to remain idle.
