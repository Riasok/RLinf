# Local Plus-Spatial evaluation

Requested three-way comparison: our 40-task PPO step80 (CSD diagnostics only,
no CSD training loss), our 40-task PPO+CSD beta1 step80, and their shared
RLinf-Pi05-LIBERO-SFT few-shot initialization. Not the official PPO130/fullshot
checkpoints. Exact checkpoint locations are in each command/result artifact.

Protocol matches the completed official HF evaluations: all2402 installed
Plus-Spatial variants, one fixed trial each, seed42, horizon240, generation10,
execution5, denoise3, ODE. Forty vector environments per two-GPU lane and61
rollout epochs; the evaluation deduplicates padding and verifies2402 trajectories.
No500-environment launch. Results are stored in logs/20260906-local-plus-spatial.

GPU0-1: PPO step80 then few-shot SFT. GPU2-3: PPO+CSD step80.
Launch script: scripts/run_local_plus_spatial_20260906.py --gpus0-1 or --gpus2-3
(insert a space between --gpus and its value). Each lane stops on failure.
GPUs4-7 are not used. ManiSkill had independently failed with GPU OOM at18:07
and was cleaned up by its supervisor before these evaluations were requested.
