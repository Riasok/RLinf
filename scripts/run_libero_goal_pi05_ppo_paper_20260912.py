# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Launch the official pi-RL pi0.5 PPO recipe on LIBERO-Goal, training only."""

import argparse
import json
import os
import subprocess

import run_pirl130_20260906 as runtime
from hydra import compose, initialize_config_dir

CONFIG = "libero_goal_ppo_openpi_pi05_gpu03_20260912"
OUT = runtime.ROOT / "logs/20260912-libero-goal-pi05-ppo-paper-gpu03"
MODEL = runtime.ROOT / "checkpoints/RLinf-Pi05-LIBERO-SFT"
GPU_IDS = (0, 1, 2, 3)
EXPERIMENT = "pi05_ppo_libero_goal_pirl_paper_seed42"

OVERRIDES = (
    f"runner.logger.log_path={OUT / 'train'}",
    "runner.logger.project_name=rlinf-ppo-csd",
    f"runner.logger.experiment_name={EXPERIMENT}",
    "runner.logger.logger_backends=[tensorboard,wandb]",
    "+runner.logger.wandb_entity=riasok",
    "runner.max_epochs=1000",
    "runner.max_steps=-1",
    "runner.val_check_interval=-1",
    "runner.save_interval=10",
    f"actor.model.model_path={MODEL}",
    f"rollout.model.model_path={MODEL}",
    "actor.model.trainability_mask=all",
)


def validate_config():
    """Compose the effective config and reject any drift from the official recipe."""
    with initialize_config_dir(
        config_dir=str(runtime.ROOT / "examples/embodiment/config"),
        version_base="1.1",
    ):
        cfg = compose(config_name=CONFIG, overrides=list(OVERRIDES))

    placement = cfg.cluster.component_placement["actor,env,rollout"]
    assert placement == "0-3"
    assert cfg.runner.max_epochs == 1000 and cfg.runner.max_steps == -1
    assert cfg.runner.val_check_interval == -1 and cfg.runner.save_interval == 10
    assert cfg.algorithm.update_epoch == 4
    assert cfg.algorithm.loss_type == "actor_critic"
    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.kl_beta == 0.0
    assert cfg.env.train.total_num_envs == 64
    assert cfg.env.train.rollout_epoch == 8
    assert cfg.env.train.max_episode_steps == 320
    assert cfg.actor.micro_batch_size == 128
    assert cfg.actor.global_batch_size == 2048
    assert cfg.actor.model.model_type == "openpi"
    assert cfg.actor.model.num_action_chunks == 5
    assert cfg.actor.model.num_steps == 5
    assert cfg.actor.model.openpi.noise_method == "flow_sde"
    assert cfg.actor.model.openpi.noise_level == 0.3
    assert cfg.actor.model.trainability_mask == "all"
    assert not cfg.actor.model.is_lora
    assert cfg.actor.optim.lr == 5e-6
    assert cfg.actor.optim.value_lr == 1e-4
    assert cfg.actor.optim.adam_eps == 1e-8
    assert cfg.actor.optim.clip_grad == 1.0
    return cfg


def assert_ready() -> None:
    """Fail before launch if the model is absent or a requested GPU is occupied."""
    if not MODEL.is_dir():
        raise FileNotFoundError(MODEL)
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    for row in rows:
        gpu, memory = map(int, row.split(","))
        if gpu in GPU_IDS and memory > 1000:
            raise RuntimeError(f"GPU{gpu} is occupied ({memory} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    runtime.setup_environment()
    # Do not attach to another experiment's Ray head. This prevents one run's
    # normal cleanup from terminating unrelated workers, as happened previously.
    os.environ["RLINF_FORCE_LOCAL_RAY"] = "1"
    # The official config places components on `all` GPUs. Restrict visibility
    # first, making that set exactly physical GPUs 0-3 for this private cluster.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    cfg = validate_config()
    manifest = {
        "source_config": CONFIG,
        "suite": "libero_goal",
        "model": str(MODEL),
        "algorithm": "ppo",
        "policy_trainability": "all",
        "training_only": True,
        "max_epochs": cfg.runner.max_epochs,
        "max_steps": cfg.runner.max_steps,
        "save_interval": cfg.runner.save_interval,
        "gpus": list(GPU_IDS),
        "ray_isolation": "forced_local",
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.launch:
        return

    assert_ready()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runtime.OUT = OUT
    runtime.sanity_evaluation = lambda: None
    runtime.train(
        config_name=CONFIG,
        gpus=GPU_IDS,
        extra_overrides=OVERRIDES,
        min_free_gib=40,
    )


if __name__ == "__main__":
    main()
