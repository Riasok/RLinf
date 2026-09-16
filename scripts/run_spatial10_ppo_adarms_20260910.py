# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Launch all-task LIBERO-Spatial PPO with an AdaRMS-only policy body."""

import argparse
import json
import os
import subprocess

import run_pirl130_20260906 as runtime
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG = "libero_spatial10_ppo_adarms_only_20260910"
OUT = runtime.ROOT / "logs/20260910-spatial10-ppo-adarms-only"
GPU_IDS = (4, 5, 6, 7)


def validate_config():
    """Compose the run config and fail closed on protocol drift."""
    with initialize_config_dir(
        config_dir=str(runtime.ROOT / "examples/embodiment/config"),
        version_base="1.1",
    ):
        cfg = compose(config_name=CONFIG)
    expected_tasks = list(range(10))
    assert cfg.cluster.component_placement["actor,env,rollout"] == "4-7"
    assert list(cfg.env.train.task_id_filter) == expected_tasks
    assert list(cfg.env.eval.task_id_filter) == expected_tasks
    assert cfg.env.eval.num_trials_per_task == 50
    assert cfg.env.eval.rollout_epoch == 25
    assert cfg.actor.model.trainability_mask == "adarms_only"
    assert not cfg.actor.model.is_lora
    assert not cfg.actor.fsdp_config.use_orig_params
    assert cfg.actor.model.openpi.train_expert_only
    assert cfg.actor.model.add_value_head
    assert cfg.algorithm.loss_type == "actor_critic"
    assert not cfg.algorithm.csd_enabled
    assert cfg.runner.max_steps == 100
    assert cfg.actor.optim.lr == 5e-6
    assert cfg.actor.optim.value_lr == 1e-4
    assert cfg.actor.optim.clip_grad == 1.0
    return cfg


def assert_gpus_free() -> None:
    """Refuse to collide with any process already using GPUs 4-7."""
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
    """Validate and optionally launch the persistent PPO driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    runtime.setup_environment()
    # Share the existing eight-GPU Ray cluster while placement isolates 4-7.
    os.environ["RLINF_FORCE_LOCAL_RAY"] = "0"
    os.environ["LIBERO_TYPE"] = "standard"
    cfg = validate_config()
    manifest = {
        "task_suite": "libero_spatial",
        "train_task_ids": list(range(10)),
        "eval_task_ids": list(range(10)),
        "eval_trials_per_task": 50,
        "policy_trainability": "adarms_only",
        "adarms_parameter_tensors": 74,
        "adarms_parameters": 116_505_600,
        "value_head_parameters": 2_754_561,
        "steps": 100,
        "gpus": list(GPU_IDS),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.launch:
        return
    assert_gpus_free()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    runtime.OUT = OUT
    runtime.sanity_evaluation = lambda: None
    runtime.train(
        config_name=CONFIG,
        gpus=GPU_IDS,
        extra_overrides=(
            "runner.max_steps=100",
            "runner.save_interval=10",
            "runner.val_check_interval=10",
        ),
    )


if __name__ == "__main__":
    main()
