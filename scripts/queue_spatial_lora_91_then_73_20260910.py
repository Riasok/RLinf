# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Run Spatial 7/3 action-expert LoRA PPO."""

import json
import shutil
import time

import run_pirl130_20260906 as runtime

ROOT = runtime.ROOT
JOBS = (
    (
        "spatial73",
        "libero_spatial73_ppo_lora_20260910",
        ROOT / "logs/20260910-spatial73-ppo-action-lora-r32",
        (),
    ),
)
QUEUE = ROOT / "logs/20260910-spatial-lora-queue"


def write_status(state, **extra):
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / "status.json").write_text(
        json.dumps(
            {"state": state, "time": time.strftime("%F %T %Z"), **extra},
            indent=2,
        )
    )


def validate_configs():
    runtime.setup_environment()
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg73 = compose(config_name=JOBS[0][1], overrides=list(JOBS[0][3]))
    for cfg in (cfg73,):
        assert cfg.runner.max_steps == 80
        assert cfg.runner.save_interval == cfg.runner.val_check_interval == 10
        assert cfg.actor.model.is_lora
        assert cfg.actor.model.lora_target == "action_expert"
        assert cfg.actor.model.lora_rank == 32
        assert cfg.actor.model.openpi.train_expert_only
        assert not cfg.actor.fsdp_config.use_orig_params
        assert not cfg.algorithm.csd_enabled
    assert list(cfg73.env.train.task_id_filter) == [0, 2, 3, 4, 5, 6, 7]
    assert list(cfg73.env.eval.task_id_filter) == [1, 8, 9]


def main():
    validate_configs()
    if shutil.disk_usage(ROOT).free < 180 * 2**30:
        raise RuntimeError("less than 180 GiB free; refusing checkpointed training")
    runtime.sanity_evaluation = lambda: None
    for label, config_name, output, job_overrides in JOBS:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        write_status("launching", job=label, config=config_name)
        runtime.OUT = output
        runtime.train(
            config_name=config_name,
            gpus=(0, 1, 2, 3),
            extra_overrides=(
                "runner.max_steps=80",
                "runner.save_interval=10",
                "runner.val_check_interval=10",
                *job_overrides,
            ),
        )
        stopped_path = output / "train/stopped.json"
        outcome = json.loads(stopped_path.read_text())
        if outcome.get("reason") != "finished":
            write_status("blocked_after_failure", job=label, outcome=outcome)
            return
        write_status("completed", job=label, outcome=outcome)
    write_status("finished_all")


if __name__ == "__main__":
    main()
