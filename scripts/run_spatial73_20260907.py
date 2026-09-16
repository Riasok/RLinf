# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Launch paper-style Spatial PPO with a fixed RL-only task holdout."""

import argparse
import json
import os
import subprocess

import numpy as np
import run_pirl130_20260906 as runtime

CONFIG = "libero_spatial_7train3heldout_pi05_20260907"
OUT = runtime.ROOT / "logs/20260907-spatial73"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    runtime.setup_environment()
    os.environ["LIBERO_TYPE"] = "standard"
    from hydra import compose, initialize_config_dir
    from libero.libero import benchmark
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(runtime.ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name=CONFIG)
    split = np.random.default_rng(42).permutation(10)
    train, heldout = sorted(split[:7].tolist()), sorted(split[7:].tolist())
    assert list(cfg.env.train.task_id_filter) == train
    assert list(cfg.env.eval.task_id_filter) == heldout
    assert not set(train) & set(heldout)
    assert cfg.actor.global_batch_size == 2048 and cfg.algorithm.update_epoch == 1
    assert cfg.actor.model.num_action_chunks == 5 and cfg.actor.model.num_steps == 3
    assert cfg.env.train.total_num_envs == 64 and cfg.env.train.rollout_epoch == 8
    assert cfg.env.train.max_steps_per_rollout_epoch == 240
    assert cfg.actor.optim.lr == 5e-6 and cfg.actor.optim.value_lr == 1e-4
    assert cfg.actor.optim.lr_scheduler == "constant" and cfg.runner.max_steps == 150
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    manifest = {
        "split_seed": 42,
        "train_ids": train,
        "heldout_ids": heldout,
        "tasks": {i: suite.get_task(i).name for i in range(10)},
        "caveat": "Held out only from RL; SFT may contain all ten tasks.",
        "eval": "Every10 iterations: heldout3,50 fixed trials each,150 total, ODE10/5 denoise3",
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.launch:
        print("CONFIG AND SPLIT CHECK PASSED")
        return
    for row in subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines():
        gpu, memory = map(int, row.split(","))
        if gpu in range(4) and memory > 1000:
            raise RuntimeError(f"GPU {gpu} occupied")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "split.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    runtime.OUT = OUT
    runtime.sanity_evaluation = lambda: None
    runtime.train(
        config_name=CONFIG,
        gpus=(0, 1, 2, 3),
        extra_overrides=[
            "runner.max_steps=150",
            "runner.save_interval=50",
            "runner.val_check_interval=10",
        ],
    )


if __name__ == "__main__":
    main()
