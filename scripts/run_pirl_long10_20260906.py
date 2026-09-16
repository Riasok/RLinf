# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Ten iterations of paper-style Long PPO, followed by a 500-episode Long eval."""

import argparse
import json
import subprocess

import run_csd_diagnosis as diag
import run_pirl130_20260906 as runtime

OUT = runtime.ROOT / "logs/20260906-pirl-long10"
EXPERIMENT = "pi05_ppo_long_paper_seed42_10steps_20260906"
GPUS = "4-7"
OVERRIDES = [
    "env.train.task_suite_name=libero_10",
    "env.eval.task_suite_name=libero_10",
    "rollout.unnorm_key=libero_10",
    f"runner.logger.experiment_name={EXPERIMENT}",
]


def evaluate_long():
    """Called only after successful checkpoint save and trainer cleanup."""
    checkpoint = (
        OUT
        / "train"
        / EXPERIMENT
        / "checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    diag.OUT = OUT / "step10_eval"
    diag.OUT.mkdir(parents=True, exist_ok=True)
    (diag.OUT / "plan.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "suite": "libero_10",
                "gpus": GPUS,
                "episodes": 500,
                "trials_per_task": 50,
                "envs": 20,
                "epochs": 25,
                "prediction_horizon": 10,
                "execution_horizon": 10,
                "denoising_steps": 5,
                "episode_horizon": 520,
                "seed": 42,
                "sampling": "ODE, fixed reset trials",
            },
            indent=2,
        )
    )
    result = diag.evaluate(
        ("long_step10", "libero_10", 10, 5, str(checkpoint)),
        GPUS,
        trials=50,
        envs=20,
        epochs=25,
        expected=500,
        timeout=14400,
    )
    (diag.OUT / "results.json").write_text(json.dumps(result, indent=2))


def main():
    global OUT, EXPERIMENT, GPUS
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", choices=["0-3", "4-7"], default="4-7")
    args = parser.parse_args()
    GPUS = args.gpus
    lo, hi = map(int, GPUS.split("-"))
    usage = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for row in usage.splitlines():
        gpu, memory = map(int, row.split(","))
        if lo <= gpu <= hi and memory > 1000:
            raise RuntimeError(f"GPU {gpu} occupied: {memory} MiB")
    if GPUS == "0-3":
        OUT = runtime.ROOT / "logs/20260906-pirl-long10-gpu03"
        EXPERIMENT += "_gpu03"
    runtime.setup_environment()
    runtime.OUT = OUT
    runtime.sanity_evaluation = evaluate_long
    runtime.train(
        extra_overrides=[
            *OVERRIDES,
            f"runner.logger.experiment_name={EXPERIMENT}",
            f"cluster.component_placement={{actor\\,env\\,rollout:{GPUS}}}",
        ],
        gpus=tuple(range(lo, hi + 1)),
    )


if __name__ == "__main__":
    main()
