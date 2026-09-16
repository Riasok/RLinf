# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Evaluate local step80 PPO/CSD and few-shot SFT on all Plus-Spatial variants."""

import argparse
import json
import subprocess
import time

import run_csd_diagnosis as diag
import run_pirl130_20260906 as runtime

OUT = runtime.ROOT / "logs/20260906-local-plus-spatial"
LANES = {"0-1": ["ppo", "base"], "2-3": ["csd"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", choices=LANES, required=True)
    args = parser.parse_args()
    runtime.setup_environment()
    diag.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = map(int, args.gpus.split("-"))
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
    jobs = []
    for arm in LANES[args.gpus]:
        checkpoint = (
            diag.CHECKPOINTS[arm]
            / "global_step_80/actor/model_state_dict/full_weights.pt"
            if arm != "base"
            else None
        )
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        jobs.append(
            (
                f"{arm}_step80" if checkpoint else "fewshot_sft",
                "libero_spatial",
                5,
                3,
                str(checkpoint) if checkpoint else "",
            )
        )
    status = OUT / f"lane_{args.gpus}.json"
    results = []
    try:
        for job in jobs:
            status.write_text(
                json.dumps(
                    {
                        "state": "running",
                        "job": job,
                        "gpus": args.gpus,
                        "time": time.strftime("%F %T %Z"),
                    },
                    indent=2,
                )
            )
            results.append(
                diag.evaluate(
                    job,
                    args.gpus,
                    mode="plus",
                    trials=1,
                    envs=40,
                    epochs=61,
                    expected=2402,
                    timeout=43200,
                )
            )
        (OUT / f"results_{args.gpus}.json").write_text(json.dumps(results, indent=2))
        status.write_text(json.dumps({"state": "finished", "gpus": args.gpus}))
    except Exception as exc:
        status.write_text(json.dumps({"state": "failed", "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
