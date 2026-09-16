# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Launch Spatial PPO+CSD only after BC and its final evaluations complete."""

import argparse
import json
import shutil
import subprocess
import time

import run_pirl130_20260906 as runtime

ROOT = runtime.ROOT
BC = ROOT / "logs/20260908-spatial73-bc"
OUT = ROOT / "logs/20260908-spatial73-csd-b05"
CONFIG = "libero_spatial73_ppo_csd_b05_20260908"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume-queue", action="store_true")
    args = parser.parse_args()
    runtime.setup_environment()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name=CONFIG)
    assert cfg.runner.max_steps == 80
    assert cfg.runner.save_interval == cfg.runner.val_check_interval == 10
    assert cfg.algorithm.csd_enabled and cfg.algorithm.csd_beta == 0.5
    assert cfg.algorithm.csd_gate_mode == "constant"
    assert list(cfg.env.train.task_id_filter) == [0, 2, 3, 4, 5, 6, 7]
    assert list(cfg.env.eval.task_id_filter) == [1, 8, 9]
    assert cfg.actor.optim.lr == 5e-6 and cfg.actor.global_batch_size == 2048
    assert cfg.actor.model.openpi.train_expert_only
    assert cfg.actor.model.model_path.endswith("RLinf-Pi05-LIBERO-SFT")
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
    print(
        "CONFIG CHECK PASSED: 80 PPO+CSD iterations; beta0.5; save/eval10; GPUs0-3",
        flush=True,
    )
    if not args.launch:
        return
    if args.resume_queue:
        assert not (OUT / "train").exists(), (
            "Cannot rewrite an already launched training run"
        )
        previous = read_json(OUT / "queue_status.json") or {}
        assert previous.get("state", "").startswith("waiting_"), previous
    OUT.mkdir(parents=True, exist_ok=args.resume_queue)
    OmegaConf.save(cfg, OUT / "resolved.yaml", resolve=True)
    (OUT / "plan.json").write_text(
        json.dumps(
            {
                "prerequisite": str(BC),
                "gpus": [0, 1, 2, 3],
                "initialization": str(cfg.actor.model.model_path),
                "max_steps": 80,
                "save_interval": 10,
                "eval_interval": 10,
                "csd_beta": 0.5,
                "csd_gate_mode": "constant",
                "note": "Fresh PPO+CSD from original few-shot model, not BC continuation",
            },
            indent=2,
        )
    )
    shutil.copy2(ROOT / "logs/20260907-spatial73/split.json", OUT / "split.json")

    def status(state, **extra):
        (OUT / "queue_status.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "time": time.strftime("%F %T %Z"),
                    **extra,
                },
                indent=2,
            )
        )

    while True:
        failure = read_json(BC / "failure.json")
        if failure:
            status("blocked_BC_failed", failure=failure)
            return
        state = read_json(BC / "status.json") or {}
        if state.get("state") != "finished" or state.get("iteration") != 150:
            status("waiting_for_BC_and_final_evals", bc_status=state)
            time.sleep(30)
            continue
        for split, expected in (("heldout", 150), ("seen", 350)):
            result = read_json(BC / "eval/iteration_150" / split / "result.json")
            if (
                not result
                or result.get("metrics", {}).get("eval/num_trajectories") != expected
            ):
                status("blocked_missing_final_eval", split=split)
                return
        if not (
            BC / "checkpoints/global_step_150/actor/model_state_dict/full_weights.pt"
        ).is_file():
            status("blocked_missing_BC_final_checkpoint")
            return
        usage = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        busy = [
            int(row.split(",")[0])
            for row in usage.splitlines()
            if int(row.split(",")[0]) < 4 and int(row.split(",")[1]) > 1000
        ]
        if busy:
            status("waiting_for_GPU_release", busy=busy)
            time.sleep(30)
            continue
        if shutil.disk_usage(ROOT).free < 235 * 2**30:
            status("waiting_for_checkpoint_disk_budget", required_free_GiB=235)
            time.sleep(30)
            continue
        break
    status("launching")
    runtime.OUT = OUT
    runtime.sanity_evaluation = lambda: (
        None
    )  # The PPO runner evaluates at step80 itself.
    runtime.train(
        config_name=CONFIG,
        gpus=(0, 1, 2, 3),
        extra_overrides=[
            "runner.max_steps=80",
            "runner.save_interval=10",
            "runner.val_check_interval=10",
        ],
    )
    ended = read_json(OUT / "train/stopped.json") or {}
    status(
        "finished" if ended.get("reason") == "finished" else "stopped", outcome=ended
    )


if __name__ == "__main__":
    main()
