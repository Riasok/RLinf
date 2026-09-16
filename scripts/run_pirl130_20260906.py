# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Persistent, separately scoped PPO-130 training and official Plus evaluations."""

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/20260906-pirl130"
RELEASES = {
    "ppo": (
        "RLinf-Pi05-PPO-LIBERO-130",
        "df129b1bf47201ab9490d6645dfcc08133cbd689",
        "0-1",
    ),
    "sft": (
        "RLinf-Pi05-LIBERO-130-fullshot-SFT",
        "6222623f635769bfc73c9472e29fab9b7fd8e027",
        "2-3",
    ),
}


def setup_environment():
    from dotenv import load_dotenv

    load_dotenv("/data/minjaeoh/.env", override=False)
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "RAY_ADDRESS",
        "LIBERO_SUFFIX",
        "LIBERO_PERTURBATION",
    ):
        os.environ.pop(key, None)
    os.environ.update(
        {
            "REPO_PATH": str(ROOT),
            "EMBODIED_PATH": str(ROOT / "examples/embodiment"),
            "PYTHONPATH": str(ROOT),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MAGICK_HOME": str(ROOT / ".native-deps"),
            "LD_LIBRARY_PATH": str(ROOT / ".native-deps/lib")
            + ":"
            + os.environ.get("LD_LIBRARY_PATH", ""),
            "RLINF_FORCE_LOCAL_RAY": "1",
            "RAY_local_fs_capacity_threshold": "0.995",
            "RLINF_LOCAL_RAY_NUM_CPUS": "64",
            "RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES": "16000000000",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "WANDB_MODE": "online",
        }
    )


def train(
    extra_overrides=(),
    gpus=(4, 5, 6, 7),
    config_name="libero_130_ppo_pi05_gpu47_20260906",
    min_free_gib=80,
):
    dest = OUT / "train"
    dest.mkdir(parents=True, exist_ok=True)
    # Short socket paths; bulky spill objects go on the data volume.
    runtime = tempfile.mkdtemp(prefix="p130-", dir="/tmp")
    spill = dest / "ray_spill"
    spill.mkdir(exist_ok=True)
    os.environ.update(
        {
            "RAY_TMPDIR": runtime,
            "TMPDIR": runtime,
            "LIBERO_TYPE": "standard",
            "RAY_object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": str(spill)}}
            ),
        }
    )
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "examples/embodiment/train_embodied_agent.py",
        "--config-name",
        config_name,
        "runner.max_steps=10",
        "runner.save_interval=10",
        "runner.val_check_interval=-1",
        f"runner.logger.log_path={dest}",
        *extra_overrides,
    ]
    (dest / "command.json").write_text(json.dumps(cmd, indent=2))
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    owned = {}
    reason = "finished"
    with (dest / "launcher.log").open("x") as log:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        owned[proc.pid] = psutil.Process(proc.pid)
        try:
            while True:
                for process in list(owned.values()):
                    try:
                        owned.update(
                            {p.pid: p for p in process.children(recursive=True)}
                        )
                    except psutil.Error:
                        pass
                free = shutil.disk_usage(ROOT).free / 2**30
                status = {
                    "time": time.strftime("%F %T %Z"),
                    "pid": proc.pid,
                    "exit_code": proc.poll(),
                    "free_GiB": free,
                    "gpus": list(gpus),
                }
                (dest / "status.json").write_text(json.dumps(status, indent=2))
                if proc.poll() is not None:
                    reason = "finished" if proc.returncode == 0 else "worker_failure"
                    break
                if (
                    stopped
                    or free < min_free_gib
                    or shutil.disk_usage("/tmp").free < 5 * 2**30
                ):
                    reason = "requested_stop" if stopped else "disk_guard"
                    break
                time.sleep(20)
        finally:
            live = []
            for process in owned.values():
                try:
                    if process.is_running():
                        process.terminate()
                        live.append(process)
                except psutil.Error:
                    pass
            _, alive = psutil.wait_procs(live, timeout=20)
            for process in alive:
                try:
                    process.kill()
                except psutil.Error:
                    pass
            (dest / "stopped.json").write_text(
                json.dumps(
                    {"reason": reason, "time": time.strftime("%F %T %Z")}, indent=2
                )
            )
    if reason == "finished":
        sanity_evaluation()


def sanity_evaluation():
    """Evaluate the saved step10 policy only after successful trainer cleanup."""
    import run_csd_diagnosis as diag

    checkpoint = (
        OUT
        / "train/pi05_ppo_libero130_longrecipe_seed42_20260906"
        / "checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Completed training has no step10 weights: {checkpoint}"
        )
    diag.OUT = OUT / "step10_sanity"
    diag.OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "checkpoint": str(checkpoint),
        "trials_per_suite": 500,
        "prediction_horizon": 10,
        "execution_horizon": 10,
        "denoising_steps": 5,
        "mode": "standard, ODE, fixed reset trials, seed42",
        "lanes": {
            "4-5": ["libero_spatial", "libero_object"],
            "6-7": ["libero_goal", "libero_10"],
        },
    }
    (diag.OUT / "plan.json").write_text(json.dumps(plan, indent=2))

    def lane(gpus, suites):
        return [
            diag.evaluate(
                (suite, suite, 10, 5, str(checkpoint)),
                gpus,
                trials=50,
                envs=20,
                epochs=25,
                expected=500,
                timeout=14400,
            )
            for suite in suites
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(lane, gpus, suites) for gpus, suites in plan["lanes"].items()
        ]
        results = [result for future in futures for result in future.result()]
    (diag.OUT / "results.json").write_text(json.dumps(results, indent=2))


def evaluate(arm):
    import run_csd_diagnosis as diag
    from huggingface_hub import snapshot_download

    name, revision, gpus = RELEASES[arm]
    model = ROOT / "checkpoints" / name
    snapshot_download(
        "RLinf/" + name,
        revision=revision,
        local_dir=model,
        allow_patterns=["model.safetensors", "physical-intelligence/libero/*"],
    )
    diag.OUT = OUT / "plus_spatial"
    diag.OUT.mkdir(parents=True, exist_ok=True)
    (diag.OUT / f"{arm}_provenance.json").write_text(
        json.dumps(
            {
                "repo": "RLinf/" + name,
                "revision": revision,
                "gpus": gpus,
                "protocol": "All 2402 installed Plus-Spatial variants, first fixed trial each, seed42, horizon240, ODE, generate10/execute5, denoise3; no suffix remapping",
            },
            indent=2,
        )
    )
    diag.evaluate(
        (arm, "libero_spatial", 5, 3, ""),
        gpus,
        model_path=model,
        mode="plus",
        trials=1,
        envs=40,
        epochs=61,
        expected=2402,
        timeout=43200,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["train", "ppo", "sft"])
    args = parser.parse_args()
    setup_environment()
    train() if args.job == "train" else evaluate(args.job)
