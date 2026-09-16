# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Check the official ManiSkill recipe; launch only with explicit --launch."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import psutil
import run_pirl130_20260906 as runtime

ROOT = runtime.ROOT
MODEL = ROOT / "checkpoints/RLinf-Pi05-ManiSkill-25Main-SFT"
PYTHON = ROOT / ".venv-maniskill-prep/bin/python"
CONFIG = "maniskill_ppo_openpi_pi05"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--gpus", default="4-7")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--memory-safe", action="store_true")
    args = parser.parse_args()
    runtime.setup_environment()
    os.environ["MS_ASSET_DIR"] = "/data/minjaeoh/.maniskill"
    os.environ["PYTHONPATH"] = str(ROOT)
    output = ROOT / (
        "logs/maniskill_pi05_memsafe"
        if args.memory_safe
        else "logs/maniskill_pi05_next"
    )
    overrides = [
        f"cluster.component_placement={{actor\\,env\\,rollout:{args.gpus}}}",
        f"runner.max_steps={args.steps}",
        f"runner.logger.log_path={output}",
        "runner.logger.project_name=rlinf-ppo-csd",
        "runner.logger.experiment_name=pi05_maniskill_official_flownoise",
        "runner.logger.logger_backends=[tensorboard,wandb]",
        f"actor.model.model_path={MODEL}",
        f"rollout.model.model_path={MODEL}",
        "env.train.video_cfg.save_video=false",
        "env.eval.video_cfg.save_video=false",
        # Upstream YAML spells None as a string; ManiSkill requires null.
        "env.train.init_params.control_mode=null",
        "env.eval.init_params.control_mode=null",
    ]
    if args.memory_safe:
        overrides.extend(
            [
                "actor.micro_batch_size=8",
                "actor.enable_offload=true",
                "rollout.enable_offload=true",
                "runner.logger.experiment_name=pi05_maniskill_flownoise_memsafe",
            ]
        )
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from safetensors import safe_open

    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        config = compose(config_name=CONFIG, overrides=overrides)
    assert config.env.train.init_params.id == "PutOnPlateInScene25Main-v3"
    assert config.actor.global_batch_size == 5120
    assert config.actor.model.openpi.noise_method == "flow_noise"
    assert config.actor.model.openpi.action_horizon == 8
    assert config.actor.model.num_action_chunks == 5
    assert config.actor.optim.lr == 7.91e-6
    if args.memory_safe:
        assert config.actor.micro_batch_size == 8
        assert config.actor.enable_offload and config.rollout.enable_offload
        assert config.env.train.total_num_envs == 320
    with safe_open(str(MODEL / "model.safetensors"), framework="pt") as weights:
        print("Checkpoint tensors:", len(list(weights.keys())))
    assert (MODEL / "physical-intelligence/maniskill/norm_stats.json").is_file()
    assert (ROOT / "rlinf/envs/maniskill/assets").is_dir()
    assert Path("/data/minjaeoh/.maniskill/data/robots/widowx/wx250s.urdf").is_file()
    assert Path(
        "/data/minjaeoh/.maniskill/data/tasks/bridge_v2_real2sim_dataset"
    ).is_dir()
    # Register tasks without constructing a GPU simulation or allocating a model.
    import gymnasium as gym

    import rlinf.envs.maniskill  # noqa: F401

    print("Environment:", gym.spec("PutOnPlateInScene25Main-v3").id)
    cmd = [
        str(PYTHON),
        "examples/embodiment/train_embodied_agent.py",
        "--config-name",
        CONFIG,
        *overrides,
    ]
    print("Prepared command:", json.dumps(cmd))
    print("CHECK PASSED: CPU imports/config/weights only; GPU smoke test pending.")
    if not args.launch:
        return
    usage = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    lo, hi = map(int, args.gpus.split("-"))
    for row in usage.splitlines():
        gpu, memory = map(int, row.split(","))
        if lo <= gpu <= hi and memory > 1000:
            raise RuntimeError(f"GPU {gpu} is occupied; refusing to launch")
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    (output / "command.json").write_text(json.dumps(cmd, indent=2))
    ray_temp = tempfile.mkdtemp(prefix="msnext-", dir="/tmp")
    spill = output / "ray_spill"
    spill.mkdir()
    os.environ.update(
        {
            "RAY_TMPDIR": ray_temp,
            "TMPDIR": ray_temp,
            "RAY_object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": str(spill)}}
            ),
        }
    )
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    owned = {}
    reason = "supervisor_error"
    with (output / "launcher.log").open("x") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
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
                ram = psutil.virtual_memory().available / 2**30
                (output / "status.json").write_text(
                    json.dumps(
                        {
                            "time": time.strftime("%F %T %Z"),
                            "pid": proc.pid,
                            "exit_code": proc.poll(),
                            "free_GiB": free,
                            "available_ram_GiB": ram,
                            "gpus": list(range(lo, hi + 1)),
                        },
                        indent=2,
                    )
                )
                if proc.poll() is not None:
                    reason = "finished" if proc.returncode == 0 else "worker_failure"
                    break
                if stopped:
                    reason = "requested_stop"
                    break
                if free < 80 or shutil.disk_usage("/tmp").free < 5 * 2**30 or ram < 40:
                    reason = "resource_guard"
                    break
                time.sleep(10)
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
            (output / "stopped.json").write_text(
                json.dumps(
                    {"reason": reason, "time": time.strftime("%F %T %Z")}, indent=2
                )
            )


if __name__ == "__main__":
    main()
