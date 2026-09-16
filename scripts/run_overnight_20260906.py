# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded overnight, single-suite PPO execution-length control experiment."""

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
OUT = ROOT / "logs/20260906-overnight-ppo-controls"


def command(k, gpus):
    """Keep reference training settings identical except execution length."""
    dest = OUT / f"long_k{k}"
    return [
        str(ROOT / ".venv/bin/python"),
        "examples/embodiment/train_embodied_agent.py",
        "--config-path",
        str(ROOT / "examples/embodiment/config"),
        "--config-name",
        "libero_10_overnight_controls",
        f"runner.logger.log_path={dest}",
        f"runner.logger.experiment_name=overnight_long_k{k}_seed42",
        "runner.logger.project_name=rlinf-ppo-csd",
        "runner.max_steps=60",
        "runner.save_interval=10",
        "runner.val_check_interval=5",
        f"actor.model.model_path={ROOT / 'checkpoints/RLinf-Pi05-LIBERO-SFT'}",
        f"rollout.model.model_path={ROOT / 'checkpoints/RLinf-Pi05-LIBERO-SFT'}",
        f"actor.model.num_action_chunks={k}",
        "+actor.model.openpi.action_horizon=10",
        "actor.micro_batch_size=64",
        "env.train.video_cfg.save_video=false",
        "env.eval.video_cfg.save_video=false",
        "env.eval.total_num_envs=20",
        "env.eval.rollout_epoch=5",
        "env.eval.max_episode_steps=520",
        "env.eval.max_steps_per_rollout_epoch=520",
        "+env.eval.num_trials_per_task=10",
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + 12 * 3600
    lanes = []
    owned = {}
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for k, gpus in ((10, "4-5"), (5, "6-7")):
            dest = OUT / f"long_k{k}"
            dest.mkdir(exist_ok=True)
            env = os.environ.copy()
            for key in ("CUDA_VISIBLE_DEVICES", "RAY_ADDRESS"):
                env.pop(key, None)
            runtime = tempfile.mkdtemp(prefix=f"overnight{k}-", dir="/tmp")
            env.update(
                {
                    "OVERNIGHT_GPU_RANGE": gpus,
                    "RLINF_FORCE_LOCAL_RAY": "1",
                    "RLINF_LOCAL_RAY_NUM_CPUS": "32",
                    "RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES": "16000000000",
                    "REPO_PATH": str(ROOT),
                    "EMBODIED_PATH": str(ROOT / "examples/embodiment"),
                    "PYTHONPATH": str(ROOT),
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "LIBERO_TYPE": "standard",
                    "LIBERO_SUFFIX": "",
                    "LIBERO_PERTURBATION": "",
                    "MAGICK_HOME": str(ROOT / ".native-deps"),
                    "LD_LIBRARY_PATH": str(ROOT / ".native-deps/lib")
                    + ":"
                    + env.get("LD_LIBRARY_PATH", ""),
                    "RAY_TMPDIR": runtime,
                    "TMPDIR": runtime,
                    "PYTHONUNBUFFERED": "1",
                }
            )
            cmd = command(k, gpus)
            (dest / "command.json").write_text(json.dumps(cmd, indent=2))
            log = (dest / "launcher.log").open("x")
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            lanes.append((k, gpus, proc, log))
            owned[proc.pid] = psutil.Process(proc.pid)
            print(f"START K{k} GPUs {gpus} PID {proc.pid}", flush=True)
        (OUT / "plan.json").write_text(
            json.dumps(
                {
                    "started": started,
                    "deadline": deadline,
                    "deadline_local": time.strftime(
                        "%F %T %Z", time.localtime(deadline)
                    ),
                    "purpose": "Single-suite reference-style PPO K10 versus matched K5, no CSD",
                    "seed": 42,
                    "denoise_steps": 5,
                    "update_epochs": 4,
                    "train_horizon": 480,
                    "eval_horizon": 520,
                    "eval_trials": 100,
                    "eval_interval": 5,
                    "save_interval": 10,
                    "max_iterations": 60,
                    "minimum_free_GiB": 60,
                },
                indent=2,
            )
        )
        while True:
            for process in list(owned.values()):
                try:
                    for child in process.children(recursive=True):
                        owned[child.pid] = child
                except psutil.Error:
                    pass
            free = shutil.disk_usage(ROOT).free / 2**30
            status = {
                "updated": time.strftime("%F %T %Z"),
                "remaining_hours": max(0, (deadline - time.time()) / 3600),
                "free_GiB": free,
                "lanes": [
                    {"k": k, "gpus": g, "pid": p.pid, "exit_code": p.poll()}
                    for k, g, p, _ in lanes
                ],
            }
            (OUT / "status.json").write_text(json.dumps(status, indent=2))
            if (
                stopping
                or time.time() >= deadline
                or free < 60
                or all(p.poll() is not None for _, _, p, _ in lanes)
            ):
                break
            time.sleep(30)
    finally:
        # Only descendants of our two drivers: never global Ray/GPU process kills.
        live = []
        for process in owned.values():
            try:
                if process.is_running():
                    process.terminate()
                    live.append(process)
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(live, timeout=30)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        for _, _, _, log in lanes:
            log.close()
        (OUT / "stopped.json").write_text(
            json.dumps(
                {
                    "time": time.strftime("%F %T %Z"),
                    "deadline_reached": time.time() >= deadline,
                    "signal_requested": stopping,
                    "free_GiB": shutil.disk_usage(ROOT).free / 2**30,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
