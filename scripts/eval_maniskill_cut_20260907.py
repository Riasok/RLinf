# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Matched SFT and available ManiSkill PPO checkpoint evaluations on GPU4-7."""

import json
import os
import signal
import subprocess
import tempfile
import time

import psutil
import run_pirl130_20260906 as runtime
from omegaconf import OmegaConf

ROOT = runtime.ROOT
OUT = ROOT / "logs/20260907-maniskill-cut-eval"
SOURCE = ROOT / "logs/maniskill_pi05_memsafe"
CKPT = (
    SOURCE
    / "pi05_maniskill_flownoise_memsafe/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
)
PYTHON = ROOT / ".venv-maniskill-prep/bin/python"
VARIANT = None


def main():
    runtime.setup_environment()
    os.environ["MS_ASSET_DIR"] = "/data/minjaeoh/.maniskill"
    assert CKPT.is_file()
    for row in subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines():
        gpu, memory = map(int, row.split(","))
        if gpu >= 4 and memory > 1000:
            raise RuntimeError(f"GPU{gpu} occupied")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "plan.json").write_text(
        json.dumps(
            {
                "arms": ["sft", "step50_best_and_latest_saved"],
                "unavailable": "Step60 peak and last in-memory iteration were not saved",
                "protocol": "Matched original training eval:320 envs,1 epoch,horizon80,gen8/ex5,denoise4,seed0,ODE,fixed reset IDs",
                "gpus": [4, 5, 6, 7],
                "variant_override": VARIANT,
            },
            indent=2,
        )
    )
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    results = []
    for arm, ckpt in [("sft", None), ("step50_best_and_latest_saved", str(CKPT))]:
        if stopped:
            break
        dest = OUT / arm
        dest.mkdir()
        cfg = OmegaConf.load(SOURCE / "resolved.yaml")
        if VARIANT is not None:
            cfg.env.eval.init_params.id = VARIANT[0]
            cfg.env.eval.init_params.obj_set = VARIANT[1]
        cfg.rollout.model = OmegaConf.to_container(cfg.actor.model, resolve=True)
        cfg.cluster.component_placement = {"env,rollout": "4-7"}
        cfg.runner.only_eval = True
        cfg.runner.task_type = "embodied_eval"
        cfg.runner.ckpt_path = ckpt
        cfg.runner.logger.log_path = str(dest)
        cfg.runner.logger.experiment_name = "maniskill_" + OUT.name + "_" + arm
        cfg.rollout.enable_offload = False
        OmegaConf.save(cfg, dest / "eval.yaml", resolve=True)
        env = os.environ.copy()
        temp = tempfile.mkdtemp(prefix="mseval-", dir="/tmp")
        env.update({"TMPDIR": temp, "RAY_TMPDIR": temp})
        cmd = [
            str(PYTHON),
            "evaluations/eval_embodied_agent.py",
            "--config-path",
            str(dest),
            "--config-name",
            "eval",
        ]
        (dest / "command.json").write_text(json.dumps(cmd))
        owned = {}
        reason = "failure"
        with (dest / "launcher.log").open("x") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            owned[proc.pid] = psutil.Process(proc.pid)
            start = time.monotonic()
            try:
                while True:
                    for p in list(owned.values()):
                        try:
                            owned.update({c.pid: c for c in p.children(recursive=True)})
                        except psutil.Error:
                            pass
                    (OUT / "status.json").write_text(
                        json.dumps(
                            {
                                "arm": arm,
                                "pid": proc.pid,
                                "exit_code": proc.poll(),
                                "time": time.strftime("%F %T %Z"),
                            }
                        )
                    )
                    if proc.poll() is not None:
                        reason = "finished" if proc.returncode == 0 else "failure"
                        break
                    if (
                        stopped
                        or time.monotonic() - start > 14400
                        or psutil.virtual_memory().available < 40 * 2**30
                    ):
                        reason = "requested_stop_or_guard"
                        break
                    time.sleep(10)
            finally:
                live = []
                for p in owned.values():
                    try:
                        p.terminate()
                        live.append(p)
                    except psutil.Error:
                        pass
                _, alive = psutil.wait_procs(live, timeout=20)
                for p in alive:
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
        if reason != "finished":
            raise RuntimeError(f"{arm}: {reason}")
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        events = EventAccumulator(str(dest / "tensorboard"))
        events.Reload()
        metrics = {
            tag: events.Scalars(tag)[-1].value
            for tag in events.Tags()["scalars"]
            if tag.startswith("eval/")
        }
        assert "eval/success_once" in metrics
        assert metrics["eval/num_trajectories"] == 320
        result = {"arm": arm, "checkpoint": ckpt, "metrics": metrics}
        (dest / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "status.json").write_text(
        json.dumps({"state": "stopped" if stopped else "finished"})
    )


if __name__ == "__main__":
    main()
