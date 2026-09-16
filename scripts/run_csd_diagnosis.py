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

"""Run fixed-trial inference and early-checkpoint diagnostics without training."""

import concurrent.futures
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/20260905-csd-diagnosis"
SUITES = {
    "libero_spatial": 240,
    "libero_object": 280,
    "libero_goal": 320,
    "libero_10": 520,
}
CHECKPOINTS = {
    "ppo": ROOT
    / "logs/20260902-libero_40_ppo_csddiag_gpu67_gen10_exec5/libero_40_ppo_csddiag_seed42_gpu67_gen10_exec5_20260902/checkpoints",
    "csd": ROOT
    / "logs/20260901-libero_40_ppo_csd_b1_gpu45_gen10_exec5/libero_40_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260901/checkpoints",
}


def evaluate(
    job,
    gpus,
    *,
    model_path=None,
    config_name="pi05_libero",
    mode="standard",
    trials=10,
    envs=20,
    epochs=5,
    expected=100,
    timeout=3600,
    extra_overrides=(),
):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    name, suite, k, ns, checkpoint = job
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    result_path = dest / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if checkpoint and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    env = os.environ.copy()
    for key in ("CUDA_VISIBLE_DEVICES", "RAY_ADDRESS"):
        env.pop(key, None)
    runtime = tempfile.mkdtemp(prefix="csddiag-", dir="/tmp")
    env.update(
        {
            "RLINF_FORCE_LOCAL_RAY": "1",
            "RLINF_LOCAL_RAY_NUM_CPUS": "32",
            "RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES": "16000000000",
            "REPO_PATH": str(ROOT),
            "EMBODIED_PATH": str(ROOT / "examples/embodiment"),
            "PYTHONPATH": str(ROOT),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "LIBERO_TYPE": mode,
            "LIBERO_SUFFIX": "",
            "LIBERO_PERTURBATION": "",
            "MAGICK_HOME": str(ROOT / ".native-deps"),
            "LD_LIBRARY_PATH": str(ROOT / ".native-deps/lib")
            + ":"
            + env.get("LD_LIBRARY_PATH", ""),
            "RAY_TMPDIR": runtime,
            "TMPDIR": runtime,
            "EVAL_GPU_RANGE": gpus,
            "EVAL_LOG_DIR": str(dest),
            "EVAL_NAME": name,
            "EVAL_SUITE": suite,
            "EVAL_EPISODE_STEPS": str(SUITES[suite]),
            "EVAL_ROLLOUT_STEPS": str(SUITES[suite]),
            "EVAL_TRIALS_PER_TASK": str(trials),
            "EVAL_TOTAL_ENVS": str(envs),
            "EVAL_ROLLOUT_EPOCH": str(epochs),
            "EVAL_MODEL_PATH": str(
                model_path or ROOT / "checkpoints/RLinf-Pi05-LIBERO-SFT"
            ),
            "EVAL_CKPT_PATH": str(checkpoint),
        }
    )
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "evaluations/eval_embodied_agent.py",
        "--config-path",
        str(ROOT / "evaluations/libero"),
        "--config-name",
        "libero_pi05_queued_eval",
        f"rollout.model.num_action_chunks={k}",
        f"rollout.model.openpi.action_chunk={k}",
        f"rollout.model.num_steps={ns}",
        f"rollout.model.openpi.num_steps={ns}",
        f"rollout.model.openpi.config_name={config_name}",
    ]
    if config_name == "pi0_libero":
        cmd.append("rollout.model.openpi.detach_critic_input=false")
    cmd.extend(extra_overrides)
    (dest / "command.json").write_text(json.dumps(cmd, indent=2))
    print(f"START {name} GPUs={gpus}", flush=True)
    with (dest / "launcher.log").open("w") as log:
        subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    events = EventAccumulator(str(dest / "tensorboard"))
    events.Reload()
    metrics = {
        tag: events.Scalars(tag)[-1].value
        for tag in events.Tags()["scalars"]
        if tag.startswith("eval/")
    }
    if "eval/success_once" not in metrics:
        raise RuntimeError(f"Missing evaluation metrics: {name}")
    if metrics.get("eval/num_trajectories") != expected:
        raise RuntimeError(f"Expected {expected} trials: {name}: {metrics}")
    result = {
        "name": name,
        "suite": suite,
        "k": k,
        "num_steps": ns,
        "checkpoint": str(checkpoint),
        "model_path": env["EVAL_MODEL_PATH"],
        "model_config": config_name,
        "benchmark_mode": mode,
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(f"DONE {name}: {metrics}", flush=True)
    return result


def lane(jobs, gpus):
    return [evaluate(job, gpus) for job in jobs]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Stage 1: 2x2 inference settings, same checkpoint and same 100 Long trials.
    parity = [
        (f"base_long_k{k}_ns{ns}", "libero_10", k, ns, "")
        for k, ns in [(5, 3), (10, 5), (5, 5), (10, 3)]
    ]
    # Stage 2: common fixed-trial panel including baseline and step-80 anchors.
    early = [
        (f"base_{suite}", suite, 5, 3, "") for suite in SUITES if suite != "libero_10"
    ]
    for step in (10, 20, 80):
        for suite in SUITES:
            for arm, root in CHECKPOINTS.items():
                ckpt = (
                    root / f"global_step_{step}/actor/model_state_dict/full_weights.pt"
                )
                early.append((f"{arm}_step{step}_{suite}", suite, 5, 3, str(ckpt)))
    results = []
    for jobs in (parity, early):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(lane, jobs[::2], "4-5"),
                pool.submit(lane, jobs[1::2], "6-7"),
            ]
            for future in futures:
                results.extend(future.result())
        (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
