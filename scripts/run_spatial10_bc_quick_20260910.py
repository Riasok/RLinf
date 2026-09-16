# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Run a short full-shot BC job on all ten LIBERO-Spatial tasks."""

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import run_pirl130_20260906 as runtime

ROOT = runtime.ROOT
OUT = ROOT / "logs/20260910-spatial10-bc-quick"
GPU_IDS = (4, 5, 6, 7)
COMPARISON_ITERATIONS = 10
UPDATES_PER_ITERATION = 12
STOP = False


def write(path: Path, value: object) -> None:
    """Write a JSON run artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def owned_process(
    cmd: list[str], dest: Path, env: dict[str, str], timeout: int
) -> None:
    """Run and clean up only the subprocesses owned by this launcher."""
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / "command.json", cmd)
    owned: dict[int, psutil.Process] = {}
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
        started = time.monotonic()
        try:
            while proc.poll() is None:
                for parent in list(owned.values()):
                    try:
                        owned.update(
                            {p.pid: p for p in parent.children(recursive=True)}
                        )
                    except psutil.Error:
                        pass
                if STOP or time.monotonic() - started > timeout:
                    raise RuntimeError("Requested stop or timeout")
                if (
                    psutil.disk_usage(str(ROOT)).free < 80 * 2**30
                    or psutil.virtual_memory().available < 40 * 2**30
                ):
                    raise RuntimeError("Disk/RAM guard")
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f"Child failed ({proc.returncode}); see {dest}")
        finally:
            live = []
            for process in owned.values():
                try:
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


def evaluate(checkpoint: Path) -> dict[str, float]:
    """Evaluate the final checkpoint on 50 trials for each Spatial task."""
    from omegaconf import OmegaConf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    dest = OUT / "eval" / "iteration_010" / "all_spatial"
    dest.mkdir(parents=True)
    cfg = OmegaConf.load(ROOT / "logs/20260907-spatial73/resolved.yaml")
    cfg.cluster.component_placement = {"env,rollout": "4-7"}
    cfg.runner.task_type = "embodied_eval"
    cfg.runner.only_eval = True
    cfg.runner.ckpt_path = str(checkpoint)
    cfg.runner.logger.log_path = str(dest)
    cfg.runner.logger.experiment_name = "bc_spatial10_quick_step10_eval"
    cfg.runner.logger.logger_backends = ["tensorboard"]
    cfg.rollout.model = copy.deepcopy(cfg.actor.model)
    cfg.rollout.model.add_value_head = False
    cfg.rollout.model.openpi.add_value_head = False
    cfg.rollout.enable_offload = False
    cfg.env.eval.task_id_filter = list(range(10))
    cfg.env.eval.rollout_epoch = 25
    OmegaConf.save(cfg, dest / "eval.yaml", resolve=True)
    env = os.environ.copy()
    temp = tempfile.mkdtemp(prefix="bc10eval-", dir="/tmp")
    env.update({"TMPDIR": temp, "RAY_TMPDIR": temp})
    write(OUT / "status.json", {"state": "evaluating", "iteration": 10})
    owned_process(
        [
            sys.executable,
            "evaluations/eval_embodied_agent.py",
            "--config-path",
            str(dest),
            "--config-name",
            "eval",
        ],
        dest,
        env,
        10_800,
    )
    events = EventAccumulator(str(dest / "tensorboard"))
    events.Reload()
    values = {
        key: events.Scalars(key)[-1].value
        for key in events.Tags()["scalars"]
        if key.startswith("eval/")
    }
    if values["eval/num_trajectories"] != 500:
        raise RuntimeError(f"Expected 500 eval trajectories, got {values}")
    write(dest / "result.json", {"checkpoint": str(checkpoint), "metrics": values})
    return values


def driver() -> None:
    """Train for 120 updates, save once, then evaluate once."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.metric_logger import MetricLogger
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.sft.spatial_bc_worker import SpatialBCWorker

    with initialize_config_dir(
        config_dir=str(ROOT / "examples/sft/config"), version_base="1.1"
    ):
        cfg = validate_cfg(compose(config_name="spatial10_bc_quick_20260910"))
    OmegaConf.save(cfg, OUT / "resolved.yaml", resolve=True)
    logger = MetricLogger(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = SpatialBCWorker.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    actor.init_worker().wait()
    started = time.monotonic()
    for iteration in range(1, COMPARISON_ITERATIONS + 1):
        write(
            OUT / "status.json",
            {
                "state": "training",
                "iteration": iteration,
                "optimizer_steps_completed": (iteration - 1) * UPDATES_PER_ITERATION,
            },
        )
        for update in range(UPDATES_PER_ITERATION):
            actor.set_global_step(
                (iteration - 1) * UPDATES_PER_ITERATION + update
            ).wait()
            results = actor.run_training().wait()
        logger.log(
            {f"train/{key}": value for key, value in results[0].items()}, iteration
        )
        logger.log(
            {
                "budget/optimizer_steps": iteration * UPDATES_PER_ITERATION,
                "budget/demo_chunks": iteration * UPDATES_PER_ITERATION * 2048,
                "budget/elapsed_seconds": time.monotonic() - started,
            },
            iteration,
        )

    checkpoint_dir = OUT / "checkpoints/global_step_10"
    actor.save_checkpoint(
        str(checkpoint_dir / "actor"),
        COMPARISON_ITERATIONS * UPDATES_PER_ITERATION,
    ).wait()
    write(
        checkpoint_dir / "budget.json",
        {"comparison_iteration": 10, "optimizer_steps": 120},
    )
    actor.pause_for_eval().wait()
    checkpoint = checkpoint_dir / "actor/model_state_dict/full_weights.pt"
    metrics = evaluate(checkpoint)
    logger.log(
        {
            f"eval/all_spatial/{key.removeprefix('eval/')}": value
            for key, value in metrics.items()
        },
        10,
    )
    logger.finish()
    write(OUT / "status.json", {"state": "finished", "iteration": 10})


def build_manifest() -> dict[str, object]:
    """Audit and return the all-Spatial demonstration manifest."""
    from rlinf.workers.sft.spatial_bc_worker import select_episodes

    split = json.loads((ROOT / "logs/20260907-spatial73/split.json").read_text())
    allowed = {name.replace("_", " ") for name in split["tasks"].values()}
    source = ROOT / "datasets/physical-intelligence-libero"
    episodes = [
        json.loads(line)
        for line in (source / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = select_episodes(episodes, allowed)
    return {
        "train_task_ids": list(range(10)),
        "heldout_task_ids": [],
        "instructions": sorted(allowed),
        "episode_ids": [episode["episode_index"] for episode in selected],
        "frames": sum(episode["length"] for episode in selected),
        "episodes_per_task": {
            task: sum(task in episode["tasks"] for episode in selected)
            for task in sorted(allowed)
        },
        "initial_checkpoint": str(ROOT / "checkpoints/RLinf-Pi05-LIBERO-SFT"),
        "protocol": "10 comparison iterations x 12 Adam updates; all 10 Spatial tasks; expert-only BC",
    }


def main() -> None:
    """Validate resources and launch the persistent driver."""
    global STOP

    runtime.setup_environment()
    os.environ["RLINF_FORCE_LOCAL_RAY"] = "0"
    os.environ.update({"JAX_PLATFORMS": "cpu", "LIBERO_TYPE": "standard"})
    manifest = build_manifest()
    print(json.dumps(manifest, indent=2), flush=True)
    if "--launch" not in sys.argv:
        return
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    occupancy = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for row in occupancy.splitlines():
        gpu, memory = map(int, row.split(","))
        if gpu in GPU_IDS and memory > 1000:
            raise RuntimeError(f"GPU{gpu} is occupied ({memory} MiB)")
    write(OUT / "manifest.json", manifest)

    def stop(_signum: int, _frame: object) -> None:
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    temp = tempfile.mkdtemp(prefix="bc10-", dir="/tmp")
    env.update({"TMPDIR": temp, "RAY_TMPDIR": temp})
    try:
        owned_process(
            [sys.executable, "scripts/run_spatial10_bc_quick_20260910.py", "--driver"],
            OUT / "driver",
            env,
            2 * 86_400,
        )
    except Exception as exc:
        write(
            OUT / "failure.json", {"error": str(exc), "time": time.strftime("%F %T %Z")}
        )
        raise


if __name__ == "__main__":
    if "--driver" in sys.argv:
        runtime.setup_environment()
        os.environ["RLINF_FORCE_LOCAL_RAY"] = "0"
        os.environ.update({"JAX_PLATFORMS": "cpu", "LIBERO_TYPE": "standard"})
        driver()
    else:
        main()
