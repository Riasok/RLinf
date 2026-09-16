# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""PPO-matched BC updates with sequential same-GPU simulator validation."""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import psutil
import run_pirl130_20260906 as runtime

ROOT = runtime.ROOT
OUT = ROOT / "logs/20260908-spatial73-bc"
STOP = False


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def owned_process(cmd, dest, env, timeout):
    """Manage only this child's descendants; never stop another Ray cluster."""
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / "command.json", cmd)
    owned = {}
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
                for parent in list(owned.values()):
                    try:
                        owned.update(
                            {p.pid: p for p in parent.children(recursive=True)}
                        )
                    except psutil.Error:
                        pass
                if proc.poll() is not None:
                    if proc.returncode:
                        raise RuntimeError(
                            f"Child failed ({proc.returncode}); see {dest}"
                        )
                    break
                if STOP or time.monotonic() - start > timeout:
                    raise RuntimeError("Requested stop or timeout")
                if (
                    psutil.disk_usage(str(ROOT)).free < 80 * 2**30
                    or psutil.virtual_memory().available < 40 * 2**30
                ):
                    raise RuntimeError("Disk/RAM guard")
                time.sleep(2)
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


def evaluate(iteration, checkpoint, logger):
    from omegaconf import OmegaConf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    for split, ids, episodes in [
        ("heldout", [1, 8, 9], 150),
        ("seen", [0, 2, 3, 4, 5, 6, 7], 350),
    ]:
        dest = OUT / "eval" / f"iteration_{iteration:03d}" / split
        dest.mkdir(parents=True)
        cfg = OmegaConf.load(ROOT / "logs/20260907-spatial73/resolved.yaml")
        cfg.cluster.component_placement = {"env,rollout": "0-3"}
        cfg.runner.task_type = "embodied_eval"
        cfg.runner.only_eval = True
        cfg.runner.ckpt_path = str(checkpoint) if checkpoint else None
        cfg.runner.logger.log_path = str(dest)
        cfg.runner.logger.experiment_name = f"bc_spatial73_{iteration}_{split}"
        cfg.runner.logger.logger_backends = ["tensorboard"]
        cfg.rollout.model = copy.deepcopy(cfg.actor.model)
        cfg.rollout.model.add_value_head = False
        cfg.rollout.model.openpi.add_value_head = False
        cfg.rollout.enable_offload = False
        cfg.env.eval.task_id_filter = ids
        cfg.env.eval.rollout_epoch = (episodes + 19) // 20
        OmegaConf.save(cfg, dest / "eval.yaml", resolve=True)
        env = os.environ.copy()
        temp = tempfile.mkdtemp(prefix="bceval-", dir="/tmp")
        env.update({"TMPDIR": temp, "RAY_TMPDIR": temp})
        write(
            OUT / "status.json",
            {"state": "evaluating", "iteration": iteration, "split": split},
        )
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
            7200,
        )
        events = EventAccumulator(str(dest / "tensorboard"))
        events.Reload()
        values = {
            k: events.Scalars(k)[-1].value
            for k in events.Tags()["scalars"]
            if k.startswith("eval/")
        }
        assert values["eval/num_trajectories"] == episodes, values
        write(
            dest / "result.json",
            {
                "comparison_iteration": iteration,
                "optimizer_steps": iteration * 12,
                "checkpoint": str(checkpoint),
                "metrics": values,
            },
        )
        metrics.update(
            {f"eval/{split}/{k.removeprefix('eval/')}": v for k, v in values.items()}
        )
    logger.log(metrics, iteration)


def driver():
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
        cfg = validate_cfg(compose(config_name="spatial73_bc_20260908"))
    OmegaConf.save(cfg, OUT / "resolved.yaml", resolve=True)
    logger = MetricLogger(cfg)
    # Preserve a true shared-checkpoint baseline before any BC update.
    evaluate(0, None, logger)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = SpatialBCWorker.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    actor.init_worker().wait()
    started = time.monotonic()
    for iteration in range(1, 151):
        write(
            OUT / "status.json",
            {
                "state": "training",
                "iteration": iteration,
                "optimizer_steps_completed": (iteration - 1) * 12,
            },
        )
        for update in range(12):
            actor.set_global_step((iteration - 1) * 12 + update).wait()
            results = actor.run_training().wait()
        logger.log({f"train/{k}": v for k, v in results[0].items()}, iteration)
        logger.log(
            {
                "budget/optimizer_steps": iteration * 12,
                "budget/demo_chunks": iteration * 12 * 2048,
                "budget/elapsed_seconds": time.monotonic() - started,
            },
            iteration,
        )
        if iteration % 10 == 0:
            dest = OUT / "checkpoints" / f"global_step_{iteration}"
            actor.save_checkpoint(str(dest / "actor"), iteration * 12).wait()
            write(
                dest / "budget.json",
                {"comparison_iteration": iteration, "optimizer_steps": iteration * 12},
            )
            actor.pause_for_eval().wait()
            evaluate(iteration, dest / "actor/model_state_dict/full_weights.pt", logger)
            actor.resume_after_eval().wait()
    logger.finish()
    write(OUT / "status.json", {"state": "finished", "iteration": 150})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--driver", action="store_true")
    args = parser.parse_args()
    runtime.setup_environment()
    os.environ.update({"JAX_PLATFORMS": "cpu", "LIBERO_TYPE": "standard"})
    if args.driver:
        driver()
        return
    from rlinf.workers.sft.spatial_bc_worker import select_episodes

    split = json.loads((ROOT / "logs/20260907-spatial73/split.json").read_text())
    allowed = {split["tasks"][str(i)].replace("_", " ") for i in split["train_ids"]}
    source = ROOT / "datasets/physical-intelligence-libero"
    episodes = [
        json.loads(line)
        for line in (source / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = select_episodes(episodes, allowed)
    manifest = {
        "train_task_ids": split["train_ids"],
        "heldout_task_ids": split["heldout_ids"],
        "instructions": sorted(allowed),
        "episode_ids": [e["episode_index"] for e in selected],
        "frames": sum(e["length"] for e in selected),
        "episodes_per_task": {
            t: sum(t in e["tasks"] for e in selected) for t in sorted(allowed)
        },
        "normalization": "Unchanged from shared few-shot SFT checkpoint, as in PPO",
        "protocol": "150 comparison iterations x 12 Adam updates x global batch 2048; expert-only BC",
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.launch:
        return
    assert not OUT.exists(), "Use a new run ID; do not overwrite prior artifacts"
    for row in subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines():
        gpu, memory = map(int, row.split(","))
        if gpu < 4 and memory > 1000:
            raise RuntimeError(f"GPU{gpu} occupied")
    write(OUT / "split.json", manifest)

    def stop(_signum, _frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    temp = tempfile.mkdtemp(prefix="bc73-", dir="/tmp")
    env.update({"TMPDIR": temp, "RAY_TMPDIR": temp})
    try:
        owned_process(
            [sys.executable, "scripts/run_spatial73_bc_20260908.py", "--driver"],
            OUT / "driver",
            env,
            7 * 86400,
        )
    except Exception as exc:
        write(
            OUT / "failure.json", {"error": str(exc), "time": time.strftime("%F %T %Z")}
        )
        raise


if __name__ == "__main__":
    main()
