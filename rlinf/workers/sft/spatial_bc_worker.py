# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Seven-task BC worker with explicit episode filtering and eval offload."""

import json
from pathlib import Path

import torch

from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def select_episodes(episodes, allowed):
    """Select complete episodes whose instruction belongs to the approved split."""
    selected = [e for e in episodes if set(e["tasks"]).issubset(allowed) and e["tasks"]]
    observed = {task for e in selected for task in e["tasks"]}
    if observed != set(allowed):
        raise ValueError(
            f"Missing approved task demonstrations: {set(allowed) - observed}"
        )
    return selected


class SpatialBCWorker(FSDPVlaSftWorker):
    """Use all selected demonstrations, with the original SFT normalization."""

    def build_dataloader(self, data_paths, eval_dataset=False):
        import openpi.training.data_loader as dl
        import openpi.transforms as transforms
        from torch.utils.data.distributed import DistributedSampler

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        manifest = json.loads(Path(self.cfg.data.split_manifest).read_text())
        config = get_openpi_config(
            "pi05_libero", model_path=self.cfg.actor.model.model_path
        )
        data_config = config.data.create(config.assets_dirs, config.model)
        module = dl._import_lerobot_dataset()
        metadata = module.LeRobotDatasetMetadata(str(data_paths))
        dataset = module.LeRobotDataset(
            str(data_paths),
            episodes=manifest["episode_ids"],
            delta_timestamps={"actions": [t / metadata.fps for t in range(10)]},
        )
        actual = {int(i) for i in dataset.hf_dataset["episode_index"]}
        if actual != set(manifest["episode_ids"]):
            raise ValueError(
                "LeRobot episode filtering did not match the audited manifest"
            )
        if len(dataset) != manifest["frames"]:
            raise ValueError("Filtered frame count differs from manifest")
        # LeRobot v2.1 indexes this table by original episode ID, even when the
        # table returned for an episode subset is compact. Expand only the lookup
        # table; underlying frames remain filtered and episode-local.
        for key in ("from", "to"):
            compact = dataset.episode_data_index[key]
            expanded = torch.full((metadata.total_episodes,), -1, dtype=compact.dtype)
            expanded[torch.tensor(manifest["episode_ids"])] = compact
            dataset.episode_data_index[key] = expanded
        dataset = dl.TransformedDataset(
            dataset, [transforms.PromptFromLeRobotTask(metadata.tasks)]
        )
        dataset = dl.transform_dataset(dataset, data_config)
        sampler = DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=True,
            seed=42,
        )

        class EpochLoader(dl.TorchDataLoader):
            def __iter__(self):
                epoch = 0
                while True:
                    sampler.set_epoch(epoch)
                    yield from super().__iter__()
                    epoch += 1

        loader = EpochLoader(
            dataset,
            local_batch_size=self.micro_batch_size,
            sampler=sampler,
            num_workers=2,
            num_batches=len(sampler) // self.micro_batch_size,
            seed=42,
            framework="pytorch",
        )
        return dl.DataLoaderImpl(data_config, loader), data_config

    def pause_for_eval(self):
        """Release policy/optimizer GPU memory while preserving training state."""
        self.offload_param_and_grad()
        self.offload_optimizer()
        torch.cuda.empty_cache()

    def resume_after_eval(self):
        """Restore the same actor and optimizer after simulator evaluation."""
        self.load_param_and_grad(self.device)
        self.load_optimizer(self.device)
