# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static checks for action-expert LoRA PPO experiment configs."""

from pathlib import Path

from omegaconf import OmegaConf


def test_action_expert_lora_experiment_targets_only_standard_expert_lora():
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root / "examples/embodiment/config/libero_spatial73_ppo_lora_20260910.yaml"
    )

    assert cfg.actor.model.is_lora
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.lora_rank == 32
    assert cfg.actor.model.openpi.train_expert_only
    assert not cfg.actor.fsdp_config.use_orig_params
    assert not cfg.algorithm.csd_enabled
    assert list(cfg.env.train.task_id_filter) == [0, 2, 3, 4, 5, 6, 7]
    assert list(cfg.env.eval.task_id_filter) == [1, 8, 9]
