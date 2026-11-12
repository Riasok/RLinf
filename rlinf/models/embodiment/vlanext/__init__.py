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
"""VLANeXt model factory for RLinf embodied workers."""

import dataclasses

from omegaconf import DictConfig, OmegaConf


def get_model(cfg: DictConfig, torch_dtype=None):
    from rlinf.models.embodiment.vlanext.vlanext_action_model import (
        VLANeXtForRLActionPrediction,
        VLANeXtRLConfig,
    )

    overrides = OmegaConf.to_container(cfg.get("vlanext", {}) or {}, resolve=True)
    valid = {f.name for f in dataclasses.fields(VLANeXtRLConfig)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(f"Unknown actor.model.vlanext keys: {unknown}")
    overrides["checkpoint_path"] = str(cfg.model_path)
    overrides["fp32_action_head"] = bool(cfg.get("fp32_action_head", True))
    if "value_head_hidden_sizes" in overrides:
        overrides["value_head_hidden_sizes"] = tuple(overrides["value_head_hidden_sizes"])
    model_config = VLANeXtRLConfig(**overrides)
    return VLANeXtForRLActionPrediction(model_config)
