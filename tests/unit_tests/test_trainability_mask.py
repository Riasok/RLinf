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

"""Tests for AdaRMS-selective policy training."""

import pytest
import torch
from torch import nn

from rlinf.models.trainability import (
    apply_trainability_mask,
    collect_adarms_parameter_names,
)


class GemmaRMSNorm(nn.Module):
    """Minimal structural stand-in for OpenPI's patched GemmaRMSNorm."""

    def __init__(self, *, adaptive: bool):
        super().__init__()
        self.cond_dim = 4 if adaptive else None
        if adaptive:
            self.dense = nn.Linear(4, 12)
            self.register_parameter("weight", None)
        else:
            self.dense = None
            self.weight = nn.Parameter(torch.ones(4))


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.adarms = GemmaRMSNorm(adaptive=True)
        self.regular_rms = GemmaRMSNorm(adaptive=False)
        self.value_head = nn.Linear(4, 1)


def trainable_names(model):
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def test_collects_only_adaptive_rms_dense_parameters():
    model = ToyPolicy()
    assert collect_adarms_parameter_names(model) == (
        "adarms.dense.bias",
        "adarms.dense.weight",
    )


def test_adarms_only_keeps_value_head_for_actor_critic_ppo():
    model = ToyPolicy()
    report = apply_trainability_mask(model, "adarms_only")
    assert trainable_names(model) == {
        "adarms.dense.bias",
        "adarms.dense.weight",
        "value_head.bias",
        "value_head.weight",
    }
    assert report.policy_trainable_parameters == 60
    assert report.value_head_trainable_parameters == 5


def test_exclude_adarms_preserves_other_trainable_parameters():
    model = ToyPolicy()
    apply_trainability_mask(model, "exclude_adarms")
    assert trainable_names(model) == {
        "backbone.bias",
        "backbone.weight",
        "regular_rms.weight",
        "value_head.bias",
        "value_head.weight",
    }


def test_non_adarms_model_fails_closed():
    with pytest.raises(ValueError, match="matched zero parameters"):
        apply_trainability_mask(nn.Linear(2, 2), "adarms_only")


def test_unknown_mask_fails_closed():
    with pytest.raises(ValueError, match="Unknown trainability_mask"):
        apply_trainability_mask(ToyPolicy(), "typo")
