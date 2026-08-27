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
from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.workers.actor.fsdp_actor_worker import (
    _select_nested_batch,
    attach_online_csd_pairs,
)


def test_attach_online_csd_pairs_is_shifted_and_episode_safe():
    time_dim, batch_dim = 3, 2
    model_action = torch.arange(time_dim * batch_dim * 4).reshape(
        time_dim, batch_dim, 4
    )
    rollout_batch = {
        "forward_inputs": {
            "model_action": model_action,
            "chains": torch.randn(time_dim, batch_dim, 2, 4),
            "tokenized_prompt": torch.ones(time_dim, batch_dim, 3),
            "tokenized_prompt_mask": torch.ones(time_dim, batch_dim, 3),
            "observation/image": torch.randn(time_dim, batch_dim, 3, 2, 2),
        },
        "dones": torch.zeros(time_dim + 1, batch_dim, 5, dtype=torch.bool),
        "loss_mask": torch.ones(time_dim, batch_dim, 1, dtype=torch.bool),
    }
    rollout_batch["dones"][0, 1, 2] = True
    rollout_batch["loss_mask"][1, 0] = False

    result = attach_online_csd_pairs(rollout_batch)

    teacher_action = result["csd_teacher_action"]
    assert "csd_teacher_forward_inputs" not in result
    torch.testing.assert_close(teacher_action[:-1], model_action[1:])
    torch.testing.assert_close(teacher_action[-1], model_action[-1])
    expected_mask = torch.tensor(
        [[[False], [False]], [[False], [True]], [[False], [False]]]
    )
    torch.testing.assert_close(result["csd_pair_mask"], expected_mask)


def test_select_nested_batch_preserves_nested_alignment():
    batch = {
        "x": torch.tensor([[0], [1], [2]]),
        "nested": {"y": torch.tensor([[10], [11], [12]])},
    }

    selected = _select_nested_batch(batch, torch.tensor([2, 0]))

    torch.testing.assert_close(selected["x"], torch.tensor([[2], [0]]))
    torch.testing.assert_close(selected["nested"]["y"], torch.tensor([[12], [10]]))


def test_attach_online_csd_pairs_requires_multiple_steps():
    batch = {
        "forward_inputs": {"model_action": torch.zeros(1, 2, 4)},
        "dones": torch.zeros(2, 2, 5, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="at least two"):
        attach_online_csd_pairs(batch)


class _DummyCSDModel:
    def __init__(self):
        self.config = SimpleNamespace(
            action_horizon=4,
            action_dim=3,
            action_env_dim=2,
        )
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def parameters(self):
        yield self.scale

    def sample_noise(self, shape, device):
        return torch.ones(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.full((batch_size,), 0.5, device=device)

    def _online_csd_velocity(self, forward_inputs, x_t, flow_time):
        del forward_inputs, flow_time
        return self.scale * torch.ones_like(x_t)


def test_online_csd_uses_detached_flow_target_and_student_only_gradients():
    model = _DummyCSDModel()
    student_action = torch.zeros(2, 12, requires_grad=True)
    teacher_action = torch.zeros(2, 12, requires_grad=True)

    output = OpenPi0ForRLActionPrediction.online_csd_forward(
        model,
        {"model_action": student_action},
        teacher_action,
        offset=2,
    )

    torch.testing.assert_close(output["csd_loss"], torch.tensor(1.0))
    output["csd_loss"].backward()
    assert model.scale.grad is not None
    assert model.scale.grad.item() < 0
    assert student_action.grad is None
    assert teacher_action.grad is None
