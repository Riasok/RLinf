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
    apply_online_csd_gate,
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
    rollout_batch["dones"][1, 1, 2] = True
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
    torch.testing.assert_close(result["csd_candidate_pair_mask"], expected_mask)
    torch.testing.assert_close(
        result["csd_teacher_action_mask"][:-1],
        rollout_batch["loss_mask"][1:].expand(-1, -1, 5),
    )


def test_select_nested_batch_preserves_nested_alignment():
    batch = {
        "x": torch.tensor([[0], [1], [2]]),
        "nested": {"y": torch.tensor([[10], [11], [12]])},
    }

    selected = _select_nested_batch(batch, torch.tensor([2, 0]))

    torch.testing.assert_close(selected["x"], torch.tensor([[2], [0]]))
    torch.testing.assert_close(selected["nested"]["y"], torch.tensor([[12], [10]]))


def test_csd_preserves_partial_teacher_mask_with_chunk_level_ppo():
    batch = _gate_batch(time_dim=3, batch_dim=1)
    batch["dones"] = torch.zeros(4, 1, 5, dtype=torch.bool)
    batch["dones"][2, 0, 1:] = True
    batch["dones"][3] = True
    batch["loss_mask"] = torch.tensor([[[True]], [[True]], [[False]]])
    result = attach_online_csd_pairs(batch)
    assert result["csd_pair_mask"][0, 0].item()
    assert not result["csd_pair_mask"][1, 0].item()
    assert result["csd_teacher_action_mask"][0, 0].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_csd_excludes_reset_crossing_using_outcome_not_previous_done():
    batch = _gate_batch(time_dim=3, batch_dim=1)
    batch["dones"][1, 0, -1] = True
    result = attach_online_csd_pairs(batch)
    assert not result["csd_pair_mask"][0, 0].item()
    assert result["csd_pair_mask"][1, 0].item()


def test_attach_online_csd_pairs_requires_multiple_steps():
    batch = {
        "forward_inputs": {"model_action": torch.zeros(1, 2, 4)},
        "dones": torch.zeros(2, 2, 5, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="at least two"):
        attach_online_csd_pairs(batch)


def _gate_batch(time_dim=4, batch_dim=2):
    return {
        "forward_inputs": {"model_action": torch.zeros(time_dim, batch_dim, 4, 3)},
        "dones": torch.zeros(time_dim + 1, batch_dim, 2, dtype=torch.bool),
        "loss_mask": torch.ones(time_dim, batch_dim, 2, dtype=torch.bool),
        "rewards": torch.zeros(time_dim, batch_dim, 2),
    }


def test_success_gate_uses_future_chunks_and_stays_within_episode():
    batch = _gate_batch()
    batch["dones"][2, 0] = True
    batch["rewards"][1, 0, 0] = 1.0
    batch["rewards"][3, 1, 0] = 1.0
    batch = attach_online_csd_pairs(batch)

    result = apply_online_csd_gate(batch, gate_mode="success")

    expected = torch.tensor(
        [
            [[True], [True]],
            [[False], [True]],
            [[False], [True]],
            [[False], [False]],
        ]
    )
    torch.testing.assert_close(result["csd_pair_mask"], expected)
    assert result["csd_gate"][0, 0].item()
    assert not result["csd_gate"][2, 0].item()


def test_positive_advantage_gate_reads_teacher_chunk():
    batch = _gate_batch(time_dim=3, batch_dim=1)
    batch = attach_online_csd_pairs(batch)
    batch["advantages"] = torch.tensor([[[-1.0, -1.0]], [[2.0, 0.0]], [[-1.0, 0.0]]])

    result = apply_online_csd_gate(batch, gate_mode="positive_advantage")

    expected = torch.tensor([[[True]], [[False]], [[False]]])
    torch.testing.assert_close(result["csd_pair_mask"], expected)


def test_constant_gate_reproduces_candidate_pairs():
    batch = _gate_batch()
    batch = attach_online_csd_pairs(batch)

    result = apply_online_csd_gate(batch, gate_mode="constant")

    torch.testing.assert_close(
        result["csd_pair_mask"], result["csd_candidate_pair_mask"]
    )


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
        self.last_x_t = x_t.detach().clone()
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


def test_online_csd_masks_unexecuted_teacher_actions():
    model = _DummyCSDModel()
    student_action = torch.zeros(1, 12)
    teacher_action = torch.zeros(1, 12)
    changed_unexecuted_action = teacher_action.clone().reshape(1, 4, 3)
    changed_unexecuted_action[:, 1] = 100.0

    baseline = OpenPi0ForRLActionPrediction.online_csd_forward(
        model,
        {"model_action": student_action},
        teacher_action,
        offset=2,
        teacher_action_mask=torch.tensor([[True, False]]),
    )
    baseline_x_t = model.last_x_t
    masked = OpenPi0ForRLActionPrediction.online_csd_forward(
        model,
        {"model_action": student_action},
        changed_unexecuted_action,
        offset=2,
        teacher_action_mask=torch.tensor([[True, False]]),
    )

    torch.testing.assert_close(baseline["csd_loss"], masked["csd_loss"])
    torch.testing.assert_close(baseline_x_t, model.last_x_t)
    torch.testing.assert_close(
        baseline["csd_action_discrepancy"],
        masked["csd_action_discrepancy"],
    )
