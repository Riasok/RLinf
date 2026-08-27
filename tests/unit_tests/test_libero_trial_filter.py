from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.libero.libero_env import LiberoEnv


class _FakeTaskSuite:
    def __init__(self, trial_counts):
        self._trial_counts = trial_counts

    def get_num_tasks(self):
        return len(self._trial_counts)

    def get_task_init_states(self, task_id):
        return [None] * self._trial_counts[task_id]


def _make_env(trial_counts, task_id_filter=None, num_trials_per_task=None):
    env = object.__new__(LiberoEnv)
    env.task_suite = _FakeTaskSuite(trial_counts)
    env.task_id_filter = task_id_filter
    env.num_trials_per_task = num_trials_per_task
    env.cfg = SimpleNamespace()
    env._compute_total_num_group_envs()
    return env


def test_num_trials_per_task_keeps_first_trial_from_each_task():
    env = _make_env([3, 2, 4], num_trials_per_task=1)

    np.testing.assert_array_equal(env._valid_reset_state_ids, [0, 3, 5])
    assert env.total_num_group_envs == 9
    np.testing.assert_array_equal(env.cumsum_trial_id_bins, [3, 5, 9])


def test_trial_cap_composes_with_task_filter():
    env = _make_env([3, 2, 4], task_id_filter=[2, 0], num_trials_per_task=2)

    np.testing.assert_array_equal(env._valid_reset_state_ids, [0, 1, 5, 6])


@pytest.mark.parametrize("value", [0, -1, 1.5, "1"])
def test_num_trials_per_task_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="num_trials_per_task"):
        _make_env([3], num_trials_per_task=value)
