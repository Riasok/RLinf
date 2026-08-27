# Copyright 2025 The RLinf Authors.
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
import random

import numpy as np
import torch

from rlinf.utils.utils import seed_everything


def test_seed_everything_reproduces_rollout_rng_streams():
    seed_everything(43)
    first = (random.random(), np.random.random(), torch.rand(4))

    seed_everything(43)
    second = (random.random(), np.random.random(), torch.rand(4))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
