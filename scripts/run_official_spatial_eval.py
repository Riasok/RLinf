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

"""Evaluate the official pi0 Spatial PPO release on standard and Plus Spatial."""

import json

import run_csd_diagnosis as diag
from huggingface_hub import snapshot_download


def main():
    repo = "RLinf/RLinf-Pi0-PPO-LIBERO-spatial"
    revision = "d56f925c8ebf97b2ae5efc1930400543b15dce41"
    model = diag.ROOT / "checkpoints/RLinf-Pi0-PPO-LIBERO-spatial"
    snapshot_download(repo, revision=revision, local_dir=model)
    diag.OUT = diag.ROOT / "logs/20260905-official-pi0-spatial-ppo"
    diag.OUT.mkdir(parents=True, exist_ok=True)
    (diag.OUT / "provenance.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "revision": revision,
                "model": "pi0",
                "executed_actions": 5,
                "denoising_steps": 4,
                "plus_protocol": "2402 installed variants, one trial each; no suffix remapping",
            },
            indent=2,
        )
    )
    diag.evaluate(
        ("standard_spatial", "libero_spatial", 5, 4, ""),
        "2-3",
        model_path=model,
        config_name="pi0_libero",
        trials=50,
        envs=20,
        epochs=25,
        expected=500,
    )
    diag.evaluate(
        ("plus_spatial", "libero_spatial", 5, 4, ""),
        "2-3",
        model_path=model,
        config_name="pi0_libero",
        mode="plus",
        trials=1,
        envs=40,
        epochs=61,
        expected=2402,
        timeout=21600,
    )


if __name__ == "__main__":
    main()
