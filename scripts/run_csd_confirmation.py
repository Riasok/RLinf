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

"""Confirm screening findings on 500 trials, exclusively on GPUs 4 through 7."""

import concurrent.futures
import json
import time

import run_csd_diagnosis as diag


def lane(jobs, gpus):
    results = []
    for job in jobs:
        results.append(
            diag.evaluate(
                job, gpus, trials=50, envs=20, epochs=25, expected=500, timeout=7200
            )
        )
    return results


def main():
    diag.OUT = diag.ROOT / "logs/20260905-csd-confirmation"
    diag.OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("base_long_k10_ns3_n500", "libero_10", 10, 3, ""),
        ("base_long_k10_ns5_n500", "libero_10", 10, 5, ""),
    ]
    for suite in ("libero_goal", "libero_10"):
        for arm, root in diag.CHECKPOINTS.items():
            ckpt = str(root / "global_step_10/actor/model_state_dict/full_weights.pt")
            jobs.append((f"{arm}_step10_{suite}_k5_n500", suite, 5, 3, ckpt))
    for arm, root in diag.CHECKPOINTS.items():
        ckpt = str(root / "global_step_10/actor/model_state_dict/full_weights.pt")
        jobs.append((f"{arm}_step10_long_k10_n500", "libero_10", 10, 3, ckpt))
    (diag.OUT / "plan.json").write_text(
        json.dumps(
            {
                "created": time.strftime("%F %T %Z"),
                "gpu_lanes": ["4-5", "6-7"],
                "jobs": jobs,
                "trials_per_suite": 500,
                "training_queued": False,
                "baseline_500_logs": "logs/20260904-step80-eval-ns3-then-csdgate",
            },
            indent=2,
        )
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(lane, jobs[::2], "4-5"),
            pool.submit(lane, jobs[1::2], "6-7"),
        ]
        results = [row for future in futures for row in future.result()]
    (diag.OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("All 8 confirmation checks complete.", flush=True)


if __name__ == "__main__":
    main()
