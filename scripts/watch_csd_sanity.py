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

"""Supervise the finite sanity queues and maintain a durable results report."""

import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "logs/20260905-sanity-status.md"
QUEUES = [
    (
        "csd-diagnosis-20260905",
        "run_csd_diagnosis.py",
        "20260905-csd-diagnosis",
        31,
        [4, 5, 6, 7],
    ),
    (
        "csd-confirmation-20260905",
        "run_csd_confirmation.py",
        "20260905-csd-confirmation",
        8,
        [4, 5, 6, 7],
    ),
]


def main():
    retries = {q[0]: 0 for q in QUEUES}
    # Finite watchdog: no new training, at most two restarts per evaluation queue.
    deadline = time.monotonic() + 8 * 3600
    while time.monotonic() < deadline:
        lines = ["# Sanity-check status", "", time.strftime("Updated %F %T %Z"), ""]
        complete = True
        for session, script, folder, target, gpus in QUEUES:
            records = [
                json.loads(p.read_text())
                for p in sorted((ROOT / "logs" / folder).glob("*/result.json"))
            ]
            alive = (
                subprocess.run(
                    ["tmux", "has-session", "-t", session], capture_output=True
                ).returncode
                == 0
            )
            done = len(records) == target
            complete &= done
            state = "complete" if done else "running" if alive else "stopped"
            lines.extend(
                [
                    f"## {session}: {len(records)}/{target} ({state})",
                    "",
                    "| Evaluation | Success | Episodes |",
                    "|---|---:|---:|",
                ]
            )
            for row in records:
                m = row["metrics"]
                lines.append(
                    f"| {row['name']} | {100 * m['eval/success_once']:.1f}% | {m['eval/num_trajectories']:.0f} |"
                )
            lines.append("")
            if done or alive or retries[session] >= 2:
                continue
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            used = {
                int(a): int(b)
                for a, b in (line.split(",") for line in result.stdout.splitlines())
            }
            if any(used[g] > 512 for g in gpus):
                lines.append(
                    "Restart deferred: a required GPU still has allocated memory.\n"
                )
                continue
            # Preserve failed cell logs before the driver's resumable retry.
            for log in (ROOT / "logs" / folder).glob("*/launcher.log"):
                if not (log.parent / "result.json").exists():
                    log.rename(
                        log.with_name(
                            f"launcher.retry{retries[session]}.{time.time_ns()}.log"
                        )
                    )
            cmd = f"cd {shlex.quote(str(ROOT))} && .venv/bin/python scripts/{script} >> /tmp/{session}.log 2>&1"
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, cmd], check=True
            )
            retries[session] += 1
            lines.append(
                f"Restarted incomplete queue (attempt {retries[session]}/2).\n"
            )
        REPORT.write_text("\n".join(lines))
        if complete:
            print("All sanity evaluations complete; see", REPORT, flush=True)
            return
        time.sleep(60)
    print("Watchdog reached its 8-hour limit; see", REPORT, flush=True)


if __name__ == "__main__":
    main()
