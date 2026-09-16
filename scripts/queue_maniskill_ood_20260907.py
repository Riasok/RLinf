# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Queue all twelve Table7 OOD settings after the ongoing ID comparison."""

import json
import os
import time

import eval_maniskill_cut_20260907 as evaluation

OUT = evaluation.ROOT / "logs/20260907-maniskill-ood"
ID = evaluation.OUT
VARIANTS = (
    [
        ("vision_language", name, "test")
        for name in [
            "Instruct",
            "VisionImage",
            "VisionTexture03",
            "VisionTexture05",
            "VisionWhole03",
            "VisionWhole05",
        ]
    ]
    + [
        ("semantic", name, split)
        for name in ["MultiCarrot", "MultiPlate"]
        for split in ["train", "test"]
    ]
    + [("execution", name, "test") for name in ["PositionChangeTo", "Position"]]
)


def main():
    os.environ["MS_ASSET_DIR"] = "/data/minjaeoh/.maniskill"
    import gymnasium as gym

    import rlinf.envs.maniskill  # noqa: F401

    for _, name, _ in VARIANTS:
        assert gym.spec(f"PutOnPlateInScene25{name}-v1")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "plan.json").write_text(
        json.dumps(
            {
                "variants": VARIANTS,
                "models": ["sft", "step50_best_and_latest_saved"],
                "episodes_per_cell": 320,
                "gpus": [4, 5, 6, 7],
                "prerequisite": str(ID / "status.json"),
                "caveat": "Same horizon80 and default single receptacle as current ID, except variants overriding it; not full paper coverage",
            },
            indent=2,
        )
    )
    while True:
        status = json.loads((ID / "status.json").read_text())
        if status.get("state") == "finished":
            break
        if status.get("state") == "stopped" or status.get("exit_code") not in (None, 0):
            raise RuntimeError(f"ID prerequisite did not finish: {status}")
        (OUT / "status.json").write_text(
            json.dumps({"state": "waiting_for_ID", "time": time.strftime("%F %T %Z")})
        )
        time.sleep(30)
    collected = []
    for category, name, split in VARIANTS:
        key = f"{name}_{split}"
        (OUT / "status.json").write_text(
            json.dumps({"state": "running", "variant": key})
        )
        evaluation.OUT = OUT / key
        evaluation.VARIANT = (f"PutOnPlateInScene25{name}-v1", split)
        evaluation.main()
        state = json.loads((evaluation.OUT / "status.json").read_text())
        if state.get("state") != "finished":
            raise RuntimeError(f"Variant stopped: {key}")
        for result in json.loads((evaluation.OUT / "results.json").read_text()):
            collected.append({"category": category, "variant": key, **result})
        (OUT / "results.json").write_text(json.dumps(collected, indent=2))
    (OUT / "status.json").write_text(
        json.dumps({"state": "finished", "cells": len(collected)})
    )


if __name__ == "__main__":
    main()
