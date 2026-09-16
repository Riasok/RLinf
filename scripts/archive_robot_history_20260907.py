"""One-shot, explicitly reviewed historical artifact migration; dry-run by default."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "deprecated"
ACTIVE = "20260907-spatial73"
FAMILIES = (
    "20260828-libero_long_ppo_gpu67_gen10_exec5",
    "20260828-libero_long_ppo_csd_b1_gpu45_gen10_exec5",
    "20260901-libero_40_ppo_csd_b1_gpu45_gen10_exec5",
    "20260902-libero_40_ppo_csddiag_gpu67_gen10_exec5",
)


def plan():
    """Protect first/last saves and every externally referenced historical save."""
    saves = []
    for family in FAMILIES:
        saves.extend((ROOT / "logs" / family).glob("*/checkpoints/global_step_*"))
    assert len(saves) == 34, "Unexpected checkpoint inventory; stop for review"
    reasons = {str(p): [] for p in saves}
    for family in FAMILIES:
        group = sorted(
            (p for p in saves if family in p.parts),
            key=lambda p: int(p.name.rsplit("_", 1)[1]),
        )
        for p in (group[0], group[-1]):
            reasons[str(p)].append("first_or_last_saved")
    # Scan raw evaluation configs/logs too: older evals have no result.json.
    # Any reference outside the owning training directory protects the save,
    # even if that particular evaluation was partial or unsuccessful.
    for base, dirs, files in os.walk(ROOT / "logs", followlinks=False):
        dirs[:] = [d for d in dirs if d not in {"checkpoints", "wandb", "video", "ray_spill"}]
        for name in files:
            p = Path(base) / name
            if p.suffix not in {".json", ".yaml", ".log", ".jsonl", ".out"}:
                continue
            if p.stat().st_size > 32 * 1024**2:
                raise RuntimeError(f"Oversized evidence file needs manual audit: {p}")
            content = p.read_text(errors="replace")
            if "global_step_" not in content:
                continue
            for checkpoint in saves:
                owner = ROOT / "logs" / checkpoint.relative_to(ROOT / "logs").parts[0]
                if not p.is_relative_to(owner) and str(checkpoint) in content:
                    reasons[str(checkpoint)].append(str(p.relative_to(ROOT)))
    rows = []
    for p in sorted(saves):
        size = sum(f.stat().st_blocks * 512 for f in p.rglob("*") if f.is_file())
        rows.append({"path": str(p), "keep": bool(reasons[str(p)]),
                     "reasons": reasons[str(p)], "allocated_bytes": size})
    return rows


def migrate(source, dest):
    """Rename without copying data, leaving an old-path compatibility symlink."""
    assert source.exists() and not source.is_symlink(), source
    assert not dest.exists() and not dest.is_symlink(), dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    source.rename(dest)
    source.symlink_to(os.path.relpath(dest, source.parent), target_is_directory=dest.is_dir())
    return {"old": str(source), "new": str(dest)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = plan()
    payload = {"checkpoints": rows, "delete_count": sum(not r["keep"] for r in rows),
               "reclaim_gib": sum(r["allocated_bytes"] for r in rows if not r["keep"]) / 1024**3}
    print(json.dumps(payload, indent=2))
    if not args.apply:
        return
    audit = ROOT / "research/archive_execution_20260907.json"
    assert not audit.exists(), "One-shot operation already started; inspect its journal"
    payload.update({"state": "started", "moves": [], "deleted": []})
    audit.write_text(json.dumps(payload, indent=2) + "\n")
    # Move inactive run directories, including retained checkpoints and all results.
    for source in sorted((ROOT / "logs").iterdir()):
        if source.is_dir() and not source.is_symlink() and source.name != ACTIVE:
            payload["moves"].append(migrate(source, ARCHIVE / "results" / source.name))
            audit.write_text(json.dumps(payload, indent=2) + "\n")
    # Keep the current and next-direction SFT initializations at their original paths.
    for name in ("RLinf-Pi05-PPO-LIBERO-130", "RLinf-Pi05-LIBERO-130-fullshot-SFT",
                 "RLinf-Pi0-PPO-LIBERO-spatial", "lerobot-pi05-libero-base"):
        payload["moves"].append(migrate(ROOT / "checkpoints" / name, ARCHIVE / "checkpoints" / name))
        audit.write_text(json.dumps(payload, indent=2) + "\n")
    for source in sorted(ROOT.glob("*.sh")):
        payload["moves"].append(migrate(source, ARCHIVE / "launchers" / source.name))
    names = ("CSD_DIAGNOSIS_20260905.md", "LOCAL_PLUS_SPATIAL_20260906.md",
             "OVERNIGHT_20260906.md", "PIRL_130_20260906.md", "PIRL_LONG10_20260906.md")
    for name in names:
        payload["moves"].append(migrate(ROOT / name, ARCHIVE / "notes" / name))
    # Historical config paths stay valid for Hydra through symlinks. Active configs
    # and Python modules remain untouched because they have shared runtime imports.
    config = ROOT / "examples/embodiment/config"
    for source in sorted(config.glob("*.yaml")):
        if re.match(r"libero_(10_|40_|130_).*", source.name) and any(
            tag in source.name for tag in ("gpu", "overnight_controls")
        ):
            payload["moves"].append(migrate(source, ARCHIVE / "configs" / source.name))
    audit.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        if row["keep"]:
            continue
        original = Path(row["path"])
        target = original.resolve(strict=True)
        expected = ARCHIVE / "results" / original.relative_to(ROOT / "logs")
        assert target == expected and target.is_relative_to(ARCHIVE / "results")
        assert re.fullmatch(r"global_step_\d+", target.name)
        assert not any(p.is_symlink() for p in target.rglob("*")), target
        # Exact validated checkpoint directory only, never a run/workspace root.
        shutil.rmtree(target)
        payload["deleted"].append(str(target))
        audit.write_text(json.dumps(payload, indent=2) + "\n")
    payload["state"] = "finished"
    audit.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Finished: {len(payload['moves'])} moves; {len(payload['deleted'])} checkpoints deleted")


if __name__ == "__main__":
    main()
