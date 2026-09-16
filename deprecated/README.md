# Historical robot experiments

Archived on 2026-09-07 by user request. Deprecated means inactive, not invalid.

- `results/`: inactive raw run directories, metrics, configs, videos and retained
  training checkpoints (still nested inside their originating run).
- `checkpoints/`: downloaded historical reference models, not deleted.
- `launchers/`: historical shell launch/resume/queue scripts. Do not execute
  blindly: some refer to intermediate checkpoints removed in this cleanup.
- `configs/`: historical Long/40-task/130-task local profiles.
- `notes/`: historical handoffs and diagnostics; status text is time-specific.

Old paths have compatibility symlinks so retained artifact references still work.
Archive moves are reversible; deleted intermediate checkpoints have no verified
backup and are not recoverable here. No results, metrics or videos were deleted.

Python helper modules remain in `scripts/`: active jobs and reusable evaluators
import dated runtime/diagnostic modules. Moving those requires a later dependency
refactor. Core PPO/CSD code stays in place. Active Spatial output and LIBERO SFT,
and the ManiSkill SFT initialization, remain at their original locations.

Authoritative journal:
[archive_execution_20260907.json](../research/archive_execution_20260907.json).
See [the project guide](../READ_THIS_FIRST.md). Heavy archive directories are
ignored by Git; publishing this README does not publish results or weights.
