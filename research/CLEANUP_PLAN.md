# Cleanup proposal — approval required

UPDATE: the user subsequently approved archival and pruning. Execution completed:
65 moves, 24 intermediate checkpoint deletions, 454.86 GiB removed. Retention is
more conservative than the original proposal: Long saves 10/90 and four-task-suite
saves 10/20/80 remain, plus gated10, newLong10 and ManiSkill50. Results and retained
checkpoints moved into deprecated with compatibility symlinks. Active Spatial
and shared Python helpers remain untouched. See archive_execution_20260907.json.
The remainder below records the ORIGINAL proposal, not current disk state.

Inventory date: 2026-09-07. This document proposes changes; no moves, checkpoint
deletions, commits or pushes are authorized by this file.

## Disk snapshot

The shared `/data` filesystem is 96% used, with about 404 GiB available at scan.
This is shared storage, not all owned by this project.

| Local area | Allocated size, rounded | Recommendation |
|---|---:|---|
| RLinf/logs | 741 GiB | Preserve evidence; selectively prune weights after approval |
| Saved training checkpoint directories inside logs | 701.25 GiB, 37 checkpoints | Main space-recovery opportunity |
| RLinf/checkpoints | 50 GiB | Preserve base/reference models for now |
| RLinf/datasets | 33 GiB | Preserve dependencies; no deletion proposed |
| RLinf/.runtime_cache | 33 GiB | Mostly HF dataset cache; check reuse before any pruning |
| RLinf virtualenvs | roughly 34 GiB | Preserve working runtimes |
| vOPD | roughly 94 GiB | Separate reasoning project; leave untouched |

The vOPD directory layout changed during inspection (the former top-level
vOPD_reference appeared under vOPD). This cleanup did not perform that move.
Do not reorganize that separate project without a separate inventory/approval.

## Proposed classification: what is what

| Family / files | Purpose | Proposed treatment |
|---|---|---|
| run_spatial73_20260907.py; matching Spatial YAML and SPATIAL73 handoff | Active 7/3 PPO experiment | KEEP ACTIVE, paths unchanged |
| prepare_maniskill_pi05.py; eval_maniskill_cut_20260907.py; queue_maniskill_ood_20260907.py | Working ManiSkill reference and evaluation | KEEP; later parameterize task, run ID and checkpoint |
| run_pirl130_20260906.py | Old PPO130 experiment AND shared runtime helpers | KEEP IN PLACE until helpers extracted; active Spatial and ManiSkill import it |
| run_csd_diagnosis.py | CSD investigation AND reusable evaluation helper | KEEP IN PLACE until helper extraction; Plus/Long/official eval import it |
| run_local_plus_spatial_20260906.py; run_official_spatial_eval.py; lerobot_libero_eval_entry.py | Local and official baseline Plus/Spatial evaluation | RETAIN as reference; make reusable before publication |
| run_pirl_long10_20260906.py; run_overnight_20260906.py | Completed/canceled Long and overnight orchestration | Candidate deprecated/launchers after dependency audit |
| run_csd_confirmation.py; watch_csd_sanity.py | Historical CSD diagnostics/monitoring | Candidate deprecated/csd, keep diagnostic evidence |
| root run_libero_long_*.sh; run_libero_40_*.sh; run_robot_eval_then_goal_queue.sh | Historical Long/40-task/job queues | Candidate deprecated/launchers; inspect each before moves |
| root stop_runs_at_step80.sh; switch_to_csdgate_at_step80.sh; resume_csdgate_padv_from10.sh; run_step80_eval*.sh | One-off stop/switch/resume orchestration | Candidate deprecated/launchers; not generic instructions |
| local Long/40-task/CSD/gated-CSD/queued-Goal YAMLs | Historical experiment profiles | Candidate deprecated/configs with dependency-preserving references |
| CSD_DIAGNOSIS, LOCAL_PLUS_SPATIAL, OVERNIGHT, PIRL_130, PIRL_LONG10 dated notes | Experiment history and caveats | Candidate deprecated/notes; preserve links and corrected status |
| rlinf/envs/libero/libero_env.py | Includes Plus/PRO malformed-BDDL filtering | KEEP; document skipped variants and add focused tests before push |
| rlinf/scheduler/cluster/cluster.py | Local Ray CPU/object-store memory caps | KEEP; active resource-safety support |
| openpi_action_model.py; fsdp_actor_worker.py; test_online_ppo_csd.py | Shared PPO path plus CSD/gating modifications | KEEP; do NOT move core files or remove CSD opportunistically |
| upstream examples, algorithms, docs, tests | General RLinf framework | KEEP; unrelated to current experiment does not mean deprecated |
| third_party/lerobot_latest | Local third-party dependency source | Review pin/license/local changes; do not blindly stage or move |

Proposed eventual layout (not created by moving files yet):

```text
READ_THIS_FIRST.md
research/                 # protocols, artifact/result registry, cleanup decisions
scripts/                  # maintained launch/eval entrypoints and shared helpers
deprecated/
  README.md               # archive map, reasons, replacement paths
  launchers/
  configs/
  notes/
logs/                     # ignored; raw evidence stays in place initially
checkpoints/              # ignored; downloaded base/reference models
```

Use registry status `archived` for historical results/checkpoints rather than
physically moving them now. Hardcoded paths, imports, W&B references and resume
metadata can break on moves. Archiving on the same disk saves zero bytes.

## Checkpoint approval options

Every checkpoint averages 18.95 GiB (weights plus training state). The exact
families and checkpoint directory roots are in RESULTS_AND_CHECKPOINTS.md.

Conservative retention proposal:

- KEEP active Spatial outputs and its LIBERO SFT initialization.
- KEEP ManiSkill step 50: only saved trained checkpoint, used in all new evals.
- KEEP new Long step 10: evaluated short-run reference.
- KEEP historical Long PPO and CSD step 90, one final evaluated checkpoint each.
- KEEP historical 40-task PPO-diagnostic and CSD step 80, one evaluated checkpoint each.
- KEEP gated-CSD step 10 pending an explicit decision about revisiting that idea.
- KEEP all metrics/configs/split manifests even when weights are later removed.

Possible staged deletion, NOT performed:

| Option | Exact step selection within the registered roots | Approximate recovery |
|---|---|---:|
| Small first batch | Long PPO and Long CSD steps 10,20,30 only (6 directories) | 114 GiB |
| Full historical-intermediate pruning | Long PPO/CSD steps 10,20,30,40,50,60,70,80; 40-task PPO/CSD steps 10,20,30,40,50,60,70 (30 directories total) | 569 GiB |

These options overlap; do not add their recoveries. Intermediate checkpoints may
include a best-performing step not yet identified by a full training-curve audit.
Before deletion, check that possibility, scheduled dependencies, and backup
availability; present the expanded absolute-path list for approval. Local trained
weights are not assumed recoverable from Hugging Face. Removing optimizer state
alone also forfeits exact resume and needs its own approval.

Historical rollout videos are another roughly 40 GiB candidate, but preserve a
small failure/success sample before asking to remove them. No video deletion yet.

## Publication gates

1. User approves classification and archive structure; checkpoint deletion is a
   separate decision, not bundled with GitHub publication.
2. Wait for a safe point before refactoring anything used by the running job.
3. Extract shared runtime/evaluation utilities; parameterize machine-specific
   paths, GPU IDs, output directories, models and W&B project. Existing launchers
   mutate configs and import dated experiment modules; they are not clean APIs.
4. Preserve resolved historical configs; mark failed/canceled/partial runs. Export
   small sanitized result tables with provenance, not raw artifact directories.
5. Review existing dirty changes, do not overwrite them. Extend ignore rules for
   `.venv-*`, artifact/archive weights and credentials as needed. Never `git add .`.
6. Audit staged paths for secrets and large files without displaying credential
   values. Review third-party licensing/pins. No `.env`, weights or raw W&B uploads.
7. Run targeted tests for task filtering, padding exclusion, variant filtering,
   Ray overrides, PPO/CSD compatibility; lint changed code and validate docs.
8. Review a dedicated branch and explicit file list, make signed-off commits,
   then obtain approval for push to the user's origin (not upstream).

Current delivery is documentation only; no launcher refactor or runtime change.
