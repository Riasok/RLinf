# PPO/CSD diagnosis, 2026-09-05

## 21:35 follow-up

The 31-cell screening panel and both official pi0 evaluations completed.
Official pi0 Spatial PPO: standard 86.4% (500 trials), installed Plus Spatial
32.89% (2,402 variants, one trial each, suffix remapping disabled).
The separately resumed gated-CSD trainer was stopped again at user request.

`scripts/run_csd_confirmation.py` runs eight 500-trial checks on GPUs 4-5 and
6-7 only: SFT Long K=10 with 3/5 denoising steps; PPO/CSD step-10 Goal and Long
with K=5; PPO/CSD step-10 Long with K=10 as an inference ablation. Historical
500-trial SFT K=5/N=3 results supply the reference with the same 20-env/25-epoch
protocol. Output: logs/20260905-csd-confirmation. No training restart is queued.
The watchdog's restart targets now use GPUs 4-7 exclusively.

The positive-advantage-gated beta=1 trainer was stopped at the user's request.
The last observed completed iteration was 18; the last saved checkpoint is 10.
All saved checkpoints and prior logs are retained. No training restart is queued.

## Confirmed implementation fixes

In `attach_online_csd_pairs`, the rollout stores T+1 done chunks: index zero
precedes the actions and index t+1 contains the outcomes of action chunk t.
Pair construction and the success gate had used index t instead, shifting
episode boundaries. Both now use the outcome chunk.

PPO's chunk-level loss mask discards within-chunk termination positions.
Passing that mask to CSD expanded a single true flag to every teacher action,
including actions after termination. CSD now reconstructs an action-level mask
from outcome dones and combines it with PPO validity. The terminating action
is included; subsequent actions in that chunk are excluded. A new episode in
a later chunk can still be used when auto-reset is enabled.

Regression coverage includes a partially executed teacher chunk and a pair
crossing a reset, in addition to the existing student-only gradient tests.
The focused suite passed 17 tests after these changes.

## Diagnostic order

`scripts/run_csd_diagnosis.py` runs two isolated evaluation lanes on GPUs 4-5
and 6-7. Each cell uses the same first ten reset trials for each of ten tasks,
giving 100 episodes, with 20 environments and five rollout epochs. These are
screening results; compare cells within this panel, not directly to prior
500-trial estimates as if they were paired measurements.

1. Long SFT: execution interval 5 versus 10 crossed with denoising steps 3
   versus 5. Keep the 520-step time limit identical across these four cells.
   This isolates inference effects; it is not an exact reproduction of the
   bundled Long training recipe, whose time limit is 480.
2. SFT baselines for Spatial/Object/Goal, then PPO and CSD checkpoints 10, 20,
   and 80 on all four suites with execution interval 5 and three denoising steps.
   Including step 80 provides the same-trial anchor for the historical result.
3. Interpret the panel before choosing full 500-trial confirmation evaluations
   or a reference PPO training run. Do not select beta from auxiliary loss alone.

There are 31 cells. Each writes its command, resolved Hydra config, launcher
log, TensorBoard events and result.json under logs/20260905-csd-diagnosis.
The driver fails on unsuccessful processes, missing success metrics, or an
episode count other than 100. It skips cells with an existing result.json.
It never launches training. Driver log: /tmp/csd-diagnosis-20260905.log.
Detached session: csd-diagnosis-20260905.

## Interpretation corrections

Runner steps count rollout/update iterations, not individual optimizer steps.
The actor can perform several optimizer steps per iteration; its learning-rate
scheduler advances once per iteration. Earlier run documentation referring to
100 runner steps as 100 optimizer updates should not be used for compute matching.

The PPO baseline also degrades under the historical evaluation, so CSD masking
cannot explain the entire loss of performance. Investigate inference, PPO
recipe changes and multitask interference separately. Diagnostic-only CSD also
consumes actor RNG despite having no gradient; it is not guaranteed bitwise
identical to PPO without diagnostics.
