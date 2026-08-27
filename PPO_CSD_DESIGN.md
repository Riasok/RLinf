# PPO-CSD: method and experimental protocol

## Summary

PPO-CSD augments on-policy PPO for a flow-based VLA with a consistency
self-distillation loss on the physically aligned overlap between consecutive
action chunks. The later prediction is the detached teacher because it is
conditioned on a more recent observation. The earlier prediction is trained as
the student with OpenPI's native conditional flow-matching objective.

The method is meaningful only when the policy predicts more actions than the
environment executes before replanning. If the prediction horizon is `H` and
the execution/replanning interval is `K`, the required invariant is:

```text
0 < K < H
csd_offset == num_action_chunks == K
number of aligned actions == H - K
```

For the active `pi05_libero` model, `H=10`. A valid PPO-CSD experiment must
therefore use `K<10`; the recommended setting is `K=5`, giving five aligned
actions per consecutive pair.

## Motivation

A chunked policy predicts future actions from observation `o_t`, executes only
the first `K` actions, observes the environment again at `o_{t+K}`, and then
replans. Consecutive predictions should agree about their shared physical time
indices, while the later prediction has better information about the state the
robot actually reached.

CSD regularizes this local temporal inconsistency without differentiating
through the environment, stored rollout actions, or the iterative denoising
sampler. PPO remains responsible for reward optimization; CSD supplies a dense
self-supervised gradient on on-policy trajectories.

## Consecutive-chunk alignment

Let each policy prediction contain `H` actions:

```text
A_t       = [a_t, ..., a_{t+H-1}]       conditioned on o_t
A_{t+K}   = [a_{t+K}, ..., a_{t+K+H-1}] conditioned on o_{t+K}
```

After executing `K` actions, the aligned overlap is:

```text
student indices: A_t[K:H]
teacher indices: A_{t+K}[0:H-K]
```

With the recommended OpenPI LIBERO setting, `H=10` and `K=5`, so the earlier
suffix `A_t[5:10]` aligns with the later prefix `A_{t+5}[0:5]`.

Pairs are constructed before PPO flattens and shuffles the rollout. A pair is
valid only when:

- both consecutive chunks pass the PPO loss mask;
- the earlier chunk does not terminate the episode; and
- a real next chunk exists.

The last chunk in a trajectory and pairs that cross a termination are excluded.
The current implementation stores only the shifted later action as the teacher;
it does not retain a second teacher observation or a live teacher graph.

## Native flow-matching target

Create a detached pseudo-target by copying the earlier sampled action and
replacing only the aligned suffix with the later action prefix:

```text
A_tilde = stopgrad(A_t)
A_tilde[K:H] = stopgrad(A_{t+K}[0:H-K])
```

For sampled Gaussian noise `epsilon` and flow time `tau`, use OpenPI's native
flow path and analytic velocity target:

```text
x_tau = tau * epsilon + (1 - tau) * A_tilde
u_tau = epsilon - A_tilde
```

The model recomputes a student velocity conditioned on the earlier observation.
The CSD loss is MSE only on the overlapping environment-action dimensions:

```text
L_CSD = mean(
  ||v_theta(x_tau, tau | o_t)[K:H, :D_env]
    - stopgrad(epsilon[K:H, :D_env]
               - A_{t+K}[0:H-K, :D_env])||^2
)
```

This is an MSE numerically, but it is not raw action regression. It is
conditional flow matching against the model's analytic velocity target. Raw
action MSE would require backpropagating through sampling or would average
stochastic/multimodal action samples. Matching two live velocity graphs would
also cost a teacher forward and make the gradient boundary less clear.

## Gradient flow

The combined objective is:

```text
L_total = L_PPO + beta * L_CSD
```

Gradients flow through the recomputed student velocity and its conditioning on
`o_t` into the current policy parameters. Gradients do not flow through:

- the earlier sampled rollout action;
- the later sampled teacher action;
- the later observation;
- flow noise or the analytic target; or
- environment transitions.

The actor performs PPO backward first and CSD backward second, then takes one
shared optimizer step. Both terms use the same gradient-accumulation divisor.
Consequently, `beta=0.05` does not mean that CSD contributes exactly five
percent of the final gradient; the relative effect depends on the two loss and
gradient scales.

Each actor microbatch uses up to `csd_micro_batch_size` valid pairs. The current
setting is `2`. If no valid pair exists, the CSD contribution is exactly zero.
The current implementation selects the first valid pairs rather than sampling
them randomly; this is a known limitation worth removing in a later revision.

## Implementation map

- Pair construction and episode mask:
  `rlinf/workers/actor/fsdp_actor_worker.py::attach_online_csd_pairs`
- PPO and CSD backward passes:
  `rlinf/workers/actor/fsdp_actor_worker.py::train_micro_batch`
- Flow target and overlap loss:
  `rlinf/models/embodiment/openpi/openpi_action_model.py::online_csd_forward`
- Deterministic rollout seeding:
  `rlinf/workers/rollout/hf/huggingface_worker.py::init_worker`
- Unit coverage:
  `tests/unit_tests/test_online_ppo_csd.py` and
  `tests/unit_tests/test_rollout_rng_seed.py`

Online CSD currently supports only OpenPI. Runtime validation requires positive
`csd_beta`, positive `csd_micro_batch_size`, and
`csd_offset == actor.model.num_action_chunks`. Model-level validation also
requires `0 < csd_offset < action_horizon`.

## Metrics

Log these metrics for every PPO-CSD step:

- `train/actor/csd_loss`: raw conditional flow-matching MSE;
- `train/actor/csd_weighted_loss`: `beta * csd_loss`;
- `train/actor/csd_pair_count`: valid pairs used per microbatch;
- `train/actor/csd_action_discrepancy`: raw overlap action MSE for diagnosis;
- `train/actor/csd_flow_target_norm`;
- `train/actor/csd_student_velocity_norm`;
- `train/actor/grad_norm` and `train/actor/total_loss`;
- PPO policy/value losses, entropy, KL diagnostics, and environment return.

Loss magnitudes alone are not enough to tune `beta`. The best diagnostic is the
ratio and cosine similarity between PPO-only and CSD-only parameter gradients,
which the implementation does not yet log. Until that instrumentation exists,
use stability, total gradient norm, CSD discrepancy, and held-out return jointly.

In an earlier Spatial run, `beta=0.05` produced mean raw CSD loss around `0.324`
and mean weighted loss around `0.016`. It was conservative but nonzero. A
`beta=0.5` arm is a useful 10x stress test, not the default treatment.

## Controlled training protocol

Train one joint policy per LIBERO suite and method. `libero_10` contains ten
tasks, so one run samples all ten tasks; do not launch ten task-specific PPO
policies unless the research question is task specialization.

Use the same starting checkpoint, task allocation, reset states, rollout seed,
batch sizes, action execution interval, optimizer, and training budget for every
matched arm. For the valid Long comparison, use:

```text
Model:                  RLinf-Pi05-LIBERO-SFT
Suite:                  standard LIBERO-Long / libero_10
Prediction horizon H:   10
Executed actions K:     5 for every matched arm
CSD offset:             5
Tasks:                  10 in one joint policy
Parallel environments:  64
Rollout epochs:         8
Episode horizon:        480
PPO update epochs:      4
Actor microbatch:       128
Global batch:           2048
PPO clip:               0.2
Gamma / GAE lambda:     0.99 / 0.95
Actor LR / value LR:    5e-6 / 1e-4
LR schedule:            1000-step cosine schedule, minimum rate 0.1
Experiment cap:         100 optimizer steps
Checkpoint interval:    every 10 steps
Primary seed:           42
CSD pair microbatch:    2
```

`K=5` is a method-enabling shared change from the official pi-RL Long recipe,
which uses `K=10`. Apply it to PPO and every PPO-CSD arm so the comparison is
controlled. An exact pi-RL `K=10` PPO run can be retained as a reproduction
reference, but it is not the matched baseline for an overlap-CSD treatment.

Recommended arms:

1. Matched PPO with `K=5` and CSD disabled.
2. Matched PPO-CSD with `K=5`, `csd_offset=5`, and `beta=0.05`.
3. Optional high-CSD diagnostic with the same settings and `beta=0.5`.

Run the high-CSD arm only through the first checkpoint initially. Continue it
only if PPO metrics are stable and CSD does not dominate the gradient. Once a
reasonable beta is selected, additional paired seeds are more valuable than
more beta values. Target three paired seeds for the final comparison.

## Deterministic paired rollouts

LIBERO reset-state sampling is deterministic, but Flow-SDE action exploration
also uses Python, NumPy, and Torch random state inside separate Ray workers.
Model construction and CUDA graph capture can consume random numbers before the
first rollout.

Reset every rollout worker's RNG after initialization with:

```text
rollout_seed = actor.seed + rollout_worker_rank
```

Matched jobs therefore use seeds `42` and `43` for rollout ranks zero and one.
This controls the first stochastic rollout across PPO and PPO-CSD. The methods
should diverge only after their first different optimizer update, subject to
low-level GPU nondeterminism.

## Evaluation protocol

Standard LIBERO-Long is the in-distribution training and comparison benchmark.
It is not sufficient evidence of robust generalization.

Evaluate the SFT checkpoint, PPO, and the selected PPO-CSD checkpoints on:

1. Standard LIBERO-Long with identical fixed reset states.
2. Held-out LIBERO-Plus perturbations, prioritizing camera viewpoint, robot
   initial state, object layout, and sensor noise.
3. LIBERO-Pro as a later confirmatory benchmark for object, position, semantic,
   task, and environment shifts.

Do not train on the same Plus/Pro perturbations used for the headline held-out
evaluation. If robustness adaptation is a separate research question, define
explicit disjoint perturbation types or severity levels for training and test.

Evaluate at step 0, step 10, and the final selected checkpoint. Report the mean,
per-task results, perturbation-category results, confidence intervals, and all
seeds. Select beta using a development split rather than the final test set.

## Experiment status on 2026-08-23

The first Long launch copied the official pi-RL action execution setting:
`H=10`, `K=10`, and `csd_offset=10`. This produces no physical overlap. The two
CSD jobs (`beta=0.05` and `beta=0.5`) were stopped during their first rollout,
before any optimizer update or checkpoint, after the incompatibility was found.
Their W&B histories are launch diagnostics only and are not experimental
results.

The exact pi-RL PPO `K=10` run remains useful as a reproduction reference. A
valid matched PPO/PPO-CSD comparison still requires new `K=5` configs and fresh
runs from the same SFT checkpoint.

## Known limitations

- CSD requires overlapping replanning (`K<H`); it cannot be added to a policy
  that executes its entire prediction horizon before observing again.
- The later action is a self-generated pseudo-target, not an oracle action.
- Only two valid pairs per actor microbatch are currently used.
- Valid pairs are chosen deterministically from the front of the microbatch.
- Separate PPO-only and CSD-only gradient norms/cosine similarity are not logged.
- A single seed is suitable for debugging, not a final performance claim.
- Standard LIBERO can reward memorized trajectories; held-out perturbation
  evaluation is required.

## References

- pi-RL: <https://arxiv.org/abs/2510.25889>
- Official RLinf pi-RL Long config:
  <https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml>
- LIBERO-Plus: <https://arxiv.org/abs/2510.13626>
- LIBERO-Pro: <https://arxiv.org/abs/2510.03827>
- Robot manipulation benchmark audit: <https://arxiv.org/abs/2606.04233>
